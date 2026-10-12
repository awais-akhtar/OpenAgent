"""Actual Q5_0 scalar and JIT execution against independent byte decoding."""

from contextlib import contextmanager
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import expert_kernels as contracts
from openagent.adaptive_engine import native_q5_0_expert as native
from openagent.adaptive_engine import q5_0_expert as reference
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore
from openagent.adaptive_engine.q5_0_conversion import convert_experts
from tests.test_q5_0_expert import oracle_decode


class CountingSource:
    def __init__(self, payloads):
        self.payloads = payloads
        self.active = self.maximum = self.calls = 0

    @contextmanager
    def acquire(self, key):
        self.calls += 1
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            yield self.payloads[key]
        finally:
            self.active -= 1


def backends():
    result = [reference.Q5_0ExpertBackend()]
    if native._compiled_kernel is not None:
        result.append(native.NativeQ5_0ExpertBackend())
    return result


class Q5ExecutionTests(unittest.TestCase):
    def test_seeded_handbuilt_full_expert_differential_routes_and_batches(self):
        rng = np.random.default_rng(5015)
        for h, i in ((32, 32), (64, 96), (96, 64)):
            packed, floating = {}, {}
            for key in ("alpha", "beta"):
                blob = b"".join(struct.pack("<e", float(rng.choice([.003, -.005, -0., 2 ** -24])))
                                + rng.integers(0, 256, 20, dtype=np.uint8).tobytes()
                                for _ in range(3 * h * i // 32))
                packed[key] = blob
                floating[key] = b"".join(matrix.astype("<f4").tobytes() for matrix in oracle_decode(blob, h, i))
            x = rng.normal(0, .2, (4, h)).astype(np.float32)
            routes = [[("alpha", .25), ("beta", .75)], [("alpha", 0)], [("beta", 1)],
                      [("beta", 2), ("alpha", .1)]]
            args = dict(hidden_size=h, intermediate_size=i)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource(floating), x,
                                                              routes, dtype="float32", **args)
            scalar = None
            for backend in backends():
                source = CountingSource(packed)
                actual = backend.execute_batch(source, x, routes, **args)
                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-7)
                np.testing.assert_array_equal(backend.execute(source, x[0], routes[0], **args), actual[0])
                self.assertEqual((source.active, source.maximum), (0, 1))
                if scalar is None:
                    scalar = actual
                else:
                    np.testing.assert_allclose(actual, scalar, rtol=5e-6, atol=1e-9)

    def test_each_high_bit_and_extreme_delta_projection(self):
        import_numba = native._compiled_kernel is not None
        compiled_dot = None
        if import_numba:
            import numba
            compiled_dot = numba.njit(cache=False, fastmath=False)(native._packed_dot)
        for lane in range(32):
            for delta in (65504., -65504., 2 ** -24, -(2 ** -24), 0., -0.):
                high = bytearray(4)
                high[lane // 8] = 1 << (lane % 8)
                block = struct.pack("<e", delta) + high + bytes(16)
                payload = bytes(22) + block
                raw = np.frombuffer(payload, np.uint8)
                for selected in (lane, (lane + 1) % 32):
                    x = np.zeros(32, np.float32)
                    x[selected] = 1
                    expected = np.float32((0 if selected == lane else -16) * delta)
                    with memoryview(payload) as view:
                        self.assertEqual(reference._dot(view, 22, x, 32), expected)
                    self.assertEqual(native._packed_dot(raw, 22, x, 32), expected)
                    if compiled_dot is not None:
                        self.assertEqual(compiled_dot(raw, 22, x, 32), expected)

    def test_every_binary16_bit_pattern(self):
        reader = native._read_half
        if native._compiled_kernel is not None:
            import numba
            reader = numba.njit(cache=False, fastmath=False)(reader)
        for bits in range(65536):
            expected = struct.unpack("<e", struct.pack("<H", bits))[0]
            actual = reader(np.array([bits & 255, bits >> 8], np.uint8), 0)
            if math.isfinite(expected):
                self.assertEqual(actual, expected)
                if expected == 0:
                    self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
            else:
                self.assertTrue(math.isnan(actual))

    def test_prelease_validation_and_exact_scratch_limit(self):
        blob = reference.pack_expert(*[np.ones((32, 32))] * 3)
        args = dict(hidden_size=32, intermediate_size=32)
        self.assertEqual(reference.scratch_bytes(1024, 32), 128)
        for backend in backends():
            source = CountingSource({"a": blob})
            small = type(backend)(max_scratch_bytes=127)
            with self.assertRaises(MemoryError):
                small.execute(source, np.ones(32), [("a", 1)], **args)
            for inputs in (np.full(32, np.nan), np.ones(31), np.ones(32, complex)):
                with self.assertRaises(ValueError):
                    backend.execute(source, inputs, [("a", 1)], **args)
            for routes in ([], [("a", -1)], [("a", 1e100)], [("a", float("nan"))],
                           [("a", 1), ("a", 1)], [("a", True)]):
                with self.assertRaises(ValueError):
                    backend.execute(source, np.ones(32), routes, **args)
            with self.assertRaises(ValueError):
                backend.execute(source, np.ones(32), [("a", 1)], dtype="q4_0", **args)
            with self.assertRaises(ValueError):
                backend.execute_batch(source, np.ones((2, 32)), [[("a", 1)]], **args)
            self.assertEqual(source.calls, 0)
            self.assertEqual(backend.execute_batch(source, np.empty((0, 32)), [], **args).shape, (0, 32))
            limited = type(backend)(max_scratch_bytes=128)
            self.assertTrue(np.isfinite(limited.execute(source, np.zeros(32), [("a", 1)], **args)).all())

    def test_malformed_nonfinite_zero_routes_and_arithmetic_overflow(self):
        blob = reference.pack_expert(*[np.ones((32, 32))] * 3)
        args = dict(hidden_size=32, intermediate_size=32)
        for backend in backends():
            for bad in (blob[:-1], blob + b"x", memoryview(blob)[::2]):
                source = CountingSource({"a": bad})
                with self.assertRaises(ValueError):
                    backend.execute(source, np.ones(32), [("a", 0)], **args)
                self.assertEqual(source.active, 0)
            for position in (0, len(blob) - 22):
                for bits in (0x7c00, 0xfc00, 0x7e00):
                    bad = bytearray(blob)
                    struct.pack_into("<H", bad, position, bits)
                    source = CountingSource({"a": bad})
                    with self.assertRaisesRegex(ValueError, "finite"):
                        backend.execute(source, np.ones(32), [("a", 0)], **args)
                    self.assertEqual(source.active, 0)
            source = CountingSource({"a": blob})
            for value in (np.finfo(np.float32).max, 1e18):
                with self.assertRaises(ArithmeticError):
                    backend.execute(source, np.full(32, value), [("a", 1)], **args)
                self.assertEqual(source.active, 0)
            with self.assertRaises(ArithmeticError):
                backend.execute(source, np.ones(32), [("a", float(np.finfo(np.float32).max))], **args)
            np.testing.assert_array_equal(backend.execute(source, np.full(32, -1e18), [("a", 1)], **args), 0)

    def test_converter_actual_disk_leases_and_retained_exception_cleanup(self):
        args = dict(hidden_size=32, intermediate_size=32)
        weights = [np.full((32, 32), .1)] * 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = convert_experts({"a": weights}, root / "experts.q50", **args)
            size = reference.payload_size(32, 32)
            for backend in backends():
                for mode in ("buffered", "mmap"):
                    with DiskExpertStore(root, result.regions, capacity_bytes=size, io_mode=mode) as store:
                        actual = backend.execute(store, np.ones(32), [("a", 1)], **args)
                        self.assertTrue(np.isfinite(actual).all())
                        for fail_kind in ("overflow", "injected"):
                            retained = None
                            originals = []
                            def fail(raw, *unused):
                                view = np.frombuffer(raw, np.uint8)[::2]
                                assert view.size
                                try:
                                    raise ValueError("inner borrowed buffer")
                                except ValueError as cause:
                                    originals.append(cause)
                                    raise RuntimeError("injected kernel failure") from cause
                            target = "_scalar_expert" if isinstance(backend, reference.Q5_0ExpertBackend) and not isinstance(
                                backend, native.NativeQ5_0ExpertBackend) else "_compiled_kernel"
                            module = reference if target == "_scalar_expert" else native
                            context = patch.object(module, target, fail) if fail_kind == "injected" else patch.object(
                                module, target, getattr(module, target))
                            with context:
                                try:
                                    backend.execute(store, np.full(32, 1e30) if fail_kind == "overflow" else np.ones(32),
                                                    [("a", 1)], **args)
                                except (ArithmeticError, RuntimeError) as error:
                                    retained = error
                            self.assertIsNotNone(retained)
                            self.assertIsNone(retained.__cause__)
                            self.assertIsNone(retained.__context__)
                            self.assertEqual(store.status()["active_leases"], 0)
                            if originals:
                                self.assertIsNone(originals[0].__traceback__)
                        store.close()
                    self.assertTrue(store.status()["closed"])

    @unittest.skipUnless(native._compiled_kernel is not None, "active Numba required")
    def test_actual_compiled_signatures_no_weight_allocations_no_fastmath(self):
        blob = reference.pack_expert(*[np.zeros((32, 32))] * 3)
        native.NativeQ5_0ExpertBackend().execute(MappingPayloadSource({"a": blob}), np.ones(32),
                                               [("a", 1)], hidden_size=32, intermediate_size=32)
        self.assertTrue(native._compiled_kernel.nopython_signatures)
        self.assertFalse(native._compiled_kernel.targetoptions["fastmath"])
        self.assertTrue(native._compiled_kernel.targetoptions["nogil"])
        for signature in native._compiled_kernel.signatures:
            self.assertNotIn("NRT_MemInfo_alloc", native._compiled_kernel.inspect_llvm(signature))

    def test_explicit_custom_registry_after_parent_dtype_integration(self):
        if "q5_0" not in contracts.REGISTERED_PAYLOAD_DTYPES:
            self.skipTest("parent q5_0 dtype integration pending; numerical tests execute directly")
        for backend in backends():
            registry = contracts.ExpertKernelRegistry()
            registry.register(backend, replace=True)
            self.assertIs(registry.resolve(dtype="q5_0", preferred=backend.NAME, allow_fallback=False), backend)
            self.assertEqual(backend.info().capabilities.compiled, isinstance(backend, native.NativeQ5_0ExpertBackend))
            self.assertFalse(backend.info().capabilities.native_simd)

    def test_disabled_or_missing_numba_never_fakes_native(self):
        code = """
import importlib.abc
import os
import sys
if os.environ.get('Q5_TEST_NO_NUMBA'):
    class BlockNumba(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'numba' or fullname.startswith('numba.'):
                raise ModuleNotFoundError('numba intentionally unavailable')
    sys.meta_path.insert(0, BlockNumba())
import numpy as np
from openagent.adaptive_engine.q5_0_expert import Q5_0ExpertBackend, pack_expert
from openagent.adaptive_engine.native_q5_0_expert import NativeQ5_0ExpertBackend, _compiled_kernel
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, REGISTERED_PAYLOAD_DTYPES
assert _compiled_kernel is None
backend = NativeQ5_0ExpertBackend()
if 'q5_0' in REGISTERED_PAYLOAD_DTYPES:
    assert not backend.info().available
    assert not backend.info().capabilities.compiled
try:
    backend.execute_batch(None, [], [], hidden_size=32, intermediate_size=32)
except RuntimeError:
    pass
else:
    raise AssertionError('native silently fell back')
blob = pack_expert(*[np.zeros((32, 32))] * 3)
actual = Q5_0ExpertBackend().execute(MappingPayloadSource({'a': blob}), np.ones(32),
                                  [('a', 1)], hidden_size=32, intermediate_size=32)
assert np.all(actual == 0)
"""
        for overrides in ({"NUMBA_DISABLE_JIT": "1"}, {"Q5_TEST_NO_NUMBA": "1"}):
            process = subprocess.run([sys.executable, "-W", "error", "-c", code],
                                     env=dict(os.environ, **overrides), capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
