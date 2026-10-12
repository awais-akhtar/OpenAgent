"""Affine Q5_1 scalar/native differentials, leases, validation and unavailable JIT."""

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
from openagent.adaptive_engine import q5_1_expert as scalar
from openagent.adaptive_engine import native_q5_1_expert as native
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore
from openagent.adaptive_engine.q5_1_conversion import convert_experts
from tests.test_q5_1_expert import oracle_decode


class ObservedSource:
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


def active_backends():
    result = [scalar.Q5_1ExpertBackend()]
    if native._compiled_kernel is not None:
        result.append(native.NativeQ5_1ExpertBackend())
    return result


class Q5AffineExecutionTests(unittest.TestCase):
    def test_random_handbuilt_multiblock_nonsquare_expert_differential(self):
        rng = np.random.default_rng(51081)
        for h, i in ((32, 32), (64, 96), (96, 64)):
            packed, decoded = {}, {}
            for key in ("a", "b"):
                blob = b"".join(struct.pack("<ee", float(rng.choice([.002, -.003, -0., 2 ** -24])),
                                           float(rng.choice([-.02, .015, 0.])))
                                + rng.integers(0, 256, 20, dtype=np.uint8).tobytes()
                                for _ in range(3 * h * i // 32))
                packed[key] = blob
                decoded[key] = b"".join(matrix.astype("<f4").tobytes() for matrix in oracle_decode(blob, h, i))
            x = rng.normal(0, .2, (4, h)).astype(np.float32)
            routes = [[("a", .2), ("b", .8)], [("a", 0)], [("b", 1)], [("b", 2), ("a", .1)]]
            args = dict(hidden_size=h, intermediate_size=i)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource(decoded), x, routes,
                                                              dtype="float32", **args)
            baseline = None
            for backend in active_backends():
                source = ObservedSource(packed)
                actual = backend.execute_batch(source, x, routes, **args)
                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-7)
                np.testing.assert_array_equal(backend.execute(source, x[0], routes[0], **args), actual[0])
                self.assertEqual((source.active, source.maximum), (0, 1))
                if baseline is None:
                    baseline = actual
                else:
                    np.testing.assert_allclose(actual, baseline, rtol=5e-6, atol=1e-9)

    def test_all_codes_and_high_bit_impulses_affine_projection_with_metadata_boundaries(self):
        jit_dot = None
        if native._compiled_kernel is not None:
            import numba
            jit_dot = numba.njit(cache=False, fastmath=False)(native._packed_dot)
        cases = ((65504., -65504.), (-65504., 65504.), (2 ** -24, -(2 ** -24)),
                 (2 ** -24, 65504.), (0., -.75), (-0., 0.))
        for delta, minimum in cases:
            blocks = [(struct.pack("<eeI16B", delta, minimum, 0xffff0000,
                                   *[17 * j for j in range(16)]), np.arange(32))]
            for selected in range(32):
                high = bytearray(4)
                high[selected // 8] = 1 << (selected % 8)
                codes = np.zeros(32)
                codes[selected] = 16
                blocks.append((struct.pack("<ee", delta, minimum) + high + bytes(16), codes))
            for block, codes in blocks:
                raw = np.frombuffer(bytes(24) + block, np.uint8)
                for lane in range(32):
                    x = np.zeros(32, np.float32)
                    x[lane] = 1
                    expected = np.float32(float(codes[lane]) * delta + minimum)
                    with memoryview(bytes(24) + block) as view:
                        actual = scalar._dot(view, 24, x, 32)
                    self.assertEqual(actual, expected)
                    self.assertEqual(native._packed_dot(raw, 24, x, 32), expected)
                    if jit_dot is not None:
                        self.assertEqual(jit_dot(raw, 24, x, 32), expected)

    def test_affine_metadata_read_once_per_block_and_ordered_fp32_dot(self):
        blocks = [struct.pack("<eeI16B", -.125, .5, 0x13579bdf, *range(16)),
                  struct.pack("<eeI16B", 2 ** -24, -.25, 0x2468ace0, *range(240, 256))]
        blob = b"".join(blocks)
        x = np.resize(np.array([1e4, -1e4, .001, -.003, .25, -2.], np.float32), 64)
        expected = np.float32(0)
        weights = oracle_decode(blob * 48, 32, 32)[0].reshape(-1)[:64]
        for value, input_value in zip(weights, x):
            expected = np.float32(expected + np.float32(value * input_value))
        with patch.object(native, "_read_half", wraps=native._read_half) as reader:
            actual = native._packed_dot(np.frombuffer(blob, np.uint8), 0, x, 64)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual([call.args[1] for call in reader.call_args_list], [0, 2, 24, 26])

    def test_binary16_reader_reuse_every_pattern_both_fields(self):
        reader = native._read_half
        if native._compiled_kernel is not None:
            import numba
            reader = numba.njit(cache=False, fastmath=False)(reader)
        for bits in range(65536):
            expected = struct.unpack("<e", struct.pack("<H", bits))[0]
            raw = np.array([bits & 255, bits >> 8] * 2, np.uint8)
            for position in (0, 2):
                actual = reader(raw, position)
                if math.isfinite(expected):
                    self.assertEqual(actual, expected)
                    if expected == 0:
                        self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
                else:
                    self.assertTrue(math.isnan(actual))

    def test_invalid_routes_inputs_dtype_and_scratch_before_lease(self):
        blob = scalar.pack_expert(*[np.ones((32, 32))] * 3)
        args = dict(hidden_size=32, intermediate_size=32)
        self.assertEqual(scalar.scratch_bytes(1024, 32), 128)
        self.assertEqual(native.scratch_bytes(1024, 32), 128)
        for backend in active_backends():
            source = ObservedSource({"a": blob})
            with self.assertRaises(MemoryError):
                type(backend)(max_scratch_bytes=127).execute(source, np.ones(32), [("a", 1)], **args)
            for x in (np.ones(31), np.ones(32, complex), np.full(32, np.nan)):
                with self.assertRaises(ValueError):
                    backend.execute(source, x, [("a", 1)], **args)
            for routes in ([], [("a", -1)], [("a", float("nan"))], [("a", 1e100)],
                           [("a", True)], [("a", 1), ("a", 1)]):
                with self.assertRaises(ValueError):
                    backend.execute(source, np.ones(32), routes, **args)
            with self.assertRaises(ValueError):
                backend.execute(source, np.ones(32), [("a", 1)], dtype="q5_0", **args)
            with self.assertRaises(ValueError):
                backend.execute_batch(source, np.ones((2, 32)), [[("a", 1)]], **args)
            self.assertEqual(source.calls, 0)
            self.assertEqual(backend.execute_batch(source, np.empty((0, 32)), [], **args).shape, (0, 32))
            type(backend)(max_scratch_bytes=128).execute(source, np.zeros(32), [("a", 1)], **args)

    def test_nonfinite_both_fields_zero_routes_malformed_and_arithmetic_overflow(self):
        blob = scalar.pack_expert(*[np.ones((32, 32))] * 3)
        args = dict(hidden_size=32, intermediate_size=32)
        for backend in active_backends():
            for malformed in (blob[:-1], blob + b"x", memoryview(blob)[::2]):
                source = ObservedSource({"a": malformed})
                with self.assertRaises(ValueError):
                    backend.execute(source, np.ones(32), [("a", 0)], **args)
                self.assertEqual(source.active, 0)
            for offset in (0, 2, len(blob) - 24, len(blob) - 22):
                for bits in (0x7c00, 0xfc00, 0x7e00):
                    malformed = bytearray(blob)
                    struct.pack_into("<H", malformed, offset, bits)
                    source = ObservedSource({"a": malformed})
                    with self.assertRaisesRegex(ValueError, "finite"):
                        backend.execute(source, np.ones(32), [("a", 0)], **args)
                    self.assertEqual(source.active, 0)
            source = ObservedSource({"a": blob})
            for value in (np.finfo(np.float32).max, 1e18):
                with self.assertRaises(ArithmeticError):
                    backend.execute(source, np.full(32, value), [("a", 1)], **args)
                self.assertEqual(source.active, 0)
            with self.assertRaises(ArithmeticError):
                backend.execute(source, np.ones(32), [("a", float(np.finfo(np.float32).max))], **args)
            np.testing.assert_array_equal(backend.execute(source, np.full(32, -1e18), [("a", 1)], **args), 0)

    def test_real_disk_converter_and_retained_scalar_native_exception_cleanup(self):
        args = dict(hidden_size=32, intermediate_size=32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = convert_experts({"a": [np.full((32, 32), .1)] * 3}, root / "weights.q51", **args)
            for backend in active_backends():
                for mode in ("buffered", "mmap"):
                    with DiskExpertStore(root, result.regions, capacity_bytes=2304, io_mode=mode) as store:
                        actual = backend.execute(store, np.ones(32), [("a", 1)], **args)
                        self.assertTrue(np.isfinite(actual).all())
                        with self.assertRaises(ArithmeticError):
                            backend.execute(store, np.full(32, 1e30), [("a", 1)], **args)
                        inner = []
                        def fail(raw, *unused):
                            view = np.frombuffer(raw, np.uint8)[::2]
                            assert view.size
                            try:
                                raise ValueError("inner borrowed frame")
                            except ValueError as cause:
                                inner.append(cause)
                                raise RuntimeError("injected execution failure") from cause
                        module, name = (native, "_compiled_kernel") if isinstance(backend, native.NativeQ5_1ExpertBackend) else (
                            scalar, "_scalar_expert")
                        retained = None
                        with patch.object(module, name, fail):
                            try:
                                backend.execute(store, np.ones(32), [("a", 1)], **args)
                            except RuntimeError as error:
                                retained = error
                        self.assertIsNotNone(retained)
                        self.assertIsNone(retained.__cause__)
                        self.assertIsNone(retained.__context__)
                        self.assertIsNone(inner[0].__traceback__)
                        self.assertEqual(store.status()["active_leases"], 0)
                        store.close()
                    self.assertTrue(store.status()["closed"])

    @unittest.skipUnless(native._compiled_kernel is not None, "active optional Numba required")
    def test_actual_native_compilation_and_no_numeric_allocations(self):
        blob = scalar.pack_expert(*[np.zeros((32, 32))] * 3)
        native.NativeQ5_1ExpertBackend().execute(MappingPayloadSource({"a": blob}), np.ones(32), [("a", 1)],
                                               hidden_size=32, intermediate_size=32)
        self.assertTrue(native._compiled_kernel.nopython_signatures)
        self.assertFalse(native._compiled_kernel.targetoptions["fastmath"])
        self.assertTrue(native._compiled_kernel.targetoptions["nogil"])
        for signature in native._compiled_kernel.signatures:
            self.assertNotIn("NRT_MemInfo_alloc", native._compiled_kernel.inspect_llvm(signature))

    def test_custom_registry_when_parent_integrates_dtype(self):
        if "q5_1" not in contracts.REGISTERED_PAYLOAD_DTYPES:
            self.skipTest("parent q5_1 shared dtype integration pending")
        for backend in active_backends():
            registry = contracts.ExpertKernelRegistry()
            registry.register(backend, replace=True)
            self.assertIs(registry.resolve(dtype="q5_1", preferred=backend.NAME, allow_fallback=False), backend)
            self.assertEqual(backend.info().capabilities.compiled, isinstance(backend, native.NativeQ5_1ExpertBackend))
            self.assertFalse(backend.info().capabilities.native_simd)

    def test_missing_disabled_or_failed_jit_never_executes_scalar(self):
        code = """
import importlib.abc
import os
import sys
if os.environ.get('Q51_NO_NUMBA'):
    class NoNumba(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'numba' or fullname.startswith('numba.'):
                raise ModuleNotFoundError('intentionally unavailable')
    sys.meta_path.insert(0, NoNumba())
import numpy as np
from openagent.adaptive_engine import native_q5_1_expert as n
from openagent.adaptive_engine.q5_1_expert import Q5_1ExpertBackend, pack_expert
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, REGISTERED_PAYLOAD_DTYPES
assert n._compiled_kernel is None
backend = n.NativeQ5_1ExpertBackend()
if 'q5_1' in REGISTERED_PAYLOAD_DTYPES:
    assert not backend.info().available
    assert not backend.info().capabilities.compiled
try:
    backend.execute_batch(None, [], [], hidden_size=32, intermediate_size=32)
except RuntimeError:
    pass
else:
    raise AssertionError('native silently fell back')
blob = pack_expert(*[np.zeros((32,32))] * 3)
actual = Q5_1ExpertBackend().execute(MappingPayloadSource({'a':blob}), np.ones(32), [('a',1)],
                                  hidden_size=32, intermediate_size=32)
assert np.all(actual == 0)
"""
        for overrides in ({"NUMBA_DISABLE_JIT": "1"}, {"Q51_NO_NUMBA": "1"}):
            process = subprocess.run([sys.executable, "-W", "error", "-c", code], env=dict(os.environ, **overrides),
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        if native._compiled_kernel is not None:
            blob = scalar.pack_expert(*[np.zeros((32, 32))] * 3)
            with patch.object(native, "_compiled_kernel", side_effect=RuntimeError("lazy JIT failure")):
                with patch.object(scalar, "_scalar_expert", side_effect=AssertionError("fake fallback")):
                    with self.assertRaisesRegex(RuntimeError, "lazy JIT failure"):
                        native.NativeQ5_1ExpertBackend().execute(MappingPayloadSource({"a": blob}), np.ones(32),
                            [("a", 1)], hidden_size=32, intermediate_size=32)


if __name__ == "__main__":
    unittest.main()
