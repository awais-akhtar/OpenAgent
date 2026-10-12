"""Actual Numba CPU execution over Q8_0 bytes; no execution mocks."""

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

from openagent.adaptive_engine import native_q8_0_expert as native
from openagent.adaptive_engine.expert_kernels import ExpertKernelRegistry, MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion
from openagent.adaptive_engine.q8_0_expert import Q8_0ExpertBackend, pack_expert
from tests.test_q8_0_expert import TrackingSource, scalar_decode


@unittest.skipUnless(native.NativeQ8_0ExpertBackend().info().available, "optional Numba unavailable")
class NativeQ8_0Tests(unittest.TestCase):
    def test_all_binary16_bit_patterns_against_struct(self):
        for bits in range(65536):
            expected = struct.unpack("<e", struct.pack("<H", bits))[0]
            actual = native._half_bits(bits)
            if math.isfinite(expected):
                self.assertEqual(actual, expected)
                if expected == 0:
                    self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
            else:
                self.assertTrue(math.isnan(actual))
        self.assertTrue(native._half_bits.nopython_signatures)

    def test_handbuilt_differential_and_actual_compilation(self):
        rng = np.random.default_rng(808)
        backend = native.NativeQ8_0ExpertBackend()
        for h, i in ((32, 32), (64, 96), (96, 64)):
            raw_payloads, float_payloads = {}, {}
            for key in ("a", "b"):
                raw = b"".join(struct.pack("<e32b", float(rng.choice([-.002, .003, -0.0, 2 ** -24])),
                                           *rng.integers(-128, 128, 32).tolist())
                               for _ in range(3 * h * i // 32))
                raw_payloads[key] = raw
                float_payloads[key] = b"".join(a.astype("<f4").tobytes() for a in scalar_decode(raw, h, i))
            x = rng.normal(0, .2, (4, h)).astype(np.float32)
            routes = [[("a", .3), ("b", .7)], [("b", 1)], [("a", 0)], [("b", 2), ("a", .1)]]
            args = dict(hidden_size=h, intermediate_size=i)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(
                MappingPayloadSource(float_payloads), x, routes, dtype="float32", **args)
            source = TrackingSource(raw_payloads)
            actual = backend.execute_batch(source, x, routes, **args)
            np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=2e-7)
            np.testing.assert_array_equal(backend.execute(source, x[0], routes[0], **args), actual[0])
            self.assertEqual(source.maximum, 1)
            self.assertEqual(source.active, 0)
        self.assertTrue(native._compiled_loop.nopython_signatures)
        # The numerical function receives its activation scratch; it must not allocate arrays.
        for signature in native._compiled_loop.signatures:
            self.assertNotIn("NRT_MemInfo_alloc", native._compiled_loop.inspect_llvm(signature))
        self.assertFalse(native._compiled_loop.targetoptions["fastmath"])
        self.assertTrue(backend.info().capabilities.compiled)
        self.assertFalse(backend.info().capabilities.native_simd)

    def test_explicit_registry_and_real_store_cleanup(self):
        registry = ExpertKernelRegistry()
        self.assertEqual(registry.resolve(dtype="q8_0").info().name, "q8_0-numpy")
        backend = native.NativeQ8_0ExpertBackend()
        registry.register(backend)
        self.assertIs(registry.resolve(dtype="q8_0", preferred=backend.NAME, allow_fallback=False), backend)
        for dtype, name in (("int8-row", "numpy-reference"), ("int4-row", "packed-int4-reference")):
            self.assertEqual(registry.resolve(dtype=dtype).info().name, name)
        raw = pack_expert(*[np.full((32, 32), .1)] * 3)
        args = dict(hidden_size=32, intermediate_size=32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("buffered", "mmap"):
                path = root / (mode + ".q8")
                path.write_bytes(raw)
                with DiskExpertStore(root, {"a": ExpertRegion(path.name, 0, len(raw))},
                                     capacity_bytes=len(raw), io_mode=mode) as store:
                    actual = backend.execute(store, np.ones(32), [("a", 1)], **args)
                    expected = Q8_0ExpertBackend().execute(store, np.ones(32), [("a", 1)], **args)
                    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=2e-7)
                    with self.assertRaises(ArithmeticError):
                        backend.execute(store, np.full(32, np.finfo(np.float32).max), [("a", 1)], **args)
                    self.assertEqual(store.status()["active_leases"], 0)
                self.assertTrue(store.status()["closed"])
                damaged = raw[:-34] + struct.pack("<e", math.nan) + raw[-32:]
                path.write_bytes(damaged)
                with DiskExpertStore(root, {"a": ExpertRegion(path.name, 0, len(raw))},
                                     capacity_bytes=len(raw), io_mode=mode) as store:
                    with self.assertRaisesRegex(ValueError, "scales"):
                        backend.execute(store, np.ones(32), [("a", 0)], **args)
                    self.assertEqual(store.status()["active_leases"], 0)

    def test_finite_extreme_scale_and_code_decoding(self):
        args = dict(hidden_size=32, intermediate_size=32)
        backend = native.NativeQ8_0ExpertBackend()
        for scale in (65504, -65504, 2 ** -24, -(2 ** -24), 0, -0.0):
            raw = struct.pack("<e32b", scale, *([-128, 127] * 16)) * 96
            source = MappingPayloadSource({"a": raw})
            x = np.linspace(-1e-12, 1e-12, 32, dtype=np.float32)
            expected = Q8_0ExpertBackend().execute(source, x, [("a", 1)], **args)
            np.testing.assert_allclose(backend.execute(source, x, [("a", 1)], **args),
                                       expected, rtol=5e-5, atol=1e-15)

    def test_lazy_compilation_failure_retained_exception_does_not_pin_mmap(self):
        originals = []

        class CompilationFailure(RuntimeError):
            pass

        def callback(raw, rows):
            borrowed_argument_view = raw[::2]
            self.assertTrue(np.shares_memory(raw, borrowed_argument_view))
            self.assertEqual(rows.shape, (32,))
            try:
                raise ValueError("inner compiler failure")
            except ValueError as cause:
                error = CompilationFailure("injected lazy compilation failure")
                originals.extend((error, cause))
                raise error from cause

        def fail(raw, *args):
            callback(raw, args[0])

        raw = pack_expert(*[np.ones((32, 32))] * 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "expert.q8").write_bytes(raw)
            store = DiskExpertStore(root, {"a": ExpertRegion("expert.q8", 0, len(raw))},
                                    capacity_bytes=len(raw), io_mode="mmap")
            retained = None
            try:
                with patch.object(native, "_compiled_loop", fail):
                    try:
                        native.NativeQ8_0ExpertBackend().execute(
                            store, np.ones(32), [("a", 1)], hidden_size=32, intermediate_size=32)
                    except RuntimeError as error:
                        retained = error
                self.assertIsNotNone(retained)
                self.assertEqual(store.status()["active_leases"], 0)
                store.close()
                self.assertIsNone(retained.__cause__)
                self.assertIsNone(retained.__context__)
                self.assertIn("injected lazy compilation failure", str(retained))
                self.assertIsInstance(retained, CompilationFailure)
                self.assertIs(retained, originals[0])
                self.assertIsNone(originals[1].__traceback__)
                for original in originals:
                    self.assertIsNone(original.__cause__)
                    self.assertIsNone(original.__context__)
            finally:
                # Cleanup the pre-fix failure so the regression does not leak its mapping.
                error = retained
                while error is not None:
                    error.__traceback__ = None
                    error = error.__cause__ or error.__context__
                for original in originals:
                    original.__traceback__ = None
                store.close()

    def test_overflow_corruption_and_scratch_limit_before_lease(self):
        args = dict(hidden_size=32, intermediate_size=32)
        raw = pack_expert(*[np.ones((32, 32))] * 3)
        backend = native.NativeQ8_0ExpertBackend(max_scratch_bytes=128)
        source = TrackingSource({"a": raw})
        self.assertEqual(native.scratch_bytes(32, 32), 128)
        self.assertEqual(native.scratch_bytes(1024, 32), 128)
        for value in (np.finfo(np.float32).max, 1e18, 2e17):
            with self.assertRaises(ArithmeticError):
                backend.execute(source, np.full(32, value), [("a", 1)], **args)
            self.assertEqual(source.active, 0)
        with self.assertRaises(ArithmeticError):
            backend.execute(source, np.ones(32), [("a", float(np.finfo(np.float32).max))], **args)
        np.testing.assert_array_equal(backend.execute(source, np.full(32, -1e18), [("a", 1)], **args), 0)
        for damaged in (raw[:-1], raw + b"\0", raw[:-34] + struct.pack("<e", math.inf) + raw[-32:]):
            source = TrackingSource({"a": damaged})
            with self.assertRaises(ValueError):
                backend.execute(source, np.ones(32), [("a", 0)], **args)
            self.assertEqual(source.active, 0)
        source = TrackingSource({"a": raw})
        with self.assertRaises(MemoryError):
            native.NativeQ8_0ExpertBackend(max_scratch_bytes=127).execute(source, np.ones(32), [("a", 1)], **args)
        self.assertEqual(source.calls, [])
        self.assertEqual(backend.execute_batch(source, np.empty((0, 32)), [], **args).shape, (0, 32))
        for routes in ([], [("a", -1)], [("a", 1e100)], [("a", 1), ("a", 1)]):
            with self.assertRaises(ValueError):
                backend.execute(source, np.ones(32), routes, **args)
        self.assertEqual(source.calls, [])


class NativeQ8_0UnavailableTests(unittest.TestCase):
    def test_disabled_jit_is_explicit_without_python_execution_fallback(self):
        code = """
from openagent.adaptive_engine.native_q8_0_expert import NativeQ8_0ExpertBackend
backend = NativeQ8_0ExpertBackend()
assert not backend.info().available
assert not backend.info().capabilities.compiled
try:
    backend.execute_batch(None, [], [], hidden_size=32, intermediate_size=32)
except RuntimeError:
    pass
else:
    raise AssertionError('disabled JIT silently executed')
"""
        result = subprocess.run([sys.executable, "-c", code],
                                env=dict(os.environ, NUMBA_DISABLE_JIT="1"),
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
