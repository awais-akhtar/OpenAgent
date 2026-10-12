"""Numerical differential and leased direct-byte execution, no mocked math."""

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

from openagent.adaptive_engine import native_q4_0_expert as native
from openagent.adaptive_engine import expert_kernels as contracts
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion
from openagent.adaptive_engine.q4_0_expert import Q4_0ExpertBackend, pack_expert
from tests.test_q4_0_expert import scalar_decode


class TrackedSource:
    def __init__(self, blobs):
        self.blobs = blobs
        self.active = self.maximum = self.calls = 0

    @contextmanager
    def acquire(self, key):
        self.calls += 1
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            yield self.blobs[key]
        finally:
            self.active -= 1


@unittest.skipUnless(native._compiled, "optional active Numba unavailable")
class Q4KernelTests(unittest.TestCase):
    def test_handbuilt_numerical_differential(self):
        rng = np.random.default_rng(407)
        backend = native.NativeQ4_0ExpertBackend()
        for h, i in ((32, 32), (64, 96), (96, 64)):
            packed, floating = {}, {}
            for key in ("a", "b"):
                blob = b"".join(struct.pack("<e", float(rng.choice([-.01, .02, -0., 2 ** -24])))
                                + rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
                                for _ in range(3 * h * i // 32))
                packed[key] = blob
                floating[key] = b"".join(m.astype("<f4").tobytes() for m in scalar_decode(blob, h, i))
            x = rng.normal(0, .2, (4, h)).astype(np.float32)
            routes = [[("a", .3), ("b", .7)], [("b", 1)], [("a", 0)], [("b", 2), ("a", .1)]]
            args = dict(hidden_size=h, intermediate_size=i)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(
                MappingPayloadSource(floating), x, routes, dtype="float32", **args)
            source = TrackedSource(packed)
            actual = backend.execute_batch(source, x, routes, **args)
            scalar = Q4_0ExpertBackend().execute_batch(source, x, routes, **args)
            np.testing.assert_allclose(actual, scalar, rtol=5e-6, atol=1e-10)
            np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-7)
            np.testing.assert_array_equal(backend.execute(source, x[0], routes[0], **args), actual[0])
            self.assertEqual(source.active, 0)
            self.assertEqual(source.maximum, 1)
        if native._compiled:
            self.assertTrue(native._compiled_evaluate.nopython_signatures)
            self.assertFalse(native._compiled_evaluate.targetoptions["fastmath"])
            for signature in native._compiled_evaluate.signatures:
                self.assertNotIn("NRT_MemInfo_alloc", native._compiled_evaluate.inspect_llvm(signature))

    def test_binary16_exhaustive_finite_and_nonfinite(self):
        for bits in range(65536):
            expected = struct.unpack("<e", struct.pack("<H", bits))[0]
            actual = native._delta(np.array([bits & 255, bits >> 8], np.uint8), 0)
            if math.isfinite(expected):
                self.assertEqual(actual, expected)
                if expected == 0:
                    self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
            else:
                self.assertTrue(math.isnan(actual))

    def test_extreme_finite_deltas_and_zero_codes(self):
        backend = native.NativeQ4_0ExpertBackend()
        for delta in (65504, -65504, 2 ** -24, -(2 ** -24), 0., -0.):
            blob = (struct.pack("<e", delta) + b"\xf0" * 16) * 96
            matrices = scalar_decode(blob, 32, 32)
            source = MappingPayloadSource({"a": blob})
            float_source = MappingPayloadSource({"a": b"".join(m.astype("<f4").tobytes() for m in matrices)})
            args = dict(hidden_size=32, intermediate_size=32)
            x = np.linspace(-1e-12, 1e-12, 32, dtype=np.float32)
            expected = NumPyRoutedSwiGLUBackend().execute(float_source, x, [("a", 1)], dtype="float32", **args)
            np.testing.assert_allclose(backend.execute(source, x, [("a", 1)], **args),
                                       expected, rtol=5e-5, atol=1e-15)

    def test_prelease_validation_overflow_and_malformed(self):
        blob = pack_expert(*[np.ones((32, 32))] * 3)
        source = TrackedSource({"a": blob})
        backend = native.NativeQ4_0ExpertBackend(max_scratch_bytes=128)
        args = dict(hidden_size=32, intermediate_size=32)
        self.assertEqual(native.scratch_bytes(1024, 32), 128)
        with self.assertRaises(MemoryError):
            native.NativeQ4_0ExpertBackend(max_scratch_bytes=127).execute(source, np.ones(32), [("a", 1)], **args)
        for routes in ([], [("a", -1)], [("a", 1e100)], [("a", 1), ("a", 1)]):
            with self.assertRaises(ValueError):
                backend.execute(source, np.ones(32), routes, **args)
        for x in (np.full(32, np.nan), np.ones(31)):
            with self.assertRaises(ValueError):
                backend.execute(source, x, [("a", 1)], **args)
        with self.assertRaises(ValueError):
            backend.execute(source, np.ones(32), [("a", 1)], dtype="q8_0", **args)
        with self.assertRaises(ValueError):
            backend.execute_batch(source, np.ones((2, 32)), [[("a", 1)]], **args)
        self.assertEqual(source.calls, 0)
        self.assertEqual(backend.execute_batch(source, np.empty((0, 32)), [], **args).shape, (0, 32))
        for value in (np.finfo(np.float32).max, 1e18):
            with self.assertRaises(ArithmeticError):
                backend.execute(source, np.full(32, value), [("a", 1)], **args)
            self.assertEqual(source.active, 0)
        for damaged in (blob[:-1], blob + b"x", memoryview(blob)[::2]):
            with self.assertRaises(ValueError):
                backend.execute(TrackedSource({"a": damaged}), np.ones(32), [("a", 0)], **args)
        for bits in (0x7c00, 0xfc00, 0x7e00):
            for position in (0, len(blob) - 18):
                damaged = bytearray(blob)
                struct.pack_into("<H", damaged, position, bits)
                source = TrackedSource({"a": damaged})
                with self.assertRaisesRegex(ValueError, "finite"):
                    backend.execute(source, np.ones(32), [("a", 0)], **args)
                self.assertEqual(source.active, 0)

    def test_real_disk_mmap_and_retained_failure_cleanup(self):
        blob = pack_expert(*[np.full((32, 32), .1)] * 3)
        args = dict(hidden_size=32, intermediate_size=32)
        backend = native.NativeQ4_0ExpertBackend()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.q4").write_bytes(blob)
            for mode in ("buffered", "mmap"):
                with DiskExpertStore(root, {"a": ExpertRegion("a.q4", 0, len(blob))},
                                     capacity_bytes=len(blob), io_mode=mode) as store:
                    result = backend.execute(store, np.ones(32), [("a", 1)], **args)
                    self.assertTrue(np.isfinite(result).all())
                    with self.assertRaises(ArithmeticError):
                        backend.execute(store, np.full(32, 1e30), [("a", 1)], **args)
                    self.assertEqual(store.status()["active_leases"], 0)
                    def fail(raw, *unused):
                        borrowed = raw[::2]
                        assert borrowed.size
                        try:
                            raise ValueError("inner")
                        except ValueError as cause:
                            raise RuntimeError("compiler failure") from cause
                    retained = None
                    with patch.object(native, "_compiled_evaluate", fail):
                        try:
                            backend.execute(store, np.ones(32), [("a", 1)], **args)
                        except RuntimeError as error:
                            retained = error
                    self.assertIsNotNone(retained)
                    self.assertIsNone(retained.__cause__)
                    self.assertIsNone(retained.__context__)
                    self.assertEqual(store.status()["active_leases"], 0)
                self.assertTrue(store.status()["closed"])

    def test_custom_registry_after_dtype_integration(self):
        if "q4_0" not in contracts.REGISTERED_PAYLOAD_DTYPES:
            self.skipTest("parent has not added q4_0 to registered dtype contract")
        backend = native.NativeQ4_0ExpertBackend()
        registry = contracts.ExpertKernelRegistry([backend])
        selected = registry.resolve(dtype="q4_0", preferred=backend.NAME, allow_fallback=False)
        self.assertIs(selected, backend)
        self.assertEqual(selected.info().capabilities.compiled, native._compiled)
        self.assertFalse(selected.info().capabilities.native_simd)
        blob = pack_expert(*[np.zeros((32, 32))] * 3)
        np.testing.assert_array_equal(selected.execute(MappingPayloadSource({"a": blob}),
                                      np.ones(32), [("a", 1)], hidden_size=32, intermediate_size=32), 0)

class Q4ScalarTests(unittest.TestCase):
    def test_scalar_direct_byte_oracle_and_cleanup(self):
        blob = (struct.pack("<e", .02) + bytes(range(16))) * 96
        decoded = scalar_decode(blob, 32, 32)
        source = TrackedSource({"a": blob})
        float_source = MappingPayloadSource({"a": b"".join(m.astype("<f4").tobytes() for m in decoded)})
        args = dict(hidden_size=32, intermediate_size=32)
        expected = NumPyRoutedSwiGLUBackend().execute(float_source, np.ones(32), [("a", 1)], dtype="float32", **args)
        np.testing.assert_allclose(Q4_0ExpertBackend().execute(source, np.ones(32), [("a", 1)], **args),
                                   expected, rtol=5e-5, atol=2e-7)
        self.assertEqual(source.active, 0)
        def fail(raw, *unused):
            borrowed = raw[::2]
            assert borrowed.size
            raise RuntimeError("scalar failure")
        with patch.object(native, "_evaluate", fail):
            with self.assertRaisesRegex(RuntimeError, "scalar failure"):
                Q4_0ExpertBackend().execute(source, np.ones(32), [("a", 1)], **args)
        self.assertEqual(source.active, 0)

    def test_disabled_jit_native_unavailable_scalar_explicit(self):
        code = """
import numpy as np
from openagent.adaptive_engine import native_q4_0_expert as n
from openagent.adaptive_engine.q4_0_expert import Q4_0ExpertBackend, pack_expert
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource
assert not n._compiled
raw = np.array([0, 60] + [0xf0] * 16, np.uint8)
assert n._projection(raw, 0, np.ones(32, np.float32), 32) == -16
try:
    n.NativeQ4_0ExpertBackend().execute_batch(None, [], [], hidden_size=32, intermediate_size=32)
except RuntimeError:
    pass
else:
    raise AssertionError('native silently fell back')
blob = pack_expert(*[np.zeros((32, 32))] * 3)
result = Q4_0ExpertBackend().execute(MappingPayloadSource({'a': blob}), np.ones(32),
                                  [('a', 1)], hidden_size=32, intermediate_size=32)
assert np.all(result == 0)
"""
        process = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, NUMBA_DISABLE_JIT="1"),
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
