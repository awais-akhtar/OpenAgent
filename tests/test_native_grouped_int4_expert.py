"""Real direct-byte grouped INT4 scalar/native execution and adversarial leases."""

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
from openagent.adaptive_engine import grouped_int4_expert as scalar
from openagent.adaptive_engine import native_grouped_int4_expert as native
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore
from openagent.adaptive_engine.grouped_int4_conversion import convert_experts
from tests.test_grouped_int4_expert import fixture, oracle_decode


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


def backends(gs):
    result = [scalar.GroupedInt4ExpertBackend(group_size=gs)]
    if native._compiled_kernel is not None:
        result.append(native.NativeGroupedInt4ExpertBackend(group_size=gs))
    return result


class GroupedExecutionTests(unittest.TestCase):
    def test_independent_handbuilt_numerical_differentials_odd_and_partial_groups(self):
        rng = np.random.default_rng(414)
        for h, i, gs in ((1, 1, 1), (3, 5, 2), (65, 129, 64), (65, 33, 7), (18, 11, 100)):
            blobs = {"a": fixture(h, i, gs, 415), "b": fixture(h, i, gs, 416)}
            floats = {key: b"".join(m.astype("<f4").tobytes() for m in oracle_decode(blob, h, i, gs))
                      for key, blob in blobs.items()}
            x = rng.normal(0, .2, (3, h)).astype(np.float32)
            routes = [[("a", .3), ("b", .7)], [("b", 0)], [("a", 2), ("b", .1)]]
            args = dict(hidden_size=h, intermediate_size=i)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource(floats), x, routes,
                                                              dtype="float32", **args)
            baseline = None
            for backend in backends(gs):
                source = ObservedSource(blobs)
                actual = backend.execute_batch(source, x, routes, **args)
                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-7)
                np.testing.assert_array_equal(backend.execute(source, x[0], routes[0], **args), actual[0])
                self.assertEqual((source.active, source.maximum), (0, 1))
                if baseline is None:
                    baseline = actual
                else:
                    np.testing.assert_allclose(actual, baseline, rtol=5e-6, atol=1e-9)

    def test_f32_reader_boundaries_and_seeded_bitpatterns(self):
        rng = np.random.default_rng(417)
        bits = list(rng.integers(0, 2 ** 32, 10000, dtype=np.uint32))
        bits += [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
                 0x7f800000, 0xff800000, 0x7fc00000]
        reader = native._read_f32
        if native._compiled_kernel is not None:
            import numba
            reader = numba.njit(cache=False, fastmath=False)(reader)
        for word in bits:
            expected = struct.unpack("<f", struct.pack("<I", int(word)))[0]
            actual = reader(np.frombuffer(struct.pack("<I", int(word)), np.uint8), 0)
            if math.isfinite(expected):
                self.assertEqual(actual, expected)
                if expected == 0:
                    self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
            else:
                self.assertTrue(math.isnan(actual))

    def test_all_nibbles_signed_scales_and_scale_before_reduction(self):
        compiled_dot = None
        if native._compiled_kernel is not None:
            import numba
            compiled_dot = numba.njit(cache=False, fastmath=False)(native._packed_dot)
        codes = bytes(j + 16 * (j + 1) for j in range(0, 16, 2))
        for scale in (0., -0., 2 ** -149, -(2 ** -149), .25, -.5):
            blob = codes + struct.pack("<f", scale)
            x = np.array([1e30, -1e30, .2, -.4] * 4, np.float32)
            expected = np.float32(0)
            for lane in range(16):
                value = lane if lane < 8 else lane - 16
                expected = np.float32(expected + np.float32(np.float32(value * scale) * x[lane]))
            with memoryview(blob) as view:
                actual = scalar._dot(view, 0, 8, x, 16, 16)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            if compiled_dot is not None:
                self.assertEqual(compiled_dot(np.frombuffer(blob, np.uint8), 0, 8, x, 16, 16), expected)

    def test_malformed_nonfinite_padding_zero_routes_and_overflow(self):
        blob = fixture(3, 5, 2)
        args = dict(hidden_size=3, intermediate_size=5)
        damaged = [blob[:-1], blob + b"x", memoryview(blob)[::2]]
        for offset in (29, len(blob) - 4):
            for value in (float("nan"), float("inf"), -float("inf")):
                bad = bytearray(blob)
                struct.pack_into("<f", bad, offset, value)
                damaged.append(bad)
        for offset in (1, 11, 22):
            bad = bytearray(blob)
            bad[offset] |= 0x80
            damaged.append(bad)
        for backend in backends(2):
            for bad in damaged:
                source = ObservedSource({"a": bad})
                with self.assertRaises(ValueError):
                    backend.execute(source, np.ones(3), [("a", 0)], **args)
                self.assertEqual(source.active, 0)

    def test_finite_extreme_scale_zero_codes_and_weight_overflow(self):
        maximum = float(np.finfo(np.float32).max)
        # Three one-byte rows, then three unaligned FP32 scales.
        zeros = bytes(3) + struct.pack("<3f", maximum, -maximum, maximum)
        ones = bytes([1] * 3) + struct.pack("<3f", maximum, maximum, maximum)
        overflowing = bytes([8] * 3) + struct.pack("<3f", maximum, maximum, maximum)
        args = dict(hidden_size=1, intermediate_size=1)
        for backend in backends(1):
            np.testing.assert_array_equal(backend.execute(MappingPayloadSource({"a": zeros}), np.ones(1),
                                          [("a", 1)], **args), 0)
            result = backend.execute(MappingPayloadSource({"a": ones}), np.array([1e-40], np.float32),
                                     [("a", 1)], **args)
            self.assertTrue(np.isfinite(result).all())
            with self.assertRaises(ArithmeticError):
                backend.execute(MappingPayloadSource({"a": overflowing}), np.zeros(1), [("a", 0)], **args)
        with self.assertRaises(ArithmeticError):
            scalar.decode_expert(overflowing, 1, 1, group_size=1)

    def test_activation_overflow_rejects_finite_inputs(self):
        args = dict(hidden_size=3, intermediate_size=5)
        matrices = [np.ones((5, 3)), np.ones((5, 3)), np.ones((3, 5))]
        source = ObservedSource({"a": scalar.pack_expert(*matrices, group_size=2)})
        for backend in backends(2):
            for x in (np.full(3, np.finfo(np.float32).max), np.full(3, 1e20)):
                with self.assertRaises(ArithmeticError):
                    backend.execute(source, x, [("a", 1)], **args)
                self.assertEqual(source.active, 0)

    def test_prelease_routes_group_size_scratch_inputs_and_empty_batch(self):
        blob = fixture(3, 5, 2)
        args = dict(hidden_size=3, intermediate_size=5)
        self.assertEqual(scalar.scratch_bytes(3, 5, group_size=2), 20)
        for backend in backends(2):
            source = ObservedSource({"a": blob})
            with self.assertRaises(MemoryError):
                type(backend)(group_size=2, max_scratch_bytes=19).execute(source, np.ones(3), [("a", 1)], **args)
            for routes in ([], [("a", -1)], [("a", 1e100)], [("a", True)], [("a", 1), ("a", 1)]):
                with self.assertRaises(ValueError):
                    backend.execute(source, np.ones(3), routes, **args)
            for x in (np.ones(4), np.ones(3, complex), np.full(3, np.nan)):
                with self.assertRaises(ValueError):
                    backend.execute(source, x, [("a", 1)], **args)
            with self.assertRaises(ValueError):
                backend.execute(source, np.ones(3), [("a", 1)], dtype="int4-row", **args)
            with self.assertRaises(ValueError):
                backend.execute_batch(source, np.ones((2, 3)), [[("a", 1)]], **args)
            self.assertEqual(source.calls, 0)
            self.assertEqual(backend.execute_batch(source, np.empty((0, 3)), [], **args).shape, (0, 3))
            type(backend)(group_size=2, max_scratch_bytes=20).execute(source, np.zeros(3), [("a", 1)], **args)

    def test_actual_mmap_leases_and_retained_chained_failures(self):
        args = dict(hidden_size=3, intermediate_size=5)
        matrices = [np.full((5, 3), .1), np.full((5, 3), .1), np.full((3, 5), .1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = convert_experts({"a": matrices}, root / "weights.gi4", group_size=2, **args)
            for backend in backends(2):
                for mode in ("buffered", "mmap"):
                    with DiskExpertStore(root, result.regions, capacity_bytes=145, io_mode=mode) as store:
                        actual = backend.execute(store, np.ones(3), [("a", 1)], **args)
                        self.assertTrue(np.isfinite(actual).all())
                        inner = []
                        def fail(raw, *unused):
                            view = np.frombuffer(raw, np.uint8)[::2]
                            assert view.size
                            try:
                                raise ValueError("inner borrowed frame")
                            except ValueError as cause:
                                inner.append(cause)
                                raise RuntimeError("injected failure") from cause
                        module, name = (native, "_compiled_kernel") if isinstance(backend, native.NativeGroupedInt4ExpertBackend) else (
                            scalar, "_scalar_expert")
                        retained = None
                        with patch.object(module, name, fail):
                            try:
                                backend.execute(store, np.ones(3), [("a", 1)], **args)
                            except RuntimeError as error:
                                retained = error
                        self.assertIsNotNone(retained)
                        self.assertIsNone(retained.__cause__)
                        self.assertIsNone(retained.__context__)
                        self.assertIsNone(inner[0].__traceback__)
                        self.assertEqual(store.status()["active_leases"], 0)
                        store.close()

    @unittest.skipUnless(native._compiled_kernel is not None, "active optional Numba required")
    def test_actual_nopython_kernel_no_allocations(self):
        native.NativeGroupedInt4ExpertBackend(group_size=2).execute(MappingPayloadSource({"a": fixture(3, 5, 2)}),
            np.ones(3), [("a", 1)], hidden_size=3, intermediate_size=5)
        self.assertTrue(native._compiled_kernel.nopython_signatures)
        self.assertFalse(native._compiled_kernel.targetoptions["fastmath"])
        for signature in native._compiled_kernel.signatures:
            self.assertNotIn("NRT_MemInfo_alloc", native._compiled_kernel.inspect_llvm(signature))

    def test_registry_after_parent_dtype_integration(self):
        if "int4-grouped" not in contracts.REGISTERED_PAYLOAD_DTYPES:
            self.skipTest("parent grouped dtype integration pending")
        for backend in backends(64):
            registry = contracts.ExpertKernelRegistry()
            registry.register(backend, replace=True)
            self.assertIs(registry.resolve(dtype="int4-grouped", preferred=backend.NAME, allow_fallback=False), backend)

    def test_disabled_or_missing_numba_native_unavailable_scalar_explicit(self):
        code = """
import importlib.abc
import os
import sys
if os.environ.get('GROUP_NO_NUMBA'):
    class NoNumba(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'numba' or fullname.startswith('numba.'):
                raise ModuleNotFoundError('intentionally unavailable')
    sys.meta_path.insert(0, NoNumba())
import numpy as np
from openagent.adaptive_engine import native_grouped_int4_expert as n
from openagent.adaptive_engine.grouped_int4_expert import GroupedInt4ExpertBackend, pack_expert
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, REGISTERED_PAYLOAD_DTYPES
assert n._compiled_kernel is None
if 'int4-grouped' in REGISTERED_PAYLOAD_DTYPES:
    assert not n.NativeGroupedInt4ExpertBackend().info().available
try:
    n.NativeGroupedInt4ExpertBackend().execute_batch(None, [], [], hidden_size=1, intermediate_size=1)
except RuntimeError:
    pass
else:
    raise AssertionError('silent fallback')
blob = pack_expert(*[np.zeros((1,1))]*3)
actual = GroupedInt4ExpertBackend().execute(MappingPayloadSource({'a':blob}), np.ones(1), [('a',1)],
                                          hidden_size=1, intermediate_size=1)
assert np.all(actual == 0)
"""
        for overrides in ({"NUMBA_DISABLE_JIT": "1"}, {"GROUP_NO_NUMBA": "1"}):
            result = subprocess.run([sys.executable, "-W", "error", "-c", code], env=dict(os.environ, **overrides),
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
