from __future__ import annotations

from contextlib import contextmanager
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import native_expert
from openagent.adaptive_engine.expert_kernels import (
    ExpertKernelRegistry, MappingPayloadSource, NumPyRoutedSwiGLUBackend,
)
from openagent.adaptive_engine.native_expert import NativeExpertBackend


def encode(matrices, dtype):
    if dtype != "int8-row":
        return b"".join(m.astype("<f4" if dtype == "float32" else "<f2").tobytes() for m in matrices)
    scales = [np.maximum(np.abs(m).max(axis=1), 1e-8).astype(np.float32) / 127 for m in matrices]
    quantized = [np.rint(m / s[:, None]).astype(np.int8) for m, s in zip(matrices, scales)]
    return b"".join(m.tobytes() for m in quantized) + b"".join(s.astype("<f4").tobytes() for s in scales)


class AvailabilityTests(unittest.TestCase):
    def test_missing_numba_is_explicit_and_registry_falls_back(self):
        with patch.object(native_expert, "_compiled_loop", None), patch.object(native_expert, "_unavailable", "Numba missing"):
            backend = NativeExpertBackend()
            info = backend.info()
            self.assertFalse(info.available)
            self.assertFalse(info.capabilities.compiled)
            self.assertFalse(info.capabilities.native_simd)
            self.assertIn("Numba", info.unavailable_reason)
            registry = ExpertKernelRegistry([backend])
            self.assertIsInstance(registry.resolve(dtype="int8-row"), NumPyRoutedSwiGLUBackend)
            with self.assertRaises(RuntimeError):
                registry.resolve(dtype="int8-row", preferred=backend.NAME, allow_fallback=False)
            with self.assertRaisesRegex(RuntimeError, "Numba"):
                backend.execute_batch(None, None, None, hidden_size=1, intermediate_size=1, dtype="int8-row")


@unittest.skipUnless(NativeExpertBackend().info().available, "optional Numba JIT unavailable")
class NativeTests(unittest.TestCase):
    def test_supported_dtypes_odd_sizes_scalar_and_grouped_batch(self):
        rng = np.random.default_rng(107)
        backend = NativeExpertBackend()
        reference = NumPyRoutedSwiGLUBackend()
        for hidden, intermediate in ((1, 1), (3, 7), (17, 11)):
            matrices = {key: [rng.normal(0, .3, size=shape).astype(np.float32) for shape in
                              ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))]
                        for key in ("a", "b")}
            batch = rng.normal(size=(4, hidden)).astype(np.float32)
            routes = [[("b", .7), ("a", .3)], [("a", 1.)], [("b", 0.)], [("a", .2), ("b", .8)]]
            for dtype in sorted(backend.info().capabilities.payload_dtypes):
                with self.subTest(hidden=hidden, intermediate=intermediate, dtype=dtype):
                    source = MappingPayloadSource({k: encode(v, dtype) for k, v in matrices.items()})
                    args = dict(hidden_size=hidden, intermediate_size=intermediate, dtype=dtype)
                    expected = reference.execute_batch(source, batch, routes, **args)
                    actual = backend.execute_batch(source, batch, routes, **args)
                    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
                    self.assertEqual(actual.dtype, np.float32)
                    for row in range(4):
                        np.testing.assert_allclose(backend.execute(source, batch[row], routes[row], **args), actual[row], rtol=3e-5, atol=3e-6)
                    np.testing.assert_array_equal(actual, backend.execute_batch(source, batch, routes, **args))

    def test_int8_quantization_accuracy_and_one_lease_per_expert(self):
        rng = np.random.default_rng(31)
        h, i = 9, 13
        matrices = [rng.normal(0, .2, size=s).astype(np.float32) for s in ((i, h), (i, h), (h, i))]
        raw = encode(matrices, "int8-row")
        leases = []

        class Source:
            @contextmanager
            def acquire(self, key):
                leases.append(key)
                yield memoryview(raw)

        x = rng.normal(size=(6, h)).astype(np.float32)
        args = dict(hidden_size=h, intermediate_size=i)
        backend = NativeExpertBackend()
        actual = backend.execute_batch(Source(), x, [[("a", 1.)]] * 6, dtype="int8-row", **args)
        self.assertEqual(leases, ["a"])
        expected = NumPyRoutedSwiGLUBackend().execute_batch(
            MappingPayloadSource({"a": encode(matrices, "float32")}), x,
            [[("a", 1.)]] * 6, dtype="float32", **args,
        )
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), .03)
        self.assertTrue(native_expert._compiled_loop.nopython_signatures)
        signatures = tuple(native_expert._compiled_loop.nopython_signatures)
        again = NativeExpertBackend().execute_batch(
            Source(), x, [[("a", 1.)]] * 6, dtype="int8-row", **args,
        )
        np.testing.assert_array_equal(again, actual)
        self.assertEqual(tuple(native_expert._compiled_loop.nopython_signatures), signatures)

    def test_corrupt_payloads_and_routes(self):
        backend = NativeExpertBackend()
        args = dict(hidden_size=3, intermediate_size=5, dtype="int8-row")
        raw = encode([np.ones(s, np.float32) for s in ((5, 3), (5, 3), (3, 5))], "int8-row")
        bad = bytearray(raw)
        bad[45:49] = np.array([0], dtype="<f4").tobytes()
        for payload in (raw[:-1], bad):
            with self.assertRaises(ValueError):
                backend.execute(MappingPayloadSource({"a": payload}), np.ones(3), [("a", 1.)], **args)
        source = MappingPayloadSource({"a": raw})
        for routes in ([], [("a", -1.)], [("a", 1.), ("a", 0.)]):
            with self.assertRaises(ValueError):
                backend.execute(source, np.ones(3), routes, **args)
        empty = backend.execute_batch(source, np.empty((0, 3)), [], **args)
        self.assertEqual(empty.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
