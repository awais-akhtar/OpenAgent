from __future__ import annotations

from contextlib import contextmanager
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.expert_kernels import (
    MappingPayloadSource, NumPyRoutedSwiGLUBackend, RoutedSwiGLUKernelBackend,
)
from openagent.adaptive_engine.torch_expert import TorchExpertBackend


class CountingSource(MappingPayloadSource):
    def __init__(self, payloads):
        super().__init__(payloads)
        self.keys = []
        self.active = 0

    @contextmanager
    def acquire(self, key):
        self.keys.append(key)
        self.active += 1
        try:
            with super().acquire(key) as payload:
                yield payload
        finally:
            self.active -= 1


def payloads(dtype):
    rng = np.random.default_rng(42)
    result = {}
    for key in ("a", "b", "unused"):
        if dtype == "int8-row":
            values = rng.integers(-8, 9, size=36, dtype=np.int8)
            scales = rng.uniform(.01, .1, size=11).astype("<f4")
            result[key] = values.tobytes() + scales.tobytes()
        else:
            result[key] = rng.normal(0, .2, 36).astype(
                "<f4" if dtype == "float32" else "<f2").tobytes()
    return result


class TorchDiscoveryTests(unittest.TestCase):
    def test_missing_optional_dependency(self):
        with patch("openagent.adaptive_engine.torch_expert.importlib.import_module",
                   side_effect=ImportError("torch not installed")):
            backend = TorchExpertBackend()
        self.assertFalse(backend.info().available)
        self.assertIn("torch not installed", backend.info().unavailable_reason)
        self.assertIsInstance(backend, RoutedSwiGLUKernelBackend)
        with self.assertRaises(RuntimeError):
            backend.execute(MappingPayloadSource(payloads("float32")), [1, 2, 3],
                            [("a", 1)], hidden_size=3, intermediate_size=4,
                            dtype="float32")

    def test_cpu_requires_explicit_test_opt_in(self):
        with self.assertRaises(ValueError):
            TorchExpertBackend(device="cpu")

    def test_invalid_device(self):
        for device in ("gpu", "cuda:-1", "cpu:0", "mps:1"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                TorchExpertBackend(device=device)

    def test_real_discovery_never_defaults_to_cpu(self):
        backend = TorchExpertBackend()
        if backend.info().available:
            self.assertTrue(backend.device.startswith("cuda:") or backend.device == "mps")
        else:
            self.assertIsNone(backend.device)
            self.assertTrue(backend.info().unavailable_reason)

    def test_unavailable_devices_and_indices(self):
        runtime = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        for device in (None, "cuda:0", "mps"):
            with self.subTest(device=device), patch(
                "openagent.adaptive_engine.torch_expert.importlib.import_module",
                return_value=runtime,
            ):
                backend = TorchExpertBackend(device=device)
                self.assertFalse(backend.info().available)
                self.assertIsNone(backend.device)

    def test_failed_real_probe_is_unavailable(self):
        runtime = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
            backends=SimpleNamespace(mps=None), float32=None,
            ones=unittest.mock.Mock(side_effect=RuntimeError("allocation failed")),
        )
        with patch("openagent.adaptive_engine.torch_expert.importlib.import_module",
                   return_value=runtime):
            backend = TorchExpertBackend(device="cpu", allow_cpu_for_tests=True)
        self.assertFalse(backend.info().available)
        self.assertIn("allocation failed", backend.info().unavailable_reason)


class TorchNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = TorchExpertBackend(device="cpu", allow_cpu_for_tests=True)
        if not cls.backend.info().available:
            raise unittest.SkipTest(cls.backend.info().unavailable_reason)

    def test_explicit_cpu_label(self):
        self.assertEqual(self.backend.info().name, "torch-cpu-test")
        self.assertIn("CPU differential tests only", self.backend.info().evidence[0].detail)

    def check_numerics(self, backend):
        for dtype in ("float32", "float16", "int8-row"):
            with self.subTest(dtype=dtype, device=backend.device):
                source = CountingSource(payloads(dtype))
                reference = NumPyRoutedSwiGLUBackend()
                config = dict(hidden_size=3, intermediate_size=4, dtype=dtype)
                x = np.array([[.2, -.7, .9], [-.5, .3, 1], [1, 0, -.2]], dtype=np.float32)
                routes = [[("b", .3), ("a", 1.7)], [("a", 0)], [("a", .8), ("b", .2)]]
                actual = backend.execute_batch(source, x, routes, **config)
                self.assertEqual(source.keys, ["b", "a"])
                self.assertEqual(source.active, 0)
                np.testing.assert_allclose(actual, reference.execute_batch(
                    source, x, routes, **config), rtol=2e-5, atol=2e-6)
                for row, route in zip(x, routes):
                    np.testing.assert_allclose(backend.execute(source, row, route, **config),
                                               reference.execute(source, row, route, **config),
                                               rtol=2e-5, atol=2e-6)
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_array_equal(actual[1], 0)
                self.assertNotIn("unused", source.keys)

    def test_scalar_and_grouped_batch_all_payload_types(self):
        self.check_numerics(self.backend)

    def test_real_accelerator_when_available(self):
        backend = TorchExpertBackend()
        if not backend.info().available:
            self.skipTest(backend.info().unavailable_reason)
        self.check_numerics(backend)

    def test_validation_and_lease_cleanup(self):
        config = dict(hidden_size=3, intermediate_size=4, dtype="float32")
        source = CountingSource(payloads("float32"))
        for routes in ([], [("a", -1)], [("a", 1), ("a", 2)], [("a", float("nan"))]):
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                self.backend.execute(source, [1, 2, 3], routes, **config)
        self.assertEqual(source.keys, [])
        with self.assertRaises(ValueError):
            self.backend.execute_batch(source, [[1, 2, 3]], [], **config)
        for dtype, data in (("float32", b"bad"),
                            ("float32", np.full(36, np.nan, dtype="<f4").tobytes()),
                            ("int8-row", bytes(36) + np.zeros(11, dtype="<f4").tobytes())):
            bad = CountingSource({"a": data})
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                self.backend.execute(bad, [1, 2, 3], [("a", 1)],
                                     **dict(config, dtype=dtype))
            self.assertEqual(bad.active, 0)

    def test_empty_batch_and_close(self):
        backend = TorchExpertBackend(device="cpu", allow_cpu_for_tests=True)
        config = dict(hidden_size=3, intermediate_size=4, dtype="float32")
        source = MappingPayloadSource({})
        self.assertEqual(backend.execute_batch(source, np.empty((0, 3)), [], **config).shape, (0, 3))
        backend.close()
        backend.close()
        self.assertFalse(backend.info().available)
        with self.assertRaises(RuntimeError):
            backend.execute_batch(source, np.empty((0, 3)), [], **config)

    def test_no_cached_weights_and_nonfinite_output(self):
        config = dict(hidden_size=3, intermediate_size=4, dtype="float32")
        source = CountingSource(payloads("float32"))
        first = self.backend.execute(source, [1, 2, 3], [("a", 1)], **config)
        source._payloads["a"] = np.zeros(36, dtype="<f4").tobytes()
        second = self.backend.execute(source, [1, 2, 3], [("a", 1)], **config)
        self.assertEqual(source.keys, ["a", "a"])
        self.assertTrue(np.any(first != 0))
        np.testing.assert_array_equal(second, 0)
        source._payloads["a"] = np.full(36, 1e30, dtype="<f4").tobytes()
        with self.assertRaises(ArithmeticError):
            self.backend.execute(source, [1, 2, 3], [("a", 1)], **config)
        self.assertEqual(source.active, 0)


if __name__ == "__main__":
    unittest.main()
