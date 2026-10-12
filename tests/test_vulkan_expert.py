from __future__ import annotations

from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.expert_kernels import (
    ExpertKernelRegistry, MappingPayloadSource, NumPyRoutedSwiGLUBackend,
)
from openagent.adaptive_engine.vulkan_expert import VulkanRoutedSwiGLUBackend


def encode(rng, hidden, intermediate, dtype):
    width = hidden * intermediate
    if dtype == "int8-row":
        values = rng.integers(-25, 26, 3 * width, dtype=np.int8)
        scales = rng.uniform(0.01, 0.06, 2 * intermediate + hidden).astype("<f4")
        return values.tobytes() + scales.tobytes()
    return rng.normal(0, 0.3, 3 * width).astype(
        "<f4" if dtype == "float32" else "<f2"
    ).tobytes()


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
            if self.active != 1:
                raise AssertionError("experts must be streamed individually")
            with super().acquire(key) as view:
                yield view
        finally:
            self.active -= 1


class VulkanProbeTests(unittest.TestCase):
    def test_missing_dependency_is_optional(self):
        with patch("openagent.adaptive_engine.vulkan_expert.importlib.import_module",
                   side_effect=ModuleNotFoundError("No module named 'wgpu'")):
            backend = VulkanRoutedSwiGLUBackend()
        self.assertFalse(backend.info().available)
        self.assertIn("wgpu", backend.info().unavailable_reason)
        registry = ExpertKernelRegistry([backend])
        self.assertIsInstance(registry.resolve(dtype="float32"), NumPyRoutedSwiGLUBackend)
        with self.assertRaises(RuntimeError):
            registry.resolve(dtype="float32", preferred=backend.NAME, allow_fallback=False)
        with self.assertRaises(RuntimeError):
            backend.execute(MappingPayloadSource({}), [1], [("a", 1)],
                            hidden_size=1, intermediate_size=1, dtype="float32")

    def test_request_is_explicit_and_wrong_backend_or_software_is_rejected(self):
        for backend_type, adapter_type in (("D3D12", "DiscreteGPU"), ("Vulkan", "CPU")):
            with self.subTest(backend_type=backend_type, adapter_type=adapter_type):
                def request(**kwargs):
                    self.assertEqual(os.environ["WGPU_BACKEND_TYPE"], "Vulkan")
                    self.assertFalse(kwargs["force_fallback_adapter"])
                    return SimpleNamespace(info={"backend_type": backend_type,
                                                 "adapter_type": adapter_type})
                fake = SimpleNamespace(gpu=SimpleNamespace(request_adapter_sync=request))
                with patch.dict(os.environ, {"WGPU_BACKEND_TYPE": "D3D12"}), patch(
                    "openagent.adaptive_engine.vulkan_expert.importlib.import_module", return_value=fake
                ):
                    backend = VulkanRoutedSwiGLUBackend()
                    self.assertEqual(os.environ["WGPU_BACKEND_TYPE"], "D3D12")
                self.assertFalse(backend.info().available)
                self.assertFalse(backend.info().evidence[0].verified)


class VulkanHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = VulkanRoutedSwiGLUBackend()
        if not cls.backend.info().available:
            raise unittest.SkipTest(cls.backend.info().unavailable_reason)

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def setUp(self):
        self.rng = np.random.default_rng(9512)
        self.hidden, self.intermediate = 13, 19
        self.options = dict(hidden_size=self.hidden, intermediate_size=self.intermediate)

    def source(self, dtype):
        return CountingSource({key: encode(self.rng, self.hidden, self.intermediate, dtype)
                               for key in ("a", "b", "c")})

    def test_actual_hardware_adapter(self):
        self.assertEqual(self.backend.adapter_info["backend_type"], "Vulkan")
        self.assertIn(self.backend.adapter_info["adapter_type"], ("IntegratedGPU", "DiscreteGPU"))
        self.assertTrue(self.backend.info().evidence[0].verified)

    def test_scalar_all_payload_dtypes_and_exact_router_weights(self):
        for dtype in ("float32", "float16", "int8-row"):
            with self.subTest(dtype=dtype):
                source = self.source(dtype)
                x = self.rng.normal(size=self.hidden).astype(np.float32)
                routes = [("b", 2.75), ("a", 0.125), ("c", 0.0)]
                expected = NumPyRoutedSwiGLUBackend().execute(
                    source, x, routes, dtype=dtype, **self.options)
                source.keys.clear()
                actual = self.backend.execute(source, x, routes, dtype=dtype, **self.options)
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(source.keys, ["b", "a", "c"])
                self.assertEqual(source.active, 0)

    def test_grouped_all_payload_dtypes_streamed_and_no_cache(self):
        for dtype in ("float32", "float16", "int8-row"):
            with self.subTest(dtype=dtype):
                source = self.source(dtype)
                x = self.rng.normal(size=(7, self.hidden)).astype(np.float32)
                routes = [[("b", 1.7), ("a", 0.2)] if i % 2 else [("c", 0.4), ("b", 0.0)]
                          for i in range(7)]
                for repeat in range(2):
                    if repeat:
                        source._payloads["b"] = encode(self.rng, self.hidden, self.intermediate, dtype)
                    expected = NumPyRoutedSwiGLUBackend().execute_batch(
                        source, x, routes, dtype=dtype, **self.options)
                    source.keys.clear()
                    with patch.object(self.backend, "_compute", wraps=self.backend._compute) as compute:
                        actual = self.backend.execute_batch(source, x, routes, dtype=dtype, **self.options)
                    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)
                    self.assertEqual(source.keys, ["c", "b", "a"])
                    self.assertEqual(compute.call_count, 3)
                    self.assertEqual(source.active, 0)

    def test_stable_activation_large_positive_and_negative_gate(self):
        source = MappingPayloadSource({"a": np.asarray([1, 1, 1], dtype="<f4").tobytes()})
        options = dict(hidden_size=1, intermediate_size=1, dtype="float32")
        for value in (-1000, 1000):
            actual = self.backend.execute(source, [value], [("a", 1)], **options)
            expected = NumPyRoutedSwiGLUBackend().execute(source, [value], [("a", 1)], **options)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_malformed_inputs_and_unsupported_configuration(self):
        source = self.source("float32")
        x = np.ones(self.hidden, dtype=np.float32)
        cases = [
            (x[:-1], [("a", 1)], {}),
            (np.full(self.hidden, np.nan), [("a", 1)], {}),
            (x.astype(complex), [("a", 1)], {}),
            (x, [], {}), (x, [("a", -1)], {}),
            (x, [("a", np.inf)], {}), (x, [("a", True)], {}),
            (x, [("a", 1), ("a", 0)], {}), (x, [("", 1)], {}),
            (x, [("a", 1)], {"dtype": "bfloat16"}),
            (x, [("a", 1)], {"hidden_size": 0}),
            (x, [("a", 1)], {"intermediate_size": True}),
        ]
        for inputs, routes, overrides in cases:
            with self.subTest(overrides=overrides, routes=routes):
                with self.assertRaises((ValueError, TypeError)):
                    self.backend.execute(source, inputs, routes,
                                         **(self.options | {"dtype": "float32"} | overrides))
        self.assertEqual(source.keys, [])
        with self.assertRaisesRegex(ValueError, "one route list"):
            self.backend.execute_batch(source, x[None], [], dtype="float32", **self.options)
        with self.assertRaises(TypeError):
            self.backend.execute_batch(source, x[None], None, dtype="float32", **self.options)

    def test_malformed_payloads_and_lease_cleanup(self):
        width = self.hidden * self.intermediate
        invalid = [
            ("float32", b"short"),
            ("float32", np.full(3 * width, np.nan, dtype="<f4").tobytes()),
            ("float16", np.full(3 * width, np.inf, dtype="<f2").tobytes()),
        ]
        for scale in (0.0, -1.0, np.inf, np.nan):
            invalid.append(("int8-row", bytes(3 * width) + np.full(
                2 * self.intermediate + self.hidden, scale, dtype="<f4").tobytes()))
        for dtype, payload in invalid:
            with self.subTest(dtype=dtype, size=len(payload)):
                source = CountingSource({"a": payload})
                with self.assertRaises(ValueError):
                    self.backend.execute(source, np.ones(self.hidden), [("a", 1)],
                                         dtype=dtype, **self.options)
                self.assertEqual(source.active, 0)
        source = CountingSource({"a": memoryview(bytearray(2000))[::2]})
        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            self.backend.execute(source, np.ones(self.hidden), [("a", 1)],
                                 dtype="float32", **self.options)
        self.assertEqual(source.active, 0)

    def test_empty_batch(self):
        result = self.backend.execute_batch(MappingPayloadSource({}), np.empty((0, self.hidden)),
                                            [], dtype="float32", **self.options)
        self.assertEqual(result.shape, (0, self.hidden))

    def test_larger_noncontiguous_group(self):
        hidden, intermediate = 129, 257
        options = dict(hidden_size=hidden, intermediate_size=intermediate, dtype="float32")
        source = MappingPayloadSource({"a": encode(self.rng, hidden, intermediate, "float32")})
        x = self.rng.normal(size=(5, hidden * 2)).astype(np.float32)[:, ::2]
        routes = [[("a", 1.25)] for _ in x]
        expected = NumPyRoutedSwiGLUBackend().execute_batch(source, x, routes, **options)
        actual = self.backend.execute_batch(source, x, routes, **options)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)

    def test_readback_failure_releases_buffers_and_payload(self):
        source = self.source("float32")
        device = self.backend._device
        created = []
        create = device.create_buffer

        def record(*args, **kwargs):
            buffer = create(*args, **kwargs)
            spy = patch.object(buffer, "destroy", wraps=buffer.destroy)
            destroy = spy.start()
            self.addCleanup(spy.stop)
            created.append(destroy)
            return buffer

        with patch.object(device, "create_buffer", side_effect=record), patch.object(
            device.queue, "read_buffer", side_effect=RuntimeError("readback failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "readback failed"):
                self.backend.execute(source, np.ones(self.hidden), [("a", 1)],
                                     dtype="float32", **self.options)
        self.assertEqual(source.active, 0)
        self.assertEqual(len(created), 6)
        for destroy in created:
            destroy.assert_called_once()

    def test_nonfinite_compute_output_rejected(self):
        source = MappingPayloadSource({"a": np.asarray([1, 1, 1], dtype="<f4").tobytes()})
        with self.assertRaises(ArithmeticError):
            self.backend.execute(source, [1e20], [("a", 1)],
                                 hidden_size=1, intermediate_size=1, dtype="float32")

    def test_device_limits_reject_before_upload(self):
        with self.assertRaisesRegex(ValueError, "device buffer or dispatch limits"):
            self.backend._compute(np.ones((1, 1), dtype=np.float32),
                                  np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32),
                                  1, 2**32)

    def test_close_is_idempotent(self):
        backend = VulkanRoutedSwiGLUBackend()
        self.assertTrue(backend.info().available)
        backend.close()
        backend.close()
        self.assertFalse(backend.info().available)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.execute(MappingPayloadSource({}), [1], [("a", 1)],
                            hidden_size=1, intermediate_size=1, dtype="float32")


if __name__ == "__main__":
    unittest.main()
