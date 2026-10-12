"""Opt-in real Metal execution; a skipped run is never certification."""

from contextlib import contextmanager
import gc
import os
import platform
import unittest

import numpy as np

from openagent.adaptive_engine.expert_kernels import (
    MappingPayloadSource, NumPyRoutedSwiGLUBackend,
)
from openagent.adaptive_engine.torch_expert import TorchExpertBackend


class LeaseSource(MappingPayloadSource):
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


def payloads(dtype, hidden=17, intermediate=31):
    rng = np.random.default_rng(42)
    result = {}
    for key in ("a", "b", "unused"):
        count = 3 * hidden * intermediate
        if dtype == "int8-row":
            values = rng.integers(-8, 9, count, dtype=np.int8)
            scales = rng.uniform(.01, .04, 2 * intermediate + hidden).astype("<f4")
            result[key] = values.tobytes() + scales.tobytes()
        else:
            result[key] = rng.normal(0, .15, count).astype(
                "<f4" if dtype == "float32" else "<f2").tobytes()
    return result


@unittest.skipUnless(os.environ.get("OPENAGENT_CERTIFY_METAL") == "1",
                     "Metal certification not requested; NOT hardware verification")
class MetalCertificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if platform.system() != "Darwin":
            raise AssertionError("Metal certification requires a real macOS MPS host; "
                                 f"cannot verify Metal on {platform.system()}")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise AssertionError("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before Python starts")
        import torch
        cls.torch = torch
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise AssertionError("PyTorch MPS must be built and available; no CPU fallback")
        probe = torch.ones((17, 17), device="mps", dtype=torch.float32)
        result = probe @ probe
        if probe.device.type != "mps" or result.device.type != "mps":
            raise AssertionError("Probe did not execute on MPS")
        torch.mps.synchronize()
        np.testing.assert_array_equal(result.cpu().numpy(), np.full((17, 17), 17))
        print(f"Metal host: {platform.platform()}; Python {platform.python_version()}; "
              f"torch {torch.__version__}; machine {platform.machine()}", flush=True)

    def setUp(self):
        self.backend = TorchExpertBackend(device="mps")
        self.addCleanup(self.cleanup_backend)
        self.assertTrue(self.backend.info().available, self.backend.info().unavailable_reason)
        self.assertEqual(self.backend.device, "mps")
        self.assertTrue(self.backend.info().evidence[0].verified)

    def cleanup_backend(self):
        self.backend.close()
        self.assertFalse(self.backend.info().available)
        self.torch.mps.synchronize()
        gc.collect()
        self.torch.mps.empty_cache()

    def assert_accuracy(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.isfinite(expected).all())
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)

    def run_device(self, source, inputs, routes, config, *, batch=False):
        from torch.utils._python_dispatch import TorchDispatchMode

        owner = self
        observed = set()

        class PlacementAudit(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = str(func)
                if name in {"aten.mm.default", "aten.silu.default"}:
                    for value in args:
                        if isinstance(value, owner.torch.Tensor):
                            owner.assertEqual(value.device.type, "mps", name)
                            owner.assertEqual(value.dtype, owner.torch.float32, name)
                    observed.add(name)
                result = func(*args, **(kwargs or {}))
                if name in {"aten.mm.default", "aten.silu.default"}:
                    owner.assertEqual(result.device.type, "mps", name)
                    owner.assertEqual(result.dtype, owner.torch.float32, name)
                return result

        with PlacementAudit():
            method = self.backend.execute_batch if batch else self.backend.execute
            result = method(source, inputs, routes, **config)
        self.torch.mps.synchronize()
        self.assertEqual(observed, {"aten.mm.default", "aten.silu.default"})
        return result

    def test_scalar_and_grouped_batch_all_payloads(self):
        reference = NumPyRoutedSwiGLUBackend()
        x = np.random.default_rng(47).normal(0, .4, (4, 17)).astype(np.float32)
        routes = [[("b", .3), ("a", 1.7)], [("a", 0)],
                  [("a", .8), ("b", .2)], [("b", 1)]]
        for dtype in ("float32", "float16", "int8-row"):
            with self.subTest(dtype=dtype):
                config = dict(hidden_size=17, intermediate_size=31, dtype=dtype)
                data = payloads(dtype)
                source = LeaseSource(data)
                actual = self.run_device(source, x, routes, config, batch=True)
                self.assertEqual(source.keys, ["b", "a"])
                self.assertEqual(source.active, 0)
                expected = reference.execute_batch(MappingPayloadSource(data), x, routes, **config)
                self.assert_accuracy(actual, expected)
                self.assertTrue(np.any(expected != 0))
                np.testing.assert_array_equal(actual[1], 0)
                for row, route, target in zip(x, routes, expected):
                    self.assert_accuracy(self.run_device(source, row, route, config), target)
                self.assertNotIn("unused", source.keys)
                self.assertEqual(source.active, 0)

    def test_cleanup_errors_and_weight_refresh(self):
        config = dict(hidden_size=17, intermediate_size=31, dtype="float32")
        source = LeaseSource(payloads("float32"))
        x = np.full(17, .3, dtype=np.float32)
        first = self.backend.execute(source, x, [("a", 1)], **config)
        self.assertTrue(np.any(first != 0))
        source._payloads["a"] = bytes(3 * 17 * 31 * 4)
        np.testing.assert_array_equal(self.backend.execute(source, x, [("a", 1)], **config), 0)
        self.assertEqual(source.keys, ["a", "a"])
        for data in (b"bad", np.full(3 * 17 * 31, np.nan, dtype="<f4").tobytes()):
            bad = LeaseSource({"a": data})
            with self.assertRaises(ValueError):
                self.backend.execute(bad, x, [("a", 1)], **config)
            self.assertEqual(bad.active, 0)
        overflow = LeaseSource({"a": np.full(3 * 17 * 31, 1e30, dtype="<f4").tobytes()})
        with self.assertRaises(ArithmeticError):
            self.backend.execute(overflow, x, [("a", 1)], **config)
        self.assertEqual(overflow.active, 0)
        self.assertEqual(source.active, 0)
        self.backend.close()
        self.backend.close()
        with self.assertRaises(RuntimeError):
            self.backend.execute(source, x, [("a", 1)], **config)


if __name__ == "__main__":
    from tests.test_platform_certification_runner import run_certification

    raise SystemExit(run_certification(
        unittest.defaultTestLoader.loadTestsFromTestCase(MetalCertificationTests)))
