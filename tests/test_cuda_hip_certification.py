"""Opt-in real-device certification; enabled runs never skip missing hardware."""

from contextlib import contextmanager
import json
import os
import platform
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.expert_kernels import (
    MappingPayloadSource, NumPyRoutedSwiGLUBackend, SUPPORTED_PAYLOAD_DTYPES,
)
from openagent.adaptive_engine.torch_expert import TorchExpertBackend


class LeaseSource(MappingPayloadSource):
    def __init__(self, payloads):
        super().__init__(payloads)
        self.keys = []
        self.active = 0
        self.peak = 0

    @contextmanager
    def acquire(self, key):
        self.keys.append(key)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            with super().acquire(key) as view:
                yield view
        finally:
            self.active -= 1


def make_payloads(dtype, hidden, intermediate):
    rng = np.random.default_rng(734)
    count = 3 * hidden * intermediate
    payloads = {}
    for key in ("a", "b", "unused"):
        if dtype == "int8-row":
            values = rng.integers(-128, 128, count, dtype=np.int16).astype(np.int8)
            values[:2] = [-128, 127]
            scales = np.linspace(.0001, .01, 2 * intermediate + hidden).astype("<f4")
            payloads[key] = values.tobytes() + scales.tobytes()
        else:
            values = rng.uniform(-.15, .15, count).astype(
                "<f4" if dtype == "float32" else "<f2")
            payloads[key] = values.tobytes()
    return payloads


@unittest.skipUnless(os.environ.get("OPENAGENT_CUDA_HIP_CERTIFY") == "1",
                     "opt in with OPENAGENT_CUDA_HIP_CERTIFY=1")
class CudaHipCertificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report = {"python": platform.python_version(), "platform": platform.platform()}
        try:
            import torch
            report.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                          hip=torch.version.hip, available=torch.cuda.is_available(),
                          device_count=torch.cuda.device_count())
            print("CUDA_HIP_RUNTIME " + json.dumps(report, sort_keys=True), flush=True)
            if not report["available"] or not report["device_count"]:
                raise RuntimeError("no usable CUDA/HIP device (CPU and Metal are disallowed)")
            if not (report["cuda"] or report["hip"]):
                raise RuntimeError("runtime has neither CUDA nor HIP identity")
            device = os.environ.get("OPENAGENT_CUDA_HIP_DEVICE", "cuda:0")
            if not device.startswith("cuda:") or not device[5:].isdigit():
                raise RuntimeError("OPENAGENT_CUDA_HIP_DEVICE must be cuda:N, including on HIP")
            cls.backend = TorchExpertBackend(device=device)
            if not cls.backend.info().available or cls.backend.device != device:
                raise RuntimeError(cls.backend.info().unavailable_reason or "device mismatch")
            cls.addClassCleanup(cls.backend.close)
            cls.torch = torch
            props = torch.cuda.get_device_properties(device)
            report.update(device=device, name=props.name, total_memory=props.total_memory,
                          runtime="HIP" if report["hip"] else "CUDA",
                          capability=list(torch.cuda.get_device_capability(device)))
            cls.addClassCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32",
                                torch.backends.cuda.matmul.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = False
            print("CUDA_HIP_DEVICE " + json.dumps(report, sort_keys=True), flush=True)
        except Exception as exc:
            print("CUDA_HIP_BLOCKED " + json.dumps(dict(report, reason=str(exc))), flush=True)
            raise RuntimeError(f"strict hardware certification blocked: {exc}") from exc

    def run_device(self, source, inputs, routes, config, *, batch=True, compute=True):
        from torch.utils._python_dispatch import TorchDispatchMode

        expected = self.torch.device(self.backend.device)
        observed = set()
        owner = self

        class PlacementAudit(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # Inspect real dispatched math, not only the backend's device label.
                name = str(func)
                if name in {"aten.mm.default", "aten.silu.default"}:
                    for value in args:
                        if isinstance(value, owner.torch.Tensor):
                            owner.assertEqual(value.device, expected, name)
                            owner.assertEqual(value.dtype, owner.torch.float32, name)
                    observed.add(name)
                result = func(*args, **(kwargs or {}))
                if name in {"aten.mm.default", "aten.silu.default"}:
                    owner.assertEqual(result.device, expected, name)
                return result

        with PlacementAudit():
            method = self.backend.execute_batch if batch else self.backend.execute
            result = method(source, inputs, routes, **config)
        self.torch.cuda.synchronize(expected)
        if compute:
            self.assertEqual(observed, {"aten.mm.default", "aten.silu.default"})
        self.assertEqual(self.backend.device, str(expected))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(source.active, 0)
        self.assertLessEqual(source.peak, 1)
        return result

    def test_payloads_scalar_batch_routes_and_dimensions(self):
        self.assertEqual(SUPPORTED_PAYLOAD_DTYPES, {"float32", "float16", "int8-row"})
        for dtype in sorted(SUPPORTED_PAYLOAD_DTYPES):
            for hidden, intermediate in ((1, 1), (3, 4), (17, 33), (64, 65)):
                with self.subTest(dtype=dtype, hidden=hidden, intermediate=intermediate):
                    config = dict(hidden_size=hidden, intermediate_size=intermediate, dtype=dtype)
                    payloads = make_payloads(dtype, hidden, intermediate)
                    x = np.random.default_rng(19).uniform(-1, 1, (4, hidden)).astype(np.float32)
                    # Non-contiguous input and unnormalized, zero, reversed and shared routes.
                    x = np.repeat(x, 2, axis=1)[:, ::2]
                    routes = [[("b", .25), ("a", 1.75)], [("a", 0)],
                              [("a", .8), ("b", .2)], [("b", 1)]]
                    source = LeaseSource(payloads)
                    actual = self.run_device(source, x, routes, config)
                    self.assertEqual(source.keys, ["b", "a"])
                    expected = NumPyRoutedSwiGLUBackend().execute_batch(
                        MappingPayloadSource(payloads), x, routes, **config)
                    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-6)
                    self.assertEqual(actual.shape, x.shape)
                    np.testing.assert_array_equal(actual[1], 0)
                    for index, route in enumerate(routes):
                        single = self.run_device(LeaseSource(payloads), x[index], route,
                                                 config, batch=False)
                        scalar = NumPyRoutedSwiGLUBackend().execute(
                            MappingPayloadSource(payloads), x[index], route, **config)
                        np.testing.assert_allclose(single, scalar, rtol=5e-5, atol=5e-6)
                        np.testing.assert_allclose(single, actual[index], rtol=5e-5, atol=5e-6)
                    print(f"CUDA_HIP_PASS payload={dtype} shape={hidden}x{intermediate} "
                          "scalar,batch,routes,placement,leases", flush=True)

    def test_payload_boundaries_and_lease_cleanup(self):
        for dtype in sorted(SUPPORTED_PAYLOAD_DTYPES):
            config = dict(hidden_size=3, intermediate_size=4, dtype=dtype)
            valid = make_payloads(dtype, 3, 4)["a"]
            bad_payloads = [valid[:-1], valid + b"\0"]
            if dtype == "int8-row":
                for scale in (0, -1, float("nan"), float("inf")):
                    bad_payloads.append(valid[:36] + np.full(11, scale, dtype="<f4").tobytes())
            else:
                for value in (float("nan"), float("inf")):
                    bad_payloads.append(np.full(36, value, dtype=
                                                "<f4" if dtype == "float32" else "<f2").tobytes())
            for payload in bad_payloads:
                with self.subTest(dtype=dtype, size=len(payload)):
                    source = LeaseSource({"a": payload})
                    with self.assertRaises(ValueError):
                        self.run_device(source, [[1, 2, 3]], [[("a", 1)]], config)
                    self.assertEqual(source.active, 0)
            source = LeaseSource({"a": valid})
            with self.assertRaises(KeyError):
                self.run_device(source, [[1, 2, 3]], [[("missing", 1)]], config)
            self.assertEqual(source.active, 0)
        print("CUDA_HIP_PASS malformed sizes,nonfinite payloads,invalid scales,missing keys,cleanup",
              flush=True)

    def test_input_route_configuration_boundaries(self):
        config = dict(hidden_size=3, intermediate_size=4, dtype="float32")
        cases = [([[1, 2]], [[("a", 1)]], config),
                 ([[1, 2, np.nan]], [[("a", 1)]], config),
                 ([[1, 2, np.inf]], [[("a", 1)]], config),
                 ([[1, 2, 3]], [], config)]
        for routes in ([], [("a", -1)], [("a", np.inf)], [("a", np.nan)],
                       [("a", True)], [("a", 1), ("a", 2)], [("", 1)]):
            cases.append(([[1, 2, 3]], [routes], config))
        for field, value in (("hidden_size", 0), ("intermediate_size", -1),
                             ("hidden_size", True), ("dtype", "int4")):
            cases.append(([[1, 2, 3]], [[("a", 1)]], dict(config, **{field: value})))
        for inputs, routes, settings in cases:
            with self.subTest(settings=settings, routes=routes):
                source = LeaseSource(make_payloads("float32", 3, 4))
                with self.assertRaises(ValueError):
                    self.run_device(source, inputs, routes, settings)
                self.assertEqual(source.keys, [])
        source = LeaseSource({})
        empty = self.run_device(source, np.empty((0, 3)), [], config, compute=False)
        self.assertEqual(empty.shape, (0, 3))
        self.assertEqual(source.keys, [])
        print("CUDA_HIP_PASS input,route,configuration rejection; empty batch", flush=True)

    def test_reloads_overflow_and_closed_backend(self):
        config = dict(hidden_size=3, intermediate_size=4, dtype="float32")
        source = LeaseSource(make_payloads("float32", 3, 4))
        args = (source, [[1, 2, 3]], [[("a", 1)]], config)
        first = self.run_device(*args)
        source._payloads["a"] = bytes(36 * 4)
        np.testing.assert_array_equal(self.run_device(*args), 0)
        self.assertTrue(np.any(first != 0))
        self.assertEqual(source.keys, ["a", "a"])
        source._payloads["a"] = np.full(36, 1e30, dtype="<f4").tobytes()
        with self.assertRaises(ArithmeticError):
            self.run_device(*args)
        self.assertEqual(source.active, 0)
        backend = TorchExpertBackend(device=self.backend.device)
        self.assertTrue(backend.info().available, backend.info().unavailable_reason)
        backend.close()
        backend.close()
        with self.assertRaises(RuntimeError):
            backend.execute_batch(LeaseSource({}), np.empty((0, 3)), [], **config)
        print("CUDA_HIP_PASS reloads,nonfinite output rejection,idempotent close,no closed fallback",
              flush=True)

    def test_device_execution_failure_never_falls_back(self):
        source = LeaseSource(make_payloads("float32", 3, 4))
        with patch.object(self.torch, "tensor", side_effect=RuntimeError("device allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "device allocation failed"):
                self.backend.execute_batch(
                    source, [[1, 2, 3]], [[("a", 1)]], hidden_size=3,
                    intermediate_size=4, dtype="float32")
        self.assertEqual(source.keys, [])
        self.assertEqual(self.backend.device, os.environ.get("OPENAGENT_CUDA_HIP_DEVICE", "cuda:0"))
        print("CUDA_HIP_PASS injected allocation failure propagates without CPU fallback", flush=True)


if __name__ == "__main__":
    from tests.test_platform_certification_runner import run_certification

    raise SystemExit(run_certification(
        unittest.defaultTestLoader.loadTestsFromTestCase(CudaHipCertificationTests)))
