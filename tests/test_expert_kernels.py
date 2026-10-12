from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from openagent.adaptive_engine.expert_kernels import (
    ExpertKernelRegistry,
    KernelBackendInfo,
    KernelCapabilities,
    KernelEvidence,
    MappingPayloadSource,
    NumPyRoutedSwiGLUBackend,
)


def payload(weights: np.ndarray, dtype: str) -> bytes:
    return np.asarray(weights, dtype="<f4" if dtype == "float32" else "<f2").tobytes()


def int8_payload(weights: np.ndarray, hidden: int, intermediate: int) -> tuple[bytes, np.ndarray]:
    matrices = [
        weights[: hidden * intermediate].reshape(intermediate, hidden),
        weights[hidden * intermediate:2 * hidden * intermediate].reshape(intermediate, hidden),
        weights[2 * hidden * intermediate:].reshape(hidden, intermediate),
    ]
    quantized = []
    scales = []
    reconstructed = []
    for matrix in matrices:
        scale = np.maximum(np.max(np.abs(matrix), axis=1), np.float32(1e-12)) / np.float32(127)
        values = np.clip(np.rint(matrix / scale[:, None]), -128, 127).astype(np.int8)
        quantized.append(values)
        scales.append(scale.astype("<f4"))
        reconstructed.append(values.astype(np.float32) * scale[:, None])
    raw = b"".join(item.tobytes() for item in quantized + scales)
    return raw, np.concatenate([item.reshape(-1) for item in reconstructed])


def resident_reference(x: np.ndarray, weights: np.ndarray, hidden: int, intermediate: int) -> np.ndarray:
    width = hidden * intermediate
    gate = weights[:width].reshape(intermediate, hidden) @ x
    up = weights[width:2 * width].reshape(intermediate, hidden) @ x
    return weights[2 * width:].reshape(hidden, intermediate) @ (gate / (1 + np.exp(-gate)) * up)


class NumPyKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(823)
        self.hidden = 5
        self.intermediate = 9
        self.vector = self.rng.normal(size=self.hidden).astype(np.float32)
        self.weights = [
            self.rng.normal(size=3 * self.hidden * self.intermediate).astype(np.float32)
            for _ in range(3)
        ]
        self.backend = NumPyRoutedSwiGLUBackend()

    def test_float_payloads_match_resident_reference(self) -> None:
        routes = [("b", 0.7), ("a", 0.3)]
        for dtype, tolerance in (("float32", 1e-5), ("float16", 2e-3)):
            source = MappingPayloadSource({name: payload(self.weights[index], dtype) for index, name in enumerate("abc")})
            actual = self.backend.execute(
                source, self.vector, routes,
                hidden_size=self.hidden, intermediate_size=self.intermediate, dtype=dtype,
            )
            expected = sum(
                scale * resident_reference(
                    self.vector,
                    self.weights["abc".index(name)].astype("<f4" if dtype == "float32" else "<f2").astype(np.float32),
                    self.hidden,
                    self.intermediate,
                )
                for name, scale in routes
            )
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            self.assertEqual(actual.dtype, np.float32)

    def test_row_int8_matches_explicit_dequantized_reference(self) -> None:
        encoded = {}
        dequantized = {}
        for index, name in enumerate("abc"):
            encoded[name], dequantized[name] = int8_payload(self.weights[index], self.hidden, self.intermediate)
        routes = [("a", 0.2), ("c", 0.8)]
        actual = self.backend.execute(
            MappingPayloadSource(encoded), self.vector, routes,
            hidden_size=self.hidden, intermediate_size=self.intermediate, dtype="int8-row",
        )
        expected = sum(
            scale * resident_reference(self.vector, dequantized[name], self.hidden, self.intermediate)
            for name, scale in routes
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_batch_is_differentially_equal_to_individual_execution(self) -> None:
        source = MappingPayloadSource({name: payload(self.weights[index], "float32") for index, name in enumerate("abc")})
        batch = self.rng.normal(size=(4, self.hidden)).astype(np.float32)
        routes = [[("a", 1.0)], [("b", 0.4), ("c", 0.6)], [("a", 0.2), ("c", 0.8)], [("b", 1.0)]]
        actual = self.backend.execute_batch(
            source, batch, routes,
            hidden_size=self.hidden, intermediate_size=self.intermediate, dtype="float32",
        )
        expected = np.stack([
            self.backend.execute(
                source, row, row_routes,
                hidden_size=self.hidden, intermediate_size=self.intermediate, dtype="float32",
            )
            for row, row_routes in zip(batch, routes)
        ])
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_validation_rejects_ambiguous_or_corrupt_inputs(self) -> None:
        valid = payload(self.weights[0], "float32")
        source = MappingPayloadSource({"a": valid, "short": valid[:-1], "nan": payload(np.full_like(self.weights[0], np.nan), "float32")})
        arguments = dict(hidden_size=self.hidden, intermediate_size=self.intermediate, dtype="float32")
        invalid = [
            (self.vector[:-1], [("a", 1.0)]),
            (np.full(self.hidden, np.inf), [("a", 1.0)]),
            (self.vector, []),
            (self.vector, [("a", 1.0), ("a", 0.0)]),
            (self.vector, [("a", -0.1)]),
            (self.vector, [("a", float("nan"))]),
        ]
        for vector, routes in invalid:
            with self.subTest(routes=routes), self.assertRaises((TypeError, ValueError)):
                self.backend.execute(source, vector, routes, **arguments)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.backend.execute(source, self.vector, [("short", 1.0)], **arguments)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.backend.execute(source, self.vector, [("nan", 1.0)], **arguments)
        with self.assertRaises(KeyError):
            self.backend.execute(source, self.vector, [("missing", 1.0)], **arguments)
        with self.assertRaisesRegex(ValueError, "one route list"):
            self.backend.execute_batch(source, [self.vector], [], **arguments)

    def test_invalid_int8_scales_are_rejected(self) -> None:
        raw, _ = int8_payload(self.weights[0], self.hidden, self.intermediate)
        corrupted = bytearray(raw)
        scale_offset = 3 * self.hidden * self.intermediate
        corrupted[scale_offset:scale_offset + 4] = np.asarray([0.0], dtype="<f4").tobytes()
        with self.assertRaisesRegex(ValueError, "scales"):
            self.backend.execute(
                MappingPayloadSource({"a": corrupted}), self.vector, [("a", 1.0)],
                hidden_size=self.hidden, intermediate_size=self.intermediate, dtype="int8-row",
            )


class StubBackend:
    def __init__(self, info: KernelBackendInfo) -> None:
        self._info = info

    def info(self) -> KernelBackendInfo:
        return self._info

    def execute(self, *args, **kwargs):
        return np.asarray([1.0], dtype=np.float32)

    def execute_batch(self, *args, **kwargs):
        return np.asarray([[1.0]], dtype=np.float32)


def backend_info(name: str, *, available: bool = True, priority: int = 10,
                 batch: bool = True, dtypes=frozenset({"float32"})) -> KernelBackendInfo:
    return KernelBackendInfo(
        name=name,
        provider="test-provider",
        available=available,
        priority=priority,
        unavailable_reason="dependency missing" if not available else "",
        capabilities=KernelCapabilities(dtypes, batch_execution=batch, deterministic=True, compiled=True),
        evidence=(KernelEvidence("binary", f"{name}.pyd loaded", verified=True),),
    )


class RegistryTests(unittest.TestCase):
    def test_selection_is_priority_then_name_and_fallback_is_reference(self) -> None:
        registry = ExpertKernelRegistry([
            StubBackend(backend_info("zeta", priority=30)),
            StubBackend(backend_info("alpha", priority=30)),
            StubBackend(backend_info("offline", available=False, priority=100)),
        ])
        self.assertEqual(registry.resolve(dtype="float32").info().name, "alpha")
        self.assertEqual(registry.resolve(dtype="float16").info().name, "numpy-reference")
        self.assertEqual(
            registry.resolve(dtype="float32", preferred="offline").info().name,
            "alpha",
        )
        self.assertEqual([item.name for item in registry.describe()], sorted(item.name for item in registry.describe()))

    def test_strict_resolution_and_registration_errors_are_explicit(self) -> None:
        registry = ExpertKernelRegistry()
        with self.assertRaises(KeyError):
            registry.resolve(dtype="float32", preferred="absent", allow_fallback=False)
        limited = StubBackend(backend_info("scalar-only", batch=False))
        registry.register(limited)
        with self.assertRaises(RuntimeError):
            registry.resolve(dtype="float32", batch=True, preferred="scalar-only", allow_fallback=False)
        self.assertEqual(registry.resolve(dtype="float32", batch=True, preferred="scalar-only").info().name, "numpy-reference")
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(limited)
        with self.assertRaisesRegex(ValueError, "cannot be removed"):
            registry.unregister("numpy-reference")

    def test_native_simd_claim_requires_verified_provider_evidence(self) -> None:
        capabilities = KernelCapabilities(
            frozenset({"float32"}), batch_execution=True, deterministic=True,
            compiled=True, native_simd=True,
        )
        with self.assertRaisesRegex(ValueError, "verified provider evidence"):
            KernelBackendInfo(
                name="unproven-simd", provider="test", available=True, priority=50,
                capabilities=capabilities,
                evidence=(KernelEvidence("assertion", "claims AVX2", verified=False),),
            )
        proven = StubBackend(KernelBackendInfo(
            name="proven-simd", provider="test", available=True, priority=50,
            capabilities=capabilities,
            evidence=(KernelEvidence("loaded-binary", "provider module reports AVX2 kernel", verified=True),),
        ))
        registry = ExpertKernelRegistry([proven])
        self.assertTrue(registry.resolve(dtype="float32").info().capabilities.native_simd)

    def test_metadata_invariants_reject_false_or_incomplete_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "compiled backend"):
            KernelCapabilities(frozenset({"float32"}), True, True, native_simd=True)
        with self.assertRaisesRegex(ValueError, "explain why"):
            replace(backend_info("broken"), available=False, unavailable_reason="")
        with self.assertRaises(ValueError):
            KernelCapabilities(frozenset({"int4"}), True, True)


if __name__ == "__main__":
    unittest.main()
