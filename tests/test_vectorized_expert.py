from __future__ import annotations

from contextlib import contextmanager
import re
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import native_expert, vectorized_expert
from openagent.adaptive_engine.expert_kernels import (
    ExpertKernelRegistry, MappingPayloadSource, NumPyRoutedSwiGLUBackend,
)
from openagent.adaptive_engine.vectorized_expert import VectorizedExpertBackend
from tests.test_native_expert import encode


class AvailabilityTests(unittest.TestCase):
    def test_unavailable_fallback(self):
        with patch.object(vectorized_expert, "_compiled_loop", None), patch.object(
            vectorized_expert, "_unavailable", "Numba unavailable",
        ):
            backend = VectorizedExpertBackend()
            self.assertFalse(backend.info().available)
            registry = ExpertKernelRegistry([backend])
            self.assertIsInstance(registry.resolve(dtype="float32"), NumPyRoutedSwiGLUBackend)
            with self.assertRaises(RuntimeError):
                registry.resolve(dtype="float32", preferred=backend.NAME, allow_fallback=False)
            with self.assertRaisesRegex(RuntimeError, "Numba"):
                backend.execute_batch(None, None, None, hidden_size=1, intermediate_size=1, dtype="float32")


@unittest.skipUnless(VectorizedExpertBackend().info().available, "optional Numba unavailable")
class VectorizedTests(unittest.TestCase):
    def test_wide_odd_all_dtypes_weighted_routes(self):
        rng = np.random.default_rng(107)
        backend = VectorizedExpertBackend()
        reference = NumPyRoutedSwiGLUBackend()
        self.assertIn("non-bit-exact", backend.NUMERICAL_CLASS)
        self.assertFalse(backend.info().capabilities.native_simd)
        registry = ExpertKernelRegistry([backend])
        self.assertIs(registry.resolve(dtype="float32", preferred=backend.NAME), backend)
        for h, i in ((1, 1), (3, 7), (257, 129), (512, 256)):
            matrices = {key: [rng.normal(0, .05, size=s).astype(np.float32)
                              for s in ((i, h), (i, h), (h, i))] for key in ("a", "b", "c")}
            x = rng.normal(size=(4, h)).astype(np.float32)
            routes = [[("b", .7), ("a", .3), ("c", .2)], [("a", 1.)],
                      [("b", 0.)], [("c", .2), ("b", .8)]]
            for dtype in sorted(backend.info().capabilities.payload_dtypes):
                with self.subTest(h=h, i=i, dtype=dtype):
                    source = MappingPayloadSource({k: encode(v, dtype) for k, v in matrices.items()})
                    args = dict(hidden_size=h, intermediate_size=i, dtype=dtype)
                    actual = backend.execute_batch(source, x, routes, **args)
                    np.testing.assert_allclose(actual, reference.execute_batch(source, x, routes, **args), rtol=5e-5, atol=5e-6)
                    for row in range(4):
                        np.testing.assert_allclose(backend.execute(source, x[row], routes[row], **args), actual[row], rtol=5e-5, atol=5e-6)
                    np.testing.assert_array_equal(actual, backend.execute_batch(source, x, routes, **args))

    def test_borrowing_releases_and_no_native_dispatch(self):
        for dtype in ("int8-row", "float32"):
            raw = bytearray(encode([np.ones((3, 3), np.float32) * .1] * 3, dtype))
            leases = []
            original = vectorized_expert._compiled_loop

            class Source:
                @contextmanager
                def acquire(self, key):
                    leases.append(key)
                    try:
                        yield memoryview(raw)
                    finally:
                        leases.append("released")

            def check(*args):
                for matrix in args[3:6]:
                    self.assertTrue(np.shares_memory(matrix, np.frombuffer(raw, np.uint8)))
                return original(*args)

            backend = VectorizedExpertBackend()
            args = dict(hidden_size=3, intermediate_size=3, dtype=dtype)
            with patch.object(vectorized_expert, "_compiled_loop", side_effect=check), patch.object(
                native_expert, "_compiled_loop", side_effect=AssertionError("native dispatch"),
            ):
                first = backend.execute_batch(Source(), np.ones((2, 3)), [[("a", 1.)]] * 2, **args)
            self.assertEqual(leases, ["a", "released"])
            raw[:] = encode([np.ones((3, 3), np.float32) * .2] * 3, dtype)
            second = backend.execute(Source(), np.ones(3), [("a", 1.)], **args)
            self.assertFalse(np.array_equal(first[0], second))
            self.assertEqual(vars(backend), {})

    def test_validation_and_overflow(self):
        backend = VectorizedExpertBackend()
        args = dict(hidden_size=3, intermediate_size=3, dtype="float32")
        raw = encode([np.ones((3, 3), np.float32)] * 3, "float32")
        source = MappingPayloadSource({"a": raw})
        for routes in ([], [("a", -1.)], [("a", float("nan"))], [("a", 1.), ("a", 0.)]):
            with self.assertRaises(ValueError):
                backend.execute(source, np.ones(3), routes, **args)
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                backend.execute(source, [value, 1, 1], [("a", 1.)], **args)
            bad = encode([np.full((3, 3), value, np.float32)] * 3, "float32")
            with self.assertRaises(ValueError):
                backend.execute(MappingPayloadSource({"a": bad}), np.ones(3), [("a", 1.)], **args)
        with self.assertRaises(ValueError):
            backend.execute(MappingPayloadSource({"a": raw[:-1]}), np.ones(3), [("a", 1.)], **args)
        with self.assertRaises(ValueError):
            backend.execute_batch(source, np.ones((2, 3)), [[("a", 1.)]], **args)
        with self.assertRaises(ArithmeticError):
            backend.execute(source, np.full(3, 3e38, np.float32), [("a", 1.)], **args)
        self.assertEqual(backend.execute_batch(source, np.empty((0, 3)), [], **args).shape, (0, 3))
        quantized = bytearray(encode([np.ones((3, 3), np.float32)] * 3, "int8-row"))
        quantized[27:31] = np.array([0], dtype="<f4").tobytes()
        with self.assertRaises(ValueError):
            backend.execute(MappingPayloadSource({"a": quantized}), np.ones(3), [("a", 1.)], **dict(args, dtype="int8-row"))

    def test_compiler_flags_and_inspectable_vector_evidence(self):
        dispatcher = vectorized_expert._compiled_loop
        self.assertEqual(dispatcher.targetoptions["fastmath"], {"reassoc", "contract"})
        backend = VectorizedExpertBackend()
        raw = encode([np.ones((64, 64), np.float32) * .01] * 3, "float32")
        backend.execute(MappingPayloadSource({"a": raw}), np.ones(64), [("a", 1.)],
                        hidden_size=64, intermediate_size=64, dtype="float32")
        llvm = "\n".join(dispatcher.inspect_llvm(sig) for sig in dispatcher.signatures)
        assembly = "\n".join(dispatcher.inspect_asm(sig) for sig in dispatcher.signatures)
        self.assertTrue(dispatcher.nopython_signatures)
        self.assertNotRegex(llvm, r"\b(?:nnan|ninf|fast)\b")
        # Target-dependent proof is not a portable capability guarantee.
        vector_ir = re.search(r"f(?:add|mul)\s+[^\n]*<\d+ x float>", llvm)
        packed_asm = re.search(r"\b(?:v?addps|v?mulps|vfmadd\d*ps|fadd\s+v\d+\.4s|fmul\s+v\d+\.4s)\b", assembly)
        if not (vector_ir and packed_asm):
            self.skipTest("target did not prove vector FP32 arithmetic in both LLVM and assembly")


if __name__ == "__main__":
    unittest.main()
