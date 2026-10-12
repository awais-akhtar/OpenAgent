"""Differential tests of actual JIT execution over packed weight bytes."""

import unittest

import numpy as np

from openagent.adaptive_engine.expert_kernels import MappingPayloadSource
from openagent.adaptive_engine.native_packed_expert import NativePackedExpertBackend
from openagent.adaptive_engine.packed_expert import PackedInt4Backend, pack_expert


class NativePackedTests(unittest.TestCase):
    def test_unavailable_execution_is_explicit(self):
        from unittest.mock import patch

        with patch("openagent.adaptive_engine.native_packed_expert._compiled_loop", None):
            self.assertFalse(NativePackedExpertBackend().info().available)
            with self.assertRaises(RuntimeError):
                NativePackedExpertBackend().execute_batch(None, [], [], hidden_size=1, intermediate_size=1)

    @unittest.skipUnless(NativePackedExpertBackend().info().available, "optional Numba unavailable")
    def test_tiny_scales_do_not_overflow_unscaled_accumulation(self):
        gate = np.full((1, 2), 1e-38, np.float32)
        down = np.ones((2, 1), np.float32)
        source = MappingPayloadSource({"a": pack_expert(gate, gate, down)})
        x = np.full(2, 1e38, np.float32)
        args = dict(hidden_size=2, intermediate_size=1)
        expected = PackedInt4Backend().execute(source, x, [("a", 1)], **args)
        self.assertTrue(np.isfinite(expected).all())
        actual = NativePackedExpertBackend().execute(source, x, [("a", 1)], **args)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)

    @unittest.skipUnless(NativePackedExpertBackend().info().available, "optional Numba unavailable")
    def test_jit_matches_reference_for_odd_and_even_shapes(self):
        rng = np.random.default_rng(447)
        reference, backend = PackedInt4Backend(), NativePackedExpertBackend()
        for h, i in ((1, 1), (5, 7), (6, 8), (17, 31)):
            with self.subTest(hidden=h, intermediate=i):
                payloads = {
                    key: pack_expert(*(rng.normal(0, .2, shape).astype(np.float32)
                                       for shape in ((i, h), (i, h), (h, i))))
                    for key in ("a", "b")
                }
                source = MappingPayloadSource(payloads)
                x = rng.normal(0, .3, (4, h)).astype(np.float32)
                routes = [[("a", .3), ("b", .7)], [("b", 1)], [("a", 0)], [("a", 1)]]
                args = dict(hidden_size=h, intermediate_size=i)
                expected = reference.execute_batch(source, x, routes, **args)
                actual = backend.execute_batch(source, x, routes, **args)
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)
                np.testing.assert_array_equal(actual[2], 0)
                np.testing.assert_allclose(backend.execute(source, x[0], routes[0], **args), expected[0], rtol=3e-5, atol=2e-7)
