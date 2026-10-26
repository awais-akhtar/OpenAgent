"""Numerical and admission invariants for the independent media reference."""
import math
import unittest

import numpy as np

from openagent.adaptive_engine.independent_media_math import NumpyMediaMath


class IndependentMediaMathTests(unittest.TestCase):
    def test_attention_never_mixes_separate_frames(self):
        qkv = np.zeros((3, 12), np.float32)
        qkv[:, 8:] = [[1, 2, 3, 4], [3, 4, 5, 6], [99, 98, 97, 96]]
        actual = NumpyMediaMath().vision_attention(
            qkv, np.zeros((3, 2), np.int64), np.array([0, 2, 3], np.uint64), heads=1)
        np.testing.assert_array_equal(actual[:2], [[2, 3, 4, 5], [2, 3, 4, 5]])
        np.testing.assert_array_equal(actual[2], [99, 98, 97, 96])

    def test_attention_rotation_uses_both_spatial_axes(self):
        qkv = np.zeros((2, 12), np.float32)
        qkv[:, :8] = np.array([1, 2, 3, 4, 1, 2, 3, 4], np.float32)
        qkv[1, 8:] = 1
        math_ops = NumpyMediaMath()
        offsets = np.array([0, 2], np.uint64)
        flat = math_ops.vision_attention(qkv, np.zeros((2, 2), np.int64), offsets, heads=1)
        row = math_ops.vision_attention(qkv, np.array([[0, 0], [1, 0]], np.int64), offsets, heads=1)
        column = math_ops.vision_attention(qkv, np.array([[0, 0], [0, 1]], np.int64), offsets, heads=1)
        self.assertFalse(np.array_equal(flat, row))
        self.assertFalse(np.array_equal(flat, column))

    def test_attention_rejects_memory_and_malformed_frame_boundaries(self):
        qkv = np.ones((2, 12), np.float32)
        positions = np.zeros((2, 2), np.int64)
        with self.assertRaises(MemoryError):
            NumpyMediaMath(max_attention_bytes=1).vision_attention(
                qkv, positions, np.array([0, 2], np.uint64), heads=1)
        with self.assertRaises(ValueError):
            NumpyMediaMath().vision_attention(
                qkv, positions, np.array([0, 2, 1], np.uint64), heads=1)

    def test_linear_norm_and_exact_gelu_equations(self):
        ops = NumpyMediaMath()
        x = np.array([[1, 3]], np.float32)
        np.testing.assert_array_equal(ops.linear(x, np.array([[2, -1]], np.float32)), [[-1]])
        expected = np.array([[-1, 1]], np.float32) / np.float32(math.sqrt(1 + 1e-6))
        np.testing.assert_allclose(ops.norm(x, np.ones(2, np.float32)), expected, atol=1e-7)
        result = ops.activation(x)
        np.testing.assert_allclose(result, [[.5 * v * (1 + math.erf(v / math.sqrt(2)))
                                            for v in [1, 3]]], rtol=1e-7)

    def test_swiglu_clamps_positive_gate_and_both_up_limits(self):
        ops = NumpyMediaMath()
        gate = np.array([[-100, 100]], np.float32)
        up = np.array([[-100, 100]], np.float32)
        actual = ops.clamped_swiglu(gate, up, limit=2.)
        expected = ops.activation(np.array([[-100, 2]], np.float32), kind='silu') * [-2, 2]
        np.testing.assert_allclose(actual, expected, rtol=1e-7)
        self.assertTrue(np.isfinite(actual).all())


if __name__ == '__main__':
    unittest.main()
