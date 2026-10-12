import unittest
from contextlib import contextmanager
import numpy as np
from unittest.mock import patch
import openagent.adaptive_engine.packed_expert as packed_module

from openagent.adaptive_engine.packed_expert import (
    DTYPE, PackedInt4Backend, pack_matrix, pack_expert, payload_size, decode_expert,
)
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend


class PackedTests(unittest.TestCase):
    def weights(self, h=5, i=7):
        rng = np.random.default_rng(91)
        return tuple(rng.normal(0, .2, shape).astype(np.float32) for shape in ((i, h), (i, h), (h, i)))

    def test_layout_and_known_codes(self):
        raw, scales = pack_matrix(np.array([[-7, -1, 0, 1, 7], [0, 0, 0, 0, 0]], np.float32))
        self.assertEqual(raw, bytes([0xF9, 0x10, 7, 0, 0, 0]))
        np.testing.assert_array_equal(scales, [1, 1])
        for h, i in ((1, 1), (5, 7), (4, 6), (4, 7), (5, 6)):
            w = self.weights(h, i)
            parts = [pack_matrix(a, block_rows=1) for a in w]
            composed = b"".join(p for p, s in parts) + b"".join(s.tobytes() for p, s in parts)
            self.assertEqual(pack_expert(*w), composed)
            self.assertEqual(len(composed), payload_size(h, i))
            decoded = decode_expert(composed, h, i, block_rows=1)
            for a, b, (_, s) in zip(w, decoded, parts):
                bound = s[:, None].astype(np.float64) / 2 + 2 * np.finfo(np.float32).eps * np.max(np.abs(a), axis=1)[:, None]
                self.assertTrue(np.all(np.abs(a.astype(np.float64) - b) <= bound))

    def test_float_differential_and_batch(self):
        h, i = 5, 7
        w = self.weights(h, i)
        raw = pack_expert(*w)
        decoded = decode_expert(raw, h, i)
        float_raw = b"".join(a.astype("<f4").tobytes() for a in decoded)
        rng = np.random.default_rng(7)
        x = rng.normal(0, .3, (8, h)).astype(np.float32)
        routes = [[("a", .3), ("b", .7)] for _ in x]
        args = dict(hidden_size=h, intermediate_size=i)
        reference = NumPyRoutedSwiGLUBackend()
        expected = reference.execute_batch(MappingPayloadSource(dict(a=float_raw, b=float_raw)), x, routes, dtype="float32", **args)
        original = reference.execute_batch(MappingPayloadSource(dict(a=b"".join(a.tobytes() for a in w), b=b"".join(a.tobytes() for a in w))), x, routes, dtype="float32", **args)
        for block_rows in (1, 3, 32):
            backend = PackedInt4Backend(block_rows=block_rows)
            actual = backend.execute_batch(MappingPayloadSource(dict(a=raw, b=raw)), x, routes, **args)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-8)
            self.assertLess(np.linalg.norm(actual - original) / np.linalg.norm(original), .2)
            single = backend.execute(MappingPayloadSource(dict(a=raw, b=raw)), x[0], routes[0], **args)
            np.testing.assert_allclose(single, actual[0], rtol=1e-6, atol=1e-8)
        self.assertEqual(PackedInt4Backend().info().capabilities.payload_dtypes, frozenset({DTYPE}))

    def test_propagated_absolute_error_and_bounded_blocks(self):
        h, i = 17, 23
        w = self.weights(h, i)
        raw = pack_expert(*w, block_rows=2)
        d = decode_expert(raw, h, i)
        x = np.random.default_rng(12).normal(0, .3, h).astype(np.float32)
        g, u = w[0].astype(np.float64) @ x, w[1].astype(np.float64) @ x
        silu = lambda a: a * np.exp(-np.logaddexp(0, -a))
        a = silu(g) * u
        eg = np.abs(d[0].astype(np.float64) - w[0]) @ np.abs(x)
        eu = np.abs(d[1].astype(np.float64) - w[1]) @ np.abs(x)
        # SiLU is globally 2-Lipschitz (a conservative bound).
        ea = 2 * eg * (np.abs(u) + eu) + np.abs(silu(g)) * eu
        bound = np.abs(d[2].astype(np.float64)) @ ea + np.abs(d[2].astype(np.float64) - w[2]) @ np.abs(a)
        observed = []
        original_blocks = packed_module._blocks
        def blocks(*args):
            for start, block in original_blocks(*args):
                observed.append(block.shape)
                yield start, block
        with patch.object(packed_module, "_blocks", blocks):
            actual = PackedInt4Backend(block_rows=2).execute(
                MappingPayloadSource({"a": raw}), x, [("a", 1)],
                hidden_size=h, intermediate_size=i)
        expected = w[2].astype(np.float64) @ a
        self.assertTrue(np.all(np.abs(actual - expected) <= bound + 1e-6))
        self.assertTrue(observed)
        self.assertTrue(all(rows <= 2 and cols <= max(h, i) for rows, cols in observed))

    def test_reject_corruption(self):
        raw = pack_expert(*self.weights())
        bad = [raw[:-1], raw + b"\0"]
        reserved = bytearray(raw); reserved[0] = 8; bad.append(reserved)
        padding = bytearray(raw); padding[2] |= 16; bad.append(padding)
        scale_offset = 7 * 3 * 2 + 5 * 4
        for value in (0, -1, np.nan, np.inf, np.finfo(np.float32).max):
            damaged = bytearray(raw)
            damaged[scale_offset:scale_offset + 4] = np.array([value], dtype="<f4").tobytes()
            bad.append(damaged)
        for damaged in bad:
            with self.assertRaises(ValueError): decode_expert(damaged, 5, 7)
        with self.assertRaises(ValueError): decode_expert(raw, 7, 5)
        with self.assertRaises(ValueError): decode_expert(memoryview(raw)[::2], 5, 7)

    def test_finite_extremes_and_validation(self):
        for value in (np.nan, np.inf, -np.inf, 1e100):
            with self.assertRaises(ValueError): pack_matrix([[value]])
        for value in (0, np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).max):
            raw = pack_expert(*[np.array([[value]], np.float32)] * 3)
            self.assertTrue(all(np.isfinite(a).all() for a in decode_expert(raw, 1, 1)))
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): payload_size(value, 1)
            with self.assertRaises(ValueError): PackedInt4Backend(block_rows=value)
        with self.assertRaises(ValueError): pack_expert(*self.weights(), block_rows=0)
        with self.assertRaises(ValueError): pack_matrix([[1j]])
        with self.assertRaises(ValueError): pack_expert(np.ones((2, 3)), np.ones((3, 2)), np.ones((3, 2)))

    def test_lease_cleanup_and_execution_errors(self):
        class Source:
            active = False
            @contextmanager
            def acquire(self, key):
                self.active = True
                try: yield b"bad"
                finally: self.active = False
        source = Source()
        args = dict(hidden_size=1, intermediate_size=1)
        backend = PackedInt4Backend()
        with self.assertRaises(ValueError): backend.execute(source, [1], [("a", 1)], **args)
        self.assertFalse(source.active)
        valid = MappingPayloadSource({"a": pack_expert(*[np.ones((1, 1))] * 3)})
        for routes in ([], [("a", -1)], [("a", np.nan)], [("a", 1), ("a", 1)], [("a", 1e100)]):
            with self.assertRaises(ValueError): backend.execute(valid, [1], routes, **args)
        with self.assertRaises(ValueError): backend.execute(valid, [np.nan], [("a", 1)], **args)
        with self.assertRaises(ArithmeticError): backend.execute(valid, [np.finfo(np.float32).max], [("a", 1)], **args)
        with self.assertRaises(ValueError): backend.execute_batch(valid, [[1]], [], **args)
        self.assertEqual(backend.execute_batch(valid, np.empty((0, 1)), [], **args).shape, (0, 1))


if __name__ == "__main__":
    unittest.main()
