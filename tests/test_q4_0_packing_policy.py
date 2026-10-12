import struct
import unittest

import numpy as np

from openagent.adaptive_engine.q4_0_expert import pack_matrix


class Q4PackingPolicyTests(unittest.TestCase):
    def test_vectorized_chunks_preserve_scalar_policy_bytes(self):
        rng = np.random.default_rng(304)
        weights = rng.normal(size=(7, 160)).astype(np.float32)
        weights[0].fill(0)
        weights[1] *= np.float32(2 ** -24)
        weights[2] *= np.float32(10000)
        expected = bytearray()
        for block in weights.reshape(-1, 32):
            maximum = float(np.max(np.abs(block)))
            delta = np.float16(maximum / 7)
            if maximum and delta == 0:
                delta = np.float16(2 ** -24)
            codes = np.clip(np.rint(block / (float(delta) or 1)), -8, 7).astype(np.int16) + 8
            expected.extend(struct.pack('<e', float(delta)))
            expected.extend((codes[:16] | (codes[16:] << 4)).astype(np.uint8).tobytes())
        for block_count in (1, 3, 64, 128):
            with self.subTest(block_count=block_count):
                self.assertEqual(pack_matrix(weights, block_count=block_count), bytes(expected))
