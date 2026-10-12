"""Independent hand-built Q4_0 fixtures and converter contract tests."""

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.q4_0_expert import (decode_expert, pack_expert,
                                                 pack_matrix, payload_size)
from openagent.adaptive_engine.q4_0_conversion import convert_experts


def scalar_decode(blob, h, i):
    """Test oracle uses struct and scalar indexing, not production helpers."""
    values = []
    for address in range(0, len(blob), 18):
        delta, *nibbles = struct.unpack_from("<e16B", blob, address)
        values.extend((byte % 16 - 8) * delta for byte in nibbles)
        values.extend((byte // 16 - 8) * delta for byte in nibbles)
    matrix_elements = h * i
    return tuple(np.array(values[n * matrix_elements:(n + 1) * matrix_elements], np.float32).reshape(shape)
                 for n, shape in enumerate(((i, h), (i, h), (h, i))))


class Q4FormatTests(unittest.TestCase):
    def test_handbuilt_split_nibbles_all_codes_and_delta_boundaries(self):
        for delta in (65504, -65504, 2 ** -24, -(2 ** -24), 0.0, -0.0, .125):
            block = struct.pack("<e16B", delta, *[j | ((15 - j) << 4) for j in range(16)])
            blob = block * 96
            for actual, expected in zip(decode_expert(blob, 32, 32), scalar_decode(blob, 32, 32)):
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(actual.flags.owndata)
            np.testing.assert_array_equal(decode_expert(blob, 32, 32)[0][0, :16],
                                          (np.arange(16) - 8) * delta)

    def test_converter_policy_and_chunk_independence(self):
        matrix = np.tile(np.arange(-8, 24, dtype=np.float32), (3, 2))
        blob = pack_matrix(matrix, block_count=1)
        self.assertEqual(blob, pack_matrix(matrix, block_count=7))
        # A known exactly representable delta of 1 and ties-to-even values.
        block = np.array([-7, 7, .5, 1.5, 2.5, -1.5] + [0] * 26, np.float32)[None]
        known = pack_matrix(block)
        self.assertEqual(struct.unpack_from("<e", known)[0], 1)
        codes = [1, 15, 8, 10, 10, 6] + [8] * 26
        self.assertEqual(known[2:], bytes(codes[j] | (codes[j + 16] << 4) for j in range(16)))
        self.assertEqual(pack_matrix(np.zeros((1, 32))), b"\0\0" + b"\x88" * 16)
        tiny = pack_matrix(np.full((1, 32), 1e-40))
        self.assertEqual(struct.unpack_from("<e", tiny)[0], 2 ** -24)

    def test_finite_shapes_and_malformed(self):
        for h, i in ((0, 32), (32, 31), (True, 32), (32.0, 32)):
            with self.assertRaises(ValueError):
                payload_size(h, i)
        for value in (np.nan, np.inf, -np.inf, 1e100, 1e7):
            with self.assertRaises(ValueError):
                pack_matrix(np.full((1, 32), value))
        for matrix in (np.ones((1, 31)), np.ones((0, 32)), np.ones((1, 32), complex)):
            with self.assertRaises(ValueError):
                pack_matrix(matrix)
        for count in (0, True, 1.5):
            with self.assertRaises(ValueError):
                pack_matrix(np.ones((1, 32)), block_count=count)
        good = pack_expert(*[np.zeros((32, 32))] * 3)
        for blob in (good[:-1], good + b"x", good[:-18] + struct.pack("<e", float("nan")) + good[-16:]):
            with self.assertRaises(ValueError):
                decode_expert(blob, 32, 32)
        with self.assertRaises(ValueError):
            decode_expert(memoryview(good)[::2], 32, 32)
        with self.assertRaises(ValueError):
            pack_expert(np.ones((32, 32)), np.ones((32, 64)), np.ones((32, 32)))

    def test_container_hash_offsets_and_failure_atomicity(self):
        matrices = [np.full((32, 32), .05)] * 3
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "experts.q4"
            result = convert_experts({"a": matrices, "b": matrices}, target,
                                     hidden_size=32, intermediate_size=32)
            blob = target.read_bytes()
            size = payload_size(32, 32)
            self.assertEqual(len(blob), 2 * size)
            self.assertEqual(result.dtype, "q4_0")
            for index, region in enumerate(result.regions.values()):
                self.assertEqual(region.offset, index * size)
                self.assertEqual(region.size, size)
                self.assertEqual(region.sha256, hashlib.sha256(blob[index * size:(index + 1) * size]).hexdigest())
            with self.assertRaises(FileExistsError):
                convert_experts({"a": matrices}, target, hidden_size=32, intermediate_size=32)
            damaged = Path(directory) / "bad.q4"
            with self.assertRaises(ValueError):
                convert_experts({"a": matrices, "b": [np.full((32, 32), np.nan)] * 3},
                                damaged, hidden_size=32, intermediate_size=32)
            self.assertFalse(damaged.exists())
            self.assertEqual(list(Path(directory).iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
