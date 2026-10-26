"""Independent affine fixtures and container contracts."""

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.q4_1_expert import (
    decode_expert, pack_expert, pack_matrix, payload_size,
)
from openagent.adaptive_engine.q4_1_conversion import convert_experts


def scalar_decode(blob, h, i):
    values = []
    for address in range(0, len(blob), 20):
        d, m, *quants = struct.unpack_from("<ee16B", blob, address)
        values.extend((q % 16) * d + m for q in quants)
        values.extend((q // 16) * d + m for q in quants)
    extent = h * i
    return tuple(np.array(values[n * extent:(n + 1) * extent], np.float32).reshape(shape)
                 for n, shape in enumerate(((i, h), (i, h), (h, i))))


class Q4AffineTests(unittest.TestCase):
    def test_signed_metadata_and_split_lanes(self):
        for d in (65504, -65504, 2 ** -24, -(2 ** -24), 0., -0., .125):
            for m in (65504, -65504, 2 ** -24, -(2 ** -24), 0., -0.):
                blob = struct.pack("<ee16B", d, m,
                                   *[j | ((15 - j) << 4) for j in range(16)]) * 96
                for actual, expected in zip(decode_expert(blob, 32, 32),
                                            scalar_decode(blob, 32, 32)):
                    np.testing.assert_array_equal(actual, expected)
                    self.assertTrue(actual.flags.owndata)

    def test_independent_quantizer(self):
        block = np.array([-2, 13, -1.5, -.5, .5] + [-2] * 27)[None]
        blob = pack_matrix(block)
        self.assertEqual(struct.unpack_from("<ee", blob), (1, -2))
        codes = [0, 15, 0, 2, 2] + [0] * 27
        self.assertEqual(blob[4:], bytes(codes[j] | codes[j + 16] << 4 for j in range(16)))
        self.assertEqual(pack_matrix(np.zeros((1, 32))), bytes(20))
        for value in (-3., 3., 65504., -65504.):
            self.assertEqual(struct.unpack_from("<ee", pack_matrix(np.full((1, 32), value))),
                             (0, value))
        tiny = np.zeros((1, 32))
        tiny[0, 1] = 1e-40
        self.assertEqual(struct.unpack_from("<ee", pack_matrix(tiny))[0], 2 ** -24)
        matrix = np.tile(np.arange(-8, 24), (3, 2))
        self.assertEqual(pack_matrix(matrix, block_count=1), pack_matrix(matrix, block_count=7))

    def test_vectorized_packer_matches_legacy_scalar_bytes(self):
        rng = np.random.default_rng(407)
        matrix = rng.normal(0, 3, size=(5, 128)).astype(np.float32)
        matrix[0, :32] = -2.0
        matrix[1, 32:64] = np.linspace(-65504, 65504, 32, dtype=np.float32)
        matrix[2, 64:96] = 0
        matrix[3, 96] = np.nextafter(np.float32(0), np.float32(1))

        expected = bytearray()
        for row in matrix:
            for begin in range(0, matrix.shape[1], 32):
                block = row[begin:begin + 32].astype(np.float32)
                minimum = float(np.min(block))
                maximum = float(np.max(block))
                offset = np.float16(minimum)
                delta = np.float16((maximum - minimum) / 15)
                if maximum != minimum and delta == 0:
                    delta = np.float16(2 ** -24)
                codes = np.clip(np.rint((block.astype(np.float64) - float(offset))
                                        / (float(delta) or 1)), 0, 15).astype(np.uint8)
                expected.extend(np.asarray((delta, offset), dtype="<f2").tobytes())
                expected.extend((codes[:16] | (codes[16:] << 4)).tobytes())

        self.assertEqual(pack_matrix(matrix, block_count=1), bytes(expected))
        self.assertEqual(pack_matrix(matrix, block_count=64), bytes(expected))

    def test_invalid_inputs_and_metadata(self):
        for h, i in ((0, 32), (32, 31), (True, 32), (32., 32)):
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
        for blob in (good[:-1], good + b"x", memoryview(good)[::2]):
            with self.assertRaises(ValueError):
                decode_expert(blob, 32, 32)
        for position in (0, 2, len(good) - 20, len(good) - 18):
            for bits in (0x7c00, 0xfc00, 0x7e00):
                bad = bytearray(good)
                struct.pack_into("<H", bad, position, bits)
                with self.assertRaisesRegex(ValueError, "finite"):
                    decode_expert(bad, 32, 32)
        with self.assertRaises(ValueError):
            pack_expert(np.zeros((32, 32)), np.zeros((32, 64)), np.zeros((32, 32)))

    def test_atomic_container(self):
        triple = [np.full((32, 32), -.05)] * 3
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "experts.q4"
            result = convert_experts({"a": triple, "b": triple}, target,
                                     hidden_size=32, intermediate_size=32)
            blob = target.read_bytes()
            size = payload_size(32, 32)
            self.assertEqual(len(blob), 2 * size)
            self.assertEqual(result.dtype, "q4_1")
            for n, region in enumerate(result.regions.values()):
                self.assertEqual(region.offset, n * size)
                self.assertEqual(region.size, size)
                self.assertEqual(region.sha256, hashlib.sha256(blob[n * size:(n + 1) * size]).hexdigest())
            with self.assertRaises(FileExistsError):
                convert_experts({"a": triple}, target, hidden_size=32, intermediate_size=32)
            bad = Path(directory) / "bad.q4"
            with self.assertRaises(ValueError):
                convert_experts({"a": triple, "b": [np.full((32, 32), np.nan)] * 3},
                                bad, hidden_size=32, intermediate_size=32)
            self.assertEqual(list(Path(directory).iterdir()), [target])
