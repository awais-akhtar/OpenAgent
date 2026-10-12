"""Q5_1 field-contract fixtures, independent bit-plane decode and conversion."""

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.q5_1_expert import (BLOCK_BYTES, BLOCK_WIDTH,
                                                 decode_expert, pack_matrix,
                                                 pack_expert, payload_size)
from openagent.adaptive_engine.q5_1_conversion import convert_experts


def oracle_decode(blob, h, i):
    """Bit-plane oracle independent of production block/nibble loops."""
    blocks = np.frombuffer(blob, np.uint8).reshape(-1, 24)
    meta = np.array([struct.unpack("<ee", bytes(row[:4])) for row in blocks], np.float64)
    if not np.isfinite(meta).all():
        raise ValueError("oracle nonfinite metadata")
    fifth = np.unpackbits(blocks[:, 4:8], bitorder="little", axis=1)
    nibbles = np.concatenate((blocks[:, 8:] % 16, blocks[:, 8:] // 16), axis=1)
    codes = nibbles.astype(np.float64) + 16 * fifth
    weights = (codes * meta[:, :1] + meta[:, 1:]).astype(np.float32).reshape(-1)
    area = h * i
    return tuple(weights[index * area:(index + 1) * area].reshape(shape).copy()
                 for index, shape in enumerate(((i, h), (i, h), (h, i))))


class Q5AffineFormatTests(unittest.TestCase):
    def test_exact_layout_all_codes_extreme_signed_metadata_and_owned_decode(self):
        self.assertEqual((BLOCK_WIDTH, BLOCK_BYTES, payload_size(32, 32)), (32, 24, 2304))
        for d, m in ((1., -16.), (-.125, .25), (65504., -65504.), (-65504., 65504.),
                     (2 ** -24, -(2 ** -24)), (-0., 65504.), (0., -.5), (0., -0.)):
            block = struct.pack("<eeI16B", d, m, 0xffff0000, *[j * 17 for j in range(16)])
            decoded = decode_expert(block * 96, 32, 32)
            np.testing.assert_array_equal(decoded[0][0], (np.arange(32) * d + m).astype(np.float32))
            for actual, expected in zip(decoded, oracle_decode(block * 96, 32, 32)):
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(actual.flags.owndata)

    def test_each_fifth_bit_is_its_lane_and_affine_offset_is_not_centering(self):
        for lane in range(32):
            high = bytearray(4)
            high[lane // 8] = 1 << (lane % 8)
            block = struct.pack("<ee", -.25, .75) + high + bytes(16)
            expected = np.full(32, .75, np.float32)
            expected[lane] = -3.25
            np.testing.assert_array_equal(decode_expert(block * 96, 32, 32)[0][0], expected)

    def test_known_quantizer_ties_constants_underflow_and_maximum_metadata(self):
        values = np.array([-1., 30., -.5, .5, 1.5, 2.5] + [0.] * 26, np.float32)
        codes = [0, 31, 0, 2, 2, 4] + [1] * 26
        fifth = sum((code // 16) * 2 ** lane for lane, code in enumerate(codes))
        expected = struct.pack("<eeI", 1., -1., fifth) + bytes(codes[j] % 16 + 16 * (codes[j + 16] % 16)
                                                                  for j in range(16))
        self.assertEqual(pack_matrix(values[None]), expected)
        for constant in (0., -0., -.25, 65504., 1025.4):
            packed = pack_matrix(np.full((1, 32), constant))
            self.assertEqual(packed, struct.pack("<eeI16B", 0., float(np.float16(constant)), 0, *([0] * 16)))
        tiny = pack_matrix(np.array([0., 1e-40] + [0.] * 30)[None])
        self.assertEqual(struct.unpack_from("<ee", tiny), (2 ** -24, 0.))
        huge = pack_matrix(np.array([65504.] + [65504. * 32] * 31)[None])
        self.assertEqual(struct.unpack_from("<ee", huge), (65504., 65504.))

    def test_seeded_error_bound_noncontiguous_source_and_block_chunk_independence(self):
        rng = np.random.default_rng(51055)
        matrices = [rng.normal(.1, .2, shape).astype(np.float32) for shape in ((96, 64), (96, 64), (64, 96))]
        blob = pack_expert(*matrices, block_count=1)
        self.assertEqual(blob, pack_expert(*matrices, block_count=91))
        self.assertEqual(pack_matrix(matrices[0][:, ::-1]), pack_matrix(matrices[0][:, ::-1].copy()))
        for original, decoded in zip(matrices, decode_expert(blob, 64, 96)):
            blocks = original.reshape(-1, 32).astype(np.float64)
            minima, maxima = np.min(blocks, axis=1), np.max(blocks, axis=1)
            d = ((maxima - minima) / 31).astype(np.float16).astype(np.float64)
            m = minima.astype(np.float16).astype(np.float64)
            # Clipping and rounded metadata can increase endpoint error.
            bound = d / 2 + np.abs(minima - m) + np.abs((maxima - minima) - 31 * d)
            errors = np.max(np.abs(original.astype(np.float64) - decoded).reshape(-1, 32), axis=1)
            self.assertTrue(np.all(errors <= bound + 2 * np.finfo(np.float32).eps * np.max(np.abs(blocks), axis=1)))

    def test_reject_nonfinite_metadata_weights_shapes_and_payload_lengths(self):
        for h, i in ((0, 32), (32, -32), (33, 32), (True, 32), (32., 32)):
            with self.assertRaises(ValueError):
                payload_size(h, i)
        for count in (0, -1, True, .5):
            with self.assertRaises(ValueError):
                pack_matrix(np.ones((1, 32)), block_count=count)
        for matrix in (np.ones(32), np.ones((0, 32)), np.ones((1, 31)),
                       np.ones((1, 32), bool), np.ones((1, 32), complex)):
            with self.assertRaises(ValueError):
                pack_matrix(matrix)
        for bad in (np.nan, np.inf, -np.inf, 1e100, -65536., 65536.):
            with self.assertRaises(ValueError):
                pack_matrix(np.full((1, 32), bad))
        # A representable minimum does not rescue an overflowing delta.
        with self.assertRaisesRegex(ValueError, "binary16"):
            pack_matrix(np.array([0.] + [1e8] * 31)[None])
        with self.assertRaises(ValueError):
            pack_expert(np.ones((32, 32)), np.ones((32, 64)), np.ones((32, 32)))
        blob = pack_expert(*[np.zeros((32, 32))] * 3)
        for malformed in (blob[:-1], blob + b"x", memoryview(blob)[::2]):
            with self.assertRaises(ValueError):
                decode_expert(malformed, 32, 32)
        for offset in (0, 2, len(blob) - 24, len(blob) - 22):
            for bits in (0x7c00, 0xfc00, 0x7e00, 0xffff):
                malformed = bytearray(blob)
                struct.pack_into("<H", malformed, offset, bits)
                with self.assertRaisesRegex(ValueError, "finite"):
                    decode_expert(malformed, 32, 32)

    def test_atomic_conversion_metadata_and_cleanup_on_invalid_late_expert(self):
        matrices = [np.full((32, 32), .05)] * 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "experts.q51"
            result = convert_experts({"a": matrices, "b": matrices}, destination,
                                     hidden_size=32, intermediate_size=32)
            size = payload_size(32, 32)
            self.assertEqual((result.dtype, result.activation, result.path), ("q5_1", "swiglu", destination.absolute()))
            encoded = destination.read_bytes()
            self.assertEqual(len(encoded), 2 * size)
            for index, region in enumerate(result.regions.values()):
                self.assertEqual((region.path, region.offset, region.size), (destination.name, index * size, size))
                self.assertEqual(region.sha256, hashlib.sha256(encoded[index * size:(index + 1) * size]).hexdigest())
            with self.assertRaises(TypeError):
                result.regions["new"] = result.regions["a"]
            with self.assertRaises(FileExistsError):
                convert_experts({"a": matrices}, destination, hidden_size=32, intermediate_size=32)
            self.assertEqual(destination.read_bytes(), encoded)
            for invalid in ({}, {"": matrices}, {"a/b": matrices}, {"a\\b": matrices}, {"a": matrices[:2]},
                            {"a": [np.ones((32, 64))] * 3}, {"a": matrices, "b": [np.full((32, 32), np.nan)] * 3}):
                with self.assertRaises(ValueError):
                    convert_experts(invalid, root / "failed", hidden_size=32, intermediate_size=32)
            with patch("openagent.adaptive_engine.q5_1_conversion.os.link", side_effect=OSError("no hard links")):
                with self.assertRaisesRegex(OSError, "no hard links"):
                    convert_experts({"a": matrices}, root / "failed", hidden_size=32, intermediate_size=32)
            self.assertEqual(list(root.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
