"""Independent public-contract Q5_0 bytes, quantizer and publication tests."""

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.q5_0_expert import (BLOCK_BYTES, BLOCK_WIDTH,
                                                 decode_expert, pack_expert,
                                                 pack_matrix, payload_size)
from openagent.adaptive_engine.q5_0_conversion import convert_experts


def oracle_decode(blob, hidden, intermediate):
    """Separate NumPy bit-plane decoder; never calls production decoding."""
    records = np.frombuffer(blob, np.uint8).reshape(-1, 22)
    deltas = np.array([struct.unpack("<e", bytes(record[:2]))[0] for record in records], np.float32)
    if not np.isfinite(deltas).all():
        raise ValueError("oracle nonfinite delta")
    fifth = np.unpackbits(records[:, 2:6], axis=1, bitorder="little").astype(np.int16)
    low = np.concatenate((records[:, 6:] % 16, records[:, 6:] // 16), axis=1).astype(np.int16)
    weights = ((low + 16 * fifth - 16).astype(np.float32) * deltas[:, None]).reshape(-1)
    area = hidden * intermediate
    return tuple(weights[index * area:(index + 1) * area].reshape(shape).copy()
                 for index, shape in enumerate(((intermediate, hidden), (intermediate, hidden),
                                                (hidden, intermediate))))


class Q5FormatTests(unittest.TestCase):
    def test_layout_all_32_codes_signed_scales_and_owned_decode(self):
        self.assertEqual((BLOCK_WIDTH, BLOCK_BYTES, payload_size(32, 32)), (32, 22, 2112))
        for delta in (1., -.25, 65504., -65504., 2 ** -24, -(2 ** -24), 0., -0.):
            block = struct.pack("<eI16B", delta, 0xffff0000, *[lane * 17 for lane in range(16)])
            blob = block * 96
            decoded = decode_expert(blob, 32, 32)
            np.testing.assert_array_equal(decoded[0][0], np.arange(-16, 16) * delta)
            for actual, expected in zip(decoded, oracle_decode(blob, 32, 32)):
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(actual.flags.owndata)
                self.assertEqual(actual.dtype, np.float32)

    def test_every_high_bit_maps_to_its_lane(self):
        for lane in range(32):
            # One high bit, all low nibbles zero: -16 everywhere except lane=0.
            high_bytes = bytearray(4)
            high_bytes[lane // 8] = 1 << (lane % 8)
            block = struct.pack("<e", 1.) + high_bytes + bytes(16)
            row = decode_expert(block * 96, 32, 32)[0][0]
            expected = np.full(32, -16., np.float32)
            expected[lane] = 0
            np.testing.assert_array_equal(row, expected)

    def test_known_ties_even_zero_and_subnormal_policy(self):
        values = np.array([-15., 15., .5, 1.5, 2.5, -1.5] + [0.] * 26, np.float32)
        actual = pack_matrix(values[None])
        codes = [1, 31, 16, 18, 18, 14] + [16] * 26
        fifth = sum((code // 16) * 2 ** lane for lane, code in enumerate(codes))
        expected = struct.pack("<eI", 1., fifth) + bytes(codes[j] % 16 + 16 * (codes[j + 16] % 16)
                                                           for j in range(16))
        self.assertEqual(actual, expected)
        self.assertEqual(pack_matrix(np.zeros((1, 32))), b"\0\0" + b"\xff" * 4 + bytes(16))
        self.assertEqual(struct.unpack_from("<e", pack_matrix(np.full((1, 32), 1e-40)))[0], 2 ** -24)
        maximum = pack_matrix(np.full((1, 32), 65504. * 15))
        self.assertEqual(struct.unpack_from("<e", maximum)[0], 65504.)

    def test_seeded_quantization_error_and_chunk_independence(self):
        rng = np.random.default_rng(550)
        h, i = 64, 96
        matrices = [rng.normal(0, .2, shape).astype(np.float32) for shape in ((i, h), (i, h), (h, i))]
        blob = pack_expert(*matrices, block_count=1)
        self.assertEqual(blob, pack_expert(*matrices, block_count=37))
        for original, decoded in zip(matrices, decode_expert(blob, h, i)):
            peaks = np.max(np.abs(original.reshape(-1, 32)), axis=1)
            scales = (peaks.astype(np.float64) / 15).astype(np.float16).astype(np.float32)
            tolerance = scales[:, None] / 2 + 2 * np.finfo(np.float32).eps * peaks[:, None]
            self.assertTrue(np.all(np.abs(original.reshape(-1, 32) - decoded.reshape(-1, 32)) <= tolerance))

    def test_reject_invalid_values_shapes_dimensions_and_payloads(self):
        for h, i in ((0, 32), (32, -32), (31, 32), (True, 32), (32., 32)):
            with self.assertRaises(ValueError):
                payload_size(h, i)
        for count in (0, -1, True, .5):
            with self.assertRaises(ValueError):
                pack_matrix(np.ones((1, 32)), block_count=count)
        for matrix in (np.ones((0, 32)), np.ones((1, 31)), np.ones(32),
                       np.ones((1, 32), complex), np.ones((1, 32), bool)):
            with self.assertRaises(ValueError):
                pack_matrix(matrix)
        for value in (np.nan, np.inf, -np.inf, 1e100, 1e7):
            with self.assertRaises(ValueError):
                pack_matrix(np.full((1, 32), value))
        with self.assertRaises(ValueError):
            pack_expert(np.ones((32, 32)), np.ones((32, 64)), np.ones((32, 32)))
        blob = pack_expert(*[np.zeros((32, 32))] * 3)
        for bad in (blob[:-1], blob + b"x", memoryview(blob)[::2]):
            with self.assertRaises(ValueError):
                decode_expert(bad, 32, 32)
        for position in (0, len(blob) - 22):
            for bits in (0x7c00, 0xfc00, 0x7e00, 0x7fff):
                bad = bytearray(blob)
                struct.pack_into("<H", bad, position, bits)
                with self.assertRaisesRegex(ValueError, "finite"):
                    decode_expert(bad, 32, 32)

    def test_converter_ranges_hashes_exclusive_publish_and_failure_cleanup(self):
        weights = [np.ones((32, 32)) * .1] * 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "experts.q50"
            converted = convert_experts({"first": weights, "second": weights}, path,
                                        hidden_size=32, intermediate_size=32)
            size = payload_size(32, 32)
            self.assertEqual(converted.dtype, "q5_0")
            self.assertEqual(converted.activation, "swiglu")
            self.assertEqual(converted.path, path.absolute())
            with path.open("rb") as stream:
                for index, region in enumerate(converted.regions.values()):
                    self.assertEqual((region.path, region.offset, region.size), (path.name, index * size, size))
                    self.assertEqual(hashlib.sha256(stream.read(size)).hexdigest(), region.sha256)
                self.assertEqual(stream.read(), b"")
            with self.assertRaises(TypeError):
                converted.regions["new"] = next(iter(converted.regions.values()))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                convert_experts({"x": weights}, path, hidden_size=32, intermediate_size=32)
            self.assertEqual(path.read_bytes(), before)
            for experts in ({}, {"bad/name": weights}, {"x": weights, "y": [np.full((32, 32), np.nan)] * 3},
                            {"x": weights[:2]}, {"x": [np.ones((32, 64))] * 3}):
                with self.assertRaises(ValueError):
                    convert_experts(experts, root / "failure", hidden_size=32, intermediate_size=32)
            with patch("openagent.adaptive_engine.q5_0_conversion.os.link", side_effect=OSError("no links")):
                with self.assertRaisesRegex(OSError, "no links"):
                    convert_experts({"x": weights}, root / "failure", hidden_size=32, intermediate_size=32)
            self.assertEqual(list(root.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
