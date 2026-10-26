"""Group geometry, signed nibble policy, partial groups and publication tests."""

import hashlib
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.grouped_int4_expert import (decode_expert, pack_expert,
                                                         pack_matrix, payload_size)
from openagent.adaptive_engine.grouped_int4_conversion import convert_experts


def oracle_decode(blob, h, i, gs):
    """Vectorized two's-complement/group expansion, no production decoding."""
    shapes = ((i, h), (i, h), (h, i))
    scale_start = sum(rows * ((width + 1) // 2) for rows, width in shapes)
    code_cursor, scale_cursor = 0, scale_start
    result = []
    for rows, width in shapes:
        stride, groups = (width + 1) // 2, (width + gs - 1) // gs
        raw = np.frombuffer(blob, np.uint8, rows * stride, code_cursor).reshape(rows, stride)
        quants = np.empty((rows, stride * 2), np.int16)
        quants[:, ::2] = raw % 16
        quants[:, 1::2] = raw // 16
        quants[quants >= 8] -= 16
        scales = np.array(struct.unpack_from("<" + "f" * (rows * groups), blob, scale_cursor),
                          np.float64).reshape(rows, groups)
        values = (quants[:, :width].astype(np.float64) * np.repeat(scales, gs, axis=1)[:, :width]).astype(np.float32)
        result.append(values)
        code_cursor += rows * stride
        scale_cursor += 4 * rows * groups
    if scale_cursor != len(blob):
        raise AssertionError("oracle unexpected payload size")
    return tuple(result)


def fixture(h, i, gs, seed=410):
    rng = np.random.default_rng(seed)
    code_parts, scale_parts = [], []
    for rows, width in ((i, h), (i, h), (h, i)):
        stride = (width + 1) // 2
        raw = rng.integers(0, 256, (rows, stride), dtype=np.uint8)
        if width % 2:
            raw[:, -1] &= 15
        code_parts.append(raw.tobytes())
        scales = rng.choice(np.array([-.01, .02, -0., 2 ** -149], np.float32), (rows, (width + gs - 1) // gs))
        scale_parts.append(scales.astype("<f4").tobytes())
    return b"".join(code_parts + scale_parts)


def legacy_pack_matrix(matrix, group_size):
    """Original scalar policy retained as an independent byte-layout oracle."""
    weights = np.asarray(matrix)
    rows, columns = weights.shape
    stride = (columns + 1) // 2
    groups = (columns + group_size - 1) // group_size
    codes = bytearray(rows * stride)
    scales = np.empty((rows, groups), dtype="<f4")
    for row in range(rows):
        for group in range(groups):
            start, stop = group * group_size, min(columns, (group + 1) * group_size)
            values = weights[row, start:stop].astype(np.float32)
            maximum = float(np.max(np.abs(values)))
            scale = np.float32(maximum / 7) if maximum else np.float32(1)
            if maximum and scale == 0:
                scale = np.nextafter(np.float32(0), np.float32(1))
            integers = np.clip(np.rint(values.astype(np.float64) / float(scale)), -7, 7).astype(np.int8)
            scales[row, group] = scale
            for column, value in enumerate(integers, start):
                codes[row * stride + column // 2] |= (int(value) & 15) << (4 * (column % 2))
    return bytes(codes), scales


class GroupedFormatTests(unittest.TestCase):
    def test_tiled_vector_packer_matches_legacy_scalar_bytes(self):
        rng = np.random.default_rng(420)
        for group_size in (1, 2, 3, 7, 64, 100):
            for row_count in (1, 3, 64):
                for matrix in (rng.normal(0, .1, (7, 19)), np.zeros((4, 65)),
                               np.full((2, 17), np.nextafter(np.float32(0), np.float32(1)))):
                    expected_codes, expected_scales = legacy_pack_matrix(matrix, group_size)
                    actual_codes, actual_scales = pack_matrix(matrix, group_size=group_size,
                                                              row_count=row_count)
                    self.assertEqual(actual_codes, expected_codes)
                    np.testing.assert_array_equal(actual_scales, expected_scales)

    def test_payload_geometry_odd_rows_and_partial_groups(self):
        self.assertEqual(payload_size(3, 5, group_size=2), 145)
        self.assertEqual(payload_size(3, 5, group_size=64), 81)
        for h, i, gs in ((1, 1, 1), (3, 5, 2), (65, 129, 64), (65, 33, 7), (18, 11, 100)):
            blob = fixture(h, i, gs)
            self.assertEqual(len(blob), payload_size(h, i, group_size=gs))
            for actual, expected in zip(decode_expert(blob, h, i, group_size=gs), oracle_decode(blob, h, i, gs)):
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(actual.flags.owndata)

    def test_all_codes_and_group_boundary_inside_nibble_byte(self):
        # h=16, i=1: gate/up each 8 bytes, down sixteen 1-byte padded rows.
        code_row = bytes(j + 16 * (j + 1) for j in range(0, 16, 2))
        gs = 3
        scales = [1., 2., -1., .5, 0., -0.] * 2 + [1.] * 16
        blob = code_row * 2 + bytes(16) + struct.pack("<28f", *scales)
        decoded = decode_expert(blob, 16, 1, group_size=gs)[0][0]
        integers = np.arange(16)
        integers[integers >= 8] -= 16
        expected = integers * np.repeat(scales[:6], gs)[:16]
        np.testing.assert_array_equal(decoded, expected)
        self.assertEqual(decoded[2], 2)
        self.assertEqual(decoded[3], 6)

    def test_quantization_ties_zero_subnormal_and_per_group_error(self):
        matrix = np.array([[7., .5, 1.5, 2.5, -1.5, -7., 0.]], np.float32)
        codes, scales = pack_matrix(matrix, group_size=6)
        self.assertEqual(codes, bytes([0x07, 0x22, 0x9e, 0]))
        np.testing.assert_array_equal(scales, [[1., 1.]])
        tiny = np.full((1, 2), np.nextafter(np.float32(0), np.float32(1)))
        _, scales = pack_matrix(tiny, group_size=2)
        self.assertEqual(scales[0, 0], np.nextafter(np.float32(0), np.float32(1)))
        rng = np.random.default_rng(419)
        matrices = [rng.normal(0, .1, shape).astype(np.float32) for shape in ((11, 17), (11, 17), (17, 11))]
        for gs in (1, 3, 7, 64):
            blob = pack_expert(*matrices, group_size=gs)
            for original, decoded in zip(matrices, decode_expert(blob, 17, 11, group_size=gs)):
                _, scales = pack_matrix(original, group_size=gs)
                bound = np.repeat(scales, gs, axis=1)[:, :original.shape[1]] / 2
                self.assertTrue(np.all(np.abs(original - decoded) <= bound + 2e-7 * np.max(np.abs(original))))

    def test_dimensions_group_size_shapes_finite_and_quantized_overflow(self):
        for h, i, gs in ((0, 1, 1), (1, -1, 1), (1, 1, 0), (True, 1, 1), (1, 1, 2.), (1, 1, True)):
            with self.assertRaises(ValueError):
                payload_size(h, i, group_size=gs)
        for matrix in (np.ones(2), np.ones((0, 1)), np.ones((1, 2), complex), np.ones((1, 2), bool)):
            with self.assertRaises(ValueError):
                pack_matrix(matrix)
        for value in (np.nan, np.inf, -np.inf, 1e100):
            with self.assertRaises(ValueError):
                pack_matrix(np.full((1, 2), value))
        with self.assertRaises(ValueError):
            pack_expert(np.ones((3, 2)), np.ones((3, 3)), np.ones((2, 3)))

    def test_malformed_padding_and_scales(self):
        h, i, gs = 3, 5, 2
        blob = fixture(h, i, gs)
        for bad in (blob[:-1], blob + b"x", memoryview(blob)[::2]):
            with self.assertRaises(ValueError):
                decode_expert(bad, h, i, group_size=gs)
        for offset in (29, len(blob) - 4):
            for value in (float("nan"), float("inf"), -float("inf")):
                bad = bytearray(blob)
                struct.pack_into("<f", bad, offset, value)
                with self.assertRaisesRegex(ValueError, "finite"):
                    decode_expert(bad, h, i, group_size=gs)
        for offset in (1, 11, 22):
            bad = bytearray(blob)
            bad[offset] |= 0x80
            with self.assertRaisesRegex(ValueError, "padding"):
                decode_expert(bad, h, i, group_size=gs)

    def test_converter_hashes_group_size_and_atomic_cleanup(self):
        matrices = [np.full((5, 3), .1), np.full((5, 3), -.1), np.full((3, 5), .2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "weights.gi4"
            result = convert_experts({"a": matrices, "b": matrices}, path,
                                     hidden_size=3, intermediate_size=5, group_size=2)
            self.assertEqual((result.dtype, result.group_size), ("int4-grouped", 2))
            blob = path.read_bytes()
            for index, region in enumerate(result.regions.values()):
                self.assertEqual((region.offset, region.size), (145 * index, 145))
                self.assertEqual(region.sha256, hashlib.sha256(blob[145 * index:145 * (index + 1)]).hexdigest())
            with self.assertRaises(FileExistsError):
                convert_experts({"a": matrices}, path, hidden_size=3, intermediate_size=5)
            bad = [np.full((5, 3), np.nan)] * 2 + [np.full((3, 5), np.nan)]
            with self.assertRaises(ValueError):
                convert_experts({"a": matrices, "b": bad}, root / "fail", hidden_size=3, intermediate_size=5)
            with patch("openagent.adaptive_engine.grouped_int4_conversion.os.link", side_effect=OSError("no links")):
                with self.assertRaises(OSError):
                    convert_experts({"a": matrices}, root / "fail", hidden_size=3, intermediate_size=5)
            self.assertEqual(list(root.iterdir()), [path])

    def test_converter_closes_raw_descriptor_when_fdopen_fails(self):
        matrices = [np.full((5, 3), .1), np.full((5, 3), -.1), np.full((3, 5), .2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_mkstemp = tempfile.mkstemp
            opened = {}

            def capture_descriptor(*args, **kwargs):
                descriptor, name = real_mkstemp(*args, **kwargs)
                opened.update(descriptor=descriptor, path=Path(name))
                return descriptor, name

            with patch('openagent.adaptive_engine.grouped_int4_conversion.tempfile.mkstemp',
                       side_effect=capture_descriptor), \
                 patch('openagent.adaptive_engine.grouped_int4_conversion.os.fdopen',
                       side_effect=OSError('fdopen failed')):
                with self.assertRaisesRegex(OSError, 'fdopen failed'):
                    convert_experts({'a': matrices}, root / 'weights.gi4',
                                    hidden_size=3, intermediate_size=5)

            with self.assertRaises(OSError):
                os.fstat(opened['descriptor'])
            self.assertFalse(opened['path'].exists())
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
