import math
import mmap
import struct
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import q8_0_expert as q8
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend


def scalar_decode(raw, hidden, intermediate):
    """Independent struct-based byte oracle, not the production decoder."""
    values = []
    for offset in range(0, len(raw), 34):
        scale = struct.unpack_from("<e", raw, offset)[0]
        values.extend(scale * code for code in struct.unpack_from("32b", raw, offset + 2))
    width = hidden * intermediate
    return tuple(np.array(values[start:start + width], np.float32).reshape(shape)
                 for start, shape in zip((0, width, 2 * width),
                                         ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))))


class TrackingSource:
    def __init__(self, payloads):
        self.payloads = payloads
        self.active = 0
        self.maximum = 0
        self.calls = []

    @contextmanager
    def acquire(self, key):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.calls.append(key)
        storage = bytearray(self.payloads[key])
        try:
            yield storage
        finally:
            self.active -= 1
            # Resizing fails if any NumPy array still borrows the leased buffer.
            storage.clear()


class Q8_0Tests(unittest.TestCase):
    def test_handbuilt_layout_all_signed_codes_and_scales(self):
        codes = list(range(-128, 128))
        records = [struct.pack("<e32b", scale, *codes[start:start + 32])
                   for scale in (0.0, -0.0, 0.5, -2.0, 65504.0, 2 ** -24)
                   for start in range(0, 256, 32)]
        # Three 32x32 matrices contain 96 records.
        raw = b"".join(records * 2)
        self.assertEqual(len(raw), q8.payload_size(32, 32))
        expected = scalar_decode(raw, 32, 32)
        for count in (1, 7, 64):
            actual = q8.decode_expert(raw, 32, 32, block_count=count)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
                self.assertTrue(a.flags.owndata)

    def test_packer_known_bytes_ties_zero_and_subnormal(self):
        row = np.zeros((1, 32), np.float32)
        row[0, :6] = [127, -127, .5, 1.5, 2.5, -1.5]
        expected = struct.pack("<e32b", 1, 127, -127, 0, 2, 2, -2, *([0] * 26))
        self.assertEqual(q8.pack_matrix(row), expected)
        self.assertEqual(q8.pack_matrix(np.zeros((1, 32))), bytes(34))
        tiny = np.full((1, 32), 2 ** -24, np.float32)
        self.assertEqual(q8.pack_matrix(tiny), struct.pack("<e32b", 2 ** -24, *([1] * 32)))

    def test_packing_bound_shape_order_and_block_invariance(self):
        rng = np.random.default_rng(119)
        weights = tuple(rng.normal(0, .2, shape).astype(np.float32)
                        for shape in ((64, 96), (64, 96), (96, 64)))
        raw = q8.pack_expert(*weights, block_count=1)
        self.assertEqual(raw, q8.pack_expert(*weights, block_count=17))
        self.assertEqual(raw, b"".join(q8.pack_matrix(a) for a in weights))
        decoded = scalar_decode(raw, 96, 64)
        offset = 0
        for original, actual in zip(weights, decoded):
            for a, b in zip(original.reshape(-1, 32), actual.reshape(-1, 32)):
                scale = struct.unpack_from("<e", raw, offset)[0]
                # Clipping after binary16 scale rounding can exceed scale/2.
                bound = max(scale / 2, float(np.max(np.abs(a))) - 127 * scale)
                self.assertLessEqual(float(np.max(np.abs(a.astype(np.float64) - b))), bound + 1e-7)
                offset += 34

    def test_scalar_oracle_differential_batch_routes_and_bounded_leases(self):
        h, i = 96, 64
        rng = np.random.default_rng(42)
        payloads = {}
        decoded_payloads = {}
        for key in ("a", "b"):
            # Import-shaped bytes exercise signed scales and -128, independent of packer.
            raw = b"".join(struct.pack("<e32b", float(rng.choice([-.002, 0, .003])),
                                       *rng.integers(-128, 128, 32).tolist())
                           for _ in range(3 * h * i // 32))
            payloads[key] = raw
            decoded_payloads[key] = b"".join(a.astype("<f4").tobytes() for a in scalar_decode(raw, h, i))
        x = rng.normal(0, .1, (4, h)).astype(np.float32)
        routes = [[("a", .25), ("b", .75)], [("b", 0)], [("b", 2)], [("b", .4), ("a", .6)]]
        args = dict(hidden_size=h, intermediate_size=i)
        expected = NumPyRoutedSwiGLUBackend().execute_batch(
            MappingPayloadSource(decoded_payloads), x, routes, dtype="float32", **args)
        original = q8._decoded_blocks
        for count in (1, 2, 7):
            source = TrackingSource(payloads)
            observed = []
            def observe(*args):
                for start, block in original(*args):
                    self.assertEqual(source.active, 1)
                    observed.append(block.size)
                    yield start, block
            backend = q8.Q8_0ExpertBackend(block_count=count)
            with patch.object(q8, "_decoded_blocks", observe), patch.object(
                q8, "decode_expert", side_effect=AssertionError("full decode forbidden")
            ):
                actual = backend.execute_batch(source, x, routes, **args)
                single = backend.execute(source, x[0], routes[0], **args)
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)
            np.testing.assert_array_equal(single, actual[0])
            self.assertEqual(source.active, 0)
            self.assertEqual(source.maximum, 1)
            self.assertTrue(observed)
            self.assertLessEqual(max(observed), count * 32)
            self.assertEqual(source.calls[:7], ["a", "b", "b", "b", "b", "a", "a"])

    def test_corruption_validated_before_projection_and_lease_cleanup(self):
        raw = bytes(q8.payload_size(32, 32))
        damaged = [raw[:-1], raw + b"\0"]
        for value in (math.nan, math.inf, -math.inf):
            data = bytearray(raw)
            data[-34:-32] = struct.pack("<e", value)
            damaged.append(data)
        backend = q8.Q8_0ExpertBackend(block_count=1)
        for data in damaged:
            source = TrackingSource({"a": data})
            with patch.object(backend, "_project", side_effect=AssertionError("arithmetic before validation")):
                with self.assertRaises(ValueError):
                    backend.execute(source, np.ones(32), [("a", 0)], hidden_size=32, intermediate_size=32)
            self.assertEqual(source.active, 0)
            with self.assertRaises(ValueError):
                q8.decode_expert(data, 32, 32)
        with self.assertRaises(ValueError):
            q8.decode_expert(memoryview(raw)[::2], 32, 32)

    def test_validation_empty_batch_and_arithmetic_cleanup(self):
        args = dict(hidden_size=32, intermediate_size=32)
        source = TrackingSource({"a": q8.pack_expert(*[np.ones((32, 32))] * 3)})
        backend = q8.Q8_0ExpertBackend()
        self.assertEqual(backend.execute_batch(source, np.empty((0, 32)), [], **args).shape, (0, 32))
        self.assertEqual(source.calls, [])
        for value in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                q8.Q8_0ExpertBackend(block_count=value)
            with self.assertRaises(ValueError):
                q8.payload_size(value, 32)
        for h, i in ((1, 32), (32, 33)):
            with self.assertRaises(ValueError):
                q8.payload_size(h, i)
        for value in (math.nan, math.inf, 1e100, 1e8, 1j):
            with self.assertRaises(ValueError):
                q8.pack_matrix(np.full((1, 32), value))
        with self.assertRaises(ValueError):
            q8.pack_matrix(np.ones((1, 31)))
        with self.assertRaises(ValueError):
            q8.pack_expert(np.ones((32, 32)), np.ones((64, 32)), np.ones((32, 32)))
        for routes in ([], [("a", -1)], [("a", math.nan)], [("a", 1e100)], [("a", 1), ("a", 1)]):
            with self.assertRaises(ValueError):
                backend.execute(source, np.ones(32), routes, **args)
        with self.assertRaises(ValueError):
            backend.execute_batch(source, [np.ones(32)], [], **args)
        with self.assertRaises(ValueError):
            backend.execute(source, np.full(32, math.nan), [("a", 1)], **args)
        with self.assertRaises(ValueError):
            backend.execute(source, np.ones(32), [("a", 1)], dtype="int8-row", **args)
        with self.assertRaises(TypeError):
            backend.execute(object(), np.ones(32), [("a", 1)], **args)
        for value in (np.finfo(np.float32).max, 1e18, 2e17):
            with self.assertRaises(ArithmeticError):
                backend.execute(source, np.full(32, value), [("a", 1)], **args)
        self.assertEqual(source.active, 0)

    def test_real_mapped_storage_closes_on_success_and_failure(self):
        args = dict(hidden_size=32, intermediate_size=32)
        backend = q8.Q8_0ExpertBackend(block_count=1)
        class MappedSource:
            mapping = None
            @contextmanager
            def acquire(self, key):
                with tempfile.TemporaryFile() as file:
                    file.write(raw)
                    file.flush()
                    with mmap.mmap(file.fileno(), len(raw), access=mmap.ACCESS_READ) as mapping:
                        self.mapping = mapping
                        yield mapping
        source = MappedSource()
        raw = q8.pack_expert(*[np.ones((32, 32))] * 3)
        backend.execute(source, np.ones(32), [("a", 1)], **args)
        self.assertTrue(source.mapping.closed)
        with self.assertRaises(ArithmeticError):
            backend.execute(source, np.full(32, 1e18), [("a", 1)], **args)
        self.assertTrue(source.mapping.closed)
        raw = raw[:-34] + struct.pack("<e", math.nan) + raw[-32:]
        with self.assertRaises(ValueError):
            backend.execute(source, np.ones(32), [("a", 1)], **args)
        self.assertTrue(source.mapping.closed)


if __name__ == "__main__":
    unittest.main()
