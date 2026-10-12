"""Independent Q4_1 layout and conversion; no upstream implementation copied.

Metadata: https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
Each little-endian block holds binary16 delta and minimum followed by 16 nibble bytes.
Low nibbles describe columns 0..15; high nibbles describe columns 16..31.
The decoded value is nibble * delta + minimum. Finite signed metadata is legal.
"""

import struct

import numpy as np

DTYPE = "q4_1"
BLOCK_WIDTH = 32
BLOCK_BYTES = 20


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shapes(hidden, intermediate):
    for name, value in (("hidden_size", hidden), ("intermediate_size", intermediate)):
        _positive(value, name)
        if value % 32:
            raise ValueError("Q4_1 row widths must be multiples of 32")
    return ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))


def payload_size(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return 3 * hidden_size * intermediate_size // 32 * 20


def pack_matrix(matrix, *, block_count=64):
    """Independent min/range quantizer using half-rounded metadata and ties-even codes.

    Nonzero underflow deltas are floored to the smallest half subnormal.
    No upstream quantizer byte equivalence is claimed.
    """
    _positive(block_count, "block_count")
    a = np.asarray(matrix)
    if a.ndim != 2 or a.dtype.kind not in "iuf":
        raise ValueError("weights must be real matrices")
    rows, columns = a.shape
    _positive(rows, "rows")
    _positive(columns, "columns")
    if columns % 32:
        raise ValueError("Q4_1 row widths must be multiples of 32")
    encoded = bytearray()
    for row in a:
        for begin in range(0, columns, 32 * block_count):
            with np.errstate(over="ignore", invalid="ignore"):
                chunk = row[begin:begin + 32 * block_count].astype(np.float32).reshape(-1, 32)
            if not np.isfinite(chunk).all():
                raise ValueError("weights must be finite and representable as float32")
            for block in chunk:
                minimum = float(np.min(block))
                maximum = float(np.max(block))
                with np.errstate(over="ignore", under="ignore"):
                    offset = np.float16(minimum)
                    delta = np.float16((maximum - minimum) / 15)
                if not np.isfinite(delta) or not np.isfinite(offset):
                    raise ValueError("Q4_1 metadata exceeds binary16 range")
                if maximum != minimum and delta == 0:
                    delta = np.float16(2 ** -24)
                # Float64 arithmetic keeps finite FP32 endpoint differences safe.
                codes = np.clip(np.rint((block.astype(np.float64) - float(offset))
                                       / (float(delta) or 1)), 0, 15).astype(np.uint8)
                encoded.extend(struct.pack("<ee", float(delta), float(offset)))
                encoded.extend((codes[:16] | (codes[16:] << 4)).tobytes())
    return bytes(encoded)


def pack_expert(gate, up, down, *, block_count=64):
    matrices = [np.asarray(x) for x in (gate, up, down)]
    if matrices[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    intermediate, hidden = matrices[0].shape
    if [a.shape for a in matrices] != list(_shapes(hidden, intermediate)):
        raise ValueError("expert projection shape mismatch")
    return b"".join(pack_matrix(a, block_count=block_count) for a in matrices)


def decode_expert(payload, hidden_size, intermediate_size):
    """Explicit full decode returning owned FP32 matrices (not kernel scratch)."""
    shapes = _shapes(hidden_size, intermediate_size)
    with memoryview(payload) as view:
        if not view.c_contiguous or view.nbytes != payload_size(hidden_size, intermediate_size):
            raise ValueError("Q4_1 payload must be contiguous with exact size")
        with view.cast("B") as raw:
            matrices = []
            cursor = 0
            for shape in shapes:
                matrix = np.empty(shape, np.float32)
                flat = matrix.reshape(-1)
                for begin in range(0, flat.size, 32):
                    delta, offset = struct.unpack_from("<ee", raw, cursor)
                    if not np.isfinite(delta) or not np.isfinite(offset):
                        raise ValueError("Q4_1 metadata must be finite")
                    for lane in range(16):
                        byte = raw[cursor + 4 + lane]
                        flat[begin + lane] = (byte & 15) * delta + offset
                        flat[begin + 16 + lane] = (byte >> 4) * delta + offset
                    cursor += 20
                matrices.append(matrix)
    return tuple(matrices)


class Q4_1ExpertBackend:
    """Direct-byte scalar baseline; never materializes decoded weight matrices."""

    NAME = "q4_1-reference"

    def __init__(self, *, max_scratch_bytes=64 * 1024 * 1024):
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def _backend(self):
        from .native_q4_1_expert import _DirectBackend
        return _DirectBackend(max_scratch_bytes=self.max_scratch_bytes)

    def info(self):
        return self._backend().info()

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        return self._backend().execute(source, inputs, routes, hidden_size=hidden_size,
                                       intermediate_size=intermediate_size, dtype=dtype)

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        return self._backend().execute_batch(source, inputs, route_batch, hidden_size=hidden_size,
                                             intermediate_size=intermediate_size, dtype=dtype)
