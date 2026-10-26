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
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(a, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite and representable as float32")

    # Q4_1's 32-value blocks are independent. Process a bounded number of
    # blocks at a time, but vectorize extrema, binary16 metadata and nibble
    # codes instead of iterating over every block in Python. block_count keeps
    # its existing memory-bound meaning and does not affect payload ordering.
    blocks_per_chunk = max(1, block_count * max(1, columns // 32))
    rows_as_blocks = values.reshape(rows, columns // 32, 32)
    flat = rows_as_blocks.reshape(-1, 32)
    output = bytearray(flat.shape[0] * 20)
    output_view = np.frombuffer(output, dtype=np.uint8).reshape(-1, 20)
    for start in range(0, flat.shape[0], blocks_per_chunk):
        block = flat[start:start + blocks_per_chunk]
        minimum = block.min(axis=1).astype(np.float64)
        maximum = block.max(axis=1).astype(np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            offset = minimum.astype("<f2")
            delta = ((maximum - minimum) / 15.0).astype("<f2")
        if not np.isfinite(offset).all() or not np.isfinite(delta).all():
            raise ValueError("Q4_1 metadata exceeds binary16 range")
        delta = delta.copy()
        delta[(maximum != minimum) & (delta == 0)] = np.float16(2 ** -24)

        denominator = delta.astype(np.float64)
        denominator[denominator == 0] = 1.0
        codes = np.clip(np.rint(
            (block.astype(np.float64) - offset.astype(np.float64)[:, None])
            / denominator[:, None]), 0, 15).astype(np.uint8)
        target = output_view[start:start + len(block)]
        target[:, :2] = delta.view(np.uint8).reshape(-1, 2)
        target[:, 2:4] = offset.view(np.uint8).reshape(-1, 2)
        target[:, 4:] = codes[:, :16] | (codes[:, 16:] << 4)
    return bytes(output)


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
