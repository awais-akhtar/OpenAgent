"""Independent Q4_0 layout and conversion; no upstream implementation copied.

Metadata: https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
Each little-endian block holds a binary16 delta followed by 16 nibble bytes.
Low nibbles describe columns 0..15; high nibbles describe columns 16..31.
The decoded value is (nibble - 8) * delta. All finite signed deltas are legal.
"""

import struct

import numpy as np

DTYPE = "q4_0"
BLOCK_WIDTH = 32
BLOCK_BYTES = 18


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shapes(hidden, intermediate):
    for name, value in (("hidden_size", hidden), ("intermediate_size", intermediate)):
        _positive(value, name)
        if value % 32:
            raise ValueError("Q4_0 row widths must be multiples of 32")
    return ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))


def payload_size(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return 3 * hidden_size * intermediate_size // 32 * 18


def pack_matrix(matrix, *, block_count=64):
    """Positive max-abs/7 delta, half-rounded, ties-even codes clipped to [-8,7].

    Nonzero underflow deltas are floored to the smallest half subnormal.
    This is an independent policy, not upstream quantizer byte equivalence.
    """
    _positive(block_count, "block_count")
    a = np.asarray(matrix)
    if a.ndim != 2 or a.dtype.kind not in "iuf":
        raise ValueError("weights must be real matrices")
    rows, columns = a.shape
    _positive(rows, "rows")
    _positive(columns, "columns")
    if columns % 32:
        raise ValueError("Q4_0 row widths must be multiples of 32")
    encoded = bytearray()
    for row in a:
        for begin in range(0, columns, 32 * block_count):
            with np.errstate(over="ignore", invalid="ignore"):
                chunk = row[begin:begin + 32 * block_count].astype(np.float32).reshape(-1, 32)
            if not np.isfinite(chunk).all():
                raise ValueError("weights must be finite and representable as float32")
            maximum = np.max(np.abs(chunk), axis=1)
            with np.errstate(over="ignore", under="ignore"):
                delta = (maximum.astype(np.float64) / 7).astype("<f2")
            if not np.isfinite(delta).all():
                raise ValueError("Q4_0 delta exceeds binary16 range")
            delta[(maximum != 0) & (delta == 0)] = np.float16(2 ** -24)
            divisor = np.where(delta == 0, np.float32(1), delta.astype(np.float32))
            codes = np.clip(np.rint(chunk / divisor[:, None]), -8, 7).astype(np.int16) + 8
            blocks = np.empty(len(chunk), dtype=[("delta", "<f2"), ("codes", "u1", (16,))])
            blocks["delta"] = delta
            blocks["codes"] = (codes[:, :16] | (codes[:, 16:] << 4)).astype(np.uint8)
            encoded.extend(blocks.tobytes())
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
            raise ValueError("Q4_0 payload must be contiguous with exact size")
        with view.cast("B") as raw:
            matrices = []
            cursor = 0
            for shape in shapes:
                matrix = np.empty(shape, np.float32)
                flat = matrix.reshape(-1)
                for begin in range(0, flat.size, 32):
                    delta = struct.unpack_from("<e", raw, cursor)[0]
                    if not np.isfinite(delta):
                        raise ValueError("Q4_0 deltas must be finite")
                    for lane in range(16):
                        byte = raw[cursor + 2 + lane]
                        flat[begin + lane] = ((byte & 15) - 8) * delta
                        flat[begin + 16 + lane] = ((byte >> 4) - 8) * delta
                    cursor += 18
                matrices.append(matrix)
    return tuple(matrices)


class Q4_0ExpertBackend:
    """Direct-byte scalar baseline; never materializes decoded weight matrices."""

    NAME = "q4_0-reference"

    def __init__(self, *, max_scratch_bytes=64 * 1024 * 1024):
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def _backend(self):
        from .native_q4_0_expert import _DirectBackend
        return _DirectBackend(max_scratch_bytes=self.max_scratch_bytes)

    def info(self):
        return self._backend().info()

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        return self._backend().execute(source, inputs, routes, hidden_size=hidden_size,
                                       intermediate_size=intermediate_size, dtype=dtype)

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        return self._backend().execute_batch(source, inputs, route_batch, hidden_size=hidden_size,
                                             intermediate_size=intermediate_size, dtype=dtype)
