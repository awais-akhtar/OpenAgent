"""Independent Q5_1 affine format from public block field metadata only.

Metadata: ggml-org/ggml src/ggml-common.h, block_q5_1. No reference quantizer
or kernel is imported. Decode uses FP32(q * float(d) + float(m)), q in [0,31].
"""

import math
import struct

import numpy as np

from .expert_kernels import (ExpertPayloadSource, KernelBackendInfo,
                             KernelCapabilities, KernelEvidence, _finite_array,
                             _finite_matrix, _lease, _validate_routes)
from .q5_0_expert import _positive, _detach_borrowed_frames

DTYPE = "q5_1"
BLOCK_WIDTH = 32
BLOCK_BYTES = 24
_RECORD = struct.Struct("<eeI16B")


def _shapes(hidden_size, intermediate_size):
    h = _positive(hidden_size, "hidden_size")
    i = _positive(intermediate_size, "intermediate_size")
    if h % BLOCK_WIDTH or i % BLOCK_WIDTH:
        raise ValueError("Q5_1 row widths must be multiples of 32")
    return ((i, h), (i, h), (h, i))


def payload_size(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return 3 * hidden_size * intermediate_size // BLOCK_WIDTH * BLOCK_BYTES


def pack_matrix(matrix, *, block_count=64):
    """Independent half-rounded minimum/range quantizer, not upstream parity.

    d = half((maximum-minimum)/31), m = half(minimum); codes are ties-even
    against stored metadata and clipped to [0,31]. Nonzero d underflow is
    floored to 2^-24. Constant blocks store d=0 and all-zero codes.
    """
    _positive(block_count, "block_count")
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.dtype.kind not in "iuf":
        raise ValueError("weights must be real matrices")
    rows, columns = matrix.shape
    _positive(rows, "rows")
    _positive(columns, "columns")
    if columns % BLOCK_WIDTH:
        raise ValueError("Q5_1 row widths must be multiples of 32")
    data = bytearray(rows * columns // BLOCK_WIDTH * BLOCK_BYTES)
    cursor = 0
    for row in matrix:
        for first in range(0, columns, block_count * BLOCK_WIDTH):
            with np.errstate(over="ignore", invalid="ignore"):
                chunk = row[first:first + block_count * BLOCK_WIDTH].astype(np.float32).reshape(-1, BLOCK_WIDTH)
            if not np.isfinite(chunk).all():
                raise ValueError("weights must be finite and representable as float32")
            for block in chunk:
                smallest, largest = float(np.min(block)), float(np.max(block))
                with np.errstate(over="ignore", under="ignore"):
                    minimum = np.float16(smallest)
                    delta = np.float16((largest - smallest) / 31.0)
                if not np.isfinite(minimum) or not np.isfinite(delta):
                    raise ValueError("Q5_1 metadata exceeds binary16 range")
                if smallest == largest:
                    codes = np.zeros(BLOCK_WIDTH, np.uint8)
                else:
                    if delta == 0:
                        delta = np.float16(2 ** -24)
                    codes = np.clip(np.rint((block.astype(np.float64) - float(minimum))
                                           / float(delta)), 0, 31).astype(np.uint8)
                fifth = sum((int(codes[lane]) // 16) << lane for lane in range(BLOCK_WIDTH))
                paired = [int(codes[lane]) % 16 + 16 * (int(codes[lane + 16]) % 16)
                          for lane in range(16)]
                _RECORD.pack_into(data, cursor, float(delta), float(minimum), fifth, *paired)
                cursor += BLOCK_BYTES
    return bytes(data)


def pack_expert(gate, up, down, *, block_count=64):
    matrices = [np.asarray(matrix) for matrix in (gate, up, down)]
    if matrices[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    i, h = matrices[0].shape
    expected = _shapes(h, i)
    if any(matrix.shape != shape for matrix, shape in zip(matrices, expected)):
        raise ValueError("expert projection shape mismatch")
    return b"".join(pack_matrix(matrix, block_count=block_count) for matrix in matrices)


def decode_expert(payload, hidden_size, intermediate_size):
    """Full decode convenience API; returns three owned FP32 matrices."""
    shapes = _shapes(hidden_size, intermediate_size)
    with memoryview(payload) as borrowed:
        if not borrowed.c_contiguous or borrowed.nbytes != payload_size(hidden_size, intermediate_size):
            raise ValueError("Q5_1 payload must be contiguous with exact size")
        matrices = []
        offset = 0
        for shape in shapes:
            matrix = np.empty(shape, np.float32)
            flat = matrix.reshape(-1)
            for first in range(0, flat.size, BLOCK_WIDTH):
                d, m, fifth, *pairs = _RECORD.unpack_from(borrowed, offset)
                if not math.isfinite(d) or not math.isfinite(m):
                    raise ValueError("Q5_1 metadata must be finite")
                for lane in range(BLOCK_WIDTH):
                    pair = pairs[lane % 16]
                    low = pair % 16 if lane < 16 else pair // 16
                    code = low + 16 * ((fifth >> lane) & 1)
                    flat[first + lane] = code * d + m
                offset += BLOCK_BYTES
            matrices.append(matrix)
    return tuple(matrices)


def _dot(payload, offset, x, width):
    accumulator = np.float32(0)
    for start in range(0, width, BLOCK_WIDTH):
        d, m, fifth, *pairs = _RECORD.unpack_from(payload, offset)
        for lane in range(BLOCK_WIDTH):
            low = pairs[lane] % 16 if lane < 16 else pairs[lane - 16] // 16
            code = low + 16 * ((fifth >> lane) & 1)
            weight = np.float32(code * d + m)
            product = np.float32(weight * x[start + lane])
            accumulator = np.float32(accumulator + product)
        offset += BLOCK_BYTES
    return accumulator


def _scalar_expert(payload, x, h, i, coefficient, activation, output):
    for offset in range(0, payload.nbytes, BLOCK_BYTES):
        d, m = struct.unpack_from("<ee", payload, offset)
        if not math.isfinite(d) or not math.isfinite(m):
            return 1
    projection_bytes = h * i // BLOCK_WIDTH * BLOCK_BYTES
    gate_stride = h // BLOCK_WIDTH * BLOCK_BYTES
    for row in range(i):
        gate = _dot(payload, row * gate_stride, x, h)
        up = _dot(payload, projection_bytes + row * gate_stride, x, h)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        if gate >= 0:
            sigmoid = 1.0 / (1.0 + math.exp(-float(gate)))
        else:
            exp_gate = math.exp(float(gate))
            sigmoid = exp_gate / (1.0 + exp_gate)
        activation[row] = np.float32(np.float32(gate * np.float32(sigmoid)) * up)
        if not math.isfinite(activation[row]):
            return 2
    down_stride = i // BLOCK_WIDTH * BLOCK_BYTES
    for row in range(h):
        value = _dot(payload, 2 * projection_bytes + row * down_stride, activation, i)
        if not math.isfinite(value):
            return 2
        output[row] = np.float32(output[row] + np.float32(coefficient * value))
        if not math.isfinite(output[row]):
            return 2
    return 0


def scratch_bytes(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return intermediate_size * np.dtype(np.float32).itemsize


class Q5_1ExpertBackend:
    """Direct-byte scalar Q5_1 SwiGLU, one lease and one activation vector."""

    NAME = "q5_1-reference"

    def __init__(self, *, max_scratch_bytes=64 * 1024 * 1024):
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def info(self):
        return KernelBackendInfo(name=self.NAME, provider="openagent", available=True, priority=-12000,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True),
            evidence=(KernelEvidence("implementation", "Independent affine Q5_1 byte reductions"),))

    def _check_available(self):
        if (type(self.max_scratch_bytes) is not int or self.max_scratch_bytes <= 0 or
                BLOCK_WIDTH != 32 or BLOCK_BYTES != 24):
            raise RuntimeError("Q5_1 backend admission state is invalid")

    def _run_expert(self, payload, vector, h, i, weight, activation, output):
        return _scalar_expert(payload, vector, h, i, weight, activation, output)

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        self._check_available()
        _shapes(hidden_size, intermediate_size)
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        self._check_available()
        length = payload_size(hidden_size, intermediate_size)
        if scratch_bytes(hidden_size, intermediate_size) > self.max_scratch_bytes:
            raise MemoryError("Q5_1 activation scratch exceeds max_scratch_bytes")
        if dtype != DTYPE:
            raise ValueError("unsupported Q5_1 dtype")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        vectors = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
        routing = list(route_batch)
        if len(routing) != len(vectors):
            raise ValueError("one route list required per input row")
        routing = [_validate_routes(routes, "routes") for routes in routing]
        for routes in routing:
            if any(weight > float(np.finfo(np.float32).max) for _, weight in routes):
                raise ValueError("route weights must be representable as float32")
        output = np.zeros_like(vectors)
        if not len(vectors):
            return output
        activation = np.empty(intermediate_size, np.float32)
        for row, routes in enumerate(routing):
            for key, weight in routes:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != length:
                        raise ValueError("Q5_1 payload size mismatch")
                    try:
                        with np.errstate(over="ignore", invalid="ignore"):
                            status = self._run_expert(payload, vectors[row], hidden_size, intermediate_size,
                                                      np.float32(weight), activation, output[row])
                    except BaseException as error:
                        _detach_borrowed_frames(error)
                        failure = error
                if failure is not None:
                    raise failure from None
                if status == 1:
                    raise ValueError("Q5_1 metadata must be finite")
                if status != 0:
                    raise ArithmeticError("Q5_1 computation produced non-finite output")
        return output
