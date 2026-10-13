"""Q8_1 leased experts using the public 36-byte block field layout.

Layout source: ggml-org/ggml src/ggml-common.h (block_q8_1).
Packing and execution are independent; no upstream quantizer is used.
"""

import math
import struct

import numpy as np

from .expert_kernels import (
    ExpertPayloadSource, KernelBackendInfo, KernelCapabilities, KernelEvidence,
    _finite_array, _finite_matrix, _lease, _validate_routes,
)
from .q5_0_expert import _detach_borrowed_frames
from .q8_0_expert import _positive, _shapes, pack_matrix as pack_q8_0

DTYPE = "q8_1"
BLOCK_BYTES = 36
BLOCK_WIDTH = 32
_RECORD = struct.Struct("<ee32b")


def payload_size(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return 3 * hidden_size * intermediate_size // BLOCK_WIDTH * BLOCK_BYTES


def pack_matrix(matrix, *, block_count=64):
    """Use the Q8_0 independent rounding policy, plus half-rounded code sum."""
    encoded = pack_q8_0(matrix, block_count=block_count)
    result = bytearray(len(encoded) // 34 * BLOCK_BYTES)
    for index, offset in enumerate(range(0, len(encoded), 34)):
        delta, *codes = struct.unpack_from("<e32b", encoded, offset)
        with np.errstate(over="ignore"):
            total = np.float16(delta * sum(codes))
        if not np.isfinite(total):
            raise ValueError("Q8_1 sum exceeds binary16 range")
        _RECORD.pack_into(result, index * BLOCK_BYTES, delta, float(total), *codes)
    return bytes(result)


def pack_expert(gate, up, down, *, block_count=64):
    matrices = [np.asarray(value) for value in (gate, up, down)]
    if matrices[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    intermediate, hidden = matrices[0].shape
    shapes = _shapes(hidden, intermediate)
    if any(value.shape != shape for value, shape in zip(matrices, shapes)):
        raise ValueError("expert projection shape mismatch")
    return b"".join(pack_matrix(value, block_count=block_count) for value in matrices)


def decode_expert(payload, hidden_size, intermediate_size):
    """Explicit owned FP32 decode; execution does not allocate these matrices."""
    shapes = _shapes(hidden_size, intermediate_size)
    with memoryview(payload) as view:
        if not view.c_contiguous or view.nbytes != payload_size(hidden_size, intermediate_size):
            raise ValueError("Q8_1 payload size or contiguity mismatch")
        offset = 0
        result = []
        for shape in shapes:
            matrix = np.empty(shape, np.float32)
            flat = matrix.reshape(-1)
            for first in range(0, flat.size, BLOCK_WIDTH):
                delta, total, *codes = _RECORD.unpack_from(view, offset)
                if not math.isfinite(delta) or not math.isfinite(total):
                    raise ValueError("Q8_1 metadata must be finite")
                flat[first:first + BLOCK_WIDTH] = np.asarray(codes, np.float32) * delta
                offset += BLOCK_BYTES
            result.append(matrix)
    return tuple(result)


def _dot(payload, offset, vector, width):
    result = np.float32(0)
    for first in range(0, width, BLOCK_WIDTH):
        delta, _, *codes = _RECORD.unpack_from(payload, offset)
        for lane, code in enumerate(codes):
            weight = np.float32(delta * code)
            result = np.float32(result + np.float32(weight * vector[first + lane]))
        offset += BLOCK_BYTES
    return result


def _kernel(payload, vector, hidden, intermediate, coefficient, activation, output):
    for offset in range(0, payload.nbytes, BLOCK_BYTES):
        delta, total = struct.unpack_from("<ee", payload, offset)
        if not math.isfinite(delta) or not math.isfinite(total):
            return 1
    area = hidden * intermediate // BLOCK_WIDTH * BLOCK_BYTES
    stride = hidden // BLOCK_WIDTH * BLOCK_BYTES
    for row in range(intermediate):
        gate = _dot(payload, row * stride, vector, hidden)
        up = _dot(payload, area + row * stride, vector, hidden)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        exponential = math.exp(-abs(float(gate)))
        sigmoid = np.float32((1.0 if gate >= 0 else exponential) / (1.0 + exponential))
        activation[row] = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(activation[row]):
            return 2
    stride = intermediate // BLOCK_WIDTH * BLOCK_BYTES
    for row in range(hidden):
        value = _dot(payload, 2 * area + row * stride, activation, intermediate)
        output[row] = np.float32(output[row] + np.float32(coefficient * value))
        if not math.isfinite(value) or not math.isfinite(output[row]):
            return 2
    return 0


class Q8_1ExpertBackend:
    """Direct-byte SwiGLU with one leased expert and one activation vector."""

    NAME = "q8_1-reference"

    def __init__(self, *, max_scratch_bytes=64 * 1024 * 1024):
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def info(self):
        return KernelBackendInfo(
            name=self.NAME, provider="openagent", available=True, priority=-12000,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True),
            evidence=(KernelEvidence("implementation", "Direct Q8_1 scalar reductions"),),
        )

    def _check_available(self):
        if (type(self.max_scratch_bytes) is not int or self.max_scratch_bytes <= 0 or
                BLOCK_WIDTH != 32 or BLOCK_BYTES != 36):
            raise RuntimeError("Q8_1 backend admission state is invalid")

    def _run_expert(self, payload, *arguments):
        return _kernel(payload, *arguments)

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        self._check_available()
        length = payload_size(hidden_size, intermediate_size)
        if dtype != DTYPE:
            raise ValueError("unsupported Q8_1 dtype")
        if intermediate_size * 4 > self.max_scratch_bytes:
            raise MemoryError("Q8_1 activation exceeds scratch budget")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        vectors = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
        routing = list(route_batch)
        if len(routing) != len(vectors):
            raise ValueError("one route list required per input row")
        routing = [_validate_routes(routes, "routes") for routes in routing]
        if any(weight > float(np.finfo(np.float32).max) for routes in routing for _, weight in routes):
            raise ValueError("route weights must be representable as float32")
        output = np.zeros_like(vectors)
        activation = np.empty(intermediate_size, np.float32)
        for row, routes in enumerate(routing):
            for key, weight in routes:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != length:
                        raise ValueError("Q8_1 payload size mismatch")
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
                    raise ValueError("Q8_1 metadata must be finite")
                if status != 0:
                    raise ArithmeticError("Q8_1 computation produced non-finite output")
        return output
