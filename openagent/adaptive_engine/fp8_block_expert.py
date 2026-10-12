"""Independent E4M3FN/block128 FP32-scale payload contract (see ENGINE_FP8_BLOCK)."""
from __future__ import annotations

import math
import struct
import traceback
import numpy as np

from .expert_kernels import (ExpertPayloadSource, KernelBackendInfo, KernelCapabilities,
                             KernelEvidence, _finite_array, _finite_matrix, _lease, _validate_routes)

DTYPE = "fp8-e4m3-b128-f32"
BLOCK_WIDTH = 128
SCALE_ENCODING = "f32"


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _encoding(value):
    if value != SCALE_ENCODING:
        raise ValueError("unsupported scale encoding; expected explicit f32")


def _shapes(hidden, intermediate):
    _positive(hidden, "hidden_size")
    _positive(intermediate, "intermediate_size")
    return ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))


def matrix_payload_size(rows, columns):
    _positive(rows, "rows")
    _positive(columns, "columns")
    return rows * columns + 4 * ((rows + 127) // 128) * ((columns + 127) // 128)


def payload_size(hidden_size, intermediate_size):
    return sum(matrix_payload_size(*shape) for shape in _shapes(hidden_size, intermediate_size))


def decode_code(code):
    """Decode every byte, including signed zero and both NaN encodings."""
    if isinstance(code, bool) or not isinstance(code, (int, np.integer)) or not 0 <= code <= 255:
        raise ValueError("code must be a byte")
    exponent, fraction = (int(code) >> 3) & 15, int(code) & 7
    if exponent == 15 and fraction == 7:
        return math.nan
    value = math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(8 + fraction, exponent - 10)
    return -value if int(code) & 128 else value


_LEVELS = np.array([decode_code(c) for c in range(127)], np.float64)


def pack_matrix(matrix, *, scale_encoding=SCALE_ENCODING):
    """Pack finite real weights with block-bounded conversion scratch and RNE."""
    _encoding(scale_encoding)
    a = np.asarray(matrix)
    if a.ndim != 2 or a.dtype.kind not in "iuf":
        raise ValueError("weights must be a real matrix")
    rows, cols = a.shape
    result = bytearray(matrix_payload_size(rows, cols))
    grid_cols = (cols + 127) // 128
    for r in range(0, rows, 128):
        for c in range(0, cols, 128):
            with np.errstate(over="ignore", invalid="ignore"):
                block = a[r:r+128, c:c+128].astype(np.float32)
            if not np.isfinite(block).all():
                raise ValueError("weights must be finite float32")
            maximum = float(np.max(np.abs(block)))
            scale = np.float32(maximum / 448)
            if maximum and scale == 0:
                scale = np.nextafter(np.float32(0), np.float32(1))
            values = np.abs(block.astype(np.float64)) / (float(scale) if scale else 1)
            high = np.minimum(np.searchsorted(_LEVELS, values), 126)
            low = np.maximum(high - 1, 0)
            dl, dh = values - _LEVELS[low], _LEVELS[high] - values
            codes = np.where((dl < dh) | ((dl == dh) & ((low & 1) == 0)), low, high).astype(np.uint8)
            codes |= np.signbit(block).astype(np.uint8) * 128
            for j in range(block.shape[0]):
                start = (r+j) * cols + c
                result[start:start+block.shape[1]] = codes[j].tobytes()
            offset = rows * cols + 4 * ((r // 128) * grid_cols + c // 128)
            result[offset:offset+4] = np.asarray(scale, dtype="<f4").tobytes()
    return bytes(result)


def pack_expert(gate, up, down, *, scale_encoding=SCALE_ENCODING):
    _encoding(scale_encoding)
    arrays = tuple(np.asarray(a) for a in (gate, up, down))
    if arrays[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    intermediate, hidden = arrays[0].shape
    if any(a.shape != s for a, s in zip(arrays, _shapes(hidden, intermediate))):
        raise ValueError("expert projection shape mismatch")
    return b"".join(pack_matrix(a, scale_encoding=scale_encoding) for a in arrays)


def scratch_bytes(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return intermediate_size * 4


def _clear_failure(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = current.__cause__ = current.__context__ = None


def _reference_project(raw, offset, row, rows, cols, vector):
    total = np.float32(0)
    for column in range(cols):
        scale_offset = offset + rows*cols + 4*((row//128)*((cols+127)//128)+column//128)
        scale = np.float32(struct.unpack_from("<f", raw, scale_offset)[0])
        weight = np.float32(np.float32(decode_code(int(raw[offset+row*cols+column]))) * scale)
        total = np.float32(total + np.float32(weight * vector[column]))
    return total


def _reference_loop(raw, vector, route_weight, hidden, intermediate, activation, output):
    stride = matrix_payload_size(intermediate, hidden)
    offset = 0
    for rows, cols in _shapes(hidden, intermediate):
        for index in range(rows*cols):
            if (int(raw[offset+index]) & 127) == 127:
                return 1
        for index in range(((rows+127)//128)*((cols+127)//128)):
            if not math.isfinite(struct.unpack_from("<f", raw, offset+rows*cols+4*index)[0]):
                return 1
        offset += matrix_payload_size(rows, cols)
    for row in range(intermediate):
        gate = _reference_project(raw, 0, row, intermediate, hidden, vector)
        up = _reference_project(raw, stride, row, intermediate, hidden, vector)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        if gate >= 0:
            sigmoid = np.float32(1 / (1 + math.exp(-float(gate))))
        else:
            exponential = math.exp(float(gate))
            sigmoid = np.float32(exponential / (1 + exponential))
        activation[row] = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(activation[row]):
            return 2
    for row in range(hidden):
        down = _reference_project(raw, 2*stride, row, hidden, intermediate, activation)
        value = np.float32(output[row] + np.float32(route_weight * down))
        if not math.isfinite(down) or not math.isfinite(value):
            return 2
        output[row] = value
    return 0


class FP8BlockExpertBackend:
    """Explicit scalar reference backend, independent of Numba availability."""

    NAME = "fp8-e4m3-b128-f32-reference"

    def __init__(self, *, max_scratch_bytes=64*1024*1024, scale_encoding=SCALE_ENCODING):
        _encoding(scale_encoding)
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def info(self):
        return KernelBackendInfo(name=self.NAME, provider="openagent", available=True, priority=-11000,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True),
            evidence=(KernelEvidence("implementation", "Scalar direct-byte FP8 reference execution"),))

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        if dtype != DTYPE:
            raise ValueError("unsupported FP8 dtype")
        if scratch_bytes(hidden_size, intermediate_size) > self.max_scratch_bytes:
            raise MemoryError("FP8 activation scratch exceeds max_scratch_bytes")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        batch = _finite_matrix(inputs, hidden_size)
        routes = list(route_batch)
        if len(routes) != len(batch):
            raise ValueError("one route list is required per input row")
        routes = [_validate_routes(row, "routes") for row in routes]
        for assignments in routes:
            for _, weight in assignments:
                if weight > float(np.finfo(np.float32).max):
                    raise ValueError("route weight must be representable as float32")
        output = np.zeros_like(batch)
        activation = np.empty(intermediate_size, np.float32)
        expected = payload_size(hidden_size, intermediate_size)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for row, assignments in enumerate(routes):
                for key, weight in assignments:
                    failure = None
                    with _lease(source, key) as payload:
                        if payload.nbytes != expected:
                            raise ValueError("FP8 payload size mismatch")
                        raw = np.frombuffer(payload, np.uint8)
                        try:
                            status = _reference_loop(raw, batch[row], np.float32(weight), hidden_size,
                                                     intermediate_size, activation, output[row])
                        except BaseException as error:
                            _clear_failure(error)
                            failure = error
                        finally:
                            del raw
                    if failure is not None:
                        raise failure from None
                    if status == 1:
                        raise ValueError("FP8 codes and scales must be finite")
                    if status != 0:
                        raise ArithmeticError("FP8 computation produced non-finite output")
        return output
