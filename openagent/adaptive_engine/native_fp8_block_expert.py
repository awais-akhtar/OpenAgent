"""Optional genuine Numba CPU execution directly over leased FP8 bytes."""
from __future__ import annotations

import math
import numpy as np

from .fp8_block_expert import (DTYPE, SCALE_ENCODING, _encoding, _positive, payload_size,
                               scratch_bytes, _clear_failure)
from .expert_kernels import (ExpertPayloadSource, KernelBackendInfo, KernelCapabilities,
                             KernelEvidence, _finite_array, _finite_matrix, _lease, _validate_routes)

try:
    import numba as _numba
except Exception as error:
    _numba = None
    _unavailable = f"Numba unavailable: {type(error).__name__}: {error}"
else:
    _unavailable = "Numba JIT disabled" if _numba.config.DISABLE_JIT else ""


def _code(bits):
    e, f = (bits >> 3) & 15, bits & 7
    if e == 15 and f == 7:
        return np.float32(np.nan)
    value = math.ldexp(float(f), -9) if e == 0 else math.ldexp(float(8 + f), e - 10)
    return np.float32(-value if bits & 128 else value)


def _scale(raw, offset):
    bits = (int(raw[offset]) | (int(raw[offset+1]) << 8) |
            (int(raw[offset+2]) << 16) | (int(raw[offset+3]) << 24))
    e, f = (bits >> 23) & 255, bits & 8388607
    if e == 255:
        return np.float32(np.nan)
    value = math.ldexp(float(f), -149) if e == 0 else math.ldexp(float(8388608+f), e-150)
    return np.float32(-value if bits & 2147483648 else value)


def _size(rows, cols):
    return rows * cols + 4 * ((rows+127)//128) * ((cols+127)//128)


def _valid(raw, offset, rows, cols):
    for index in range(rows * cols):
        if (int(raw[offset+index]) & 127) == 127:
            return False
    scale_rows, scale_cols = (rows+127)//128, (cols+127)//128
    for scale_row in range(scale_rows):
        for scale_col in range(scale_cols):
            scale = _scale(raw, offset + rows*cols +
                           4*(scale_row*scale_cols+scale_col))
            if not math.isfinite(scale):
                return False
    return True


def _dot(raw, offset, row, rows, cols, vector):
    total = np.float32(0)
    for column in range(cols):
        scale_offset = offset + rows*cols + 4 * ((row//128)*((cols+127)//128) + column//128)
        weight = np.float32(_code(int(raw[offset+row*cols+column])) * _scale(raw, scale_offset))
        total = np.float32(total + np.float32(weight * vector[column]))
    return total


def _loop(raw, vector, route_weight, hidden, intermediate, activation, output):
    stride = _size(intermediate, hidden)
    if not (_valid(raw, 0, intermediate, hidden) and
            _valid(raw, stride, intermediate, hidden) and
            _valid(raw, 2*stride, hidden, intermediate)):
        return 1
    for row in range(intermediate):
        gate = _dot(raw, 0, row, intermediate, hidden, vector)
        up = _dot(raw, stride, row, intermediate, hidden, vector)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        if gate >= 0:
            sigmoid = np.float32(1 / (1 + math.exp(-float(gate))))
        else:
            exp = math.exp(float(gate))
            sigmoid = np.float32(exp / (1 + exp))
        activation[row] = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(activation[row]):
            return 2
    for row in range(hidden):
        down = _dot(raw, 2*stride, row, hidden, intermediate, activation)
        value = np.float32(output[row] + np.float32(route_weight * down))
        if not math.isfinite(down) or not math.isfinite(value):
            return 2
        output[row] = value
    return 0


if _numba is not None and not _unavailable:
    _code = _numba.njit(inline="always", fastmath=False)(_code)
    _scale = _numba.njit(inline="always", fastmath=False)(_scale)
    _size = _numba.njit(inline="always")(_size)
    _valid = _numba.njit(inline="always", fastmath=False)(_valid)
    _dot = _numba.njit(inline="always", fastmath=False)(_dot)
    _compiled_loop = _numba.njit(nogil=True, cache=False, fastmath=False)(_loop)
else:
    _compiled_loop = None


class NativeFP8BlockExpertBackend:
    NAME = "fp8-e4m3-b128-f32-numba"

    def __init__(self, *, max_scratch_bytes=64*1024*1024, scale_encoding=SCALE_ENCODING):
        _encoding(scale_encoding)
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def info(self):
        available = _compiled_loop is not None
        return KernelBackendInfo(name=self.NAME, provider="numba", available=available, priority=100,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True, compiled=available),
            evidence=(KernelEvidence("implementation", "Independent direct FP8 block128 byte kernel"),),
            unavailable_reason="" if available else _unavailable)

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        if _compiled_loop is None:
            raise RuntimeError(_unavailable or "FP8 JIT unavailable")
        if dtype != DTYPE:
            raise ValueError("unsupported FP8 dtype")
        if scratch_bytes(hidden_size, intermediate_size) > self.max_scratch_bytes:
            raise MemoryError("FP8 activation scratch exceeds max_scratch_bytes")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        batch = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
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
        for row, assignments in enumerate(routes):
            for key, weight in assignments:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != expected:
                        raise ValueError("FP8 payload size mismatch")
                    raw = np.frombuffer(payload, np.uint8)
                    try:
                        status = _compiled_loop(raw, batch[row], np.float32(weight), hidden_size,
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
