"""Compiled affine Q5_1 byte reductions; unavailable without active Numba."""

import math

import numpy as np

from .expert_kernels import KernelBackendInfo, KernelCapabilities, KernelEvidence
from .native_q5_0_expert import _read_half
from .q5_1_expert import DTYPE, Q5_1ExpertBackend, scratch_bytes as scratch_bytes


def _packed_dot(raw, offset, vector, width):
    total = np.float32(0)
    for block in range(width // 32):
        address = offset + 24 * block
        delta = float(_read_half(raw, address))
        minimum = float(_read_half(raw, address + 2))
        for lane in range(32):
            pair = int(raw[address + 8 + lane % 16])
            nibble = (pair & 15) if lane < 16 else (pair >> 4)
            fifth = (int(raw[address + 4 + lane // 8]) >> (lane % 8)) & 1
            value = np.float32((nibble + 16 * fifth) * delta + minimum)
            total = np.float32(total + np.float32(value * vector[32 * block + lane]))
    return total


def _kernel(raw, inputs, hidden, intermediate, coefficient, activation, output):
    for address in range(0, raw.size, 24):
        delta = _read_half(raw, address)
        minimum = _read_half(raw, address + 2)
        if not math.isfinite(delta) or not math.isfinite(minimum):
            return 1
    area = hidden * intermediate // 32 * 24
    row_bytes = hidden // 32 * 24
    for row in range(intermediate):
        gate = _packed_dot(raw, row * row_bytes, inputs, hidden)
        up = _packed_dot(raw, area + row * row_bytes, inputs, hidden)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        e = math.exp(-abs(float(gate)))
        sigmoid = np.float32((1.0 if gate >= 0 else e) / (1.0 + e))
        activation[row] = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(activation[row]):
            return 2
    row_bytes = intermediate // 32 * 24
    for row in range(hidden):
        projected = _packed_dot(raw, 2 * area + row * row_bytes, activation, intermediate)
        if not math.isfinite(projected):
            return 2
        output[row] = np.float32(output[row] + np.float32(coefficient * projected))
        if not math.isfinite(output[row]):
            return 2
    return 0


_compiled_kernel = None
_unavailable_reason = ""
try:
    import numba
    if numba.config.DISABLE_JIT:
        _unavailable_reason = "Numba JIT disabled"
    else:
        numba.extending.register_jitable(_packed_dot)
        _compiled_kernel = numba.njit(cache=False, nogil=True, fastmath=False)(_kernel)
except Exception as error:
    _unavailable_reason = f"Numba unavailable: {type(error).__name__}: {error}"


class NativeQ5_1ExpertBackend(Q5_1ExpertBackend):
    NAME = "q5_1-numba"

    def info(self):
        ready = _compiled_kernel is not None
        return KernelBackendInfo(name=self.NAME, provider="numba", available=ready, priority=100,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True, compiled=ready),
            evidence=(KernelEvidence("implementation", "Direct affine Q5_1 Numba byte reductions"),),
            unavailable_reason="" if ready else (_unavailable_reason or "Q5_1 JIT unavailable"))

    def _check_available(self):
        if _compiled_kernel is None:
            raise RuntimeError(_unavailable_reason or "Q5_1 JIT unavailable")

    def _run_expert(self, payload, vector, h, i, weight, activation, output):
        raw = np.frombuffer(payload, np.uint8)
        try:
            return _compiled_kernel(raw, vector, h, i, weight, activation, output)
        finally:
            del raw
