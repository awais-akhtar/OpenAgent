"""Optional compiled Q8_1 CPU reductions over leased bytes."""

import math

import numpy as np

from .expert_kernels import KernelBackendInfo, KernelCapabilities, KernelEvidence
from .native_q5_0_expert import _read_half
from .q8_1_expert import DTYPE, Q8_1ExpertBackend


def _dot(raw, offset, vector, width):
    value = np.float32(0)
    for block in range(width // 32):
        address = offset + block * 36
        delta = _read_half(raw, address)
        for lane in range(32):
            code = int(raw[address + 4 + lane])
            if code >= 128:
                code -= 256
            weight = np.float32(float(delta) * code)
            value = np.float32(value + np.float32(weight * vector[block * 32 + lane]))
    return value


def _kernel(raw, vector, hidden, intermediate, coefficient, activation, output):
    for address in range(0, raw.size, 36):
        if not math.isfinite(_read_half(raw, address)) or not math.isfinite(_read_half(raw, address + 2)):
            return 1
    area = hidden * intermediate // 32 * 36
    stride = hidden // 32 * 36
    for row in range(intermediate):
        gate = _dot(raw, row * stride, vector, hidden)
        up = _dot(raw, area + row * stride, vector, hidden)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        exponential = math.exp(-abs(float(gate)))
        sigmoid = np.float32((1.0 if gate >= 0 else exponential) / (1.0 + exponential))
        activation[row] = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(activation[row]):
            return 2
    stride = intermediate // 32 * 36
    for row in range(hidden):
        value = _dot(raw, 2 * area + row * stride, activation, intermediate)
        output[row] = np.float32(output[row] + np.float32(coefficient * value))
        if not math.isfinite(value) or not math.isfinite(output[row]):
            return 2
    return 0


_compiled = None
_reason = ""
try:
    import numba
    if numba.config.DISABLE_JIT:
        _reason = "Numba JIT disabled"
    else:
        numba.extending.register_jitable(_dot)
        _compiled = numba.njit(cache=False, nogil=True, fastmath=False)(_kernel)
except Exception as error:
    _reason = f"Numba unavailable: {type(error).__name__}: {error}"


class NativeQ8_1ExpertBackend(Q8_1ExpertBackend):
    NAME = "q8_1-numba"

    def info(self):
        ready = _compiled is not None
        return KernelBackendInfo(
            name=self.NAME, provider="numba", available=ready, priority=100,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True, compiled=ready),
            evidence=(KernelEvidence("implementation", "Direct Q8_1 Numba byte reductions"),),
            unavailable_reason="" if ready else (_reason or "Q8_1 JIT unavailable"),
        )

    def _check_available(self):
        if _compiled is None:
            raise RuntimeError(_reason or "Q8_1 JIT unavailable")

    def _run_expert(self, payload, *arguments):
        raw = np.frombuffer(payload, np.uint8)
        try:
            return _compiled(raw, *arguments)
        finally:
            del raw
