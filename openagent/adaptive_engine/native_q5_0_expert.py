"""Optional independent compiled CPU Q5_0 reductions, no Python fallback."""

import math

import numpy as np

from .expert_kernels import KernelBackendInfo, KernelCapabilities, KernelEvidence
from .q5_0_expert import DTYPE, Q5_0ExpertBackend, scratch_bytes as scratch_bytes


def _read_half(bytes_view, position):
    bits = int(bytes_view[position]) | int(bytes_view[position + 1]) << 8
    exponent = bits // 1024 % 32
    fraction = bits % 1024
    if exponent == 31:
        return np.float32(np.nan)
    value = math.ldexp(float(fraction), -24) if exponent == 0 else math.ldexp(
        float(1024 + fraction), exponent - 25)
    return np.float32(value if bits < 32768 else -value)


def _packed_dot(bytes_view, position, x, columns):
    sum_value = np.float32(0)
    for block in range(columns // 32):
        begin = position + 22 * block
        delta = _read_half(bytes_view, begin)
        for lane in range(32):
            pair = int(bytes_view[begin + 6 + lane % 16])
            low = (pair & 15) if lane < 16 else pair >> 4
            high_byte = int(bytes_view[begin + 2 + lane // 8])
            high = (high_byte >> (lane % 8)) & 1
            centered = low + 16 * high - 16
            weight = np.float32(np.float32(centered) * delta)
            term = np.float32(weight * x[block * 32 + lane])
            sum_value = np.float32(sum_value + term)
    return sum_value


def _kernel(bytes_view, x, hidden, intermediate, weight, activation, output):
    # Validate even blocks whose route weight is zero, before any arithmetic.
    for position in range(0, bytes_view.size, 22):
        if not math.isfinite(_read_half(bytes_view, position)):
            return 1
    stride = hidden // 32 * 22
    matrix_size = intermediate * stride
    for row in range(intermediate):
        g = _packed_dot(bytes_view, row * stride, x, hidden)
        u = _packed_dot(bytes_view, matrix_size + row * stride, x, hidden)
        if not math.isfinite(g) or not math.isfinite(u):
            return 2
        positive = g >= 0
        e = math.exp(-float(g) if positive else float(g))
        sigmoid = np.float32((1.0 if positive else e) / (1.0 + e))
        activation[row] = np.float32(np.float32(g * sigmoid) * u)
        if not math.isfinite(activation[row]):
            return 2
    stride = intermediate // 32 * 22
    for row in range(hidden):
        projection = _packed_dot(bytes_view, 2 * matrix_size + row * stride, activation, intermediate)
        if not math.isfinite(projection):
            return 2
        output[row] = np.float32(output[row] + np.float32(weight * projection))
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
        numba.extending.register_jitable(_read_half)
        numba.extending.register_jitable(_packed_dot)
        _compiled_kernel = numba.njit(nogil=True, fastmath=False, cache=False)(_kernel)
except Exception as error:
    _unavailable_reason = f"Numba unavailable: {type(error).__name__}: {error}"


class NativeQ5_0ExpertBackend(Q5_0ExpertBackend):
    """Explicit opt-in native backend; lazy compilation failures propagate."""

    NAME = "q5_0-numba"

    def info(self):
        available = _compiled_kernel is not None
        return KernelBackendInfo(
            name=self.NAME, provider="numba", available=available, priority=100,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True, compiled=available),
            evidence=(KernelEvidence("implementation", "Independent direct-byte Q5_0 Numba kernel"),),
            unavailable_reason="" if available else (_unavailable_reason or "Q5_0 JIT unavailable"),
        )

    def _check_available(self):
        if _compiled_kernel is None:
            raise RuntimeError(_unavailable_reason or "Q5_0 JIT unavailable")

    def _run_expert(self, payload, vector, h, i, weight, activation, output):
        raw = np.frombuffer(payload, np.uint8)
        try:
            return _compiled_kernel(raw, vector, h, i, weight, activation, output)
        finally:
            del raw
