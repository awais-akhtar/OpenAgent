"""Optional direct-byte grouped INT4 CPU kernel; no silent scalar fallback."""

import math

import numpy as np

from .expert_kernels import KernelBackendInfo, KernelCapabilities, KernelEvidence
from .grouped_int4_expert import DTYPE, MANIFEST_DTYPE, GroupedInt4ExpertBackend, scratch_bytes as scratch_bytes


def _read_f32(raw, offset):
    word = (int(raw[offset]) | (int(raw[offset + 1]) << 8) |
            (int(raw[offset + 2]) << 16) | (int(raw[offset + 3]) << 24))
    power, fraction = (word >> 23) & 255, word & 0x7fffff
    if power == 255:
        return np.float32(np.nan)
    value = math.ldexp(float(fraction), -149) if power == 0 else math.ldexp(float(0x800000 + fraction), power - 150)
    return np.float32(-value if word & 0x80000000 else value)


def _packed_dot(raw, codes, scales, x, width, group_size):
    total = np.float32(0)
    for group in range((width + group_size - 1) // group_size):
        scale = _read_f32(raw, scales + group * 4)
        for column in range(group * group_size, min(width, (group + 1) * group_size)):
            code = (int(raw[codes + column // 2]) >> (4 * (column % 2))) & 15
            signed = code if code < 8 else code - 16
            weight = np.float32(np.float32(signed) * scale)
            total = np.float32(total + np.float32(weight * x[column]))
    return total


def _kernel(raw, x, h, i, group_size, weight, activation, output):
    gate_stride, down_stride = (h + 1) // 2, (i + 1) // 2
    gate_groups, down_groups = (h + group_size - 1) // group_size, (i + group_size - 1) // group_size
    up_codes, down_codes = i * gate_stride, 2 * i * gate_stride
    scale_start = down_codes + h * down_stride
    up_scales, down_scales = scale_start + 4 * i * gate_groups, scale_start + 8 * i * gate_groups
    for offset in range(scale_start, raw.size, 4):
        if not math.isfinite(_read_f32(raw, offset)):
            return 1
    if h % 2:
        for row in range(2 * i):
            if int(raw[(row + 1) * gate_stride - 1]) >> 4:
                return 3
    if i % 2:
        for row in range(h):
            if int(raw[down_codes + (row + 1) * down_stride - 1]) >> 4:
                return 3
    for row in range(i):
        gate = _packed_dot(raw, row * gate_stride, scale_start + 4 * row * gate_groups, x, h, group_size)
        up = _packed_dot(raw, up_codes + row * gate_stride, up_scales + 4 * row * gate_groups, x, h, group_size)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        e = math.exp(-abs(float(gate)))
        sigmoid = np.float32((1.0 if gate >= 0 else e) / (1.0 + e))
        activation[row] = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(activation[row]):
            return 2
    for row in range(h):
        projected = _packed_dot(raw, down_codes + row * down_stride, down_scales + 4 * row * down_groups,
                                activation, i, group_size)
        if not math.isfinite(projected):
            return 2
        output[row] = np.float32(output[row] + np.float32(weight * projected))
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
        numba.extending.register_jitable(_read_f32)
        numba.extending.register_jitable(_packed_dot)
        _compiled_kernel = numba.njit(cache=False, nogil=True, fastmath=False)(_kernel)
except Exception as error:
    _unavailable_reason = f"Numba unavailable: {type(error).__name__}: {error}"


class NativeGroupedInt4ExpertBackend(GroupedInt4ExpertBackend):
    NAME = "grouped-int4-numba"

    def info(self):
        ready = _compiled_kernel is not None
        return KernelBackendInfo(name=self.NAME, provider="numba", available=ready, priority=100,
            capabilities=KernelCapabilities(frozenset({DTYPE, MANIFEST_DTYPE} if self.group_size == 64 else {DTYPE}), True, True, compiled=ready),
            evidence=(KernelEvidence("implementation", "Independent grouped INT4 Numba byte reductions"),),
            unavailable_reason="" if ready else (_unavailable_reason or "grouped JIT unavailable"))

    def _check_available(self):
        if _compiled_kernel is None:
            raise RuntimeError(_unavailable_reason or "grouped JIT unavailable")

    def _run(self, payload, x, h, i, weight, activation, output):
        raw = np.frombuffer(payload, np.uint8)
        try:
            return _compiled_kernel(raw, x, h, i, self.group_size, weight, activation, output)
        finally:
            del raw
