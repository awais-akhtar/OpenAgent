"""Optional CPU JIT multiplication directly from leased INT4 weight bytes."""

import math

import numpy as np

from . import native_expert as _native
from .expert_kernels import (
    ExpertPayloadSource, KernelBackendInfo, KernelCapabilities, KernelEvidence,
    _finite_matrix, _finite_output, _lease, _validate_routes,
)
from .packed_expert import DTYPE, PackedInt4Backend, _layout, _shapes


def _loop(x, rows, weights, raw, scales, hidden, intermediate, output):
    stride = (hidden + 1) // 2
    down_stride = (intermediate + 1) // 2
    up_offset = intermediate * stride
    down_offset = up_offset * 2
    activation = np.empty(intermediate, dtype=np.float32)
    for assignment in range(rows.size):
        row = rows[assignment]
        for i in range(intermediate):
            gate = np.float32(0)
            up = np.float32(0)
            for j in range(hidden):
                shift = 4 * (j & 1)
                g = (raw[i * stride + j // 2] >> shift) & 15
                u = (raw[up_offset + i * stride + j // 2] >> shift) & 15
                gs = int(g) - 16 if g >= 8 else int(g)
                us = int(u) - 16 if u >= 8 else int(u)
                gate += (np.float32(gs) * scales[i]) * x[row, j]
                up += (np.float32(us) * scales[intermediate + i]) * x[row, j]
            if gate >= 0:
                sigmoid = 1.0 / (1.0 + math.exp(-gate))
            else:
                exponential = math.exp(gate)
                sigmoid = exponential / (1.0 + exponential)
            activation[i] = gate * sigmoid * up
        for j in range(hidden):
            value = np.float32(0)
            for i in range(intermediate):
                q = (raw[down_offset + j * down_stride + i // 2] >> (4 * (i & 1))) & 15
                signed = int(q) - 16 if q >= 8 else int(q)
                value += (np.float32(signed) * scales[2 * intermediate + j]) * activation[i]
            output[row, j] += weights[assignment] * value


_compiled_loop = (
    _native._numba.njit(nogil=True, fastmath=False)(_loop)
    if _native._numba is not None and not _native._unavailable else None
)


class NativePackedExpertBackend(PackedInt4Backend):
    """Serial FP32 CPU reductions with packed INT4 operands, no weight cache."""

    NAME = "packed-int4-numba"

    def info(self):
        available = _compiled_loop is not None
        return KernelBackendInfo(
            name=self.NAME, provider="numba", available=available, priority=100,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True, compiled=available),
            evidence=(KernelEvidence("implementation", "Direct nibble CPU reductions; no fastmath"),),
            unavailable_reason="" if available else (_native._unavailable or "Packed INT4 JIT unavailable"),
        )

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        if _compiled_loop is None:
            raise RuntimeError(_native._unavailable)
        _shapes(hidden_size, intermediate_size)
        if dtype != DTYPE:
            raise ValueError("unsupported packed dtype")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        batch = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
        route_rows = list(route_batch)
        if len(route_rows) != len(batch):
            raise ValueError("one route list is required per input row")
        groups = {}
        for row, routes in enumerate(route_rows):
            for key, weight in _validate_routes(routes, f"route_batch[{row}]"):
                if not np.isfinite(np.float32(weight)):
                    raise ValueError("route weight must be representable as float32")
                groups.setdefault(key, []).append((row, weight))
        output = np.zeros_like(batch)
        packed_size = 2 * intermediate_size * ((hidden_size + 1) // 2) + hidden_size * ((intermediate_size + 1) // 2)
        for key, assignments in groups.items():
            with _lease(source, key) as payload:
                _layout(payload, hidden_size, intermediate_size, self.block_rows)
                raw = np.frombuffer(payload, np.uint8, packed_size)
                scales = np.frombuffer(payload, "<f4", 2 * intermediate_size + hidden_size, packed_size)
                rows = np.asarray([row for row, weight in assignments], dtype=np.intp)
                weights = np.asarray([weight for row, weight in assignments], dtype=np.float32)
                _compiled_loop(batch, rows, weights, raw, scales, hidden_size, intermediate_size, output)
        return _finite_output(output)
