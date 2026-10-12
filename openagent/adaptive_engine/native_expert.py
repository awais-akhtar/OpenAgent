"""Optional CPU JIT kernels for routed experts, without resident weight copies.

Register explicitly with ``registry.register(NativeExpertBackend())``. Numba
specializations are shared across instances and cached on disk; payload arrays
are borrowed only for the duration of each source lease.
"""

from __future__ import annotations

import math

import numpy as np

from .expert_kernels import (
    KernelBackendInfo, KernelCapabilities, KernelEvidence,
    SUPPORTED_PAYLOAD_DTYPES, _decode_payload, _finite_array, _finite_matrix,
    _finite_output, _lease, _validate_configuration, _validate_routes,
)

try:
    import numba as _numba
except Exception as exc:
    _numba = None
    _unavailable = f"Numba could not be loaded: {type(exc).__name__}: {exc}"
else:
    _unavailable = "Numba JIT is disabled" if _numba.config.DISABLE_JIT else ""


def _expert_loop(inputs, rows, weights, gate, up, down, gs, us, ds, output):
    hidden = gate.shape[1]
    intermediate = gate.shape[0]
    activation = np.empty(intermediate, dtype=np.float32)
    for assignment in range(rows.size):
        row = rows[assignment]
        for i in range(intermediate):
            g = np.float32(0)
            u = np.float32(0)
            for j in range(hidden):
                g += np.float32(gate[i, j]) * inputs[row, j]
                u += np.float32(up[i, j]) * inputs[row, j]
            g *= gs[i]
            u *= us[i]
            # Separate sigmoid branches avoid exponential overflow.
            if g >= 0:
                sigmoid = 1.0 / (1.0 + math.exp(-g))
            else:
                exponential = math.exp(g)
                sigmoid = exponential / (1.0 + exponential)
            activation[i] = g * sigmoid * u
        for j in range(hidden):
            value = np.float32(0)
            for i in range(intermediate):
                value += np.float32(down[j, i]) * activation[i]
            output[row, j] += weights[assignment] * (value * ds[j])


_compiled_loop = (
    _numba.njit(cache=True, nogil=True, fastmath=False)(_expert_loop)
    if _numba is not None and not _unavailable else None
)


class NativeExpertBackend:
    """Serial deterministic FP32 accumulation, with row-scaled INT8 weights.

    Float16 weights are promoted for Numba's array type support. INT8 weights
    remain INT8 throughout execution. No SIMD or speedup claim is made.
    """

    NAME = "native-cpu-numba"

    def info(self) -> KernelBackendInfo:
        available = _compiled_loop is not None
        return KernelBackendInfo(
            name=self.NAME, provider="numba", available=available, priority=100,
            capabilities=KernelCapabilities(
                SUPPORTED_PAYLOAD_DTYPES, batch_execution=True,
                deterministic=True, compiled=available, native_simd=False,
            ),
            evidence=(KernelEvidence(
                "implementation", "Lazy Numba nopython CPU loops; no fastmath; disk specialization cache",
            ),) if available else (),
            unavailable_reason="" if available else _unavailable,
        )

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype):
        _validate_configuration(source, hidden_size, intermediate_size, dtype)
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(
            source, vector[None, :], [routes], hidden_size=hidden_size,
            intermediate_size=intermediate_size, dtype=dtype,
        )[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype):
        if _compiled_loop is None:
            raise RuntimeError(_unavailable)
        hidden, intermediate, size = _validate_configuration(
            source, hidden_size, intermediate_size, dtype,
        )
        batch = np.ascontiguousarray(_finite_matrix(inputs, hidden))
        route_rows = list(route_batch)
        if len(route_rows) != batch.shape[0]:
            raise ValueError("one route list is required for each input row")
        groups = {}
        for row, routes in enumerate(route_rows):
            for key, weight in _validate_routes(routes, f"route_batch[{row}]"):
                groups.setdefault(key, []).append((row, weight))
        output = np.zeros(batch.shape, dtype=np.float32)
        for key, assignments in groups.items():
            with _lease(source, key) as payload:
                gate, up, down, gs, us, ds = _decode_payload(
                    payload, hidden, intermediate, dtype, size,
                )
                if dtype == "float16":
                    gate, up, down = (matrix.astype(np.float32) for matrix in (gate, up, down))
                if dtype != "int8-row":
                    gs = np.ones(intermediate, dtype=np.float32)
                    us = np.ones(intermediate, dtype=np.float32)
                    ds = np.ones(hidden, dtype=np.float32)
                rows = np.asarray([item[0] for item in assignments], dtype=np.intp)
                weights = np.asarray([item[1] for item in assignments], dtype=np.float32)
                _compiled_loop(batch, rows, weights, gate, up, down, gs, us, ds, output)
        return _finite_output(output)


__all__ = ["NativeExpertBackend"]
