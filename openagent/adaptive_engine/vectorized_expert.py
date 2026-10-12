"""Optional reassociated CPU expert kernels; register explicitly.

FP32 reductions may differ numerically from serial and NumPy reductions.
Only reassociation and contraction are enabled, never nnan/ninf or full
fastmath. Finite operands do not guarantee finite intermediates: the shared
output check remains mandatory. Payloads are borrowed within their leases.
"""

from __future__ import annotations

import numpy as np

from . import native_expert as _native
from .expert_kernels import (
    KernelBackendInfo, KernelCapabilities, KernelEvidence,
    SUPPORTED_PAYLOAD_DTYPES, _decode_payload, _finite_matrix, _finite_output,
    _lease, _validate_configuration, _validate_routes,
)


# Keep machine code inspectable: disk-loaded Numba specializations cannot
# reliably supply LLVM/assembly for an evidence audit.
_compiled_loop = (
    _native._numba.njit(nogil=True, fastmath={"reassoc", "contract"})(_native._expert_loop)
    if _native._numba is not None and not _native._unavailable else None
)
_unavailable = _native._unavailable


class VectorizedExpertBackend(_native.NativeExpertBackend):
    """Compiler-vectorizable FP32 reductions, with no resident weight cache.

    Numerical class: reassociated/contracted FP32, non-bit-exact relative to
    native serial or NumPy. Actual SIMD depends on target and specialization;
    no blanket SIMD capability is advertised. Float16 is promoted to FP32.
    """

    NAME = "vectorized-cpu-numba"
    NUMERICAL_CLASS = "reassociated-contracted-fp32-non-bit-exact"

    def info(self) -> KernelBackendInfo:
        available = _compiled_loop is not None
        return KernelBackendInfo(
            name=self.NAME, provider="numba", available=available, priority=100,
            capabilities=KernelCapabilities(
                SUPPORTED_PAYLOAD_DTYPES, batch_execution=True,
                deterministic=True, compiled=available, native_simd=False,
            ),
            evidence=(
                KernelEvidence("implementation", "Lazy nopython CPU loop; only reassoc and contract; no disk cache"),
                KernelEvidence("numerical-class", self.NUMERICAL_CLASS),
            ) if available else (),
            unavailable_reason="" if available else _unavailable,
        )

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


__all__ = ["VectorizedExpertBackend"]
