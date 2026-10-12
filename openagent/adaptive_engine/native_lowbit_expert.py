"""Optional Numba CPU kernels for OpenAgent fixed-geometry low-bit formats."""

from . import lowbit_expert as format
from .expert_kernels import KernelBackendInfo, KernelCapabilities, KernelEvidence

_compiled = None
_reason = ""
try:
    import numba
    if numba.config.DISABLE_JIT:
        _reason = "Numba JIT disabled"
    else:
        numba.extending.register_jitable(format._code)
        numba.extending.register_jitable(format._read_f32)
        numba.extending.register_jitable(format._dot)
        _compiled = numba.njit(cache=False, nogil=True, fastmath=False)(format._kernel)
except Exception as error:
    _reason = f"Numba unavailable: {type(error).__name__}: {error}"


class NativeLowBitExpertBackend(format.LowBitExpertBackend):
    def __init__(self, dtype=format.INT2, **kwargs):
        super().__init__(dtype, **kwargs)
        self.NAME = dtype + "-numba"

    def info(self):
        ready = _compiled is not None
        return KernelBackendInfo(name=self.NAME, provider="numba", available=ready, priority=100,
            capabilities=KernelCapabilities(frozenset({self.dtype}), True, True, compiled=ready),
            evidence=(KernelEvidence("implementation", "Direct low-bit Numba CPU reductions"),),
            unavailable_reason="" if ready else (_reason or "low-bit JIT unavailable"))

    def _check_available(self):
        if _compiled is None:
            raise RuntimeError(_reason or "low-bit JIT unavailable")

    def _run(self, raw, *arguments):
        return _compiled(raw, *arguments)
