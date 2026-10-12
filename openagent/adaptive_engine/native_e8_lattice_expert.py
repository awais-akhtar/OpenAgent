"""Optional compiled caller-codebook E8 CPU execution, not IQ3 parity."""

from . import e8_lattice_expert as format
from .expert_kernels import KernelBackendInfo, KernelCapabilities, KernelEvidence

_compiled = None
_reason = ""
try:
    import numba
    if numba.config.DISABLE_JIT:
        _reason = "Numba JIT disabled"
    else:
        numba.extending.register_jitable(format._read_f32)
        numba.extending.register_jitable(format._rotated_dot)
        _compiled = numba.njit(cache=False, nogil=True, fastmath=False)(format._kernel)
except Exception as error:
    _reason = f"Numba unavailable: {type(error).__name__}: {error}"


class NativeE8LatticeExpertBackend(format.E8LatticeExpertBackend):
    def __init__(self, contract, **kwargs):
        super().__init__(contract, **kwargs)
        self.NAME = format.DTYPE + "-" + contract.identity + "-numba"

    def _executor(self):
        if _compiled is None:
            raise RuntimeError(_reason or "E8 JIT unavailable")
        return _compiled

    def info(self):
        ready = _compiled is not None
        return KernelBackendInfo(name=self.NAME, provider="numba", available=ready, priority=100,
            capabilities=KernelCapabilities(frozenset({format.DTYPE}), True, True, compiled=ready),
            evidence=(KernelEvidence("contract", self.contract.identity),),
            unavailable_reason="" if ready else (_reason or "E8 JIT unavailable"))
