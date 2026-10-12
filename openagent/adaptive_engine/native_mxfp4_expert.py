"""Genuine optional Numba MXFP4 CPU kernel; no execution fallback or CUDA claim."""
from . import mxfp4_expert as _format
from .mxfp4_expert import DTYPE, MXFP4ExpertBackend, scratch_bytes as scratch_bytes
from .expert_kernels import KernelBackendInfo, KernelCapabilities, KernelEvidence

try:
    import numba as _numba
except Exception as error:
    _numba = None
    _unavailable = f"Numba unavailable: {type(error).__name__}: {error}"
else:
    _unavailable = "Numba JIT disabled" if _numba.config.DISABLE_JIT else ""

if _numba is not None and not _unavailable:
    for _helper in (_format._code,_format._scale,_format._size,_format._valid,_format._dot):
        _numba.extending.register_jitable(_helper)
    _compiled_loop = _numba.njit(nogil=True,fastmath=False,cache=False)(_format._loop)
else:
    _compiled_loop = None


class NativeMXFP4ExpertBackend(MXFP4ExpertBackend):
    NAME = "mxfp4-e2m1-g32-ue8m0-numba"

    def _kernel(self):
        if _compiled_loop is None:
            raise RuntimeError(_unavailable or "MXFP4 JIT unavailable")
        return _compiled_loop

    def info(self):
        available = _compiled_loop is not None
        return KernelBackendInfo(name=self.NAME,provider="numba",available=available,priority=100,
            capabilities=KernelCapabilities(frozenset({DTYPE}),True,True,compiled=available),
            evidence=(KernelEvidence("implementation","Direct E2M1/UE8M0 byte reductions on CPU"),),
            unavailable_reason="" if available else (_unavailable or "MXFP4 JIT unavailable"))
