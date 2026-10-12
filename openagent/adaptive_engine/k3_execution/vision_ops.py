"""Explicit-library native vision GELU and complex-pair rotary operators."""
import ctypes as C

import numpy as np


class VisionOps:
    def __init__(self, library, max_bytes):
        self.library, self.max_bytes = library, max_bytes
        fp = C.POINTER(C.c_float)
        self._gelu = library.oa_k3_vision_gelu
        self._gelu.argtypes, self._gelu.restype = [fp, C.c_uint64, C.c_int, fp, C.c_uint64], C.c_int
        self._rope = library.oa_k3_vision_rope
        self._rope.argtypes, self._rope.restype = [fp, fp, C.c_uint64, C.c_uint64, C.c_uint64, fp, C.c_uint64], C.c_int

    @staticmethod
    def _ptr(a):
        return a.ctypes.data_as(C.POINTER(C.c_float))

    @staticmethod
    def _input(a):
        if type(a) is not np.ndarray or a.dtype != np.float32 or not a.flags.c_contiguous or not a.size or not np.isfinite(a).all():
            raise ValueError("finite contiguous vision FP32 input required")

    def gelu(self, x, approximate):
        self._input(x)
        if x.nbytes * 2 > self.max_bytes:
            raise ValueError("vision GELU budget exceeded")
        out = np.empty_like(x)
        if self._gelu(self._ptr(x), x.size, int(approximate), self._ptr(out), self.max_bytes):
            raise ValueError("native vision GELU failed")
        return out

    def rotate(self, x, phase):
        self._input(x)
        self._input(phase)
        if x.ndim != 3 or phase.shape != (x.shape[0], x.shape[2] // 2) or x.nbytes * 2 > self.max_bytes:
            raise ValueError("invalid vision rotary shape/budget")
        out = np.empty_like(x)
        if self._rope(self._ptr(x), self._ptr(phase), *x.shape, self._ptr(out), self.max_bytes):
            raise ValueError("native vision rotary failed")
        return out
