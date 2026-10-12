"""Explicit loader for caller-built Qwen4 hyper-connection math."""
import ctypes as C

import numpy as np


class CompiledQwen4HyperKernel:
    def __init__(self, library):
        self.library = C.CDLL(str(library))
        pointer = C.POINTER(C.c_float)
        self.function = self.library.oa_qwen4_hyper
        self.function.argtypes = [C.c_uint64]*4+[C.c_double]+[pointer]*7
        self.function.restype = C.c_int

    def mix(self, x, norm, down, up, inject, *, heads, epsilon):
        if type(heads) is not int or heads < 2 or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('invalid Qwen4 hyper geometry')
        arrays = [np.asarray(v) for v in (x, norm, down, up)]
        x, norm, down, up = arrays
        if x.ndim < 2 or x.shape[-1] % heads or not all(x.shape):
            raise ValueError('invalid hyper input')
        width = x.shape[-1]
        hidden, rank = width//heads, down.shape[0] if down.ndim == 2 else 0
        if not rank or norm.shape != (width,) or down.shape != (rank, width) or up.shape != (width, rank):
            raise ValueError('invalid hyper weights')
        if inject is not None:
            inject = np.asarray(inject)
            if inject.shape != (heads, width):
                raise ValueError('invalid injection weights')
            arrays.append(inject)
        if any(v.dtype != np.float32 or not v.flags.c_contiguous or not np.isfinite(v).all() for v in arrays):
            raise ValueError('expected finite contiguous FP32 hyper buffers')
        mixed = np.empty((*x.shape[:-1], hidden), np.float32)
        injection = np.empty((*x.shape[:-1], heads), np.float32) if inject is not None else None
        def ptr(value):
            return None if value is None else value.ctypes.data_as(C.POINTER(C.c_float))
        result = self.function(x.size//width, heads, hidden, rank, float(epsilon),
            *(ptr(value) for value in (x, norm, down, up, inject, mixed, injection)))
        if result:
            raise ValueError(f'compiled Qwen4 hyper execution failed ({result})')
        return mixed, injection
