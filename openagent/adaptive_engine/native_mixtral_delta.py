"""Explicit compiled C primitive loader; never builds or falls back silently."""
import ctypes
from pathlib import Path

import numpy as np


class CompiledGatedDeltaKernel:
    implementation = 'c-gated-delta-step'
    native_range_execution = False

    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.function = self.library.oa_gated_delta_step
        self.function.argtypes = [ctypes.c_size_t]*3 + [ctypes.c_void_p]*7
        self.function.restype = ctypes.c_int

    def step(self, q, k, v, beta, decay, state):
        arrays = [np.asarray(a) for a in (q, k, v, beta, decay, state)]
        q, k, v, beta, decay, state = arrays
        if state.ndim != 4 or not all(state.shape):
            raise ValueError('state must be [batch, heads, key, value]')
        b, h, kd, vd = state.shape
        shapes = [(b, h, kd), (b, h, kd), (b, h, vd), (b, h), (b, h), (b, h, kd, vd)]
        if any(a.dtype != np.dtype('float32') or a.shape != shape or not a.flags.c_contiguous
               for a, shape in zip(arrays, shapes)) or not state.flags.writeable:
            raise ValueError('kernel requires contiguous FP32 buffers with exact geometry: '
                             + str([(a.shape, str(a.dtype), a.flags.c_contiguous) for a in arrays]))
        output = np.empty((b, h, vd), np.float32)
        status = self.function(b*h, kd, vd, *(a.ctypes.data for a in arrays), output.ctypes.data)
        if status:
            raise ValueError(f'compiled gated delta status {status}')
        return output
