"""Explicit-library dynamic FP32 operations; no automatic build or fallback."""
import ctypes as C
import _ctypes
from contextlib import contextmanager
from pathlib import Path
import threading
import weakref

import numpy as np


class NativeTensorOps:
    """Counted, staged CPU kernels with caller-owned input allocations."""
    implementation = 'c-dynamic-fp32-transformer-ops'

    def __init__(self, library, *, max_scratch_bytes=64 * 1024**2):
        if type(max_scratch_bytes) is not int or not 0 < max_scratch_bytes <= np.iinfo(np.intp).max:
            raise ValueError('positive native scratch budget required')
        self.max_scratch_bytes = max_scratch_bytes
        self._lock = threading.RLock()
        self._closed = False
        self._library = C.CDLL(str(Path(library).resolve(strict=True)))
        release = _ctypes.FreeLibrary if hasattr(_ctypes, 'FreeLibrary') else _ctypes.dlclose
        self._release = weakref.finalize(self, release, self._library._handle)
        version = self._library.oa_tensor_ops_version
        version.argtypes, version.restype = [], C.c_uint32
        if version() != 1:
            self.close()
            raise ValueError('unsupported dynamic tensor ABI')
        ptr, count, scalar = C.c_void_p, C.c_size_t, C.c_double
        signatures = {
            'norm': [ptr, count, ptr, count, count, count, scalar, ptr, count, count],
            'sigmoid': [ptr, count, ptr, count, count],
            'situ': [ptr, ptr, count, scalar, scalar, ptr, count, count],
            'attnres': [ptr, count, ptr, ptr, count, count, scalar, ptr, count, count],
            'rope': [ptr, count, ptr, count, count, count, count, scalar, ptr, count, count],
        }
        try:
            self._kernels = {}
            for name, signature in signatures.items():
                kernel = getattr(self._library, 'oa_tensor_' + name)
                kernel.argtypes, kernel.restype = signature, C.c_int
                self._kernels[name] = kernel
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _operation(self, required):
        with self._lock:
            if self._closed:
                raise RuntimeError('native tensor operations are closed')
            if type(self.max_scratch_bytes) is not int or not 0 < self.max_scratch_bytes <= np.iinfo(np.intp).max:
                raise ValueError('invalid native scratch budget')
            if required > self.max_scratch_bytes:
                raise MemoryError('native tensor scratch admission exceeded')
            yield

    @staticmethod
    def _array(value, ndim=None):
        if type(value) is not np.ndarray or value.dtype != np.float32 or not value.flags.c_contiguous or not value.size:
            raise ValueError('nonempty contiguous FP32 array required')
        if ndim is not None and value.ndim != ndim:
            raise ValueError('unexpected tensor rank')
        if value.ctypes.data % C.sizeof(C.c_float):
            raise ValueError('unaligned FP32 tensor')
        return value

    @staticmethod
    def _scalar(value):
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError('positive finite scalar required')
        return float(value)

    @staticmethod
    def _check(status):
        if status == 3:
            raise MemoryError('native staging admission or allocation failed')
        if status:
            raise ValueError(f'native tensor operation failed ({status})')

    def normalize(self, x, gain, *, epsilon=1e-6):
        x, gain = self._array(x), self._array(gain, 1)
        epsilon = self._scalar(epsilon)
        if x.ndim < 1 or x.shape[-1] != gain.size:
            raise ValueError('normalization gain dimension mismatch')
        with self._operation(x.nbytes):
            out = np.empty_like(x)
            self._check(self._kernels['norm'](x.ctypes.data, x.size, gain.ctypes.data,
                gain.size, x.size // gain.size, gain.size, epsilon,
                out.ctypes.data, out.size, self.max_scratch_bytes))
            return out

    def sigmoid(self, x):
        x = self._array(x)
        with self._operation(x.nbytes):
            out = np.empty_like(x)
            self._check(self._kernels['sigmoid'](x.ctypes.data, x.size,
                out.ctypes.data, out.size, self.max_scratch_bytes))
            return out

    def situ(self, gate, up, *, beta=4.0, linear_beta=25.0):
        gate, up = self._array(gate), self._array(up)
        beta, linear_beta = self._scalar(beta), self._scalar(linear_beta)
        if gate.shape != up.shape:
            raise ValueError('SiTU tensor shape mismatch')
        with self._operation(gate.nbytes):
            out = np.empty_like(gate)
            self._check(self._kernels['situ'](gate.ctypes.data, up.ctypes.data,
                gate.size, beta, linear_beta, out.ctypes.data, out.size, self.max_scratch_bytes))
            return out

    def attnres(self, values, gain, query, *, epsilon=1e-6):
        values = self._array(values, 2)
        gain, query = self._array(gain, 1), self._array(query, 1)
        epsilon = self._scalar(epsilon)
        if values.shape[1] != gain.size or query.size != gain.size:
            raise ValueError('attention residual shape mismatch')
        required = values.shape[0] * 8 + gain.nbytes
        with self._operation(required):
            out = np.empty_like(gain)
            self._check(self._kernels['attnres'](values.ctypes.data, values.size,
                gain.ctypes.data, query.ctypes.data, gain.size, values.shape[0],
                epsilon, out.ctypes.data, out.size, self.max_scratch_bytes))
            return out

    def rotate(self, x, positions, *, rotary, theta=10000.0):
        x = self._array(x, 2)
        theta = self._scalar(theta)
        if type(rotary) is not int or not 0 <= rotary <= x.shape[1] or rotary % 2:
            raise ValueError('invalid rotary dimension')
        if type(positions) is not np.ndarray or positions.dtype != np.uint64 or positions.shape != (x.shape[0],) or not positions.flags.c_contiguous:
            raise ValueError('one contiguous uint64 position per row required')
        if positions.ctypes.data % C.sizeof(C.c_uint64):
            raise ValueError('unaligned rotary positions')
        with self._operation(x.nbytes):
            out = np.empty_like(x)
            self._check(self._kernels['rope'](x.ctypes.data, x.size,
                positions.ctypes.data, positions.size, x.shape[0], x.shape[1], rotary,
                theta, out.ctypes.data, out.size, self.max_scratch_bytes))
            return out

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._release()
