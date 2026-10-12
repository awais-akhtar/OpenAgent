"""Explicit native tiled32 FP8 projection; no builds or dense decode fallback."""
import ctypes as C
import _ctypes
import os
import weakref
import threading

import numpy as np


class NativeFP8Projection:
    """Borrow exact uint8 arrays only for the duration of the caller's lease.

    max_bytes admits vector/output scratch, not provider-owned mapped storage.
    The caller supplies a library containing fp8_projection.c's versioned ABI.
    """

    def __init__(self, *, library, max_bytes=64 * 1024**2):
        if type(max_bytes) is not int or not 0 < max_bytes < 2**(8*C.sizeof(C.c_size_t)):
            raise ValueError('invalid FP8 projection scratch budget')
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        from ..native_library_identity import NativeLibraryIdentity
        self._library_identity=NativeLibraryIdentity.read(library)
        self.identity=self._library_identity.sha256+':native-v41-tiled32-fp8:'+str(max_bytes)
        self._lib = C.CDLL(os.fspath(self._library_identity.path))
        release = _ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose
        self._release = weakref.finalize(self, release, self._lib._handle)
        self._closed = False
        try:
            self._library_identity.assert_unchanged()
            version = self._lib.oa_v41_fp8_projection_version
            version.argtypes, version.restype = [], C.c_uint32
            if version() != 1:
                raise ValueError('unsupported FP8 projection ABI')
            self._matvec = self._lib.oa_v41_fp8_projection_matvec
            self._matvec.argtypes = [C.c_void_p, C.c_size_t]*2 + [C.c_size_t]*2 + [C.c_void_p, C.c_size_t]*2 + [C.c_size_t]
            self._matvec.restype = C.c_int
        except BaseException:
            self.close()
            raise

    def matvec(self, weights, scales, x, *, rows, cols, out=None):
        with self._lock:
            return self._project(weights,scales,x,rows=rows,cols=cols,out=out)

    def _project(self, weights, scales, x, *, rows, cols, out):
        if self._closed:
            raise RuntimeError('native FP8 projection closed')
        self._library_identity.assert_unchanged()
        limit = 2**(8*C.sizeof(C.c_size_t)) - 1
        if any(type(v) is not int or v <= 0 for v in (rows, cols)) or cols%32 or rows*cols > limit or 4*(cols+2*rows) > limit:
            raise ValueError('FP8 projection geometry exceeds native extent')
        if type(self.max_bytes) is not int or not 0<self.max_bytes<=limit:
            raise ValueError('invalid FP8 projection scratch budget')
        if 4*(cols+2*rows) > self.max_bytes:
            raise MemoryError('FP8 projection exceeds scratch admission')
        for array, dtype, shape in ((weights, np.uint8, (rows, cols)),
                                    (scales, np.uint8, ((rows+31)//32, (cols+31)//32)),
                                    (x, np.float32, (cols,))):
            if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous or not array.flags.aligned:
                raise ValueError('exact contiguous FP8 source and activation arrays required')
        if out is None:
            out = np.empty(rows, np.float32)
        if not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.shape != (rows,) or not out.flags.c_contiguous or not out.flags.aligned or not out.flags.writeable:
            raise ValueError('writable contiguous FP32 projection output required')
        status = self._matvec(weights.ctypes.data, weights.nbytes, scales.ctypes.data,
                              scales.nbytes, rows, cols, x.ctypes.data, x.nbytes,
                              out.ctypes.data, out.nbytes, self.max_bytes)
        if status in (4, 5):
            raise MemoryError('native FP8 projection budget or allocation failure')
        if status:
            raise ValueError(f'native FP8 projection failure {status}; output unchanged')
        return out

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._release()
