"""Explicit native C decoding of dense rowwise E2M1/E8M0 scale32 weights."""

import ctypes as C
import _ctypes
import os
import weakref

import numpy as np


def decode_bytes(rows, cols):
    """Total admitted bytes: packed input, scales, and FP32 output; no scratch."""
    if any(type(v) is not int or v <= 0 for v in (rows, cols)) or cols % 32:
        raise ValueError("positive rows and columns divisible by 32 required")
    n = rows * cols
    if n > (2 ** (8 * C.sizeof(C.c_size_t)) - 1) // 4:
        raise ValueError("geometry exceeds native size_t")
    return n // 2 + n // 32 + n * 4


class NativeMXFP4:
    """Caller supplies a compiled library; no JIT or Python execution fallback."""

    implementation = "c-rowwise-mxfp4-e2m1-e8m0-scale32"
    full_checkpoint = False

    def __init__(self, *, library, max_bytes=64 * 1024 * 1024):
        if type(max_bytes) is not int or not 0 < max_bytes < 2 ** (
            8 * C.sizeof(C.c_size_t)
        ):
            raise ValueError("invalid memory budget")
        self.max_bytes = max_bytes
        from .native_library_identity import NativeLibraryIdentity
        self._library_identity = NativeLibraryIdentity.read(library)
        self.identity = self._library_identity.sha256 + ':native-rowwise-mxfp4-v1:' + str(max_bytes)
        self.library = C.CDLL(os.fspath(self._library_identity.path))
        release = _ctypes.FreeLibrary if os.name == "nt" else _ctypes.dlclose
        self._release = weakref.finalize(self, release, self.library._handle)
        try:
            self._bind_native()
            self._library_identity.assert_unchanged()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.cleanup = self.close
                error.add_note('native MXFP4 constructor cleanup retained: '+str(cleanup))
            raise

    def _bind_native(self):
        version = self.library.oa_mxfp4_version
        version.argtypes, version.restype = [], C.c_int
        if version() != 1:
            raise ValueError("unsupported MXFP4 ABI")
        self._decode = self.library.oa_mxfp4_decode
        self._decode.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_size_t,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_size_t,
        ]
        self._decode.restype = C.c_int
        self._matvec = self.library.oa_mxfp4_matvec
        self._matvec.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                                C.c_size_t, C.c_size_t, C.c_void_p, C.c_size_t,
                                C.c_void_p, C.c_size_t, C.c_size_t]
        self._matvec.restype = C.c_int

    def close(self):
        """Release the owned library; retain its handle when release fails."""
        if self._release.alive:
            _, callback, args, kwargs = self._release.detach()
            try:
                callback(*args, **kwargs)
            except BaseException:
                self._release = weakref.finalize(self, callback, *args, **kwargs)
                raise
        self.library = None
        self._decode = self._matvec = None

    def matvec(self, packed, scales, x, *, rows, cols, out=None):
        """Execute packed weights directly with FP64 accumulation and FP32 output."""
        if self.library is None:
            raise RuntimeError('native MXFP4 owner is closed')
        self._library_identity.assert_unchanged()
        decode_bytes(rows, cols)
        if type(self.max_bytes) is not int or not 0 < self.max_bytes < 2 ** (8 * C.sizeof(C.c_size_t)):
            raise ValueError("invalid memory budget")
        required = rows * cols // 2 + rows * cols // 32 + 4 * cols + 8 * rows
        if required > self.max_bytes:
            raise MemoryError("MXFP4 matvec exceeds memory budget")
        for a, dtype, shape in ((packed, np.uint8, (rows, cols // 2)),
                                (scales, np.uint8, (rows, cols // 32)),
                                (x, np.float32, (cols,))):
            if type(a) is not np.ndarray or a.dtype != dtype or a.shape != shape or not a.flags.c_contiguous:
                raise ValueError("exact contiguous matvec arrays required")
        if out is None:
            out = np.empty(rows, np.float32)
        if type(out) is not np.ndarray or out.dtype != np.float32 or out.shape != (rows,) or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("writable contiguous FP32 matvec output required")
        status = self._matvec(packed.ctypes.data, packed.nbytes, scales.ctypes.data,
                              scales.nbytes, rows, cols, x.ctypes.data, x.nbytes,
                              out.ctypes.data, out.nbytes, self.max_bytes)
        if status in (4, 5):
            raise MemoryError("MXFP4 matvec budget or allocation failure")
        if status:
            raise ValueError(f"native MXFP4 matvec failure {status}; output unchanged")
        return out

    def decode(self, packed, scales, *, rows, cols, out=None):
        if self.library is None:
            raise RuntimeError('native MXFP4 owner is closed')
        self._library_identity.assert_unchanged()
        if type(self.max_bytes) is not int or not 0 < self.max_bytes < 2 ** (8 * C.sizeof(C.c_size_t)):
            raise ValueError("invalid memory budget")
        required = decode_bytes(rows, cols)
        if required > self.max_bytes:
            raise MemoryError("MXFP4 decode exceeds memory budget")
        for a, shape in ((packed, (rows, cols // 2)), (scales, (rows, cols // 32))):
            if (
                type(a) is not np.ndarray
                or a.dtype != np.uint8
                or a.shape != shape
                or not a.flags.c_contiguous
            ):
                raise ValueError("exact contiguous uint8 packed/scales arrays required")
        if out is None:
            out = np.empty((rows, cols), dtype=np.float32)
        if (
            type(out) is not np.ndarray
            or out.dtype != np.float32
            or out.shape != (rows, cols)
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            raise ValueError("writable contiguous FP32 output required")
        status = self._decode(
            packed.ctypes.data,
            packed.nbytes,
            scales.ctypes.data,
            scales.nbytes,
            rows,
            cols,
            out.ctypes.data,
            out.nbytes,
            self.max_bytes,
        )
        if status == 4:
            raise MemoryError("MXFP4 native budget exceeded")
        if status:
            raise ValueError(
                {
                    1: "invalid native contract",
                    2: "output aliases input",
                    3: "nonfinite MXFP4 scale or decoded weight",
                }.get(status, "native failure")
            )
        return out
