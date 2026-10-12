"""Explicit-library local FP4 codec. No auto build, fallback, or decoder registration."""

import ctypes as C
import operator
import os
from pathlib import Path
from itertools import islice
import weakref
import _ctypes
from functools import wraps
import threading

PRIVATE_ABI_VERSION = 1


def _serialized(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return guarded


def _operation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            self._ensure_open()
            self._active += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                self._active -= 1
    return guarded


def _release_library(handle):
    if os.name == "nt":
        _ctypes.FreeLibrary(handle)
    else:
        _ctypes.dlclose(handle)


def _bounded_bytes(value, count, label):
    # Check buffer byte extent before copying; do not invoke arbitrary __bytes__.
    try:
        view = memoryview(value)
    except TypeError:
        items = tuple(islice(iter(value), count + 1))
        if len(items) != count:
            raise ValueError(label + " count must match geometry")
        return bytes(items)
    try:
        if view.nbytes != count:
            raise ValueError(label + " count must match geometry")
        return view.tobytes()
    finally:
        view.release()


class NativeDeepSeekV41Quant:
    """Caller supplies an independently compiled library; see local contract."""

    def __init__(self, library_path):
        self._lock = threading.RLock()
        self._active = 0
        path = Path(library_path).resolve(strict=True)
        if os.name == "nt" and path.drive.startswith("\\\\"):
            raise ValueError("network library paths are not accepted")
        if not path.is_file():
            raise ValueError("library path must resolve to a local regular file")
        self.library_path = path
        self.library = C.CDLL(str(path))
        self._finalizer = weakref.finalize(self, _release_library, self.library._handle)
        try:
            self._bind()
        except BaseException:
            self.close()
            raise

    def _bind(self):
        try:
            version = self.library.oa_v41_quant_abi_version
        except AttributeError as error:
            raise ValueError("missing private codec ABI handshake") from error
        version.argtypes = []
        version.restype = C.c_uint32
        if version() != PRIVATE_ABI_VERSION:
            raise ValueError("unsupported private codec ABI version")
        pairs = [C.c_void_p, C.c_size_t]
        self.library.oa_v41_quant_scales.argtypes = (
            pairs + [C.c_uint32] * 2 + pairs + [C.c_size_t]
        )
        self.library.oa_v41_quant_encode.argtypes = (
            pairs * 2 + [C.c_uint32] * 2 + pairs + [C.c_size_t]
        )
        self.library.oa_v41_quant_decode.argtypes = (
            pairs + [C.c_uint32] * 2 + pairs + [C.c_size_t]
        )
        for name in ("scales", "encode", "decode"):
            getattr(self.library, "oa_v41_quant_" + name).restype = C.c_int

    @_serialized
    def close(self):
        """Release exactly once; callers must not retain/use raw native functions."""
        if self._active:
            raise RuntimeError('cannot close codec from an active operation')
        self._finalizer()

    def _ensure_open(self):
        if not self._finalizer.alive:
            raise RuntimeError("codec library is closed")

    @staticmethod
    def _shape(rows, channels, budget):
        if any(isinstance(value, bool) for value in (rows, channels, budget)):
            raise ValueError("boolean geometry or budget is not accepted")
        rows, channels, budget = map(operator.index, (rows, channels, budget))
        if not 1 <= rows <= 32 or not 16 <= channels <= 512 or channels % 16:
            raise ValueError("rows 1..32; channels 16..512, divisible by 16")
        if not 0 <= budget <= C.c_size_t(-1).value:
            raise ValueError("invalid scratch budget")
        return rows, channels, budget

    @staticmethod
    def _check(status):
        if status:
            raise ValueError(
                {
                    1: "invalid extent or dimensions",
                    2: "scratch budget exceeded",
                    3: "nonfinite input or invalid scale",
                }.get(status, "native failure")
            )

    @_operation
    def encode(self, values, rows, channels, *, scales=None, budget=65536):
        self._ensure_open()
        rows, channels, budget = self._shape(rows, channels, budget)
        n = rows * channels
        values = tuple(islice(iter(values), n + 1))
        if len(values) != n:
            raise ValueError("input count must match geometry")
        x = (C.c_float * n)(*values)
        b = n // 16
        if scales is None:
            s = (C.c_uint8 * b)()
            self._check(
                self.library.oa_v41_quant_scales(x, n, rows, channels, s, b, budget)
            )
        else:
            scales = _bounded_bytes(scales, b, "scale")
            s = (C.c_uint8 * b).from_buffer_copy(scales)
        out = (C.c_uint8 * (b * 9))()
        self._check(
            self.library.oa_v41_quant_encode(
                x, n, s, b, rows, channels, out, b * 9, budget
            )
        )
        return bytes(out)

    @_operation
    def decode(self, packed, rows, channels, *, budget=65536):
        self._ensure_open()
        rows, channels, budget = self._shape(rows, channels, budget)
        n = rows * channels
        packed = _bounded_bytes(packed, n // 16 * 9, "packed")
        src = (C.c_uint8 * len(packed)).from_buffer_copy(packed)
        out = (C.c_float * n)()
        self._check(
            self.library.oa_v41_quant_decode(
                src, len(packed), rows, channels, out, n, budget
            )
        )
        return tuple(out)
