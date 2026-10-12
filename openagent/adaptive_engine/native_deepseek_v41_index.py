"""Explicit-library bounded index equation fixture; no decoder registration."""
import ctypes as C
import operator
import os
from pathlib import Path
from itertools import islice
import weakref
import _ctypes
from functools import wraps
import threading


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
            self._check()
            self._active += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                self._active -= 1
    return guarded


def _release(handle):
    if os.name == "nt":
        _ctypes.FreeLibrary(handle)
    else:
        _ctypes.dlclose(handle)


def _integer(value, low, high):
    if isinstance(value, bool):
        raise ValueError("boolean dimensions are not accepted")
    value = operator.index(value)
    if not low <= value <= high:
        raise ValueError("integer outside bounded contract")
    return value


def _array(values, count, kind):
    values = tuple(islice(iter(values), count + 1))
    if len(values) != count:
        raise ValueError("count must match geometry")
    if kind is not C.c_float:
        values = tuple(_integer(v, 0, 32 if kind is C.c_int32 else 1) for v in values)
    return (kind * count)(*values)


class NativeDeepSeekV41Index:
    """Local mathematical fixture; caller trusts an independently built library."""

    def __init__(self, library_path):
        self._lock = threading.RLock()
        self._active = 0
        path = Path(library_path).resolve(strict=True)
        if not path.is_file() or (os.name == "nt" and path.drive.startswith("\\\\")):
            raise ValueError("library must be a local regular file")
        self.library = C.CDLL(str(path))
        self._finalizer = weakref.finalize(self, _release, self.library._handle)
        try:
            version = self.library.oa_v41_index_abi_version
            version.argtypes, version.restype = [], C.c_uint32
            if version() != 1:
                raise ValueError("unsupported private index ABI")
            pair = [C.c_void_p, C.c_size_t]
            self.library.oa_v41_index_score.argtypes = pair * 3 + [C.c_uint32] * 3 + pair + [C.c_size_t]
            self.library.oa_v41_index_select.argtypes = pair * 2 + [C.c_uint32] * 2 + [C.c_int] + pair + [C.c_uint32] + pair + [C.c_void_p, C.c_size_t]
            self.library.oa_v41_index_score.restype = C.c_int
            self.library.oa_v41_index_select.restype = C.c_int
        except BaseException:
            self.close()
            raise

    @_serialized
    def close(self):
        """Wait for foreign operations; refuse close from an active callback."""
        if self._active:
            raise RuntimeError("cannot close index from an active operation")
        self._finalizer()

    def _check(self, status=0):
        if not self._finalizer.alive:
            raise RuntimeError("index library is closed")
        if status:
            raise ValueError({1: "invalid geometry, mask or candidates", 2: "staging budget exceeded",
                              3: "nonfinite score or input", 4: "output capacity exceeded"}.get(status, "native failure"))

    @_operation
    def score(self, query, keys, weights, rows, heads, width, *, budget=4096):
        self._check()
        rows = _integer(rows, 1, 32)
        heads, width = (_integer(v, 1, 8) for v in (heads, width))
        budget = _integer(budget, 0, C.c_size_t(-1).value)
        q = _array(query, heads * width, C.c_float)
        k = _array(keys, rows * heads * width, C.c_float)
        w = _array(weights, heads, C.c_float)
        out = (C.c_float * rows)()
        self._check()
        self._check(self.library.oa_v41_index_score(q, len(q), k, len(k), w, len(w),
                                                  rows, heads, width, out, rows, budget))
        return tuple(out)

    @_operation
    def select(self, scores, admitted, rows, block_size, limit, *, candidates=None,
               capacity=32, budget=4096):
        self._check()
        rows, block_size, limit = (_integer(v, 1, 32) for v in (rows, block_size, limit))
        capacity = _integer(capacity, 0, 32)
        budget = _integer(budget, 0, C.c_size_t(-1).value)
        s, a = _array(scores, rows, C.c_float), _array(admitted, rows, C.c_uint8)
        c = None
        if candidates is not None:
            nb = (rows + block_size - 1) // block_size
            values = tuple(islice(iter(candidates), nb + 1))
            if len(values) > nb:
                raise ValueError("too many candidates")
            c = _array(values, len(values), C.c_int32) if values else None
        out, count = (C.c_int32 * capacity)(), C.c_uint32()
        self._check()
        self._check(self.library.oa_v41_index_select(s, rows, a, rows, rows, block_size,
                    int(candidates is not None), c, len(c) if c is not None else 0,
                    limit, out, capacity, C.byref(count), budget))
        return tuple(out[:count.value])
