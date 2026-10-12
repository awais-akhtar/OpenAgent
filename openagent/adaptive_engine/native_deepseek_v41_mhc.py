"""Private, explicit-library single-pass mHC equation fixture (not a decoder)."""
import ctypes as C
import _ctypes
from functools import wraps
from itertools import islice
import operator
import os
from pathlib import Path
import threading
import weakref


def _integer(value, low, high):
    if isinstance(value, bool):
        raise ValueError("boolean dimensions are not accepted")
    value = operator.index(value)
    if not low <= value <= high:
        raise ValueError("integer outside bounded contract")
    return value


def _array(values, count):
    values = tuple(islice(iter(values), count + 1))
    if len(values) != count:
        raise ValueError("tensor extent does not match geometry")
    return (C.c_float * count)(*values)


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            self._check()
            self._active += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                self._active -= 1
    return call


def _release(handle):
    if os.name == "nt":
        _ctypes.FreeLibrary(handle)
    else:
        _ctypes.dlclose(handle)


class _State(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("magic", "rows", "streams", "width", "depth", "capacity")] + [
                    ("x", C.c_float * 512), ("previous", C.c_float * 32)]


class NativeDeepSeekV41MHC:
    """One fixed token-row batch carried across block depth, never across tokens.

    ``predict`` and ``mix`` are read-only. ``commit`` accepts a caller's F output
    with an explicit depth guard. ``run`` uses an affine F fixture entirely in C.
    The library must be independently built and trusted by the caller.
    """

    def __init__(self, library_path, rows, streams, width, x, initial_mix, *, capacity=64):
        self._lock = threading.RLock()
        self._closed = False
        self._active = 0
        self.rows = _integer(rows, 1, 8)
        self.streams = _integer(streams, 1, 4)
        self.width = _integer(width, 1, 16)
        self.capacity = _integer(capacity, 1, 64)
        path = Path(library_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("library must be a regular file")
        self.library = C.CDLL(str(path))
        self._finalizer = weakref.finalize(self, _release, self.library._handle)
        pair = [C.c_void_p, C.c_size_t]
        signatures = {
            "abi_version": ([], C.c_uint32),
            "state_bytes": ([], C.c_size_t),
            "scratch_bytes": ([], C.c_size_t),
            "validate": (pair, C.c_int),
            "init": (pair + [C.c_uint32]*4 + pair*2, C.c_int),
            "mix": (pair*2, C.c_int),
            "predict": (pair*3 + [C.c_double, C.c_uint32] + pair, C.c_int),
            "commit": (pair + [C.c_uint32] + pair*2, C.c_int),
            "run": (pair + [C.c_uint32]*2 + pair*4 +
                    [C.c_double, C.c_uint32, C.c_size_t], C.c_int),
            "head": (pair*5 + [C.c_uint32, C.c_double] + pair, C.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.library, "oa_v41_mhc_" + name)
            function.argtypes, function.restype = args, result
        if self.library.oa_v41_mhc_abi_version() != 1 or self.library.oa_v41_mhc_state_bytes() != C.sizeof(_State):
            raise ValueError("unsupported private mHC ABI")
        self._state = _State()
        xa = _array(x, self.rows*self.streams*self.width)
        aa = _array(initial_mix, self.rows*self.streams)
        self._check(self.library.oa_v41_mhc_init(C.byref(self._state), C.sizeof(_State),
                    self.rows, self.streams, self.width, self.capacity, xa, len(xa), aa, len(aa)))
        self._initial = bytes(self._state)

    def _check(self, status=0):
        if self._closed:
            raise RuntimeError("mHC component is closed")
        if status:
            raise ValueError({1: "invalid geometry, buffer, coefficient or depth",
                              2: "staging budget exceeded", 3: "nonfinite tensor or arithmetic overflow",
                              4: "depth or output capacity exceeded"}.get(status, "native failure"))

    def close(self):
        with self._lock:
            if self._active:
                raise RuntimeError("cannot close from an active mHC operation")
            self._closed = True
            self._finalizer()

    @_locked
    def snapshot(self):
        """Canonical private-ABI bytes; includes delayed coefficients and depth."""
        return bytes(self._state)

    @_locked
    def restore(self, snapshot):
        if self._active > 1:
            raise RuntimeError("reentrant state mutation is not supported")
        if not isinstance(snapshot, bytes) or len(snapshot) != C.sizeof(_State):
            raise ValueError("invalid snapshot size or type")
        candidate = _State.from_buffer_copy(snapshot)
        self._check(self.library.oa_v41_mhc_validate(C.byref(candidate), C.sizeof(candidate)))
        if (candidate.rows, candidate.streams, candidate.width, candidate.capacity) != (
                self.rows, self.streams, self.width, self.capacity):
            raise ValueError("snapshot geometry differs")
        C.memmove(C.byref(self._state), C.byref(candidate), C.sizeof(candidate))

    @_locked
    def reset(self):
        if self._active > 1:
            raise RuntimeError("reentrant state mutation is not supported")
        C.memmove(C.byref(self._state), self._initial, len(self._initial))

    @property
    @_locked
    def depth(self):
        return self._state.depth

    @property
    @_locked
    def residual(self):
        return tuple(self._state.x[:self.rows*self.streams*self.width])

    @property
    @_locked
    def previous_mix(self):
        return tuple(self._state.previous[:self.rows*self.streams])

    @_locked
    def mix(self):
        out = (C.c_float * (self.rows*self.width))()
        self._check(self.library.oa_v41_mhc_mix(C.byref(self._state), C.sizeof(_State), out, len(out)))
        return tuple(out)

    @_locked
    def predict(self, weight, bias, *, epsilon=1e-6, sweeps=8):
        q = self.streams*(self.streams+2)
        weight = _array(weight, q*self.streams*self.width)
        bias = _array(bias, q)
        out = (C.c_float * (self.rows*q))()
        sweeps = _integer(sweeps, 1, 32)
        self._check(self.library.oa_v41_mhc_predict(C.byref(self._state), C.sizeof(_State),
                    weight, len(weight), bias, len(bias), epsilon, sweeps, out, len(out)))
        return tuple(out)

    @_locked
    def commit(self, coefficients, block_output, *, expected_depth):
        if self._active > 1:
            raise RuntimeError("reentrant state mutation is not supported")
        expected_depth = _integer(expected_depth, 0, 64)
        coefficients = _array(coefficients, self.rows*self.streams*(self.streams+2))
        block_output = _array(block_output, self.rows*self.width)
        self._check(self.library.oa_v41_mhc_commit(C.byref(self._state), C.sizeof(_State),
                    expected_depth, coefficients, len(coefficients), block_output, len(block_output)))
        return self.residual

    @_locked
    def run(self, blocks, h_weight, h_bias, f_weight, f_bias, *, epsilon=1e-6,
            sweeps=8, budget=65536):
        """Atomic depth chunk; H and affine F tensors are depth-major."""
        if self._active > 1:
            raise RuntimeError("reentrant state mutation is not supported")
        depth = self._state.depth
        blocks = _integer(blocks, 1, 16)
        q, w = self.streams*(self.streams+2), self.width
        hw = _array(h_weight, blocks*q*self.streams*w)
        hb = _array(h_bias, blocks*q)
        fw = _array(f_weight, blocks*w*w)
        fb = _array(f_bias, blocks*w)
        self._check(self.library.oa_v41_mhc_run(C.byref(self._state), C.sizeof(_State), depth,
                    blocks, hw, len(hw), hb, len(hb), fw, len(fw), fb, len(fb), epsilon,
                    _integer(sweeps, 1, 32), _integer(budget, 0, C.c_size_t(-1).value)))
        return self.residual

    @_locked
    def head(self, final_mix, norm, weight, bias, vocab, *, epsilon=1e-6, capacity=None):
        """Separate final stream mix -> weighted RMSNorm -> linear logits."""
        vocab = _integer(vocab, 1, 32)
        capacity = self.rows*vocab if capacity is None else _integer(capacity, 0, 256)
        a = _array(final_mix, self.rows*self.streams)
        norm = _array(norm, self.width)
        weight = _array(weight, vocab*self.width)
        bias = _array(bias, vocab)
        out = (C.c_float * capacity)()
        self._check(self.library.oa_v41_mhc_head(C.byref(self._state), C.sizeof(_State), a, len(a),
                    norm, len(norm), weight, len(weight), bias, len(bias), vocab, epsilon, out, capacity))
        return tuple(out[:self.rows*vocab])
