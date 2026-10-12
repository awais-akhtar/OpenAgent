"""Bounded K3 hidden-state mixer with native projections and recurrent KDA.

This composes equation kernels; it is not a full model or checkpoint loader.
"""

import copy
import ctypes as C
import _ctypes
from dataclasses import dataclass
import hashlib
import threading
import weakref

import numpy as np

from .native_kimi import NativeKimiKDA


@dataclass(frozen=True)
class KimiK3KDAConfig:
    hidden: int = 12
    heads: int = 2
    key: int = 4
    value: int = 4
    rank: int = 5
    kernel: int = 4
    capacity: int = 128
    epsilon: float = 1e-6
    l2_epsilon: float = 1e-6
    query_scale: float = 1.0

    def __post_init__(self):
        for name, limit in (
            ("hidden", 64),
            ("heads", 8),
            ("key", 32),
            ("value", 32),
            ("rank", 64),
            ("kernel", 8),
            ("capacity", 4096),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError("unsupported K3 KDA geometry: " + name)
        for name in ("epsilon", "l2_epsilon", "query_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError("invalid K3 KDA epsilon")


class _Projection(C.Structure):
    _fields_ = [(name, C.c_int) for name in ("hidden", "heads", "key", "value", "rank")]


def kda_tensor_shapes(config):
    c = config
    kw, vw = c.heads * c.key, c.heads * c.value
    return {
        "q_proj.weight": (kw, c.hidden),
        "k_proj.weight": (kw, c.hidden),
        "v_proj.weight": (vw, c.hidden),
        "decay_down.weight": (c.rank, c.hidden),
        "decay_up.weight": (kw, c.rank),
        "decay_bias": (kw,),
        "A_log": (c.heads,),
        "beta_proj.weight": (c.heads, c.hidden),
        "gate_proj.weight": (vw, c.hidden),
        "out_proj.weight": (c.hidden, vw),
        "q_conv.weight": (c.heads, c.key, c.kernel),
        "k_conv.weight": (c.heads, c.key, c.kernel),
        "v_conv.weight": (c.heads, c.value, c.kernel),
        "out_norm.weight": (c.value,),
    }


class NativeKimiK3KDA:
    """Batch-one FP32 hidden-state KDA; lower-bound -5 and a full output gate."""

    implementation = "c-k3-lower-bounded-full-gate-kda"
    full_decoder = False

    def __init__(self, config, tensors, *, library, max_bytes=64 * 1024 * 1024):
        if not isinstance(config, KimiK3KDAConfig):
            raise ValueError("explicit KimiK3KDAConfig required")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("invalid memory budget")
        shapes = kda_tensor_shapes(config)
        if set(tensors) != set(shapes):
            raise ValueError("exact K3 KDA tensor inventory required")
        arrays = {}
        for name, shape in shapes.items():
            a = np.asarray(tensors[name])
            if (
                a.shape != shape
                or a.dtype != np.float32
                or not a.flags.c_contiguous
                or not np.isfinite(a).all()
            ):
                raise ValueError("finite contiguous FP32 tensor required: " + name)
            arrays[name] = a
        self._config, self.max_bytes = config, max_bytes
        c = config
        self._weights_bytes = sum(a.nbytes for a in arrays.values())
        self._cache_bytes = 4 * (
            c.heads * c.key * c.value + (c.kernel - 1) * c.heads * (2 * c.key + c.value)
        )
        self._admit(0)
        arrays = {
            name: np.frombuffer(a.tobytes(), dtype=np.float32).reshape(a.shape)
            for name, a in arrays.items()
        }
        projection_names = tuple(shapes)[:9]
        self._project_weights = np.frombuffer(
            b"".join(arrays[n].tobytes() for n in projection_names), dtype=np.float32
        )
        self._output_weights = arrays["out_proj.weight"]
        self._lock = threading.RLock()
        self._mixer = NativeKimiKDA(
            tuple(arrays[n + "_conv.weight"] for n in ("q", "k", "v")),
            arrays["out_norm.weight"],
            library=library,
            capacity=c.capacity,
            epsilon=c.epsilon,
            l2_epsilon=c.l2_epsilon,
            q_scale=c.query_scale,
            max_bytes=max_bytes,
        )
        self.library = self._mixer.library
        release = (
            _ctypes.FreeLibrary if hasattr(_ctypes, "FreeLibrary") else _ctypes.dlclose
        )
        weakref.finalize(self.library, release, self.library._handle)
        version = self.library.oa_k3_kda_projection_version
        version.argtypes, version.restype = [], C.c_int
        if version() != 1:
            raise ValueError("unsupported K3 KDA projection ABI")
        fp = C.POINTER(C.c_float)
        self._project = self.library.oa_k3_kda_project
        self._project.argtypes = [
            C.POINTER(_Projection),
            C.c_int,
            fp,
            C.c_size_t,
            fp,
            C.c_size_t,
            *([fp] * 6),
            C.POINTER(C.c_size_t),
        ]
        self._project.restype = C.c_int
        self._output = self.library.oa_k3_kda_output
        self._output.argtypes = [
            C.POINTER(_Projection),
            C.c_int,
            fp,
            C.c_size_t,
            fp,
            C.c_size_t,
            fp,
            C.c_size_t,
        ]
        self._output.restype = C.c_int
        self._abi = _Projection(c.hidden, c.heads, c.key, c.value, c.rank)
        identity = hashlib.sha256(repr(c).encode("ascii"))
        for name, a in arrays.items():
            identity.update(name.encode("ascii"))
            identity.update(a.tobytes())
        self._identity = identity.hexdigest()

    def _admit(self, n):
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("invalid memory budget")
        c = self.config
        kw, vw = c.heads * c.key, c.heads * c.value
        # Weight copies, live/staged caches, projected inputs, C staging and output.
        required = (
            3 * self._weights_bytes
            + 4 * self._cache_bytes
            + 4 * n * (4 * c.hidden + 12 * kw + 10 * vw + 4 * c.heads)
        )
        required += 8 * (2 * kw + vw + 2 * c.value + c.rank)
        if required > self.max_bytes:
            raise ValueError("K3 KDA transaction memory budget exceeded")

    @property
    def config(self):
        return self._config

    @property
    def position(self):
        with self._lock:
            return self._mixer.position

    def snapshot(self):
        with self._lock:
            return self._identity, self._mixer.snapshot()

    def restore(self, snapshot):
        with self._lock:
            if (
                type(snapshot) is not tuple
                or len(snapshot) != 2
                or snapshot[0] != self._identity
            ):
                raise ValueError("snapshot belongs to another K3 mixer")
            item = snapshot[1]
            if type(item) is not tuple or len(item) != 3:
                raise ValueError("invalid K3 KDA snapshot")
            position, state, history = item
            c = self.config
            if type(position) is not int or not 0 <= position <= c.capacity:
                raise ValueError("invalid K3 KDA snapshot position")
            shapes = (
                (c.heads, c.key, c.value),
                *((c.kernel - 1, c.heads, d) for d in (c.key, c.key, c.value)),
            )
            if type(history) is not tuple or len(history) != 3:
                raise ValueError("invalid convolution history")
            buffers = (state, *history)
            for a, shape in zip(buffers, shapes):
                if not isinstance(a, np.ndarray) or a.shape != shape:
                    raise ValueError("invalid K3 KDA snapshot shape")
                NativeKimiKDA._buffer(a)
            self._admit(0)
            buffers = tuple(a.copy() for a in buffers)
            self._mixer._state, self._mixer._history = buffers[0], buffers[1:]
            self._mixer._position = position

    def reset(self):
        with self._lock:
            self._admit(0)
            self._mixer.reset()

    def forward(self, hidden_states):
        with self._lock:
            c = self.config
            x = np.asarray(hidden_states)
            if x.ndim != 2 or x.shape[1] != c.hidden or not 1 <= x.shape[0] <= 128:
                raise ValueError("hidden states must be [1..128 tokens, hidden]")
            NativeKimiKDA._buffer(x)
            n = x.shape[0]
            if self.position + n > c.capacity:
                raise ValueError("K3 KDA capacity exceeded")
            self._admit(n)
            x = x.copy()
            shapes = [(n, c.heads, c.key)] * 2 + [
                (n, c.heads, c.value),
                (n, c.heads, c.key),
                (n, c.heads),
                (n, c.heads, c.value),
            ]
            arrays = tuple(np.empty(shape, np.float32) for shape in shapes)
            fp = C.POINTER(C.c_float)

            def ptr(a):
                return a.ctypes.data_as(fp)

            counts = (C.c_size_t * 6)(*(a.size for a in arrays))
            status = self._project(
                C.byref(self._abi),
                n,
                ptr(self._project_weights),
                self._project_weights.size,
                ptr(x),
                x.size,
                *(ptr(a) for a in arrays),
                counts,
            )
            if status:
                raise ValueError(
                    f"K3 KDA projection failed ({status}); state unchanged"
                )
            candidate = copy.copy(self._mixer)
            # The existing kernel stages all cache writes; only this candidate commits.
            mixed = candidate.forward(*arrays)
            result = np.empty((n, c.hidden), np.float32)
            status = self._output(
                C.byref(self._abi),
                n,
                ptr(self._output_weights),
                self._output_weights.size,
                ptr(mixed),
                mixed.size,
                ptr(result),
                result.size,
            )
            if status:
                raise ValueError(f"K3 KDA output failed ({status}); state unchanged")
            self._mixer._state = candidate._state
            self._mixer._history = candidate._history
            self._mixer._position = candidate._position
            return result
