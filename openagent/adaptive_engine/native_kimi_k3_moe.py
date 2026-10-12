"""Original bounded FP32 inference of report-defined K3 normalized LatentMoE.

This is an explicit module fixture contract, not a checkpoint loader.
"""

import ctypes as C
import _ctypes
from dataclasses import dataclass, fields
import threading
import weakref

import numpy as np


@dataclass(frozen=True)
class KimiK3MoEConfig:
    hidden: int = 12
    latent: int = 6
    intermediate: int = 9
    experts: int = 5
    selected: int = 2
    shared: int = 2
    epsilon: float = 1e-6
    beta: float = 4.0
    linear_beta: float = 25.0

    def __post_init__(self):
        for name, upper in zip(
            ("hidden", "latent", "intermediate", "experts", "selected", "shared"),
            (64, 64, 128, 64, 64, 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("unsupported K3 MoE geometry: " + name)
        if self.shared != 2 or self.selected >= self.experts:
            raise ValueError("K3 requires two shared experts and a top-k+1 cutoff")
        for name in ("epsilon", "beta", "linear_beta"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not np.isfinite(value)
                or not 1e-12 <= value <= 1e6
            ):
                raise ValueError("invalid K3 MoE scalar: " + name)


class _Config(C.Structure):
    _fields_ = [
        (f.name, C.c_int if i < 6 else C.c_float)
        for i, f in enumerate(fields(KimiK3MoEConfig))
    ]


def tensor_shapes(c):
    if type(c) is not KimiK3MoEConfig:
        raise TypeError("explicit K3 MoE config required")
    h, latent, m, e, s = c.hidden, c.latent, c.intermediate, c.experts, c.shared
    return {
        "router.weight": (e, h),
        "router.bias": (e,),
        "latent.down.weight": (latent, h),
        "latent.norm.weight": (latent,),
        "latent.up.weight": (h, latent),
        "routed.gate.weight": (e, m, latent),
        "routed.up.weight": (e, m, latent),
        "routed.down.weight": (e, latent, m),
        "shared.gate.weight": (s, m, h),
        "shared.up.weight": (s, m, h),
        "shared.down.weight": (s, h, m),
    }


def budgets(c, tokens):
    """Conservative construction/execution bytes; multiply-accumulate count."""
    shapes = tensor_shapes(c)
    if type(tokens) is not int or not 1 <= tokens <= 128:
        raise ValueError("token budget must be in [1, 128]")
    count = sum(int(np.prod(s)) for s in shapes.values())
    # C uses a fixed bounded stack workspace plus transactional heap output/routes.
    memory = 12 * count + 4096 + tokens * (12 * c.hidden + 8 * c.selected)
    work = tokens * (
        c.experts * c.hidden
        + 2 * c.hidden * c.latent
        + 3 * c.selected * c.latent * c.intermediate
        + 3 * c.shared * c.hidden * c.intermediate
    )
    return memory, work


class NativeKimiK3MoE:
    def __init__(
        self,
        library,
        config,
        weights,
        *,
        memory_budget=16 * 1024 * 1024,
        mac_budget=100_000_000,
    ):
        shapes = tensor_shapes(config)
        for value in (memory_budget, mac_budget):
            if type(value) is not int or not 1 <= value <= 2**63 - 1:
                raise ValueError("positive bounded integer budget required")
        if budgets(config, 1)[0] > memory_budget:
            raise ValueError("K3 MoE memory budget exceeded")
        if set(weights) != set(shapes):
            raise ValueError("K3 MoE tensor inventory mismatch")
        arrays = []
        for name, shape in shapes.items():
            a = np.asarray(weights[name])
            if a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
                raise ValueError("invalid FP32 K3 MoE tensor: " + name)
            arrays.append(a.ravel(order="C"))
        self._weights = np.frombuffer(
            np.concatenate(arrays).tobytes(), dtype=np.float32
        )
        self._config = config
        self.memory_budget, self.mac_budget = memory_budget, mac_budget
        self._c = _Config(*(getattr(config, f.name) for f in fields(config)))
        self._library = C.CDLL(str(library))
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(
            self,
            _ctypes.FreeLibrary if hasattr(_ctypes, "FreeLibrary") else _ctypes.dlclose,
            self._library._handle,
        )
        self._function = self._library.oa_kimi_k3_moe
        self._function.argtypes = [
            C.POINTER(_Config),
            C.POINTER(C.c_float),
            C.c_size_t,
            C.POINTER(C.c_float),
            C.c_size_t,
            C.c_int,
            C.POINTER(C.c_float),
            C.c_size_t,
            C.POINTER(C.c_int),
            C.c_size_t,
            C.c_size_t,
            C.c_size_t,
            C.c_int,
        ]
        self._function.restype = C.c_int

    def close(self):
        """Release this instance's OS library reference, independently of peers."""
        with self._lock:
            self._finalizer()

    @property
    def config(self):
        return self._config

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def forward(self, inputs, *, _fail_after_token=-1):
        for value in (self.memory_budget, self.mac_budget):
            if type(value) is not int or not 1 <= value <= min(2**63 - 1, C.c_size_t(-1).value):
                raise ValueError("positive bounded integer budget required")
        a = np.asarray(inputs)
        c = self.config
        if (
            a.dtype != np.float32
            or a.ndim != 2
            or a.shape[1] != c.hidden
            or not np.isfinite(a).all()
        ):
            raise ValueError("finite FP32 [tokens, hidden] input required")
        memory, work = budgets(c, len(a))
        if memory > self.memory_budget or work > self.mac_budget:
            raise ValueError("K3 MoE execution budget exceeded")
        if type(_fail_after_token) is not int or not -1 <= _fail_after_token < len(a):
            raise ValueError("invalid fault position")
        a = np.array(a, copy=True, order="C")
        output = np.empty_like(a)
        routes = np.empty((len(a), c.selected), np.int32)

        def ptr(x):
            return x.ctypes.data_as(C.POINTER(C.c_float))

        with self._lock:
            if not self._finalizer.alive:
                raise RuntimeError("K3 MoE library is closed")
            status = self._function(
                C.byref(self._c),
                ptr(self._weights),
                self._weights.size,
                ptr(a),
                a.size,
                len(a),
                ptr(output),
                output.size,
                routes.ctypes.data_as(C.POINTER(C.c_int)),
                routes.size,
                self.memory_budget,
                self.mac_budget,
                _fail_after_token,
            )
        if status:
            raise RuntimeError(f"K3 MoE rejected ({status}); outputs unchanged")
        return output, routes
