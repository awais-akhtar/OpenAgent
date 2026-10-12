"""Explicit bounded FP32 K3 hybrid fixture; Python schedules native arithmetic.

No vendor config admission, downloaded weights or model-family aliases.
"""

import copy
import ctypes as C
import _ctypes
from contextlib import contextmanager
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path
import threading
import weakref

import numpy as np

from .native_kimi_k3_decoder import KimiK3DenseMLAConfig, _Config
from .native_kimi_k3_kda import KimiK3KDAConfig, NativeKimiK3KDA, kda_tensor_shapes
from .native_kimi_k3_moe import (
    KimiK3MoEConfig,
    NativeKimiK3MoE,
    tensor_shapes as moe_shapes,
)


@dataclass(frozen=True)
class KimiK3HybridConfig:
    mla: KimiK3DenseMLAConfig
    kda: KimiK3KDAConfig
    moe: KimiK3MoEConfig
    layer_types: tuple
    dense_layers: tuple = ()
    dense_intermediate: int | None = None

    def __post_init__(self):
        if (
            type(self.mla) is not KimiK3DenseMLAConfig
            or type(self.kda) is not KimiK3KDAConfig
            or type(self.moe) is not KimiK3MoEConfig
        ):
            raise ValueError("explicit native component configurations required")
        if (
            type(self.layer_types) is not tuple
            or len(self.layer_types) != self.mla.layers
        ):
            raise ValueError("one explicit mixer type per layer required")
        if any(
            type(name) is not str or name not in ("mla", "kda")
            for name in self.layer_types
        ):
            raise ValueError("only mla and kda mixers supported")
        if self.layer_types[-1] != "mla":
            raise ValueError("K3 final layer must use global MLA")
        if self.mla.hidden != self.kda.hidden or self.mla.hidden != self.moe.hidden:
            raise ValueError("component hidden widths must match")
        if self.mla.capacity != self.kda.capacity:
            raise ValueError("component cache capacities must match")
        if (
            type(self.dense_layers) is not tuple
            or any(
                type(i) is not int or not 0 <= i < self.mla.layers
                for i in self.dense_layers
            )
            or len(set(self.dense_layers)) != len(self.dense_layers)
        ):
            raise ValueError("explicit unique dense layer indices required")
        if self.dense_intermediate is not None and (
            type(self.dense_intermediate) is not int
            or not 1 <= self.dense_intermediate <= 128
        ):
            raise ValueError("unsupported dense intermediate width")

    @property
    def dense_width(self):
        return (
            self.moe.intermediate
            if self.dense_intermediate is None
            else self.dense_intermediate
        )


def _mla_shapes(c):
    return {
        "q_a_proj.weight": (c.qrank, c.hidden),
        "q_a_layernorm.weight": (c.qrank,),
        "q_b_proj.weight": (c.heads * (c.nope + c.rope), c.qrank),
        "kv_a_proj_with_mqa.weight": (c.kvrank + c.rope, c.hidden),
        "kv_a_layernorm.weight": (c.kvrank,),
        "kv_b_proj.weight": (c.heads * (c.nope + c.value), c.kvrank),
        "g_proj.weight": (c.heads * c.value, c.hidden),
        "o_proj.weight": (c.hidden, c.heads * c.value),
    }


def hybrid_tensor_shapes(config):
    if type(config) is not KimiK3HybridConfig:
        raise ValueError("explicit KimiK3HybridConfig required")
    c = config.mla
    shapes = {"model.embed_tokens.weight": (c.vocab, c.hidden)}
    for i, kind in enumerate(config.layer_types):
        prefix = f"model.layers.{i}."
        shapes.update(
            {
                prefix + name: (c.hidden,)
                for name in (
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attention_res_norm.weight",
                    "self_attention_res_proj.weight",
                    "mlp_res_norm.weight",
                    "mlp_res_proj.weight",
                )
            }
        )
        attention = _mla_shapes(c) if kind == "mla" else kda_tensor_shapes(config.kda)
        shapes.update(
            {prefix + "attention." + name: shape for name, shape in attention.items()}
        )
        feedforward = (
            {
                "gate_proj.weight": (config.dense_width, c.hidden),
                "up_proj.weight": (config.dense_width, c.hidden),
                "down_proj.weight": (c.hidden, config.dense_width),
            }
            if i in config.dense_layers
            else moe_shapes(config.moe)
        )
        namespace = "mlp." if i in config.dense_layers else "moe."
        shapes.update(
            {prefix + namespace + name: shape for name, shape in feedforward.items()}
        )
    shapes.update(
        {
            name: (c.hidden,)
            for name in (
                "model.norm.weight",
                "model.output_attn_res_norm.weight",
                "model.output_attn_res_proj.weight",
            )
        }
    )
    shapes["lm_head.weight"] = (c.vocab, c.hidden)
    return shapes


@dataclass(frozen=True)
class KimiK3HybridSnapshot:
    identity: str
    length: int
    keys: tuple
    values: tuple
    recurrent: tuple


class NativeKimiK3Hybrid:
    """Batch-one hybrid token-to-logit fixture, with atomic whole-call state."""

    implementation = "c-k3-hybrid-fp32-bounded-fixture"
    published_checkpoint_supported = False

    def __init__(self, config, tensors, *, library, max_bytes=128 * 1024 * 1024):
        shapes = hybrid_tensor_shapes(config)
        if type(max_bytes) is not int or not 0 < max_bytes <= 2**63 - 1:
            raise ValueError("invalid hybrid memory budget")
        if set(tensors) != set(shapes):
            raise ValueError("exact hybrid tensor inventory required")
        weights = {}
        for name, shape in shapes.items():
            a = np.asarray(tensors[name])
            if (
                a.dtype != np.float32
                or a.shape != shape
                or not a.flags.c_contiguous
                or not np.isfinite(a).all()
            ):
                raise ValueError("invalid hybrid FP32 tensor: " + name)
            weights[name] = a
        self.config, self.max_bytes = config, max_bytes
        c = config.mla
        k = config.kda
        self._weights_bytes = sum(a.nbytes for a in weights.values())
        self._cache_bytes = sum(
            4 * c.capacity * c.heads * (c.nope + c.rope + c.value)
            if kind == "mla"
            else 4
            * (
                k.heads * k.key * k.value
                + (k.kernel - 1) * k.heads * (2 * k.key + k.value)
            )
            for kind in config.layer_types
        )
        self._admit(1)
        self._weights = {
            n: np.frombuffer(a.tobytes(), dtype=np.float32).reshape(a.shape)
            for n, a in weights.items()
        }
        digest = hashlib.sha256(repr(config).encode("ascii"))
        for name, a in self._weights.items():
            digest.update(name.encode("ascii"))
            digest.update(a.tobytes())
        self._identity = digest.hexdigest()
        self._lock = threading.RLock()
        self.library = C.CDLL(str(Path(library).resolve(strict=True)))
        release = (
            _ctypes.FreeLibrary if hasattr(_ctypes, "FreeLibrary") else _ctypes.dlclose
        )
        weakref.finalize(self.library, release, self.library._handle)
        version = self.library.oa_k3_blocks_version
        version.argtypes, version.restype = [], C.c_int
        if version() != 1:
            raise ValueError("unsupported native block ABI")
        fp = C.POINTER(C.c_float)
        self._norm = self.library.oa_k3_blocks_norm
        self._norm.argtypes, self._norm.restype = (
            [fp, fp, C.c_int, C.c_double, fp],
            C.c_int,
        )
        self._add = self.library.oa_k3_blocks_add
        self._add.argtypes, self._add.restype = [fp, fp, C.c_int, fp], C.c_int
        self._matrix = self.library.oa_k3_blocks_matrix
        self._matrix.argtypes, self._matrix.restype = (
            [fp, C.c_size_t, fp, C.c_int, C.c_int, fp],
            C.c_int,
        )
        self._attnres = self.library.oa_kimi_k3_attnres
        self._attnres.argtypes, self._attnres.restype = (
            [fp, fp, fp, C.c_int, C.c_int, C.c_double, fp],
            C.c_int,
        )
        self._mla = self.library.oa_k3_blocks_mla
        self._mla.argtypes = [
            C.POINTER(_Config),
            fp,
            C.c_size_t,
            fp,
            C.c_size_t,
            fp,
            C.c_size_t,
            fp,
            C.c_size_t,
            C.c_int,
            fp,
            C.c_size_t,
        ]
        self._mla.restype = C.c_int
        self._dense = None
        if config.dense_layers:
            try:
                self._dense = self.library.oa_k3_dense
            except AttributeError as error:
                raise ValueError("missing native dense SwiGLU symbol") from error
            self._dense.argtypes = [fp, fp, fp, fp, C.c_int, C.c_int, fp]
            self._dense.restype = C.c_int
        self._c = _Config(*(getattr(c, f.name) for f in fields(KimiK3DenseMLAConfig)))
        self._mixers, self._experts, self._mla_weights = [], [], []
        for i, kind in enumerate(config.layer_types):
            p = f"model.layers.{i}."
            self._experts.append(
                None
                if i in config.dense_layers
                else NativeKimiK3MoE(
                    library,
                    config.moe,
                    {n: self._weights[p + "moe." + n] for n in moe_shapes(config.moe)},
                    memory_budget=max_bytes,
                )
            )
            if kind == "kda":
                self._mixers.append(
                    NativeKimiK3KDA(
                        config.kda,
                        {
                            n: self._weights[p + "attention." + n]
                            for n in kda_tensor_shapes(config.kda)
                        },
                        library=library,
                        max_bytes=max_bytes,
                    )
                )
                self._mla_weights.append(None)
            else:
                self._mixers.append(None)
                self._mla_weights.append(
                    np.frombuffer(
                        b"".join(
                            self._weights[p + "attention." + n].tobytes()
                            for n in _mla_shapes(c)
                        ),
                        dtype=np.float32,
                    )
                )
        self.reset()

    def _admit(self, n):
        c = self.config.mla
        if type(self.max_bytes) is not int or not 0 < self.max_bytes <= 2**63 - 1:
            raise ValueError("invalid hybrid memory budget")
        required = (
            5 * self._weights_bytes
            + 5 * self._cache_bytes
            + 4 * n * (c.vocab + 16 * c.hidden)
            + 1024 * 1024
        )
        if required > self.max_bytes:
            raise ValueError("hybrid transaction memory budget exceeded")

    @staticmethod
    def _ptr(a):
        return a.ctypes.data_as(C.POINTER(C.c_float))

    @staticmethod
    def _check(status):
        if status:
            raise ValueError(f"native hybrid block failed ({status}); state unchanged")

    def _normalized(self, x, gain):
        out = np.empty_like(x)
        self._check(
            self._norm(
                self._ptr(x),
                self._ptr(gain),
                x.size,
                self.config.mla.epsilon,
                self._ptr(out),
            )
        )
        return out

    def _summed(self, x, y):
        out = np.empty_like(x)
        self._check(self._add(self._ptr(x), self._ptr(y), x.size, self._ptr(out)))
        return out

    def _combined(self, rows, query, gain):
        data = np.stack(rows)
        out = np.empty(self.config.mla.hidden, np.float32)
        self._check(
            self._attnres(
                self._ptr(data),
                self._ptr(query),
                self._ptr(gain),
                len(rows),
                out.size,
                self.config.mla.epsilon,
                self._ptr(out),
            )
        )
        return out

    @contextmanager
    def _mutation(self):
        with self._lock:
            if getattr(self, "_mutation_active", False):
                raise ValueError("reentrant hybrid mutation forbidden")
            self._mutation_active = True
            try:
                yield
            finally:
                self._mutation_active = False

    def reset(self):
        with self._mutation():
            self._admit(0)
            c = self.config.mla
            keys = [
                None
                if kind == "kda"
                else np.zeros((c.capacity, c.heads, c.nope + c.rope), np.float32)
                for kind in self.config.layer_types
            ]
            values = [
                None
                if kind == "kda"
                else np.zeros((c.capacity, c.heads, c.value), np.float32)
                for kind in self.config.layer_types
            ]
            mixers = []
            for m in self._mixers:
                candidate = None if m is None else copy.copy(m)
                if candidate is not None:
                    candidate._mixer = copy.copy(candidate._mixer)
                    candidate.reset()
                mixers.append(candidate)
            self._keys, self._values, self._mixers, self._length = (
                keys,
                values,
                mixers,
                0,
            )

    @property
    def position(self):
        with self._lock:
            return self._length

    def snapshot(self):
        with self._lock:
            self._admit(0)
            return KimiK3HybridSnapshot(
                self._identity,
                self._length,
                tuple(None if a is None else a.copy() for a in self._keys),
                tuple(None if a is None else a.copy() for a in self._values),
                tuple(None if m is None else m.snapshot() for m in self._mixers),
            )

    def restore(self, snapshot):
        with self._mutation():
            c = self.config.mla
            if (
                type(snapshot) is not KimiK3HybridSnapshot
                or snapshot.identity != self._identity
            ):
                raise ValueError("foreign hybrid snapshot")
            if (
                type(snapshot.length) is not int
                or not 0 <= snapshot.length <= c.capacity
            ):
                raise ValueError("invalid hybrid snapshot position")
            for sequence in (snapshot.keys, snapshot.values, snapshot.recurrent):
                if type(sequence) is not tuple or len(sequence) != c.layers:
                    raise ValueError("invalid hybrid snapshot inventory")
            self._admit(0)
            mixers = []
            for i, kind in enumerate(self.config.layer_types):
                if kind == "mla":
                    if snapshot.recurrent[i] is not None:
                        raise ValueError("unexpected recurrent cache")
                    for a, shape in (
                        (snapshot.keys[i], self._keys[i].shape),
                        (snapshot.values[i], self._values[i].shape),
                    ):
                        if (
                            type(a) is not np.ndarray
                            or a.dtype != np.float32
                            or a.shape != shape
                            or not a.flags.c_contiguous
                            or not np.isfinite(a).all()
                        ):
                            raise ValueError("invalid MLA snapshot buffer")
                        if a[snapshot.length :].any():
                            raise ValueError("nonzero uncommitted MLA cache")
                    mixers.append(None)
                else:
                    if snapshot.keys[i] is not None or snapshot.values[i] is not None:
                        raise ValueError("unexpected MLA cache")
                    m = copy.copy(self._mixers[i])
                    m._mixer = copy.copy(m._mixer)
                    item = snapshot.recurrent[i]
                    # Do not let ndarray subclasses override component copy semantics.
                    if (
                        type(item) is not tuple
                        or len(item) != 2
                        or type(item[1]) is not tuple
                        or len(item[1]) != 3
                        or type(item[1][2]) is not tuple
                        or any(
                            type(a) is not np.ndarray for a in (item[1][1], *item[1][2])
                        )
                    ):
                        raise ValueError("invalid recurrent snapshot buffers")
                    m.restore(snapshot.recurrent[i])
                    if m.position != snapshot.length:
                        raise ValueError("inconsistent hybrid layer positions")
                    mixers.append(m)
            keys = [None if a is None else a.copy() for a in snapshot.keys]
            values = [None if a is None else a.copy() for a in snapshot.values]
            self._keys, self._values, self._mixers, self._length = (
                keys,
                values,
                mixers,
                snapshot.length,
            )

    def forward(self, token_ids, *, _fail_after_token=-1):
        with self._mutation():
            c = self.config.mla
            ids = np.asarray(token_ids)
            if (
                ids.ndim != 1
                or ids.dtype.kind not in "iu"
                or not 1 <= ids.size <= 128
                or np.any(ids >= c.vocab)
                or np.any(ids < 0)
            ):
                raise ValueError("invalid hybrid token IDs")
            if self._length + ids.size > c.capacity:
                raise ValueError("hybrid cache capacity exceeded")
            if (
                type(_fail_after_token) is not int
                or not -1 <= _fail_after_token < ids.size
            ):
                raise ValueError("invalid hybrid fault position")
            self._admit(ids.size)
            ids = ids.copy()
            keys = [None if a is None else a.copy() for a in self._keys]
            values = [None if a is None else a.copy() for a in self._values]
            mixers = []
            for m in self._mixers:
                candidate = None if m is None else copy.copy(m)
                if candidate is not None:
                    candidate._mixer = copy.copy(candidate._mixer)
                mixers.append(candidate)
            output = np.empty((ids.size, c.vocab), np.float32)
            for t, token in enumerate(ids):
                prefix = self._weights["model.embed_tokens.weight"][int(token)].copy()
                blocks = []
                for i, kind in enumerate(self.config.layer_types):
                    p = f"model.layers.{i}."
                    mixed = (
                        prefix
                        if not blocks
                        else self._combined(
                            blocks + [prefix],
                            self._weights[p + "self_attention_res_proj.weight"],
                            self._weights[p + "self_attention_res_norm.weight"],
                        )
                    )
                    if i % c.block == 0:
                        blocks.append(prefix.copy())
                        prefix = np.zeros_like(prefix)
                    norm = self._normalized(
                        mixed, self._weights[p + "input_layernorm.weight"]
                    )
                    if kind == "kda":
                        branch = mixers[i].forward(norm.reshape(1, -1))[0]
                    else:
                        branch = np.empty(c.hidden, np.float32)
                        w = self._mla_weights[i]
                        self._check(
                            self._mla(
                                C.byref(self._c),
                                self._ptr(w),
                                w.size,
                                self._ptr(norm),
                                norm.size,
                                self._ptr(keys[i]),
                                keys[i].size,
                                self._ptr(values[i]),
                                values[i].size,
                                self._length + t,
                                self._ptr(branch),
                                branch.size,
                            )
                        )
                    prefix = self._summed(prefix, branch)
                    mixed = self._combined(
                        blocks + [prefix],
                        self._weights[p + "mlp_res_proj.weight"],
                        self._weights[p + "mlp_res_norm.weight"],
                    )
                    norm = self._normalized(
                        mixed, self._weights[p + "post_attention_layernorm.weight"]
                    )
                    if i in self.config.dense_layers:
                        branch = np.empty(c.hidden, np.float32)
                        self._check(
                            self._dense(
                                self._ptr(self._weights[p + "mlp.gate_proj.weight"]),
                                self._ptr(self._weights[p + "mlp.up_proj.weight"]),
                                self._ptr(self._weights[p + "mlp.down_proj.weight"]),
                                self._ptr(norm),
                                c.hidden,
                                self.config.dense_width,
                                self._ptr(branch),
                            )
                        )
                    else:
                        branch, _ = self._experts[i].forward(norm.reshape(1, -1))
                        branch = branch[0]
                    prefix = self._summed(prefix, branch)
                mixed = self._combined(
                    blocks + [prefix],
                    self._weights["model.output_attn_res_proj.weight"],
                    self._weights["model.output_attn_res_norm.weight"],
                )
                norm = self._normalized(mixed, self._weights["model.norm.weight"])
                w = self._weights["lm_head.weight"]
                self._check(
                    self._matrix(
                        self._ptr(w),
                        w.size,
                        self._ptr(norm),
                        c.vocab,
                        c.hidden,
                        self._ptr(output[t]),
                    )
                )
                if t == _fail_after_token:
                    raise ValueError("injected hybrid failure; state unchanged")
            self._keys, self._values, self._mixers = keys, values, mixers
            self._length += ids.size
            return output
