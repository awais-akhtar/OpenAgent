"""Opt-in bounded K3 dense-MLA fixture, not published-checkpoint admission."""
import ctypes as C
from dataclasses import dataclass, fields
import hashlib
import threading

import numpy as np


@dataclass(frozen=True)
class KimiK3DenseMLAConfig:
    hidden: int = 12
    layers: int = 3
    heads: int = 2
    nope: int = 3
    rope: int = 2
    value: int = 4
    qrank: int = 5
    kvrank: int = 6
    intermediate: int = 17
    block: int = 2
    vocab: int = 23
    capacity: int = 16
    epsilon: float = 1e-6
    beta: float = 4.0
    linear_beta: float = 25.0

    def __post_init__(self):
        bounds = (64, 8, 4, 16, 16, 16, 64, 64, 128, 8, 256, 128)
        for field, upper in zip(fields(self)[:12], bounds):
            v = getattr(self, field.name)
            if type(v) is not int or not (0 if field.name == 'rope' else 1) <= v <= upper:
                raise ValueError('unsupported K3 geometry: ' + field.name)
        for name in ('epsilon', 'beta', 'linear_beta'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
                raise ValueError('invalid K3 scalar: ' + name)


class _Config(C.Structure):
    _fields_ = [(f.name, C.c_int if i < 12 else C.c_double)
                for i, f in enumerate(fields(KimiK3DenseMLAConfig))]


def tensor_shapes(c):
    """Exact FP32 tensor admission; omitted or extra contracts are errors."""
    h, q, v = c.hidden, c.heads*(c.nope+c.rope), c.heads*c.value
    result = {'model.embed_tokens.weight': (c.vocab, h)}
    for i in range(c.layers):
        prefix = f'model.layers.{i}.'
        layer = {
            'input_layernorm.weight': (h,), 'post_attention_layernorm.weight': (h,),
            'self_attention_res_norm.weight': (h,), 'self_attention_res_proj.weight': (1, h),
            'mlp_res_norm.weight': (h,), 'mlp_res_proj.weight': (1, h),
            'self_attn.q_a_proj.weight': (c.qrank, h),
            'self_attn.q_a_layernorm.weight': (c.qrank,),
            'self_attn.q_b_proj.weight': (q, c.qrank),
            'self_attn.kv_a_proj_with_mqa.weight': (c.kvrank+c.rope, h),
            'self_attn.kv_a_layernorm.weight': (c.kvrank,),
            'self_attn.kv_b_proj.weight': (c.heads*(c.nope+c.value), c.kvrank),
            'self_attn.g_proj.weight': (v, h), 'self_attn.o_proj.weight': (h, v),
            'mlp.gate_proj.weight': (c.intermediate, h),
            'mlp.up_proj.weight': (c.intermediate, h), 'mlp.down_proj.weight': (h, c.intermediate),
        }
        result.update({prefix+k: shape for k, shape in layer.items()})
    result.update({'model.norm.weight': (h,), 'model.output_attn_res_proj.weight': (1, h),
                   'model.output_attn_res_norm.weight': (h,), 'lm_head.weight': (c.vocab, h)})
    return result


def admit_weights(config, weights):
    if type(config) is not KimiK3DenseMLAConfig:
        raise TypeError('only explicit K3 dense-MLA fixture config is supported')
    shapes = tensor_shapes(config)
    if set(weights) != set(shapes):
        raise ValueError('K3 tensor inventory mismatch')
    result = {}
    for name, shape in shapes.items():
        a = np.asarray(weights[name])
        if a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
            raise ValueError('invalid FP32 K3 tensor: ' + name)
        result[name] = np.array(a, copy=True, order='C')
        result[name].flags.writeable = False
    return result


@dataclass(frozen=True)
class KimiK3Snapshot:
    identity: str
    length: int
    keys: np.ndarray
    values: np.ndarray


class NativeKimiK3DenseMLA:
    def __init__(self, library, config, weights):
        self.config = config
        tensors = admit_weights(config, weights)
        self._weights = np.concatenate([a.ravel() for a in tensors.values()])
        self._weights.flags.writeable = False
        self._identity = hashlib.sha256(repr(config).encode()+self._weights.tobytes()).hexdigest()
        self._lock = threading.RLock()
        self._library = C.CDLL(str(library))
        self._function = self._library.oa_kimi_k3_decode
        self._function.argtypes = [C.POINTER(_Config), C.POINTER(C.c_float), C.c_size_t,
            C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_float),
            C.POINTER(C.c_int), C.POINTER(C.c_float), C.c_int]
        self._function.restype = C.c_int
        self._c = _Config(*(getattr(config, f.name) for f in fields(config)))
        self._keys = np.zeros((config.layers, config.capacity, config.heads, config.nope+config.rope), np.float32)
        self._values = np.zeros((config.layers, config.capacity, config.heads, config.value), np.float32)
        self._length = C.c_int(0)

    def snapshot(self):
        with self._lock:
            return KimiK3Snapshot(self._identity, self._length.value, self._keys.copy(), self._values.copy())

    def reset(self):
        with self._lock:
            self._keys.fill(0)
            self._values.fill(0)
            self._length.value = 0

    def restore(self, snapshot):
        with self._lock:
            if type(snapshot) is not KimiK3Snapshot or snapshot.identity != self._identity:
                raise ValueError('foreign K3 snapshot')
            if type(snapshot.length) is not int or not 0 <= snapshot.length <= self.config.capacity:
                raise ValueError('invalid K3 snapshot length')
            arrays = [np.array(a, copy=True, order='C') for a in (snapshot.keys, snapshot.values)]
            for a, current in zip(arrays, (self._keys, self._values)):
                if a.shape != current.shape or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError('invalid K3 snapshot buffer')
                if np.any(a[:, snapshot.length:] != 0):
                    raise ValueError('K3 snapshot contains nonzero unused capacity')
            self._keys[:], self._values[:] = arrays
            self._length.value = snapshot.length

    def forward(self, tokens, *, _fail_after_token=-1):
        # No padding, batches, masks, RoPE, quantization, KDA, MoE or implicit conversion.
        a = np.asarray(tokens)
        if a.ndim != 1 or a.size < 1 or a.size > 128 or a.dtype.kind not in 'iu' or a.dtype == np.bool_:
            raise ValueError('expected a nonempty bounded integer token vector')
        if np.any(a < 0) or np.any(a >= self.config.vocab):
            raise ValueError('K3 token out of range')
        if type(_fail_after_token) is not int or not -1 <= _fail_after_token < a.size:
            raise ValueError('invalid test fault position')
        a = np.ascontiguousarray(a, dtype=np.int32)
        out = np.empty((len(a), self.config.vocab), np.float32)
        def ptr(x):
            return x.ctypes.data_as(C.POINTER(C.c_float))
        with self._lock:
            status = self._function(C.byref(self._c), ptr(self._weights), self._weights.size,
                a.ctypes.data_as(C.POINTER(C.c_int)), len(a), ptr(self._keys), ptr(self._values),
                C.byref(self._length), ptr(out), _fail_after_token)
            if status:
                raise RuntimeError(f'K3 native decode rejected ({status}); state unchanged')
        return out
