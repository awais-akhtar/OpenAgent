"""Original native dense CED/CSA2 equation path, not publisher checkpoint support."""
import ctypes as C
import _ctypes
from contextlib import contextmanager
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path
import threading
import weakref

import numpy as np


@dataclass(frozen=True)
class DeepSeekV41CEDConfig:
    hidden: int = 8
    layers: int = 6
    encoder: int = 3
    vocabulary: int = 23
    capacity: int = 32
    window: int = 3
    ratio: int = 2
    topk: int = 2
    block_size: int = 2
    candidate_blocks: int = 2
    intermediate: int = 13
    epsilon: float = 1e-6
    modes: tuple = ('swa', 'full', 'reuse', 'full', 'reindex', 'reuse')

    def __post_init__(self):
        limits = {'hidden': (1, 16), 'layers': (2, 12), 'encoder': (1, 11),
                  'vocabulary': (2, 256), 'capacity': (1, 64), 'window': (1, 64),
                  'ratio': (1, 4), 'topk': (1, 8), 'block_size': (1, 8),
                  'candidate_blocks': (1, 8), 'intermediate': (1, 64)}
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('unsupported CED geometry: ' + name)
        if self.encoder >= self.layers or self.window > self.capacity:
            raise ValueError('invalid encoder split or local window')
        if type(self.epsilon) not in (int, float) or not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError('invalid CED normalization epsilon')
        if type(self.modes) is not tuple or len(self.modes) != self.layers:
            raise ValueError('one explicit mode per CED layer required')
        for phase in (self.modes[:self.encoder], self.modes[self.encoder:]):
            seen = 0
            for mode in phase:
                if type(mode) is not str or mode not in ('swa', 'full', 'reindex', 'reuse'):
                    raise ValueError('unknown CED layer mode')
                if mode == 'full':
                    seen += 1
                if mode in ('reindex', 'reuse') and not seen:
                    raise ValueError('reuse requires a Full layer in the same phase')
            if not seen:
                raise ValueError('each CED phase requires a Full layer')
        if self.modes[self.encoder:].count('full') != 1:
            raise ValueError('bounded decoder uses one global KV source')


class _Config(C.Structure):
    _fields_ = [(f.name, C.c_int) for f in fields(DeepSeekV41CEDConfig)[:11]]
    _fields_ += [('epsilon', C.c_double)]


def ced_tensor_shapes(config):
    if type(config) is not DeepSeekV41CEDConfig:
        raise ValueError('explicit CED config required')
    h, m = config.hidden, config.intermediate
    shapes = {'embedding.weight': (config.vocabulary, h)}
    for layer in range(config.layers):
        prefix = f'layers.{layer}.'
        shapes.update({prefix + name: shape for name, shape in (
            ('attention.norm', (h,)), ('attention.q', (h, h)),
            ('attention.k', (h, h)), ('attention.v', (h, h)), ('attention.out', (h, h)),
            ('mlp.norm', (h,)), ('mlp.gate', (m, h)), ('mlp.up', (m, h)), ('mlp.down', (h, m)),
            ('global.q', (h, h)), ('global.kv', (h, h)), ('global.z', (h, h)),
            ('index.q', (h, h)), ('index.k', (h, h)),
        )})
    shapes['final.norm'] = (h,)
    shapes['head.weight'] = (config.vocabulary, h)
    return shapes


@dataclass(frozen=True)
class CEDSnapshot:
    identity: str
    tokens: tuple


class NativeDeepSeekV41CED:
    """Bounded dense token-to-logit equations with atomic token-history replay.

    Single-head FP32 without RoPE, mHC, Engram, MoE or cache quantization.
    These separate components are not implied by this class's identity.
    """
    implementation = 'c-dense-ced-csa2-equation-path'
    published_checkpoint_supported = False

    def __init__(self, config, tensors, *, library, max_bytes=64 * 1024**2):
        shapes = ced_tensor_shapes(config)
        if type(max_bytes) is not int or not 0 < max_bytes <= np.iinfo(np.intp).max:
            raise ValueError('invalid CED memory budget')
        if set(tensors) != set(shapes):
            raise ValueError('exact CED tensor inventory required')
        count = sum(int(np.prod(s)) for s in shapes.values())
        if count * 12 + 1024**2 > max_bytes:
            raise ValueError('CED weight budget exceeded')
        arrays = []
        for name, shape in shapes.items():
            value = np.asarray(tensors[name])
            if value.dtype != np.float32 or value.shape != shape or not value.flags.c_contiguous or not np.isfinite(value).all():
                raise ValueError('invalid CED FP32 tensor: ' + name)
            arrays.append(value.reshape(-1))
        self._config, self.max_bytes = config, max_bytes
        self._weights = np.frombuffer(np.concatenate(arrays).tobytes(), dtype=np.float32)
        digest = hashlib.sha256(repr(config).encode('ascii'))
        digest.update(self._weights.tobytes())
        self._identity = digest.hexdigest()
        self._tokens = ()
        self._trace = ()
        self._lock = threading.RLock()
        self._active_mutation = False
        self._c = _Config(*(getattr(config, f.name) for f in fields(config)[:12]))
        self._modes = np.array([('swa', 'full', 'reindex', 'reuse').index(m) for m in config.modes], dtype=np.int32)
        self._admit(1)
        self.library = C.CDLL(str(Path(library).resolve(strict=True)))
        release = _ctypes.FreeLibrary if hasattr(_ctypes, 'FreeLibrary') else _ctypes.dlclose
        weakref.finalize(self.library, release, self.library._handle)
        version = self.library.oa_v41_ced_version
        version.argtypes, version.restype = [], C.c_uint32
        if version() != 1:
            raise ValueError('unsupported CED ABI')
        self._run = self.library.oa_v41_ced_forward
        self._run.argtypes = [C.POINTER(_Config)] + [C.c_void_p, C.c_size_t] * 5 + [C.c_size_t, C.c_int]
        self._run.restype = C.c_int

    @property
    def config(self):
        return self._config

    @property
    def weights(self):
        return self._weights

    @property
    def trace(self):
        with self._lock:
            return self._trace

    @property
    def length(self):
        with self._lock:
            return len(self._tokens)

    def _admit(self, length):
        c = self.config
        if type(self.max_bytes) is not int or not 0 < self.max_bytes <= np.iinfo(np.intp).max:
            raise ValueError('invalid CED memory budget')
        # Native heap, Python ids/output/trace and detached committed output.
        heap = 4 * ((4 * c.layers + 1) * length * c.hidden + length * c.vocabulary)
        trace = 4 * c.layers * (2 + c.topk)
        required = self._weights.nbytes * 2 + heap + length * (8 * c.vocabulary + 8) + trace * 2 + 1024**2
        if required > self.max_bytes:
            raise ValueError('CED transaction memory budget exceeded')
        return heap + trace

    @contextmanager
    def _mutation(self):
        with self._lock:
            if self._active_mutation:
                raise RuntimeError('reentrant CED state mutation is not supported')
            self._active_mutation = True
            try:
                yield
            finally:
                self._active_mutation = False

    def reset(self):
        with self._mutation():
            self._tokens, self._trace = (), ()

    def snapshot(self):
        with self._lock:
            return CEDSnapshot(self._identity, self._tokens)

    def restore(self, snapshot):
        with self._mutation():
            if type(snapshot) is not CEDSnapshot or snapshot.identity != self._identity:
                raise ValueError('CED snapshot identity mismatch')
            self._validate_ids(snapshot.tokens)
            self._admit(len(snapshot.tokens))
            self._tokens, self._trace = snapshot.tokens, ()

    def _validate_ids(self, ids):
        if type(ids) is not tuple or len(ids) > self.config.capacity or any(
            type(v) is not int or not 0 <= v < self.config.vocabulary for v in ids
        ):
            raise ValueError('invalid CED token IDs or capacity')

    def forward(self, token_ids, *, fail_layer=-1):
        with self._mutation():
            if type(fail_layer) is not int or not -1 <= fail_layer < self.config.layers:
                raise ValueError('invalid CED failure-injection layer')
            # Bound iterable consumption before conversion or allocation.
            from itertools import islice
            incoming = tuple(islice(iter(token_ids), self.config.capacity + 1))
            self._validate_ids(incoming)
            if not incoming:
                raise ValueError('nonempty CED input required')
            history = self._tokens + incoming
            self._validate_ids(history)
            budget = self._admit(len(history))
            ids = np.asarray(history, dtype=np.int32)
            logits = np.empty((len(history), self.config.vocabulary), dtype=np.float32)
            trace = np.empty((self.config.layers, 2 + self.config.topk), dtype=np.int32)
            status = self._run(C.byref(self._c), self._modes.ctypes.data, self._modes.size,
                self._weights.ctypes.data, self._weights.size, ids.ctypes.data, ids.size,
                logits.ctypes.data, logits.size, trace.ctypes.data, trace.size, budget, fail_layer)
            if status:
                raise ValueError(f'native CED failure {status}; state unchanged')
            result = logits[-len(incoming):].copy()
            self._tokens = history
            self._trace = tuple(tuple(int(v) for v in row) for row in trace)
            return result
