"""Range-resident NumPy OLMoE execution, with explicitly non-native math.

Calls and session mutations are serialized by each session's reentrant owner
lock. Cancellation is a callable returning a boolean, checked at the two
transactional range boundaries.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np

from .expert_packing import _scan_directory
from .expert_runtime import SparseExpertManifest, SparseExpertRuntime
from .olmoe import (OlmoeConfig, OlmoeModel, _rms_norm, _softmax,
                    dense_execution_fingerprint, olmoe_metadata_fingerprint, validate_olmoe_tensors)
from .olmoe_batch import _token_vector


NUMERIC_CLASS = "numpy-fp32-dense-int8-row-experts"


def _integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"invalid {name}")
    return value


def _hidden(value, width):
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != width
            or array.dtype != np.dtype('float32') or not np.isfinite(array).all()):
        raise ValueError("hidden must be finite float32 rows x hidden_size")
    return array.copy()


def _cancel(check):
    if check is not None and check():
        raise InterruptedError("OLMoE range cancelled")


class OlmoeSegmentSession:
    def __init__(self, engine, capacity):
        self._engine = engine
        self.context_tokens = capacity
        config = engine.config
        shape = (config.num_key_value_heads, capacity,
                 config.hidden_size // config.num_attention_heads)
        self._k = [np.zeros(shape, np.float32) for _ in engine.layers]
        self._v = [np.zeros_like(key) for key in self._k]
        self.position = 0
        self._closed = False
        self._lock = threading.RLock()

    def _check(self):
        self._engine._check()
        if self._closed:
            raise RuntimeError("session is closed")

    def forward(self, hidden, *, cancelled=None):
        with self._lock:
            self._check()
            return self._engine.forward(self, hidden, cancelled=cancelled)

    def reset(self):
        with self._lock:
            self._check()
            for array in self._k + self._v:
                array.fill(0)
            self.position = 0

    def snapshot_metadata(self):
        """Return snapshot identity without reading or copying KV arrays."""
        with self._lock:
            self._check()
            engine = self._engine
            return dict(format="openagent.olmoe-segment.v2",
                        fingerprint=engine.manifest.fingerprint,
                        execution_identity=engine.execution_identity,
                        range=(engine.begin, engine.end), numeric_class=NUMERIC_CLASS,
                        context_tokens=self.context_tokens)

    def snapshot_state(self):
        with self._lock:
            return dict(self.snapshot_metadata(), position=self.position,
                        keys=[a[:, :self.position].copy() for a in self._k],
                        values=[a[:, :self.position].copy() for a in self._v])

    def restore_state(self, state):
        with self._lock:
            return self._restore_state_locked(state)

    def _restore_state_locked(self, state):
        self._check()
        expected = self.snapshot_metadata()
        if not isinstance(state, dict) or set(state) != set(expected) | {'position', 'keys', 'values'}:
            raise ValueError("invalid snapshot fields")
        for field in ('format', 'fingerprint', 'execution_identity', 'range', 'numeric_class', 'context_tokens'):
            if type(state[field]) is not type(expected[field]) or state[field] != expected[field]:
                raise ValueError(f"snapshot {field} mismatch")
        if any(type(v) is not int for v in state['range']):
            raise ValueError("invalid snapshot range")
        position = _integer(state['position'], 0, self.context_tokens, 'position')
        staged = []
        for field, targets in (('keys', self._k), ('values', self._v)):
            sources = state[field]
            if type(sources) is not list or len(sources) != len(targets):
                raise ValueError("invalid snapshot layer count")
            for source, target in zip(sources, targets):
                if (not isinstance(source, np.ndarray) or source.dtype != np.dtype('float32')
                        or source.shape != target[:, :position].shape
                        or not np.isfinite(source).all()):
                    raise ValueError("invalid snapshot KV array")
                array = np.zeros_like(target)
                array[:, :position] = source
                staged.append(array)
        count = len(self._k)
        self._k, self._v = staged[:count], staged[count:]
        self.position = position

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._engine._sessions.discard(self)
                self._k.clear()
                self._v.clear()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()


class OlmoeSegmentEngine:
    """Load only dense layers and expert regions in [begin, end).

    This is a real range-resident reference backend, not a compiled/native
    segment backend. Its snapshot numeric class requires NumPy expert math.
    """
    native_range_execution = False
    range_resident_weights = True
    numeric_class = NUMERIC_CLASS
    _tensor = OlmoeModel._tensor

    def __init__(self, model_directory, manifest, begin, end, context_tokens=None,
                 *, expert_cache_bytes, **runtime_options):
        if (runtime_options.get('kernels') is not None
                or runtime_options.get('preferred_kernel') not in (None, 'numpy-reference')):
            raise ValueError("segment snapshot numeric class requires NumPy expert kernels")
        self.root = Path(model_directory).resolve(strict=True)
        self.config = OlmoeConfig.read(self.root)
        self.begin = _integer(begin, 0, self.config.num_hidden_layers - 1, 'begin')
        self.end = _integer(end, begin + 1, self.config.num_hidden_layers, 'end')
        self.context_tokens = _integer(
            self.config.max_position_embeddings if context_tokens is None else context_tokens,
            1, self.config.max_position_embeddings, 'context_tokens')
        self.manifest = SparseExpertManifest.read(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.manifest.validate()
        config = self.config
        if (self.manifest.architecture != 'olmoe'
                or self.manifest.fingerprint != olmoe_metadata_fingerprint(self.root)
                or len(self.manifest.layers) != config.num_hidden_layers):
            raise ValueError("manifest does not match OLMoE checkpoint")
        for index, layer in enumerate(self.manifest.layers):
            if (layer.id != f'model.layers.{index}' or layer.hidden_size != config.hidden_size
                    or layer.intermediate_size != config.intermediate_size
                    or layer.experts_used != config.num_experts_per_tok
                    or set(layer.regions) != {str(i) for i in range(config.num_experts)}
                    or layer.dtype != 'int8-row'):
                raise ValueError("manifest layer contract mismatch")
            if any(not region.sha256 for region in layer.regions.values()):
                raise ValueError("OLMoE expert regions require SHA-256 content hashes")
        self._index = _scan_directory(self.root)
        validate_olmoe_tensors(config, self._index)
        self.layers = [OlmoeModel._load_layer(self, i) for i in range(begin, end)]
        selected = replace(self.manifest, layers=self.manifest.layers[begin:end])
        identity = [{
            'id': layer.id, 'dtype': layer.dtype,
            'hidden': layer.hidden_size, 'intermediate': layer.intermediate_size,
            'experts': {key: {'sha256': region.sha256, 'size': region.size}
                        for key, region in layer.regions.items()},
        } for layer in selected.layers]
        dense = [(f'layer.{begin + index}.{key}', value)
                 for index, layer in enumerate(self.layers) for key, value in sorted(layer.items())]
        self.execution_identity = hashlib.sha256(json.dumps({
            'experts': identity, 'dense': dense_execution_fingerprint(dense),
        }, sort_keys=True).encode('utf-8')).hexdigest()
        self.experts = SparseExpertRuntime(self.root, selected,
                                          capacity_bytes=expert_cache_bytes, **runtime_options)
        self._sessions = set()
        self._closed = False

    def _check(self):
        if self._closed:
            raise RuntimeError("engine is closed")

    def create_session(self, *, context_tokens=None):
        self._check()
        capacity = _integer(self.context_tokens if context_tokens is None else context_tokens,
                            1, self.context_tokens, 'session context_tokens')
        session = OlmoeSegmentSession(self, capacity)
        self._sessions.add(session)
        return session

    def forward(self, session, hidden, *, cancelled=None):
        self._check()
        if not isinstance(session, OlmoeSegmentSession) or session._engine is not self:
            raise ValueError("session belongs to another engine")
        with session._lock:
            return self._forward_locked(session, hidden, cancelled=cancelled)

    def _forward_locked(self, session, hidden, *, cancelled=None):
        session._check()
        x = _hidden(hidden, self.config.hidden_size)
        stop = session.position + len(x)
        if stop > session.context_tokens:
            raise ValueError("context capacity exceeded")
        _cancel(cancelled)
        # Attention writes only to private staging arrays; errors never mutate KV.
        staging = SimpleNamespace(config=self.config,
                                  _k=[a[:, :stop].copy() for a in session._k],
                                  _v=[a[:, :stop].copy() for a in session._v])
        for local, layer in enumerate(self.layers):
            normalized = _rms_norm(x, layer['input_norm'], self.config.rms_norm_eps)
            x += OlmoeModel._attention(staging, local, layer, normalized, session.position)
            normalized = _rms_norm(x, layer['post_norm'], self.config.rms_norm_eps)
            probabilities = _softmax(normalized @ layer['router'].T)
            ids = np.argsort(-probabilities, axis=1, kind='stable')[:, :self.config.num_experts_per_tok]
            weights = np.take_along_axis(probabilities, ids, axis=1)
            if self.config.norm_topk_prob:
                weights = weights / weights.sum(axis=1, keepdims=True)
            routes = [list(zip(map(str, row), map(float, weight))) for row, weight in zip(ids, weights)]
            x += self.experts.execute_batch(f'model.layers.{self.begin + local}', normalized, routes)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite range output")
        _cancel(cancelled)
        for targets, sources in ((session._k, staging._k), (session._v, staging._v)):
            for target, source in zip(targets, sources):
                target[:, session.position:stop] = source[:, session.position:stop]
        session.position = stop
        return x

    def status(self):
        return dict(range=(self.begin, self.end), numeric_class=self.numeric_class,
                    native_range_execution=False, range_resident_weights=True,
                    live_sessions=len(self._sessions), closed=self._closed)

    def close(self):
        if self._sessions:
            raise RuntimeError("close live sessions before closing engine")
        if not self._closed:
            self.experts.close()
            self.layers.clear()
            self._index.clear()
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()


class OlmoeEdge:
    """Embedding and final-head weights only; no layers, experts, or KV."""
    _tensor = OlmoeModel._tensor
    native_range_execution = False

    def __init__(self, model_directory):
        self.root = Path(model_directory).resolve(strict=True)
        self.config = OlmoeConfig.read(self.root)
        self._index = _scan_directory(self.root)
        validate_olmoe_tensors(self.config, self._index)
        shape = (self.config.vocab_size, self.config.hidden_size)
        self.embed = self._tensor('model.embed_tokens.weight', shape)
        self.lm_head = self._tensor('lm_head.weight', shape)
        self.final_norm = self._tensor('model.norm.weight', (self.config.hidden_size,))
        self._closed = False

    def _check(self):
        if self._closed:
            raise RuntimeError("edge is closed")

    def embeddings(self, token_ids):
        self._check()
        ids = _token_vector(token_ids)
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("token outside vocabulary")
        return self.embed[ids.astype(np.int64)].copy()

    def final_head(self, hidden):
        self._check()
        x = _hidden(hidden, self.config.hidden_size)
        with np.errstate(over='ignore', invalid='ignore'):
            logits = _rms_norm(x, self.final_norm, self.config.rms_norm_eps) @ self.lm_head.T
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite final logits")
        return logits

    def close(self):
        if not self._closed:
            self.embed = self.lm_head = self.final_norm = None
            self._index.clear()
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
