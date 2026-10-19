"""Exact native DSA resident or selected packed-expert source composition."""
import ctypes as C
import _ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from threading import RLock, local
from contextlib import contextmanager
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4
import weakref

import numpy as np

from .independent_glm_dsa import IndependentGlmDsaDecoder


class Tensor(C.Structure):
    _fields_ = [('data', C.POINTER(C.c_float)), ('count', C.c_uint64)]


class Config(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ('hidden', 'heads', 'qr', 'kr', 'nope', 'rope',
        'value', 'idim', 'iheads', 'itopk', 'experts', 'topk', 'groups', 'gtop',
        'dense_width', 'moe_width', 'shared_width', 'capacity', 'normalize')]
    _fields_ += [(n, C.c_double) for n in ('epsilon', 'theta', 'route_scale')]
    _fields_ += [(n, C.c_uint64) for n in ('yarn', 'original_context', 'truncate')]
    _fields_ += [(n, C.c_double) for n in ('factor', 'beta_fast', 'beta_slow',
                                         'mscale', 'mscale_all_dim', 'attention_factor')]


class Layer(C.Structure):
    _fields_ = [('full_indexer', C.c_uint64), ('dense', C.c_uint64),
                ('w', Tensor*21), ('latent', Tensor), ('rotary', Tensor), ('index_keys', Tensor)]

ExpertCallback = C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_uint64,
    C.POINTER(C.c_float),C.c_uint64,C.POINTER(C.c_float),C.c_uint64)


def tensor(a):
    if a is None:
        return Tensor()
    if a.dtype != np.float32 or not a.flags.c_contiguous:
        raise ValueError('C tensors require contiguous FP32')
    return Tensor(a.ctypes.data_as(C.POINTER(C.c_float)), a.size)


def integers(a):
    return a.ctypes.data_as(C.POINTER(C.c_int64)) if a is not None else None


@dataclass(frozen=True)
class GlmDsaSnapshot:
    identity: str
    owner: str
    snapshot_id: str
    position: int


class NativeGlmDsaDecoder(IndependentGlmDsaDecoder):
    implementation = 'compiled-c-glm-dsa-compressed-latent'
    native_range_execution = True
    native_streamed_range_execution = False

    @classmethod
    def from_streamed_directory(cls,directory,*,library,expert_provider,max_weight_bytes,
                                max_bytes,max_snapshots=4,retain_diagnostics=False,
                                read_limits=None,cancel=None):
        from .native_checkpoint import read_native_text_checkpoint,_finish_decoder
        if expert_provider is not None:
            raise ValueError('GLM-DSA packed experts are disabled until source-content identity is bound')
        from .native_glm_dsa_ranges import admit_inventory
        from .glm_dsa_streamed_experts import config_identity
        def select(config,records):
            _,shapes = admit_inventory(config,records)
            if expert_provider.config_digest != config_identity(config):
                raise ValueError('DSA expert manifest configuration differs from checkpoint')
            return tuple(sorted(name for name in shapes if not name.endswith(
                ('mlp.experts.gate_up_proj','mlp.experts.down_proj'))))
        config,weights = read_native_text_checkpoint(directory,model_type='glm_moe_dsa',
            max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,
            tensor_selection=select,source_backend='native')
        return _finish_decoder(cls(config,weights,library=library,max_bytes=max_bytes,
            max_snapshots=max_snapshots,retain_diagnostics=retain_diagnostics,
            expert_provider=expert_provider),cancel)

    def __init__(self, config, tensors, *, library, max_bytes=512*1024*1024,
                 max_snapshots=4, retain_diagnostics=True, _hidden_only=False,
                 expert_provider=None, _expert_layer_offset=0):
        self._lock = RLock()
        self._local = local()
        self._busy = False
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('invalid max_bytes')
        if expert_provider is not None:
            raise ValueError('GLM-DSA packed experts require verified checkpoint source-content identity')
        self.max_bytes = max_bytes
        if type(retain_diagnostics) is not bool:
            raise ValueError('retain_diagnostics must be boolean')
        self.retain_diagnostics = retain_diagnostics
        if type(max_snapshots) is not int or not 1 <= max_snapshots <= 64:
            raise ValueError('GLM snapshot inventory must be between one and 64')
        self.max_snapshots = max_snapshots
        self._closed = False
        self._owner = uuid4().hex
        self._snapshots = {}
        self._snapshot_bytes = 0
        self._release = None
        if type(_hidden_only) is not bool:
            raise TypeError('boolean native GLM hidden execution policy required')
        self._hidden_only = _hidden_only
        self.source_config_digest = hashlib.sha256(json.dumps(dict(config),
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self._expert_provider = expert_provider
        self._expert_layer_offset = _expert_layer_offset
        if type(_expert_layer_offset) is not int or _expert_layer_offset < 0:
            raise ValueError('nonnegative original DSA expert layer offset required')
        from .native_library_identity import NativeLibraryIdentity
        self._library_identity=NativeLibraryIdentity.read(library._name if isinstance(library,C.CDLL) else library)
        super().__init__(config, tensors, max_weight_bytes=max_bytes,
                         _inventory_only=_hidden_only or expert_provider is not None)
        if _hidden_only or expert_provider is not None:
            shapes = {name:shape for name,shape in self.tensor_shapes.items()
                      if not _hidden_only or name.startswith('model.layers.')}
            if expert_provider is not None:
                from .glm_dsa_streamed_experts import GlmDsaStreamedExperts
                if (type(expert_provider) is not GlmDsaStreamedExperts
                        or expert_provider.hidden_size != self.config.hidden_size
                        or expert_provider.intermediate_size != self.config.moe_intermediate_size
                        or expert_provider.expert_count != self.config.n_routed_experts
                        or expert_provider.selected_experts != self.config.num_experts_per_tok
                        or (_expert_layer_offset == 0 and not _hidden_only
                            and expert_provider.config_digest != self.source_config_digest)):
                    raise ValueError('exact DSA streamed expert owner required')
                expected = {_expert_layer_offset+i for i,kind in enumerate(self.config.mlp_layer_types) if kind=='sparse'}
                if not expected.issubset(expert_provider.sparse_layers):
                    raise ValueError('DSA streamed owner lacks selected original layers')
                self._expert_identity = expert_provider.identity
                self.implementation='compiled-c-glm-dsa-selected-packed-experts'
                shapes = {name:shape for name,shape in shapes.items() if not name.endswith(
                    ('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
                self.native_streamed_range_execution = True
            tensors = dict(tensors)
            if not _hidden_only and self.config.tie_word_embeddings:
                embedding = tensors.get('model.embed_tokens.weight')
                if 'lm_head.weight' in tensors and not np.array_equal(tensors['lm_head.weight'],embedding):
                    raise ValueError('inconsistent tied DSA head')
                tensors['lm_head.weight'] = embedding
            if set(tensors) != set(shapes):
                raise ValueError('complete selected GLM hidden layer inventory required')
            size = sum(math.prod(shape)*4 for shape in shapes.values())
            if size+max((math.prod(shape)*4 for shape in shapes.values()),default=0) > max_bytes:
                raise MemoryError('GLM hidden layer ownership exceeds admission')
            for name,shape in shapes.items():
                value = np.asarray(tensors[name])
                if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('invalid GLM hidden layer tensor: '+name)
                self.weights[name] = np.frombuffer(value.tobytes(order='C'),np.float32).reshape(shape)
            self.reset()
        self._config_identity = hashlib.sha256(json.dumps(vars(self.config),
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if not _hidden_only:
            for name,value in self.weights.items():
                self.weights[name] = np.frombuffer(value.tobytes(order='C'),np.float32).reshape(value.shape)
        self.weights = MappingProxyType(self.weights)
        c = self.config
        rope = c.rope_parameters
        self._c = Config(c.hidden_size, c.num_attention_heads, c.q_lora_rank, c.kv_lora_rank,
            c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim, c.index_head_dim,
            c.index_n_heads, c.index_topk, c.n_routed_experts, c.num_experts_per_tok,
            c.n_group, c.topk_group, c.intermediate_size, c.moe_intermediate_size,
            c.n_shared_experts*c.moe_intermediate_size, c.max_position_embeddings,
            int(c.norm_topk_prob), c.rms_norm_eps, self.theta, c.routed_scaling_factor,
            int(rope['rope_type'] == 'yarn'), rope.get('original_max_position_embeddings',1),
            int(rope.get('truncate',True)), rope.get('factor',1.), rope.get('beta_fast',32.),
            rope.get('beta_slow',1.), rope.get('mscale',0.), rope.get('mscale_all_dim',0.),
            rope.get('attention_factor') or 0.)
        digest = hashlib.sha256(json.dumps(dict(
            geometry={name:getattr(self._c, name) for name, _ in Config._fields_},
            mlp_layer_types=c.mlp_layer_types, indexer_types=c.indexer_types,
            hidden_only=_hidden_only),
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii'))
        for name, value in sorted(self.weights.items()):
            digest.update(name.encode('utf-8'))
            digest.update(memoryview(value).cast('B'))
        if expert_provider is not None:
            digest.update(self._expert_identity.encode())
            digest.update(str(_expert_layer_offset).encode())
        self.identity = 'oa-glm-dsa-fp32-v2:'+digest.hexdigest()
        self._check_bytes(1, 1)
        borrowed = isinstance(library, C.CDLL)
        self._library_identity.assert_unchanged()
        self.library = library if borrowed else C.CDLL(str(self._library_identity.path))
        if not borrowed:
            release = _ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose
            self._release = weakref.finalize(self, release, self.library._handle)
        try:
            self._library_identity.assert_unchanged()
            self.identity+=':library:'+self._library_identity.sha256
            self._bind_native()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note('GLM library cleanup failed: '+str(cleanup_error))
            raise

    def _bind_native(self):
        lib = self.library
        lib.oa_glm_dsa_version.restype = C.c_uint64
        lib.oa_glm_dsa_version.argtypes = []
        if lib.oa_glm_dsa_version() != 2:
            raise ValueError('unsupported private GLM ABI')
        if not self.retain_diagnostics:
            capability = getattr(lib, 'oa_glm_dsa_optional_diagnostics_version', None)
            if capability is None:
                raise ValueError('native GLM library lacks optional-diagnostics execution')
            capability.argtypes, capability.restype = [], C.c_uint64
            if capability() != 1:
                raise ValueError('unsupported GLM optional-diagnostics contract')
        u, ip = C.c_uint64, C.POINTER(C.c_int64)
        lib.oa_glm_dsa_geometry.argtypes = [C.POINTER(Config)]
        lib.oa_glm_dsa_geometry.restype = C.c_int
        if lib.oa_glm_dsa_geometry(C.byref(self._c)):
            raise ValueError('unsupported bounded C geometry')
        lib.oa_glm_dsa_embed.argtypes = [Tensor, u, u, ip, u, Tensor]
        lib.oa_glm_dsa_head.argtypes = [Tensor, Tensor, Tensor, u, u, C.c_double, Tensor, Tensor]
        lib.oa_glm_dsa_layer.argtypes = [C.POINTER(Config), C.POINTER(Layer), u, u, u,
            C.POINTER(C.c_uint8), u, ip, u, Tensor, ip, u, Tensor, Tensor, ip, u, Tensor]
        for name in ('embed', 'head', 'layer'):
            getattr(lib, 'oa_glm_dsa_'+name).restype = C.c_int
        if self._expert_provider is not None:
            capability = getattr(lib,'oa_glm_dsa_streamed_experts_version',None)
            if capability is None:
                raise ValueError('native DSA library lacks selected-expert ABI')
            capability.argtypes,capability.restype = [],C.c_uint64
            if capability() != 1:
                raise ValueError('unsupported selected-expert ABI')
            lib.oa_glm_dsa_layer_streamed.argtypes = lib.oa_glm_dsa_layer.argtypes+[ExpertCallback,C.c_void_p]
            lib.oa_glm_dsa_layer_streamed.restype = C.c_int
        capability = getattr(lib, 'oa_glm_dsa_cache_expansion_version', None)
        self._expand_cache = None
        if capability is not None:
            capability.argtypes, capability.restype = [], C.c_uint64
            if capability() != 1:
                raise ValueError('unsupported native GLM cache-expansion contract')
            self._expand_cache = lib.oa_glm_dsa_expand_cache
            self._expand_cache.argtypes = [C.POINTER(Config), Tensor, Tensor, Tensor, u, u, Tensor, Tensor]
            self._expand_cache.restype = C.c_int

    def reset(self):
        with self._lock:
            self._check_open()
            super().reset()

    def _check_open(self):
        if self._closed:
            raise RuntimeError('native GLM decoder is closed')
        if self._busy:
            raise RuntimeError('native GLM transaction is active')
        self._library_identity.assert_unchanged()
        expected = getattr(self, '_config_identity', None)
        if expected is not None and hashlib.sha256(json.dumps(vars(self.config),
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() != expected:
            raise ValueError('native GLM configuration changed after admission')
        if self._expert_provider is not None and self._expert_provider.identity != self._expert_identity:
            raise ValueError('native DSA streamed expert identity changed after admission')

    def _cancel(self):
        event = getattr(self._local, 'cancel', None)
        if event is not None and event.is_set():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('native GLM transaction cancelled')

    @contextmanager
    def cancellation_scope(self, event):
        if event is not None and not callable(getattr(event, 'is_set', None)):
            raise TypeError('event-like GLM cancellation required')
        previous = getattr(self._local, 'cancel', None)
        self._local.cancel = event
        try:
            yield self
        finally:
            self._local.cancel = previous

    def write_cache(self, output, *, max_bytes):
        from .native_glm_dsa_cache import write_cache
        return write_cache(self, output, max_bytes=max_bytes)

    def restore_cache(self, source, *, max_bytes):
        from .native_glm_dsa_cache import restore_cache
        return restore_cache(self, source, max_bytes=max_bytes)

    def save_cache(self, destination, *, max_bytes):
        from .cache_file_recovery import save_executor_cache
        return save_executor_cache(self, destination, max_bytes=max_bytes)

    def load_cache(self, source, *, max_bytes):
        from .cache_file_recovery import load_executor_cache
        return load_executor_cache(self, source, max_bytes=max_bytes)

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('native GLM transaction is active')
            if not self._closed:
                self._closed = True
                self.cache = []
                self.mask = None
                self.position = 0
                self._snapshots.clear()
                self._snapshot_bytes = 0
                self.weights = MappingProxyType({})
            if self._release is not None and self._release.alive:
                _, callback, args, kwargs = self._release.detach()
                try:
                    callback(*args, **kwargs)
                except BaseException:
                    self._release = weakref.finalize(self, callback, *args, **kwargs)
                    raise
            self.library = None
            self._expand_cache = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()

    def _state_bytes(self):
        return sum(a.nbytes for state in self.cache if state for a in state if a is not None) + (
            0 if self.mask is None else self.mask.nbytes)

    def _admit_snapshot(self, staging):
        needed = sum(a.nbytes for a in self.weights.values()) + self._state_bytes() + self._snapshot_bytes + staging
        if self._expert_provider is not None:
            needed += self._expert_provider.max_bytes
        if needed > self.max_bytes:
            raise MemoryError('GLM snapshot staging exceeds admission')

    def export_cache(self, *, max_bytes):
        """Detach every active latent/rotary/indexer leaf and exact prefix mask."""
        with self._lock:
            self._check_open()
            if type(max_bytes) is not int or max_bytes <= 0:
                raise ValueError('explicit GLM physical cache observation budget required')
            batch = 0 if self.mask is None else self.mask.shape[0]
            widths = sum(self.config.kv_lora_rank+self.config.qk_rope_head_dim+
                (self.config.index_head_dim if kind=='full' else 0)
                for kind in self.config.indexer_types)
            size = 8+batch*self.position*(1+widths*4)
            if 3*size > max_bytes:
                raise MemoryError('native GLM cache observation exceeds admission')
            self._admit_snapshot(3*size)
            self._busy = True
            try:
                self._cancel()
                result = {'position':np.asarray([self.position],np.int64),
                    'mask':self.mask.copy() if batch else np.empty((0,0),np.uint8)}
                for layer,kind in enumerate(self.config.indexer_types):
                    self._cancel()
                    state = self.cache[layer]
                    for index,(name,width) in enumerate((('latent',self.config.kv_lora_rank),
                            ('rotary',self.config.qk_rope_head_dim),('indexer',self.config.index_head_dim))):
                        if index==2 and kind!='full':continue
                        result[f'layer.{layer}.{name}'] = (state[index][:,:self.position].copy()
                            if batch else np.empty((0,0,width),np.float32))
                self._cancel()
                return result
            finally:
                self._busy = False

    def snapshot(self):
        with self._lock:
            self._check_open()
            if len(self._snapshots) >= self.max_snapshots:
                raise MemoryError('GLM snapshot inventory exceeds admission')
            size = sum(a.shape[0]*self.position*a.shape[2]*a.itemsize
                for state in self.cache if state for a in state if a is not None) + (
                0 if self.mask is None else self.mask.nbytes)
            self._admit_snapshot(size)
            cache = tuple(None if state is None else tuple(
                None if a is None else a[:, :self.position].copy() for a in state) for state in self.cache)
            mask = None if self.mask is None else self.mask.copy()
            for state in cache:
                if state is not None:
                    for array in state:
                        if array is not None:
                            array.flags.writeable = False
            if mask is not None:
                mask.flags.writeable = False
            handle = GlmDsaSnapshot(self.identity, self._owner, uuid4().hex, self.position)
            self._snapshots[handle.snapshot_id] = (handle, cache, mask, size)
            self._snapshot_bytes += size
            return handle

    def _owned_snapshot(self, snapshot):
        if (type(snapshot) is not GlmDsaSnapshot or snapshot.owner != self._owner or
                snapshot.identity != self.identity):
            raise ValueError('foreign GLM snapshot')
        owned = self._snapshots.get(snapshot.snapshot_id)
        if owned is None or owned[0] is not snapshot:
            raise ValueError('unknown or released GLM snapshot')
        return owned

    def restore(self, snapshot):
        with self._lock:
            self._check_open()
            _, cache, mask, size = self._owned_snapshot(snapshot)
            restored_bytes = sum(a.shape[0]*self.config.max_position_embeddings*a.shape[2]*a.itemsize
                for state in cache if state for a in state if a is not None) + (
                0 if mask is None else mask.nbytes)
            self._admit_snapshot(restored_bytes)
            def expand(array):
                if array is None:
                    return None
                result = np.zeros((array.shape[0], self.config.max_position_embeddings,
                                   array.shape[2]), dtype=array.dtype)
                result[:, :snapshot.position] = array
                return result
            staged = [None if state is None else tuple(expand(a) for a in state) for state in cache]
            staged_mask = None if mask is None else mask.copy()
            self.cache, self.mask, self.position = staged, staged_mask, snapshot.position

    def release_snapshot(self, snapshot):
        with self._lock:
            self._check_open()
            _, _, _, size = self._owned_snapshot(snapshot)
            del self._snapshots[snapshot.snapshot_id]
            self._snapshot_bytes -= size

    def status(self):
        with self._lock:
            self._check_open()
            return dict(model_type='glm_moe_dsa', implementation=self.implementation,
                native_range_execution=True, native_streamed_range_execution=self.native_streamed_range_execution,
                cache_layout='latent+rotary+indexer', position=self.position,
                retain_diagnostics=self.retain_diagnostics)

    def _check_bytes(self, batch, tokens):
        c, k = self.config, self._c
        rows, end, layers = batch*tokens, self.position+tokens, c.num_hidden_layers
        cache = batch*c.max_position_embeddings*sum(c.kv_lora_rank+c.qk_rope_head_dim+
            (c.index_head_dim if kind == 'full' else 0) for kind in c.indexer_types)*4
        old = sum(a.nbytes for state in self.cache if state for a in state if a is not None)
        scratch = rows*(k.hidden+k.qr+k.heads*(k.nope+k.rope)+k.iheads*k.idim+k.iheads)
        scratch += 3*k.kr+2*k.rope+k.idim+k.heads*k.value+k.hidden
        scratch += 2*max(k.dense_width, 2*k.moe_width, k.shared_width)+end+k.groups+2*k.experts
        diagnostic = 4*(rows*c.hidden_size*(layers+2 if self.retain_diagnostics else 2)+
                        (0 if self._hidden_only else rows*c.vocab_size))
        if self.retain_diagnostics:
            diagnostic += 4*batch*c.num_attention_heads*tokens*end*layers
        diagnostic += 8*rows*min(c.index_topk, end)*(layers+1 if self.retain_diagnostics else 1)
        sparse = c.mlp_layer_types.count('sparse')
        diagnostic += (sparse if self.retain_diagnostics else min(1, sparse))*rows*(
            4*c.n_routed_experts+12*c.num_experts_per_tok)
        # Originals, pending cache, committed mask, conversions and C scratch coexist.
        required = sum(a.nbytes for a in self.weights.values())+old+cache+scratch*4+diagnostic
        required += rows*32+batch*end*4+(0 if self.mask is None else self.mask.nbytes)
        required += self._snapshot_bytes
        if self._expert_provider is not None:
            required += self._expert_provider.max_bytes
        if required > self.max_bytes:
            raise MemoryError(f'GLM resident call needs {required} bytes; limit {self.max_bytes}')

    def _layer(self, i, state):
        c, p = self.config, f'model.layers.{i}.'
        names = ['input_layernorm.weight', 'post_attention_layernorm.weight']
        names += ['self_attn.'+n for n in ('q_a_proj.weight', 'q_a_layernorm.weight',
            'q_b_proj.weight', 'kv_a_proj_with_mqa.weight', 'kv_a_layernorm.weight',
            'kv_b_proj.weight', 'o_proj.weight')]
        names += ['self_attn.indexer.'+n for n in ('wq_b.weight', 'wk.weight',
            'k_norm.weight', 'k_norm.bias', 'weights_proj.weight')]
        dense = c.mlp_layer_types[i] == 'dense'
        names += (['mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']+[None]*4
            if dense else ['mlp.gate.weight', 'mlp.gate.e_score_correction_bias',
                'mlp.experts.gate_up_proj', 'mlp.experts.down_proj',
                'mlp.shared_experts.gate_proj.weight', 'mlp.shared_experts.up_proj.weight',
                'mlp.shared_experts.down_proj.weight'])
        weights = (Tensor*21)(*[tensor(self.weights.get(p+n)) if n else Tensor() for n in names])
        return Layer(int(c.indexer_types[i] == 'full'), int(dense), weights,
            *[tensor(a) for a in state])

    @staticmethod
    def _result(code):
        if code:
            raise RuntimeError(f'GLM C kernel failed ({code}); state not committed')

    def forward(self, token_ids, *, attention_mask=None, position_ids=None):
        with self._lock:
            self._check_open()
            self._busy = True
            try:
                self._cancel()
                return self._forward(token_ids, attention_mask, position_ids)
            finally:
                self._busy = False

    def forward_hidden(self, hidden, *, attention_mask=None, position_ids=None):
        with self._lock:
            self._check_open()
            if not self._hidden_only:
                raise ValueError('explicit hidden-only GLM owner required')
            values = np.asarray(hidden)
            if (values.dtype != np.float32 or values.ndim != 3 or not all(values.shape)
                    or values.shape[-1] != self.config.hidden_size or not np.isfinite(values).all()):
                raise ValueError('finite batch/token/hidden FP32 GLM rows required')
            self._busy = True
            try:
                self._cancel()
                return self._forward(np.zeros(values.shape[:2],np.int64),
                    attention_mask,position_ids,hidden=values).last_hidden_state
            finally:
                self._busy = False

    def _forward(self, token_ids, attention_mask, position_ids, hidden=None):
        c = self.config
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2 or not all(ids.shape) or ids.shape[0] > 128 or ids.dtype.kind not in 'iu':
            raise ValueError('invalid token_ids')
        b, t = ids.shape
        end = self.position+t
        if end > c.max_position_embeddings or (self.mask is not None and b != self.mask.shape[0]):
            raise ValueError('context capacity or cached batch changed')
        self._check_bytes(b, t)
        ids, mask, positions = self._inputs(ids, attention_mask, position_ids)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        positions = np.ascontiguousarray(positions, dtype=np.int64)
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        cache = []
        for i, old in enumerate(self.cache):
            if old is None:
                state = (np.zeros((b, c.max_position_embeddings, c.kv_lora_rank), np.float32),
                    np.zeros((b, c.max_position_embeddings, c.qk_rope_head_dim), np.float32),
                    np.zeros((b, c.max_position_embeddings, c.index_head_dim), np.float32)
                    if c.indexer_types[i] == 'full' else None)
            else:
                state = tuple(a.copy() if a is not None else None for a in old)
            cache.append(state)
        x = np.empty((b, t, c.hidden_size), np.float32)
        if hidden is None:
            if self._hidden_only:
                raise ValueError('GLM hidden range does not accept token input')
            self._result(self.library.oa_glm_dsa_embed(tensor(self.weights['model.embed_tokens.weight']),
                c.vocab_size, c.hidden_size, integers(ids), b*t, tensor(x)))
        else:
            x[:] = hidden
        hidden, attentions, indexers, routers, routes = [], [], [], [], []
        selected = np.empty((b, t, min(c.index_topk, end)), np.int64)
        for i, state in enumerate(cache):
            self._cancel()
            if self.retain_diagnostics:
                hidden.append(x.copy())
            layer = self._layer(i, state)
            attention = (np.empty((b, c.num_attention_heads, t, end), np.float32)
                         if self.retain_diagnostics else None)
            sparse = c.mlp_layer_types[i] == 'sparse'
            router = np.empty((b*t, c.n_routed_experts), np.float32) if sparse else None
            experts = np.empty((b*t, c.num_experts_per_tok), np.int64) if sparse else None
            weights = np.empty((b*t, c.num_experts_per_tok), np.float32) if sparse else None
            callback_errors = []
            extra = ()
            operation = self.library.oa_glm_dsa_layer
            if sparse and self._expert_provider is not None:
                def execute_expert(owner, expert, source, width, destination, count):
                    try:
                        self._cancel()
                        if width != c.hidden_size or count != width or self._expert_provider.identity != self._expert_identity:
                            raise ValueError('DSA expert binding or geometry changed')
                        value = np.ctypeslib.as_array(source,shape=(width,)).copy()
                        result = self._expert_provider.execute(i+self._expert_layer_offset,int(expert),value)
                        self._cancel()
                        np.ctypeslib.as_array(destination,shape=(count,))[:] = result
                        return 0
                    except BaseException as error:
                        callback_errors.append(error)
                        return 1
                callback = ExpertCallback(execute_expert)
                extra = (callback,None)
                operation = self.library.oa_glm_dsa_layer_streamed
            code = operation(C.byref(self._c), C.byref(layer), b, t,
                self.position, mask.ctypes.data_as(C.POINTER(C.c_uint8)), mask.size,
                integers(positions), positions.size, tensor(x), integers(selected), selected.size,
                tensor(attention), tensor(router), integers(experts), 0 if experts is None else experts.size,
                tensor(weights),*extra)
            if callback_errors:
                raise callback_errors[0]
            self._result(code)
            if self.retain_diagnostics:
                attentions.append(attention)
                indexers.append(selected.copy())
                routers.append(router)
                routes.append((experts, weights) if sparse else None)
            else:
                router = experts = weights = None
        normalized = x if self._hidden_only else np.empty_like(x)
        logits = np.empty((0,),np.float32) if self._hidden_only else np.empty((b,t,c.vocab_size),np.float32)
        if not self._hidden_only:
            self._result(self.library.oa_glm_dsa_head(tensor(x), tensor(self.weights['model.norm.weight']),
                tensor(self.weights['lm_head.weight']), c.hidden_size, c.vocab_size, c.rms_norm_eps,
                tensor(normalized), tensor(logits)))
        if self.retain_diagnostics:
            hidden.append(normalized)
        for array in (*hidden, normalized, logits, *attentions, *(a for a in routers if a is not None),
                      *(pair[1] for pair in routes if pair is not None)):
            flat = array.reshape(-1)
            for offset in range(0, flat.size, 65536):
                if not np.isfinite(flat[offset:offset+65536]).all():
                    raise FloatingPointError('nonfinite GLM result; state not committed')
        for state in cache:
            for array in state:
                if array is not None:
                    for batch_row in array[:, :end]:
                        flat = batch_row.reshape(-1)
                        for offset in range(0, flat.size, 65536):
                            if not np.isfinite(flat[offset:offset+65536]).all():
                                raise FloatingPointError('nonfinite GLM cache; state not committed')
        self._cancel()
        self.cache, self.mask, self.position = cache, mask.copy(), end
        return SimpleNamespace(logits=logits, hidden_states=tuple(hidden), attentions=tuple(attentions),
            indexer_indices=tuple(indexers), router_logits=tuple(routers), routes=tuple(routes),
            last_hidden_state=normalized)

    def generate_ids(self, prompt_ids, *, max_tokens):
        with self._lock:
            self._check_open()
            return super().generate_ids(prompt_ids, max_tokens=max_tokens)

    def expanded_cache(self, layer):
        """Return native diagnostic keys/values for the committed prefix only.

        Expansion is optional, separately admitted, and not used for generation.
        Previously returned arrays remain caller-owned and outside this budget.
        """
        with self._lock:
            self._check_open()
            if type(layer) is not int or not 0 <= layer < self.config.num_hidden_layers:
                raise ValueError('invalid GLM diagnostic layer')
            if self._expand_cache is None:
                raise ValueError('native GLM library lacks cache-expansion capability')
            state = self.cache[layer]
            if state is None:
                raise ValueError('GLM layer has no committed cache')
            latent, rotary, _ = state
            c = self.config
            shape = (latent.shape[0], c.num_attention_heads, self.position)
            staging = 4*int(np.prod(shape))*(c.qk_nope_head_dim+c.qk_rope_head_dim+c.v_head_dim)
            self._admit_snapshot(staging)
            keys = np.empty((*shape, c.qk_nope_head_dim+c.qk_rope_head_dim), np.float32)
            values = np.empty((*shape, c.v_head_dim), np.float32)
            self._result(self._expand_cache(C.byref(self._c),
                tensor(self.weights[f'model.layers.{layer}.self_attn.kv_b_proj.weight']),
                tensor(latent), tensor(rotary), latent.shape[0], self.position,
                tensor(keys), tensor(values)))
            return keys, values
