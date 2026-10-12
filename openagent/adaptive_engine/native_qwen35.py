"""Exact Qwen3.5 text decoder: all transformer arithmetic executes in C.

NumPy owns validated buffers only. Explicit caller-built library; no fallback,
JIT, dependency upgrade, reference imports or upstream Segment ABI claims.
"""
import ctypes as C
import _ctypes
import os
import weakref
from dataclasses import dataclass
from uuid import uuid4
from numbers import Real
import threading
from types import MappingProxyType, SimpleNamespace

import numpy as np

from .independent_qwen35 import IndependentQwen35TextDecoder
from .native_library_identity import NativeLibraryIdentity

ExpertFFN = C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_uint64,C.c_uint64,
                       C.POINTER(C.c_float),C.c_uint64,C.POINTER(C.c_float),C.c_uint64)


class Tensor(C.Structure):
    _fields_ = [('data', C.c_void_p), ('count', C.c_uint64)]


class Config(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
        ('hidden', 'heads', 'kv', 'dim', 'key_heads', 'value_heads', 'key_dim', 'value_dim',
         'kernel', 'experts', 'topk', 'intermediate', 'shared', 'capacity', 'rotary')] + [
        ('epsilon', C.c_double), ('theta', C.c_double)]


class Layer(C.Structure):
    _fields_ = [('kind', C.c_uint64), ('w', Tensor*19), ('left', Tensor), ('right', Tensor)]


def tensor(a):
    return Tensor(a.ctypes.data, a.size)


@dataclass(frozen=True)
class NativeQwen35Snapshot:
    identity: str
    owner: str
    key: str
    position: int


class NativeQwen35TextDecoder(IndependentQwen35TextDecoder):
    """Resident CPU FP32 exact text subset, with transactional cache commits.

    Inherits tensor validation and greedy control flow only, never NumPy forward.
    Native ranges are private family kernels, not Segment execution/ABI support.
    """
    implementation = 'c-qwen35-fp32-double-dot'
    native_range_execution = True
    routing_observation_supported = True
    upstream_abi_compatible = False

    def __init__(self, config, tensors, *, library, max_bytes=256*1024*1024,
                 layer_range=None,_edge_only=False,expert_provider=None,max_snapshots=64,context_tokens=None):
        declared=config.get('max_position_embeddings')
        self.capacity=declared if context_tokens is None else context_tokens
        if (type(declared) is not int or not 1<=declared<=2**63-1 or type(self.capacity) is not int or
                not 1<=self.capacity<=min(declared,65536)):
            raise ValueError('explicit native Qwen3.5 physical context of at most 65536 tokens required')
        if type(max_bytes) is not int or not 0 < max_bytes <= np.iinfo(np.intp).max:
            raise ValueError('invalid max_bytes')
        self.max_bytes = max_bytes
        if type(max_snapshots) is not int or not 1<=max_snapshots<=64:
            raise ValueError('bounded Qwen3.5 snapshot inventory required')
        self.max_snapshots=max_snapshots;self._snapshots={};self._snapshot_bytes=0;self._owner=uuid4().hex
        arrays = [np.asarray(value) for value in tensors.values()]
        weight_admission = sum(max(value.nbytes, value.size*4) for value in arrays)
        if layer_range is None and config.get('tie_word_embeddings', False) and 'lm_head.weight' not in tensors:
            embedding = np.asarray(tensors.get('model.embed_tokens.weight'))
            weight_admission += max(embedding.nbytes, embedding.size*4)
        self._check_bytes(weight_admission*2)
        del arrays
        self._lock = threading.RLock()
        self._closed,self._transport = False,False
        self.edge_only = _edge_only
        self._expert_provider,self._expert_identity,self._expert_error = expert_provider,None,None
        self._expert_bytes = 0
        self._executing = False
        self._expert_callback = ExpertFFN()
        if expert_provider is not None:
            from .qwen35_streamed_experts import Qwen35StreamedExperts,qwen35_metadata_fingerprint
            from .qwen35_publisher_experts import Qwen35PublisherExperts
            fingerprint=(expert_provider.fingerprint if type(expert_provider) is Qwen35PublisherExperts else
                expert_provider.runtime.manifest.fingerprint if type(expert_provider) is Qwen35StreamedExperts else None)
            if (_edge_only or type(expert_provider) not in (Qwen35StreamedExperts,Qwen35PublisherExperts) or
                    expert_provider.hidden_size!=config['hidden_size'] or
                    expert_provider.intermediate_size!=config['moe_intermediate_size'] or
                    expert_provider.expert_count!=config['num_experts'] or
                    expert_provider.selected_experts!=config['num_experts_per_tok'] or
                    len(expert_provider.layer_ids)!=config['num_hidden_layers'] or
                    fingerprint!=qwen35_metadata_fingerprint(config)):
                raise ValueError('exact Qwen3.5 packed provider geometry/fingerprint required')
            self._expert_identity = expert_provider.identity
            self._expert_bytes = expert_provider.max_bytes
        self._check_bytes(weight_admission*2+self._expert_bytes)
        super().__init__(config, tensors,layer_range=layer_range,_edge_only=_edge_only,
                         _omit_experts=expert_provider is not None)
        self._weight_owners = tuple(self.weights.values())
        self.weights = MappingProxyType(self.weights)
        c = self.config
        self._native_config = Config(c.hidden_size, c.num_attention_heads, c.num_key_value_heads,
            c.head_dim, c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim,
            c.linear_value_head_dim, c.linear_conv_kernel_dim, c.num_experts, c.num_experts_per_tok,
            c.moe_intermediate_size,c.shared_expert_intermediate_size,self.capacity,
            self.rotary_dim, c.rms_norm_eps, self.theta)
        limits = [4096, 128, 128, 4096, 128, 128, 4096, 4096, 64, 4096, 4096, 65536, 65536, 65536, 4096]
        if any(getattr(self._native_config, name) > limit for (name, _), limit in zip(Config._fields_[:15], limits)):
            raise ValueError('geometry exceeds bounded native kernel limits')
        if c.vocab_size > 1048576 or any(v.size > np.iinfo(np.intp).max//4 for v in self.weights.values()):
            raise ValueError('weights exceed bounded native kernel limits')
        self._library_identity=NativeLibraryIdentity.read(library)
        self.library = C.CDLL(str(self._library_identity.path))
        release = _ctypes.FreeLibrary if os.name=='nt' else _ctypes.dlclose
        self._release = weakref.finalize(self,release,self.library._handle)
        self._library_identity.assert_unchanged()
        self.library.oa_qwen35_version.argtypes = []
        self.library.oa_qwen35_version.restype = C.c_uint64
        if self.library.oa_qwen35_version() != 1:
            raise ValueError('unsupported native Qwen3.5 ABI version')
        self._embed = self.library.oa_qwen35_embed
        self._embed.argtypes = [Tensor, C.c_uint64, C.c_uint64, C.c_void_p, C.c_uint64, Tensor]
        self._head = self.library.oa_qwen35_head
        self._head.argtypes = [Tensor, Tensor, Tensor, C.c_uint64, C.c_uint64, C.c_double, Tensor, Tensor]
        self._layer = self.library.oa_qwen35_layer
        self._layer.argtypes = [C.POINTER(Config), C.POINTER(Layer), C.c_uint64, C.c_uint64, C.c_uint64,
            C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64, Tensor, Tensor, C.c_void_p,
            C.c_uint64, Tensor, Tensor]
        self._streamed_layer = None
        self._mrope_layer=None
        self._context_layer=None
        if self.capacity != c.max_position_embeddings:
            version=self.library.oa_qwen35_context_version
            version.argtypes,version.restype=[],C.c_uint64
            if version()!=1:raise ValueError('unsupported additive Qwen3.5 physical-context ABI')
            self._context_layer=self.library.oa_qwen35_layer_context
            self._context_layer.argtypes=self._layer.argtypes+[C.POINTER(C.c_uint64),ExpertFFN,C.c_void_p,C.c_uint64,C.c_uint64]
            self._context_layer.restype=C.c_int
        if self.mrope_section is not None:
            version=self.library.oa_qwen35_mrope_version
            version.argtypes,version.restype=[],C.c_uint64
            if version()!=1:raise ValueError('unsupported additive Qwen3.5 MRoPE ABI')
            self._mrope_layer=self.library.oa_qwen35_layer_mrope
            self._mrope_layer.argtypes=self._layer.argtypes+[C.POINTER(C.c_uint64),ExpertFFN,C.c_void_p,C.c_uint64]
            self._mrope_layer.restype=C.c_int
            self._mrope_binding=(C.c_uint64*3)(*self.mrope_section)
        if expert_provider is not None:
            version = self.library.oa_qwen35_streamed_version
            version.argtypes,version.restype = [],C.c_uint64
            if version()!=1:
                raise ValueError('unsupported additive Qwen3.5 selected-expert ABI')
            self._streamed_layer = self.library.oa_qwen35_layer_streamed
            self._streamed_layer.argtypes = self._layer.argtypes+[ExpertFFN,C.c_void_p,C.c_uint64]
            self._streamed_layer.restype = C.c_int
            def selected_expert(user,layer,expert,incoming,count,outgoing,out_count):
                try:
                    if count!=c.hidden_size or out_count!=c.hidden_size:
                        raise ValueError('native Qwen3.5 expert callback geometry mismatch')
                    vector = np.array(np.ctypeslib.as_array(incoming,shape=(count,)),copy=True)
                    result = expert_provider.execute(int(layer),int(expert),vector)
                    if result.dtype!=np.float32 or result.shape!=(c.hidden_size,) or not np.isfinite(result).all():
                        raise ArithmeticError('invalid Qwen3.5 selected expert result')
                    C.memmove(outgoing,result.ctypes.data,result.nbytes)
                    return 0
                except BaseException as error:
                    self._expert_error = error
                    return 1
            self._expert_callback = ExpertFFN(selected_expert)
        for function in (self._embed, self._head, self._layer):
            function.restype = C.c_int
        self._bindings = []
        common = ['input_layernorm.weight', 'post_attention_layernorm.weight', 'mlp.gate.weight',
            'mlp.experts.gate_up_proj', 'mlp.experts.down_proj', 'mlp.shared_expert.gate_proj.weight',
            'mlp.shared_expert.up_proj.weight', 'mlp.shared_expert.down_proj.weight', 'mlp.shared_expert_gate.weight']
        full = ['self_attn.'+name+'.weight' for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm')]
        linear = ['linear_attn.'+name for name in ('in_proj_qkv.weight', 'in_proj_z.weight',
            'in_proj_b.weight', 'in_proj_a.weight', 'conv1d.weight', 'norm.weight', 'dt_bias', 'A_log', 'out_proj.weight')]
        for i, kind in enumerate(c.layer_types):
            if _edge_only or not self.layer_begin<=i<self.layer_end:
                self._bindings.append(None)
                continue
            prefix = f'model.layers.{i}.'
            names = common+(full if kind == 'full_attention' else linear)
            layer = Layer()
            layer.kind = int(kind == 'linear_attention')
            for slot, name in enumerate(names):
                layer.w[slot] = Tensor() if expert_provider is not None and slot in (3,4) else tensor(self.weights[prefix+name])
            self._bindings.append(layer)
        import hashlib
        import json
        digest = hashlib.sha256(json.dumps(dict(config),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        digest.update(repr((self.layer_begin,self.layer_end,self.range_mode,self.edge_only)).encode())
        if self.capacity!=c.max_position_embeddings:digest.update(('physical-context:'+str(self.capacity)).encode())
        if self._expert_identity is not None:
            digest.update(self._expert_identity.encode())
        for name,value in sorted(self.weights.items()):
            digest.update(name.encode())
            digest.update(memoryview(value).cast('B'))
        self._library_identity.assert_unchanged()
        with self._library_identity.path.open('rb') as source:
            for data in iter(lambda: source.read(1024*1024),b''):
                digest.update(data)
        self._library_identity.assert_unchanged()
        self.identity = 'openagent-native-qwen35-v1:'+digest.hexdigest()

    def _check_open(self):
        if self._closed or self._transport or self._executing:
            raise RuntimeError('native Qwen3.5 owner closed or cache transport active')
        self._library_identity.assert_unchanged()

    def close(self):
        with self._lock:
            if self._transport or self._executing:
                raise RuntimeError('cannot close Qwen3.5 during cache transport')
            self._closed = True
            self.cache,self.mask,self.position = [],None,0
            self._bindings = []
            self.weights = MappingProxyType({})
            self._weight_owners = ()
            self._embed = self._head = self._layer = None
            self._streamed_layer = None
            self._mrope_layer=None
            self._context_layer=None
            self._snapshots.clear();self._snapshot_bytes=0
            self._expert_callback = ExpertFFN()
            self._expert_provider = None
            self.library = None
            release = getattr(self,'_release',None)
            if release is not None and release.alive:
                release()
            self._borrowed_library_owner=None

    def write_cache(self,output,*,max_bytes):
        from .qwen35_cache_recovery import write_qwen35_cache
        return write_qwen35_cache(self,output,max_bytes=max_bytes)

    def restore_cache(self,source,*,max_bytes):
        from .qwen35_cache_recovery import restore_qwen35_cache
        return restore_qwen35_cache(self,source,max_bytes=max_bytes)

    def save_cache(self,destination,*,max_bytes):
        from .cache_file_recovery import save_executor_cache
        return save_executor_cache(self,destination,max_bytes=max_bytes)

    def load_cache(self,source,*,max_bytes):
        from .cache_file_recovery import load_executor_cache
        return load_executor_cache(self,source,max_bytes=max_bytes)

    def _check_bytes(self, size):
        if size + self._snapshot_bytes > self.max_bytes:
            raise ValueError('native decoder memory budget exceeded')

    def snapshot(self):
        with self._lock:
            self._check_open()
            if len(self._snapshots)>=self.max_snapshots:raise MemoryError('Qwen3.5 snapshot inventory exhausted')
            arrays=[value for bank in self.cache if bank is not None for value in bank]
            extent=sum(value.nbytes for value in arrays)+(0 if self.mask is None else self.mask.nbytes)
            self._check_bytes(sum(value.nbytes for value in self.weights.values())+3*extent+self._expert_bytes)
            def freeze(value):return np.frombuffer(value.tobytes(order='C'),dtype=value.dtype).reshape(value.shape)
            cache=tuple(None if bank is None else tuple(freeze(value) for value in bank) for bank in self.cache)
            mask=None if self.mask is None else freeze(self.mask)
            receipt=NativeQwen35Snapshot(self.identity,self._owner,uuid4().hex,self.position)
            self._snapshots[receipt.key]=(receipt,cache,mask,extent);self._snapshot_bytes+=extent
            return receipt

    def _snapshot(self,receipt):
        if type(receipt) is not NativeQwen35Snapshot or receipt.identity!=self.identity or receipt.owner!=self._owner:
            raise ValueError('foreign Qwen3.5 snapshot')
        value=self._snapshots.get(receipt.key)
        if value is None or value[0] is not receipt:raise ValueError('released or forged Qwen3.5 snapshot')
        return value

    def restore(self,receipt):
        with self._lock:
            self._check_open();_,cache,mask,extent=self._snapshot(receipt)
            current=sum(value.nbytes for bank in self.cache if bank is not None for value in bank)
            current+=0 if self.mask is None else self.mask.nbytes
            self._check_bytes(sum(value.nbytes for value in self.weights.values())+current+extent+self._expert_bytes)
            staged=[None if bank is None else tuple(value.copy() for value in bank) for bank in cache]
            staged_mask=None if mask is None else mask.copy()
            self.cache,self.mask,self.position=staged,staged_mask,receipt.position

    def release_snapshot(self,receipt):
        with self._lock:
            self._check_open();state=self._snapshot(receipt)
            del self._snapshots[receipt.key];self._snapshot_bytes-=state[3]

    @staticmethod
    def _check(status):
        if status:
            raise ValueError(f'native Qwen3.5 execution status {status}')

    def reset(self):
        with self._lock:
            self._check_open()
            self.position = 0
            self.cache = [None]*self.config.num_hidden_layers
            self.mask = None

    def status(self):
        with self._lock:
            return dict(model_type='qwen3_5_moe_text', implementation=self.implementation,
                native_range_execution=True, upstream_abi_compatible=False, position=self.position,
                native_streamed_range_execution=self._expert_provider is not None,
                physical_context_tokens=self.capacity,publisher_context_tokens=self.config.max_position_embeddings,
                snapshot_bytes=self._snapshot_bytes,retained_snapshots=len(self._snapshots),
                resident_weight_bytes=sum(v.nbytes for v in self.weights.values()), max_bytes=self.max_bytes)

    def forward(self, token_ids, *, attention_mask=None, position_ids=None,inputs_embeds=None,cancelled=None,capture_routing=False):
        with self._lock:
            if type(capture_routing) is not bool:raise ValueError('explicit Qwen3.5 routing observation policy required')
            return self._execute(token_ids,attention_mask,position_ids,inputs_embeds,cancelled=cancelled)

    def generate_ids(self, prompt_ids, *, max_tokens):
        with self._lock:
            return super().generate_ids(prompt_ids, max_tokens=max_tokens)

    def _forward(self, token_ids, attention_mask, position_ids):
        return self._execute(token_ids,attention_mask,position_ids,None)

    def _execute(self, token_ids, attention_mask, position_ids, incoming_hidden, cancelled=None):
        self._check_open()
        def cancellation_boundary():
            if cancelled is not None:
                if not callable(cancelled):
                    raise TypeError('callable Qwen3.5 cancellation required')
                self._executing = True
                try:
                    if cancelled():
                        raise InterruptedError('Qwen3.5 cancelled; continuation unchanged')
                finally:
                    self._executing = False
        cancellation_boundary()
        if self.edge_only:
            raise TypeError('edge owner requires embedding/head entry points')
        c = self.config
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= c.vocab_size):
            raise ValueError('invalid token_ids')
        b, t = ids.shape
        end = self.position+t
        if b>128 or end>self.capacity:
            raise ValueError('native batch or context capacity exceeded')
        h, rows = c.hidden_size, b*t
        aw, kw = c.num_attention_heads*c.head_dim, c.linear_num_key_heads*c.linear_key_head_dim
        vw = c.linear_num_value_heads*c.linear_value_head_dim
        width = 2*kw+vw
        cache_shapes = [((b,c.num_key_value_heads,self.capacity,c.head_dim),)*2
            if kind == 'full_attention' else ((b, width, c.linear_conv_kernel_dim),
            (b, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim)) for kind in c.layer_types]
        cache_bytes = sum(int(np.prod(shape, dtype=object))*4 for index,shapes in enumerate(cache_shapes)
                          if self.layer_begin<=index<self.layer_end for shape in shapes)
        hidden_bytes = (c.num_hidden_layers+3)*rows*h*4
        diagnostic_bytes = c.num_hidden_layers*rows*(c.num_experts*4+c.num_experts_per_tok*12)
        attention_bytes = c.layer_types.count('full_attention')*b*c.num_attention_heads*t*end*4
        scratch_bytes = (rows*(h+2*aw+2*c.num_key_value_heads*c.head_dim)+width+
            max(aw, 2*vw, h)+h+1+2*c.moe_intermediate_size+2*c.shared_expert_intermediate_size+
            c.num_experts+2*c.linear_num_value_heads+2*c.linear_key_head_dim+2*c.linear_value_head_dim)*4
        # Keep admission ahead of default masks/positions and account for the
        # incoming mask, its committed copy and the previous cached mask.
        previous_mask_bytes = 0 if self.mask is None else self.mask.nbytes
        mask_bytes = 2*b*end + previous_mask_bytes
        validation_bytes = 16*min(b*end, 65536)
        position_seed_bytes = t*8 if position_ids is None else 0
        total_bytes = (sum(v.nbytes for v in self.weights.values())+cache_bytes*2+
            hidden_bytes+diagnostic_bytes+attention_bytes+scratch_bytes+rows*c.vocab_size*4+
            rows*(32 if self.mrope_section is not None else 16)+mask_bytes+validation_bytes+position_seed_bytes+self._expert_bytes)
        self._check_bytes(total_bytes)
        if scratch_bytes > 67108864*4 or rows > 1048576:
            raise ValueError('native scratch or token limit exceeded')
        if attention_mask is None:
            mask = np.ones((b, end), np.uint8)
        elif isinstance(attention_mask, np.ndarray):
            mask = attention_mask
        elif isinstance(attention_mask, (list, tuple)):
            if len(attention_mask) != b or any(
                    not isinstance(row, (list, tuple)) or len(row) != end for row in attention_mask):
                raise ValueError('invalid attention_mask')
            mask = np.empty((b, end), np.uint8)
            for row_index, row in enumerate(attention_mask):
                for column, value in enumerate(row):
                    if not isinstance(value, (Real, np.bool_)) or value not in (0, 1):
                        raise ValueError('invalid attention_mask')
                    mask[row_index, column] = value
        else:
            raise ValueError('attention_mask must be an ndarray or nested numeric sequence')
        if mask.shape != (b, end) or mask.dtype.kind not in 'biuf':
            raise ValueError('invalid attention_mask')
        chunks = np.nditer(mask, flags=['external_loop', 'buffered', 'zerosize_ok'],
                           op_flags=['readonly'], buffersize=65536)
        valid_mask = not any(np.any((chunk != 0) & (chunk != 1)) for chunk in chunks)
        del chunks
        if not valid_mask:
            raise ValueError('invalid attention_mask')
        if self.mask is not None and not np.array_equal(self.mask, mask[:, :self.position]):
            raise ValueError('cached mask or batch changed')
        positions = np.broadcast_to(np.arange(self.position, end), ids.shape) if position_ids is None else np.asarray(position_ids)
        spatial=positions.shape==(3,b,t)
        if ((positions.shape != ids.shape and not (spatial and self.mrope_section is not None)) or
                positions.dtype.kind not in 'iu' or np.any(positions<0) or np.any(positions>=c.max_position_embeddings)):
            raise ValueError('invalid text position_ids')
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        positions = np.ascontiguousarray(positions, dtype=np.int64)
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        if self.range_mode or incoming_hidden is not None:
            value = np.asarray(incoming_hidden)
            if value.dtype!=np.float32 or value.shape!=(b,t,h) or not np.isfinite(value).all():
                raise ValueError('finite FP32 Qwen3.5 range hidden rows required')
            x = np.array(value,dtype=np.float32,order='C',copy=True)
        else:
            x = np.empty((b, t, h), np.float32)
            self._check(self._embed(tensor(self.weights['model.embed_tokens.weight']), c.vocab_size, h,
                                    ids.ctypes.data, rows, tensor(x)))
        cache, hidden, routers, routes, attentions = [], [], [], [], []
        cache = [None]*c.num_hidden_layers
        for i, (kind, shapes, binding) in enumerate(zip(c.layer_types, cache_shapes, self._bindings)):
            if not self.layer_begin<=i<self.layer_end:
                continue
            cancellation_boundary()
            old = self.cache[i]
            if old is not None and (not isinstance(old, tuple) or len(old) != 2 or any(
                    not isinstance(a, np.ndarray) or a.dtype != np.dtype('float32') or
                    a.shape != shape or not a.flags.c_contiguous for a, shape in zip(old, shapes))):
                raise ValueError('invalid native cache buffers; reset required')
            states = tuple(np.zeros(shape, np.float32) for shape in shapes) if old is None else tuple(a.copy() for a in old)
            layer = Layer.from_buffer_copy(binding)
            layer.left, layer.right = (tensor(a) for a in states)
            hidden.append(x.copy())
            router = np.empty((rows, c.num_experts), np.float32)
            indices = np.empty((rows, c.num_experts_per_tok), np.int64)
            routing = np.empty(indices.shape, np.float32)
            attention = np.empty((b, c.num_attention_heads, t, end), np.float32) if kind == 'full_attention' else np.empty(0, np.float32)
            args = (C.byref(self._native_config),C.byref(layer),b,t,self.position,
                mask.ctypes.data,mask.size,positions.ctypes.data,positions.size,tensor(x),tensor(router),
                indices.ctypes.data,indices.size,tensor(routing),tensor(attention))
            self._executing = True
            try:
                if self._context_layer is not None:
                    if self._expert_provider is not None and self._expert_provider.identity!=self._expert_identity:
                        raise ValueError('Qwen3.5 packed provider changed after admission')
                    self._expert_error=None
                    status=self._context_layer(*args,self._mrope_binding if spatial else None,
                        self._expert_callback,None,i,c.max_position_embeddings)
                    if status and self._expert_error is not None:
                        error,self._expert_error=self._expert_error,None
                        raise RuntimeError('Qwen3.5 context selected expert failed; continuation unchanged') from error
                elif spatial:
                    if self._expert_provider is not None and self._expert_provider.identity!=self._expert_identity:
                        raise ValueError('Qwen3.5 packed provider changed after admission')
                    self._expert_error=None
                    status=self._mrope_layer(*args,self._mrope_binding,self._expert_callback,None,i)
                    if status and self._expert_error is not None:
                        error,self._expert_error=self._expert_error,None
                        raise RuntimeError('Qwen3.5 MRoPE selected expert failed; continuation unchanged') from error
                elif self._expert_provider is None:
                    status = self._layer(*args)
                else:
                    if self._expert_provider.identity!=self._expert_identity:
                        raise ValueError('Qwen3.5 packed provider changed after admission')
                    self._expert_error = None
                    status = self._streamed_layer(*args,self._expert_callback,None,i)
                    if status and self._expert_error is not None:
                        error,self._expert_error = self._expert_error,None
                        raise RuntimeError('Qwen3.5 selected expert failed; continuation unchanged') from error
                self._check(status)
            finally:
                self._executing = False
            cache[i] = states
            routers.append(router)
            routes.append((indices, routing))
            if kind == 'full_attention':
                attentions.append(attention)
        if self.range_mode:
            if not np.isfinite(x).all() or any(not np.isfinite(value).all() for bank in cache if bank for value in bank):
                raise ArithmeticError('nonfinite Qwen3.5 range continuation')
            cancellation_boundary()
            self.cache,self.mask,self.position = cache,mask.copy(),end
            return SimpleNamespace(hidden=x,hidden_states=tuple(hidden),router_logits=tuple(routers),
                                   routes=tuple(routes),attentions=tuple(attentions))
        normalized = np.empty_like(x)
        logits = np.empty((b, t, c.vocab_size), np.float32)
        self._check(self._head(tensor(x), tensor(self.weights['model.norm.weight']), tensor(self.weights['lm_head.weight']),
                              h, c.vocab_size, c.rms_norm_eps, tensor(normalized), tensor(logits)))
        hidden.append(normalized)
        if not np.isfinite(logits).all() or any(not np.isfinite(value).all() for bank in cache if bank for value in bank):
            raise ArithmeticError('nonfinite Qwen3.5 logits or continuation')
        cancellation_boundary()
        self.cache, self.mask, self.position = cache, mask.copy(), end
        return SimpleNamespace(logits=logits, hidden_states=tuple(hidden), router_logits=tuple(routers),
                               routes=tuple(routes), attentions=tuple(attentions))
