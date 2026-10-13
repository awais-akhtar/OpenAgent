"""Publisher-shaped incremental text backbone, distinct from private equations.

Arithmetic contracts: DeepSeek-V4.1-Flash revision
2cba9e42aa026125f3ed06c6d98c1db82f7ca027, inference/model.py and kernel.py.
The transport provider must implement quantized projection and query roundtrip;
this module deliberately does not substitute dequantized FP32 matrix products.
"""
import ctypes as C
from dataclasses import dataclass
import hashlib
import _ctypes
import os
import threading
import weakref
from contextlib import contextmanager

import numpy as np

from ..publisher_checkpoint.v41_profile import V41PublisherProfile
from .execution import V41Execution
from .publisher_cache import PackedPublisherRow, pack_row, packed_nbytes
from .owner_scope import arithmetic_scope


@dataclass(frozen=True)
class PublisherV41Snapshot:
    identity: str
    tokens: tuple
    state: tuple
    token_types: tuple = ()


class _SourcePayloadProvider:
    """Bind loader target contracts to its preserved source-name payload."""
    def __init__(self, payload, arithmetic):
        self.payload, self.arithmetic = payload, arithmetic
        self.names = {rule.target: rule.sources[0] for rule in payload.contracts}
        self.identity = payload.config_sha256+':'+arithmetic.identity
        if not callable(getattr(arithmetic, 'grouped_linear', None)):
            self.grouped_linear = None

    def tensor(self, name):
        return self.payload.tensors[self.names[name]]

    def tensor_row(self,name,index):
        source=self.names[name]
        provider=getattr(self.payload,'embedding_provider',None)
        if provider is not None and source in provider.tensor_shapes:
            return provider.read_row(source,index)
        return self.payload.tensors[source][index]

    def linear(self, name, x):
        return self.arithmetic.linear(self.names[name], x)

    def grouped_linear(self, name, x, *, groups):
        return self.arithmetic.grouped_linear(self.names[name], x, groups=groups)

    def cast(self, x):
        return self.arithmetic.cast(x)

    def roundtrip(self, x, encoding):
        return self.arithmetic.roundtrip(x, encoding)

    def engram_rows(self, layer, tokens):
        return self.arithmetic.engram_rows(layer, tokens)


class PublisherV41Execution:
    """Single-session text token-to-logit scheduler with native reductions.

    provider.identity: immutable nonempty checkpoint/policy identity.
    provider.tensor(name): finite FP32 unquantized tensor, mapped loader names.
    provider.linear(name, input): publisher storage/activation arithmetic; no
        bias inference. Input can be FP32; provider owns BF16 rounding policy.
    provider.roundtrip(row, encoding): quantize/dequantize a transient query row.
    provider.cast(row): publisher BF16 activation boundary, returned as FP32.
    provider.engram_rows(layer, tokens): flattened selected dequantized rows
        using publisher tokenizer compression, primes, offsets and hash state.

    Providers do not own mutable attention state. All rows are detached here.
    No default provider or automatic library build is supplied.
    """
    implementation = 'native-c-publisher-v41-text-contract-v1'
    published_checkpoint_supported = False  # Numerical equivalence unestablished.
    inference_supported = False  # Unexecuted; provider implementation is external.

    _kernel = V41Execution._kernel
    _pack_row = staticmethod(pack_row)

    def _target_mean(self,residual):
        return self._cast(residual.reshape(self.profile.streams,self.profile.hidden_size).mean(axis=0))

    @classmethod
    def from_payload(cls, payload, arithmetic, *, library, max_bytes=64*1024**2):
        """Use loader's V41ExecutionPayload without private-schema conversion.

        arithmetic.linear receives original checkpoint source names. It must
        retain packed weights/scale bindings independently: the payload's
        float32 logical tensors do not preserve publisher activation arithmetic.
        """
        from ..publisher_checkpoint.v41_adapter import V41ExecutionPayload
        if not isinstance(payload, V41ExecutionPayload) or payload.unsupported:
            raise ValueError('supported source-name V41ExecutionPayload required')
        if not isinstance(getattr(arithmetic, 'identity', None), str) or not arithmetic.identity:
            raise ValueError('identified source-name publisher arithmetic required')
        for name in ('linear', 'cast', 'roundtrip'):
            if not callable(getattr(arithmetic, name, None)):
                raise ValueError('missing publisher arithmetic method: '+name)
        if any(l.owns_engram for l in payload.profile.layers) and not callable(getattr(arithmetic, 'engram_rows', None)):
            raise ValueError('publisher Engram row/hash transport required')
        return cls(payload.profile, _SourcePayloadProvider(payload, arithmetic),
                   library=library, max_bytes=max_bytes)

    def __init__(self, profile, provider, *, library, max_bytes=64*1024**2):
        if not isinstance(profile, V41PublisherProfile):
            raise ValueError('publisher V41 profile required; private schema is not accepted')
        required = ('tensor', 'linear', 'roundtrip', 'cast')
        if any(not callable(getattr(provider, name, None)) for name in required):
            raise ValueError('publisher arithmetic/transport provider required')
        if any(layer.owns_engram for layer in profile.layers) and not callable(getattr(provider, 'engram_rows', None)):
            raise ValueError('publisher Engram tokenizer/hash/row provider required')
        if not isinstance(getattr(provider, 'identity', None), str) or not provider.identity:
            raise ValueError('immutable provider identity required')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive memory admission budget required')
        if (profile.streams > 8 or profile.hidden_size > 16384 or
                profile.query_rank > 65536 or profile.expert_intermediate > 65536 or
                profile.shared_experts != 1 or profile.sinkhorn_iterations > 64 or
                profile.scoring_function != 'sqrtsoftplus' or
                profile.routing_method != 'noaux_tc' or not profile.normalize_routed_weights):
            raise ValueError('publisher profile exceeds implemented arithmetic policy')
        self.profile, self.provider = profile, provider
        self.max_bytes = max_bytes
        self._cancellation=threading.local()
        # Keep the handle alive for the facade lifetime. No implicit compilation.
        self._lib = C.CDLL(str(library))
        release = _ctypes.FreeLibrary if os.name=='nt' else _ctypes.dlclose
        self._release = weakref.finalize(self,release,self._lib._handle)
        self._op = self._lib.oa_v41_execution_op
        self._op.argtypes = [C.c_int]*4+[C.c_double]+[C.c_void_p,C.c_size_t]*4
        self._op.restype = C.c_int
        capability = self._lib.oa_v41_publisher_contract_version
        capability.argtypes, capability.restype = [], C.c_uint32
        if capability() != 1:
            raise ValueError('publisher arithmetic ABI version 1 required')
        row_version=self._lib.oa_v41_attention_rows_version
        row_version.argtypes=[];row_version.restype=C.c_uint32
        if row_version()!=1:raise ValueError('native row-streamed attention ABI 1 required')
        packed_version=self._lib.oa_v41_packed_rows_version
        packed_version.argtypes=[];packed_version.restype=C.c_uint32
        if packed_version()!=1:raise ValueError('native packed-row reduction ABI 1 required')
        self._attention_rows=self._lib.oa_v41_attention_packed_rows
        self._attention_rows.argtypes=[C.POINTER(C.c_float),C.c_uint32,C.c_uint32,C.c_uint64,
            C.POINTER(C.c_float),C.c_double,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint64,C.POINTER(C.c_float)]
        self._attention_rows.restype=C.c_int
        self._index_rows=self._lib.oa_v41_index_packed_rows
        self._index_rows.argtypes=self._attention_rows.argtypes;self._index_rows.restype=C.c_int
        self._lock = threading.RLock()
        self._busy = False
        self._closed = False
        self._tokens = ()
        self._token_types = ()
        self._state = self._empty()
        self.identity = hashlib.sha256(repr((profile, provider.identity,
            'packed-cache-low-nibble-first-v2-image-engram-boundaries-native-packed-reductions-v1')).encode('utf-8')).hexdigest()

    @contextmanager
    def cancellation_scope(self,cancel):
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise ValueError('event-like publisher cancellation required')
        old=getattr(self._cancellation,'event',None);self._cancellation.event=cancel
        try:yield self
        finally:self._cancellation.event=old

    def _cancel_rows(self):
        event=getattr(self._cancellation,'event',None)
        if event is not None and event.is_set():
            from ..execution_service import ExecutionCancelled
            raise ExecutionCancelled('publisher packed-cache work cancelled')

    def _stream_attention(self,query,rows,sinks):
        p=self.profile
        return self._packed_reduce(query,rows,sinks,heads=p.attention_heads,width=p.head_dim,
            scale=p.head_dim**-0.5,indexing=False).reshape(query.shape)

    def _stream_index(self,query,rows,weights):
        p=self.profile
        return self._packed_reduce(query,rows,weights,heads=p.index_heads,width=p.index_head_dim,
            scale=(p.index_head_dim*p.index_heads)**-0.5,indexing=True)

    def _packed_reduce(self,query,rows,auxiliary,*,heads,width,scale,indexing):
        query=np.ascontiguousarray(query,dtype=np.float32)
        auxiliary=np.ascontiguousarray(auxiliary,dtype=np.float32);rows=tuple(rows)
        if query.size!=heads*width or auxiliary.size!=heads or not rows:
            raise ValueError('complete packed-row reduction geometry required')
        output=np.empty((len(rows),) if indexing else (heads,width),np.float32);failures=[]
        formats={'fp8-e4m3-g32-ue8m0':1,'fp4-e2m1-g16-e4m3':2,'fp4-e2m1-g32-ue8m0':3}
        def read(user,index,encoding,values,value_capacity,scales,scale_capacity):
            try:
                self._cancel_rows();row=rows[index]
                if (type(row) is not PackedPublisherRow or row.size!=width or row.encoding not in formats
                        or len(row.values)>value_capacity or len(row.scales)>scale_capacity):
                    raise ValueError('complete original packed cache row required')
                encoding[0]=formats[row.encoding]
                C.memmove(values,row.values,len(row.values));C.memmove(scales,row.scales,len(row.scales));return 0
            except BaseException as failure:failures.append(failure);return 1
        def cancelled(user):
            try:self._cancel_rows();return 0
            except BaseException as failure:failures.append(failure);return 1
        reader=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_uint64,C.POINTER(C.c_int),C.POINTER(C.c_uint8),C.c_uint64,
            C.POINTER(C.c_uint8),C.c_uint64)(read)
        cancellation=C.CFUNCTYPE(C.c_int,C.c_void_p)(cancelled)
        work=(len(rows)*4+width*5+width//16 if indexing else
            12*heads*width+16*heads+5*width+width//16)
        operation=self._index_rows if indexing else self._attention_rows
        status=operation(query.ctypes.data_as(C.POINTER(C.c_float)),heads,width,len(rows),
            auxiliary.ctypes.data_as(C.POINTER(C.c_float)),scale,C.cast(reader,C.c_void_p),None,
            C.cast(cancellation,C.c_void_p),None,work,output.ctypes.data_as(C.POINTER(C.c_float)))
        if failures:raise failures[0]
        if status:raise RuntimeError('native packed-row reduction failed: '+str(status))
        return output

    def _empty(self):
        return tuple(tuple(() for _ in self.profile.layers) for _ in range(4))

    @staticmethod
    def _state_bytes(state):
        return sum(sum(v.nbytes for v in row) if k == 3 else row.nbytes
                   for k, kind in enumerate(state) for rows in kind for row in rows)

    def _cache_bytes(self, length):
        p = self.profile
        total = 0
        for layer in p.layers:
            total += min(length, p.window)*packed_nbytes(p.head_dim, layer.window_encoding)
            ratio = layer.compression_ratio
            if ratio and layer.kv_owner == layer.layer:
                total += (length//ratio)*(packed_nbytes(p.head_dim, layer.compressed_encoding) +
                                         packed_nbytes(p.index_head_dim, layer.index_encoding))
                total += (length % ratio)*2*p.head_dim*4
        return total

    @property
    def length(self):
        with self._lock:
            return len(self._tokens)

    def _row(self, value, size):
        row = np.array(value, dtype=np.float32, order='C', copy=True).reshape(-1)
        if row.size != size or not np.isfinite(row).all():
            raise ValueError('publisher provider returned invalid extent or nonfinite arithmetic')
        row.flags.writeable = False
        return row

    def _tensor(self, name, size):
        return self._row(self.provider.tensor(name), size)

    def _embedding(self,token):
        reader=getattr(self.provider,'tensor_row',None)
        value=(reader('v41.text.embedding',token) if callable(reader)
            else self.provider.tensor('v41.text.embedding')[token])
        return self._row(value,self.profile.hidden_size)

    def _linear(self, name, x, size):
        return self._row(self.provider.linear(name, x), size)

    def _cast(self, x):
        return self._row(self.provider.cast(x), x.size)

    def _quant(self, x, encoding):
        return self._row(self.provider.roundtrip(x, encoding), x.size)

    def _norm(self, x, name):
        return self._cast(self._kernel(1, x, self._tensor(name, x.size), p=self.profile.norm_epsilon))

    def _rotate(self, x, layer, position, inverse=False):
        p = self.profile
        if not p.rotary_dim:
            return x.copy()
        rows = np.array(x, dtype=np.float32, copy=True).reshape(-1, x.shape[-1])
        tail = np.ascontiguousarray(rows[:, -p.rotary_dim:])
        policy = (layer.rotary_theta, layer.rotary_original_context, p.rotary_factor,
                  p.rotary_beta_fast, p.rotary_beta_slow)
        rows[:, -p.rotary_dim:] = self._kernel(17, tail, policy, a=position,
                                             b=p.rotary_dim, c=-1 if inverse else 1).reshape(tail.shape)
        return self._cast(rows.reshape(-1)).reshape(x.shape)

    def _mix(self, residual, coefficients):
        p = self.profile
        return self._cast(self._kernel(23, residual, coefficients, a=p.streams,
                                      b=p.hidden_size, n=p.hidden_size))

    def _coefficients(self, residual, prefix):
        p = self.profile
        q = p.streams*(p.streams+2)
        # hc_fn is an FP32 projection, not activation-quantized linear().
        projection = self._linear(prefix+'_fn', residual, q)
        params = np.concatenate((self._tensor(prefix+'_base', q),
                                 self._tensor(prefix+'_scale', 3),
                                 np.asarray((p.norm_epsilon, p.sinkhorn_epsilon), dtype=np.float32)))
        return self._kernel(16, residual, projection, params, a=p.streams,
                            c=p.sinkhorn_iterations, n=q)

    def _post(self, residual, coefficients, branch):
        p = self.profile
        return self._cast(self._kernel(23, residual, coefficients, branch,
                                      a=p.streams, b=p.hidden_size, c=1))

    def _engram(self, residual, layer, tokens):
        p, eg = self.profile, self.profile.engram
        prefix = f'v41.text.layers.{layer}.engram.'
        memory = self._row(self.provider.engram_rows(layer, tokens), eg.projection_input)
        kv = self._linear(prefix+'wkv', memory, (p.streams+1)*p.hidden_size)
        keys, value = kv[:residual.size], kv[residual.size:]
        scales = np.concatenate((value, self._tensor(prefix+'q_weight', residual.size),
                                 self._tensor(prefix+'k_weight', residual.size)))
        return self._cast(self._kernel(20, residual, keys, scales, a=p.streams,
                                      b=p.hidden_size, p=p.norm_epsilon))

    def _attention(self, x, layer, position, state, selections, candidates):
        p = self.profile
        local, compressed, index, pending = state
        li, ratio = layer.layer, layer.compression_ratio
        prefix = f'v41.text.layers.{li}.attention.'
        qr = self._norm(self._linear(prefix+'wq_a', x, p.query_rank), prefix+'q_norm')
        q = self._linear(prefix+'wq_b', qr, p.attention_heads*p.head_dim).reshape(p.attention_heads, p.head_dim)
        q = self._rotate(q, layer, position)
        kv = self._norm(self._linear(prefix+'wkv', x, p.head_dim), prefix+'kv_norm')
        kv = self._pack_row(self._rotate(kv, layer, position), layer.window_encoding)
        local[li].append(kv)
        local[li] = local[li][-p.window:]
        if ratio and layer.kv_owner == li:
            cp = prefix+'compressor.'
            value = self._linear(cp+'wkv', x, p.head_dim)
            gate = self._linear(cp+'wgate', x, p.head_dim) if ratio > 1 else np.zeros(p.head_dim, dtype=np.float32)
            pending[li].append((value, gate))
            if len(pending[li]) == ratio:
                values, gates = zip(*pending[li])
                latent = self._kernel(4, np.concatenate(values), np.concatenate(gates),
                                      a=ratio, b=p.head_dim, n=p.head_dim) if ratio > 1 else value
                latent = self._norm(self._cast(latent), cp+'norm')
                ip = prefix+'indexer.'
                key = self._norm(self._linear(ip+'wk', latent, p.index_head_dim), ip+'k_norm')
                first = position+1-ratio
                index[li].append(self._pack_row(self._rotate(key, layer, first), layer.index_encoding))
                compressed[li].append(self._pack_row(self._rotate(latent, layer, first), layer.compressed_encoding))
                pending[li] = []
        selected = ()
        if ratio:
            owner = layer.kv_owner
            count = len(compressed[owner])
            if layer.index_owner == li:
                ip = prefix+'indexer.'
                if count:
                    iq = self._linear(ip+'wq_b', qr, p.index_heads*p.index_head_dim).reshape(p.index_heads, p.index_head_dim)
                    iq = self._rotate(iq, layer, position)
                    iq = np.concatenate(tuple(self._quant(row, layer.index_encoding) for row in iq))
                    iw = self._linear(ip+'weights_proj', x, p.index_heads)
                    scores=self._stream_index(iq,index[owner],iw)
                    allowed = np.ones(count, dtype=np.float32)
                    # Newest block must always survive the candidate prefilter.
                    if li == self._candidate_source:
                        candidates[li] = self._kernel(24, scores, a=p.candidate_block_size,
                                                     b=p.candidate_topk_blocks, n=count)
                    elif layer.candidate_owner is not None:
                        allowed = candidates[layer.candidate_owner]
                    picks = self._kernel(25, scores, allowed, b=min(count,p.index_topk), n=min(count,p.index_topk))
                    selected = tuple(int(i) for i in picks if i >= 0)
                selections[li] = selected
            else:
                selected = selections[layer.index_owner]
        rows = tuple(local[li]) + (tuple(compressed[layer.kv_owner][i] for i in selected) if ratio else ())
        output=self._stream_attention(q,rows,self._tensor(prefix+'attn_sink',p.attention_heads)).reshape(q.shape)
        output = self._cast(output.reshape(-1)).reshape(q.shape)
        output = self._rotate(output, layer, position, inverse=True).reshape(p.output_groups, -1)
        # The stored wo_a is rank two, but each group sees ONLY its own heads.
        grouped = getattr(self.provider, 'grouped_linear', None)
        if callable(grouped):
            projected = grouped(prefix+'wo_a', output, groups=p.output_groups)
        else:
            weight = self._cast(self._tensor(prefix+'wo_a', p.output_groups*p.output_rank*output.shape[1])).reshape(p.output_groups, p.output_rank, -1)
            projected = np.concatenate(tuple(V41Execution._linear(self, weight[g], output[g]) for g in range(p.output_groups)))
        return self._linear(prefix+'wo_b', self._cast(projected), p.hidden_size)

    @property
    def _candidate_source(self):
        owners = [layer.candidate_owner for layer in self.profile.layers if layer.candidate_owner is not None]
        return owners[0] if owners else -1

    def _expert(self, x, prefix, weight, *, shared=False):
        p = self.profile
        gate_name, up_name, down_name = ('w1', 'w3', 'w2') if shared else ('gate', 'up', 'down')
        gate = self._linear(prefix+gate_name, x, p.expert_intermediate)
        up = self._linear(prefix+up_name, x, p.expert_intermediate)
        activation = self._kernel(22, gate, up, (weight,), p=p.activation_limit)
        return self._linear(prefix+down_name, self._cast(activation), p.hidden_size)

    def _moe(self, x, li, *, image_token=False):
        p = self.profile
        prefix = f'v41.text.layers.{li}.moe.'
        logits = self._linear(prefix+'router', x, p.expert_count)
        bias_name = 'router_bias_vl' if image_token else 'router_bias'
        route = self._kernel(21, logits, self._tensor(prefix+bias_name, p.expert_count),
                             (1.0, p.routed_scale, 1.0), a=p.expert_count,
                             b=p.selected_experts, c=0, n=2*p.selected_experts)
        out = self._expert(x, prefix+'shared.', 1.0, shared=True)
        for j in range(p.selected_experts):
            branch = self._expert(x, prefix+f'experts.{int(route[j])}.', float(route[p.selected_experts+j]))
            out = self._kernel(2, out, branch)
        return self._cast(out)

    def forward(self, token_ids, *, target_layers=(), images=None, vision=None, token_types=None,
                _inputs_embeds=None, _return_hidden=False):
        with self._lock:
            if self._busy or self._closed:
                raise RuntimeError('closed or reentrant publisher execution')
            self._busy = True
            try:
                with arithmetic_scope(self.provider):
                    if type(_return_hidden) is not bool:
                        raise ValueError('explicit publisher hidden output policy required')
                    if _inputs_embeds is not None and (images is not None or vision is not None or token_types is not None):
                        raise ValueError('public publisher hidden input requires occupied text rows')
                    ids = []
                    for token in token_ids:
                        if type(token) is not int or not 0 <= token < self.profile.vocabulary:
                            raise ValueError('integer vocabulary token required')
                        ids.append(token)
                        if len(self._tokens)+len(ids) > self.profile.context:
                            raise ValueError('publisher context capacity exceeded')
                    p = self.profile
                    targets = tuple(target_layers)
                    if (len(set(targets)) != len(targets) or
                            any(type(i) is not int or not 0 <= i < len(p.layers) for i in targets)):
                        raise ValueError('unique valid backbone target layers required')
                    target_set = set(targets)
                    types = [-1]*len(ids) if token_types is None else list(token_types)
                    if (len(types) != len(ids) or
                            any(type(t) is not int or not -1 <= t <= 3 for t in types)):
                        raise ValueError('one token type per token required: text -1, image 0..3')
                    if images is not None:
                        images = tuple(images)
                        if self._tokens or not ids:
                            raise ValueError('images require nonempty initial prefill')
                        from .publisher_vision import PublisherV41Vision
                        if not isinstance(vision, PublisherV41Vision) or vision.hidden != p.hidden_size:
                            raise ValueError('matching publisher vision encoder required')
                        if (vision.provider is not self.provider and
                                (getattr(vision.provider, 'arithmetic', None) is None or
                                 getattr(vision.provider, 'arithmetic', None) is not
                                 getattr(self.provider, 'arithmetic', None))):
                            raise ValueError('vision and text must share the checkpoint provider')
                        occupied = set()
                        for image in images:
                            span_types = np.asarray(image.types)
                            if (type(image.start) is not int or image.start < 0 or
                                    span_types.ndim != 1 or span_types.dtype.kind not in 'iu' or
                                    image.start+len(span_types) > len(ids) or
                                    np.any((span_types < 0) | (span_types > 3))):
                                raise ValueError('invalid initial image token span')
                            for offset, kind in enumerate(span_types):
                                index = image.start+offset
                                if index in occupied or (token_types is not None and types[index] != int(kind)):
                                    raise ValueError('overlapping image span or mismatched token types')
                                occupied.add(index)
                                types[index] = int(kind)
                        if any(kind >= 0 and i not in occupied for i, kind in enumerate(types)):
                            raise ValueError('image token has no image span')
                    elif any(kind >= 0 for kind in types):
                        raise ValueError('image token types require initial image inputs')
                    # Packed payloads and scales are resident; decode buffers are scratch.
                    # Include old state for transactional rollback and output stacking.
                    end = len(self._tokens)+len(ids)
                    scratch = 4*(2*len(ids)*p.vocabulary + 8*p.hidden_size*p.streams +
                        8*p.attention_heads*p.head_dim + 4*p.head_dim+4*p.attention_heads+4*p.expert_intermediate +
                        max((4*p.index_heads*p.index_head_dim+4*p.index_heads+4*p.index_head_dim+8*(end//l.compression_ratio) +
                             4*l.compression_ratio*p.head_dim
                             if l.compression_ratio else 0)
                            for l in p.layers))
                    # Pending compression can peak just before its group is emitted.
                    pending_peak = sum(2*l.compression_ratio*p.head_dim*4
                        for l in p.layers if l.compression_ratio and l.kv_owner==l.layer)
                    # Row-reference transport and retained selected-token inventories
                    # coexist with native numerator/score staging during reductions.
                    scratch+=24*min(end,p.window)+64*p.index_topk
                    scratch += 8*len(ids)*len(targets)*p.hidden_size
                    if images is not None:
                        scratch += 8*len(ids)*p.hidden_size
                    estimate = self._state_bytes(self._state)+self._cache_bytes(end)+pending_peak+scratch
                    if estimate > self.max_bytes:
                        raise MemoryError('publisher session admission exceeds budget; provider weights are separately budgeted')
                    staged = tuple([list(rows) for rows in kind] for kind in self._state)
                    image_embeddings = None
                    if _inputs_embeds is not None:
                        source=np.asarray(_inputs_embeds)
                        if source.shape!=(len(ids),p.hidden_size) or source.dtype!=np.float32 or not np.isfinite(source).all():
                            raise ValueError('complete finite public V4.1 embedding rows required')
                        if estimate+source.nbytes*3>self.max_bytes:
                            raise MemoryError('public V4.1 input copies exceed session admission')
                        image_embeddings=np.array(source,order='C',copy=True)
                    if images is not None:
                        embeddings = np.stack([self._embedding(token) for token in ids])[None, :, :]
                        image_embeddings = vision.merge_image_embeddings(embeddings, [images])[0]
                    results = []
                    captured = []
                    all_tokens = self._tokens+tuple(ids)
                    all_types = self._token_types+tuple(types)
                    text_start = 0
                    for i, kind in enumerate(self._token_types):
                        if kind >= 0:
                            text_start = i+1
                    for token in ids:
                        position = len(self._tokens)+len(results)
                        image_token = all_types[position] >= 0
                        if image_token:
                            text_start = position+1
                        history = all_tokens[text_start:position+1]
                        embedding = (image_embeddings[len(results)] if image_embeddings is not None else
                            self._embedding(token))
                        residual = self._cast(np.tile(embedding, p.streams))
                        pre = np.zeros(p.streams, dtype=np.float32)
                        pre[0] = 1
                        selections, candidates = {}, {}
                        token_targets = {}
                        for layer in p.layers:
                            base = f'v41.text.layers.{layer.layer}.'
                            if layer.owns_engram and not image_token:
                                residual = self._engram(residual, layer.layer, history)
                            if layer.layer in target_set:
                                token_targets[layer.layer] = self._target_mean(residual)
                            coeff = self._coefficients(residual, base+'hc_attn')
                            x = self._norm(self._mix(residual, pre), base+'attn_norm')
                            branch = self._attention(x, layer, position, staged, selections, candidates)
                            residual = self._post(residual, coeff, branch)
                            pre = coeff[:p.streams]
                            coeff = self._coefficients(residual, base+'hc_ffn')
                            x = self._norm(self._mix(residual, pre), base+'ffn_norm')
                            residual = self._post(residual, coeff, self._moe(x, layer.layer, image_token=image_token))
                            pre = coeff[:p.streams]
                        hidden = self._norm(self._mix(residual, pre), 'v41.text.final_norm')
                        results.append(hidden if _return_hidden else self._linear('v41.text.lm_head', hidden, p.vocabulary))
                        if targets:
                            captured.append(np.concatenate([token_targets[i] for i in targets]))
                    output = np.stack(results) if results else np.empty((0,p.hidden_size if _return_hidden else p.vocabulary), dtype=np.float32)
                    target_output = (np.stack(captured) if captured else
                        np.empty((0, len(targets)*p.hidden_size), dtype=np.float32))
                    frozen = tuple(tuple(tuple(rows) for rows in kind) for kind in staged)
                    if self._state_bytes(self._state)+self._state_bytes(frozen)+scratch > self.max_bytes:
                        raise MemoryError('publisher packed session staging exceeds budget')
                    self._tokens, self._state, self._token_types = all_tokens, frozen, all_types
                    return (output, target_output) if targets else output
            finally:
                self._busy = False

    def forward_hidden(self, token_ids, inputs_embeds):
        """Execute complete mHC/Engram/shared-index state before the head Edge."""
        return self.forward(token_ids,_inputs_embeds=inputs_embeds,_return_hidden=True)

    @contextmanager
    def _edge_operation(self):
        with self._lock:
            if self._busy or self._closed:
                raise RuntimeError('closed or reentrant publisher Edge')
            self._busy=True
            try:
                with arithmetic_scope(self.provider):
                    yield
            finally:
                self._busy=False

    def embedding_rows(self, token_ids):
        with self._edge_operation():
            ids=tuple(token_ids)
            p=self.profile
            if (not 0<len(ids)<=4096 or any(type(token) is not int or not 0<=token<p.vocabulary for token in ids)):
                raise ValueError('occupied V4.1 vocabulary token rows required')
            if self._state_bytes(self._state)+len(ids)*p.hidden_size*16>self.max_bytes:
                raise MemoryError('public V4.1 embeddings exceed session admission')
            result=[]
            for token in ids:
                self._cancel_rows()
                result.append(self._embedding(token))
            self._cancel_rows()
            return np.stack(result)

    def head_rows(self, hidden):
        with self._edge_operation():
            source=np.asarray(hidden)
            p=self.profile
            if (source.ndim!=2 or source.shape[1]!=p.hidden_size or not 0<len(source)<=4096 or
                    source.dtype!=np.float32 or not np.isfinite(source).all()):
                raise ValueError('finite normalized public V4.1 hidden rows required')
            if self._state_bytes(self._state)+source.nbytes*3+len(source)*p.vocabulary*8>self.max_bytes:
                raise MemoryError('public V4.1 head exceeds session admission')
            result=[]
            for row in source:
                self._cancel_rows()
                result.append(self._linear('v41.text.lm_head',row,p.vocabulary))
            self._cancel_rows()
            return np.stack(result)

    def write_cache(self, output, *, max_bytes, scratch_directory=None):
        from ..file_cache_transport import transport_cache
        from .publisher_checkpoints import save_publisher_v41_checkpoint
        return transport_cache(self,output,save_publisher_v41_checkpoint,max_bytes=max_bytes,
                               scratch_directory=scratch_directory)

    def restore_cache(self, source, *, max_bytes, scratch_directory=None):
        from ..file_cache_transport import transport_cache
        from .publisher_checkpoints import load_publisher_v41_checkpoint
        return transport_cache(self,source,load_publisher_v41_checkpoint,max_bytes=max_bytes,
                               restoring=True,scratch_directory=scratch_directory)

    def reset(self):
        with self._lock:
            if self._busy or self._closed:
                raise RuntimeError('closed or reentrant publisher execution')
            self._tokens, self._state = (), self._empty()
            self._token_types = ()

    def snapshot(self):
        with self._lock:
            if self._busy or self._closed:
                raise RuntimeError('closed or reentrant publisher execution')
            self._busy = True
            try:
                return self._snapshot_locked()
            finally:
                self._busy = False

    def _snapshot_locked(self):
        # Packed rows share immutable bytes; only FP32 pending rows are copied.
        pending_bytes = sum(v.nbytes for rows in self._state[3] for row in rows for v in row)
        if self._state_bytes(self._state)+pending_bytes > self.max_bytes:
            raise MemoryError('publisher snapshot copy exceeds payload budget')
        # Bytes-backed arrays cannot have writeability re-enabled by a caller.
        kinds = []
        for k, kind in enumerate(self._state):
            kinds.append(tuple(tuple(tuple(np.frombuffer(v.tobytes(), dtype=np.float32) for v in row)
                if k==3 else row for row in rows) for rows in kind))
        return PublisherV41Snapshot(self.identity, self._tokens, tuple(kinds), self._token_types)

    def restore(self, snapshot):
        with self._lock:
            if self._busy or self._closed:
                raise RuntimeError('closed or reentrant publisher execution')
            self._busy = True
            try:
                return self._restore_locked(snapshot)
            finally:
                self._busy = False

    def _restore_locked(self, snapshot, *, cancelled=None):
        if type(snapshot) is not PublisherV41Snapshot or snapshot.identity != self.identity:
            raise ValueError('publisher snapshot identity mismatch')
        p, tokens = self.profile, snapshot.tokens
        if type(snapshot.token_types) is not tuple:
            raise ValueError('invalid publisher snapshot token type inventory')
        token_types = snapshot.token_types or (-1,)*len(tokens)
        if (type(token_types) is not tuple or len(token_types) != len(tokens) or
                any(type(t) is not int or not -1 <= t <= 3 for t in token_types)):
            raise ValueError('invalid publisher snapshot token types')
        if (type(tokens) is not tuple or len(tokens)>p.context or
                any(type(t) is not int or not 0<=t<p.vocabulary for t in tokens) or
                type(snapshot.state) is not tuple or len(snapshot.state)!=4):
            raise ValueError('invalid publisher snapshot history/inventory')
        restored = []
        admitted = 0
        existing = self._state_bytes(self._state)
        for k, kind in enumerate(snapshot.state):
            if type(kind) is not tuple or len(kind)!=len(p.layers):
                raise ValueError('invalid publisher snapshot layer inventory')
            layers = []
            for layer, rows in zip(p.layers,kind):
                if cancelled is not None:
                    cancelled()
                ratio = layer.compression_ratio
                owned = bool(ratio and layer.kv_owner==layer.layer)
                expected = min(len(tokens),p.window) if k==0 else (
                    len(tokens)%ratio if k==3 and owned else len(tokens)//ratio if owned else 0)
                if type(rows) is not tuple or len(rows)!=expected:
                    raise ValueError('publisher snapshot cache extent/history mismatch')
                if k==3:
                    if any(type(row) is not tuple or len(row)!=2 for row in rows):
                        raise ValueError('invalid publisher compression group')
                    admitted += expected*2*p.head_dim*4
                    if existing+admitted+8*p.head_dim > self.max_bytes:
                        raise MemoryError('publisher snapshot restore exceeds staging budget')
                    layers.append(tuple(tuple(np.frombuffer(self._row(v,p.head_dim).tobytes(), dtype=np.float32)
                                              for v in row) for row in rows))
                else:
                    size = p.index_head_dim if k==2 else p.head_dim
                    encoding = (layer.window_encoding if k==0 else
                                layer.compressed_encoding if k==1 else layer.index_encoding)
                    packed = []
                    for row in rows:
                        if cancelled is not None:
                            cancelled()
                        if (type(row) is not PackedPublisherRow or row.size!=size or row.encoding!=encoding):
                            raise ValueError('publisher snapshot packed cache format mismatch')
                        # Revalidate even a forged dataclass without decoding residency.
                        validated = PackedPublisherRow(row.encoding, row.size, row.values, row.scales)
                        admitted += row.nbytes
                        if existing+admitted > self.max_bytes:
                            raise MemoryError('publisher snapshot restore exceeds staging budget')
                        packed.append(validated)
                    layers.append(tuple(packed))
            restored.append(tuple(layers))
        if cancelled is not None:
            cancelled()
        self._tokens, self._state, self._token_types = tokens, tuple(restored), token_types

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('reentrant publisher execution')
            self._tokens, self._state, self._closed = (), self._empty(), True
            self._token_types = ()
            self._release()
