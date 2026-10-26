"""Private compiled GLM5-next token engine, with atomic chunk-cache commits."""

import ctypes as C
import _ctypes
import math
import os
from dataclasses import dataclass
from pathlib import Path
import threading
import weakref
import numpy as np
from uuid import uuid4

from .independent_glm5_next import IndependentGlm5NextTextDecoder
from .native_library_identity import NativeLibraryIdentity, release_finalizer


class Buffer(C.Structure):
    _fields_ = [("p", C.POINTER(C.c_float)), ("n", C.c_uint64)]


@dataclass(frozen=True)
class NativeGlm5NextSnapshot:
    """Exact-instance capability for a complete GLM5 physical cache image."""
    owner: object
    key: str
    position: int


def buffer(array):
    if array is None:
        return Buffer()
    if array.dtype != np.float32 or not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("native buffers must be contiguous FP32")
    return Buffer(array.ctypes.data_as(C.POINTER(C.c_float)), array.size)


class NativeGlm5NextTextDecoder(IndependentGlm5NextTextDecoder):
    implementation = "c-glm5-next-text-resident-fp32"
    native_range_execution = True

    def __init__(self, config, tensors, *, library, max_tokens=256, max_bytes=512*1024*1024,
                 max_snapshots=4, expert_provider=None):
        if type(max_snapshots) is not int or not 1 <= max_snapshots <= 64:
            raise ValueError("bounded native GLM5 snapshot count required")
        super().__init__(config, tensors, max_tokens=max_tokens, max_bytes=max_bytes,_omit_experts=expert_provider is not None)
        self.max_snapshots = max_snapshots
        self._snapshot_owner = object()
        self._snapshots = {}
        self._snapshot_bytes = 0
        self._expert_provider=expert_provider;self._expert_error=None;self._executing=False;self._active_cancel=None
        if expert_provider is not None:
            from .glm5_publisher_experts import Glm5PublisherExperts
            if type(expert_provider) is not Glm5PublisherExperts or expert_provider.contract!=self.config:
                raise ValueError('exact GLM publisher expert provider contract required')
            self._resident_bytes+=expert_provider.max_bytes
        self._lock = threading.RLock()
        self._closed = False
        self._library_identity = NativeLibraryIdentity.read(library)
        self.library = C.CDLL(str(self._library_identity.path))
        release = _ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose
        self._release = weakref.finalize(self,release,self.library._handle)
        try:
            self._library_identity.assert_unchanged()
            self._bind_native(max_tokens)
            self._library_identity.assert_unchanged()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                from .failure_reporting import retain_cleanup_failure
                retain_cleanup_failure(primary,self.close,cleanup,
                    'GLM native construction cleanup retained: ')
            raise

    def _bind_native(self,max_tokens):
        self.library.oa_glm5_next_version.argtypes = []
        self.library.oa_glm5_next_version.restype = C.c_uint32
        if self.library.oa_glm5_next_version() != 3:
            raise ValueError("unsupported private GLM5-next ABI")
        self.library.oa_glm5_next_step.argtypes = [
            C.POINTER(C.c_int32), C.POINTER(C.c_float), C.POINTER(C.c_int32),
            C.POINTER(Buffer), Buffer, Buffer, Buffer, C.POINTER(Buffer),
            C.c_int32, C.c_int64, Buffer, Buffer,
        ]
        self.library.oa_glm5_next_step.restype = C.c_int
        if self._expert_provider is not None:
            self.library.oa_glm5_next_streamed_version.argtypes=[]
            self.library.oa_glm5_next_streamed_version.restype=C.c_uint32
            if self.library.oa_glm5_next_streamed_version()!=1: raise ValueError('unsupported GLM streamed expert ABI')
            callback_type=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_int32,C.c_int32,
                C.POINTER(C.c_float),C.POINTER(C.c_float))
            def evaluate(user,layer,expert,inputs,output):
                try:
                    value=self._expert_provider.evaluate(int(layer),int(expert),
                        np.ctypeslib.as_array(inputs,shape=(self.config.hidden_size,)),cancelled=self._active_cancel)
                    C.memmove(output,value.ctypes.data,value.nbytes);return 0
                except BaseException as error:
                    self._expert_error=error;return 1
            self._expert_callback=callback_type(evaluate)
            self.library.oa_glm5_next_step_streamed.argtypes=self.library.oa_glm5_next_step.argtypes+[callback_type,C.c_void_p]
            self.library.oa_glm5_next_step_streamed.restype=C.c_int
        c = self.config
        geometry = (c.hidden_size, c.vocab_size, c.num_hidden_layers, c.hc_mult,
            c.hc_sinkhorn_iters, c.intermediate_size, c.moe_intermediate_size,
            c.n_routed_experts, c.n_shared_experts, c.num_experts_per_tok,
            c.num_attention_heads, c.q_lora_rank, c.kv_lora_rank,
            c.qk_nope_head_dim, c.v_head_dim, c.index_n_heads, c.index_head_dim,
            c.index_topk, c.index_kpool, c.linear_num_heads, c.linear_head_dim,
            c.linear_conv_kernel_dim, max_tokens)
        bounds = (16384,1048576,128,8,64,65536,65536,4096,32,4096,
                  256,8192,8192,512,512,256,512,65536,32,256,512,32,65536)
        if (any(type(value) is not int or not 1<=value<=bound for value,bound in zip(geometry,bounds)) or
                c.num_experts_per_tok>c.n_routed_experts or c.n_routed_experts<2 or
                c.index_topk%c.index_kpool or
                not 1<=c.n_group<=4096 or not 1<=c.topk_group<=c.n_group or
                c.n_routed_experts%c.n_group or c.n_routed_experts//c.n_group<2 or
                c.num_experts_per_tok>c.topk_group*(c.n_routed_experts//c.n_group)):
            raise ValueError('unsupported private GLM5-next bounded geometry')
        self._geometry = (C.c_int32*26)(
            c.hidden_size, c.vocab_size, c.num_hidden_layers, c.hc_mult,
            c.hc_sinkhorn_iters, c.intermediate_size, c.moe_intermediate_size,
            c.n_routed_experts, c.n_shared_experts, c.num_experts_per_tok,
            c.num_attention_heads, c.q_lora_rank, c.kv_lora_rank,
            c.qk_nope_head_dim, c.v_head_dim, c.index_n_heads, c.index_head_dim,
            c.index_topk, c.index_kpool, c.linear_num_heads, c.linear_head_dim,
            c.linear_conv_kernel_dim, max_tokens, int(c.norm_topk_prob),
            c.n_group, c.topk_group,
        )
        self._floats = (C.c_float*6)(c.rms_norm_eps, c.hc_eps, c.swiglu_limit,
            c.routed_scaling_factor, c.linear_lower_bound or 0.0, float(c.linear_lower_bound is not None))
        if (not all(math.isfinite(value) for value in self._floats) or
                any(value<=0 for value in self._floats[:4]) or
                (self._floats[5] == 1 and self._floats[4]>=0)):
            raise ValueError('GLM5-next parameters exceed native FP32 range')
        self.library.oa_glm5_next_workspace_floats.argtypes = [
            C.POINTER(C.c_int32), C.POINTER(C.c_float)]
        self.library.oa_glm5_next_workspace_floats.restype = C.c_uint64
        workspace_floats = self.library.oa_glm5_next_workspace_floats(self._geometry, self._floats)
        if not workspace_floats:
            raise ValueError('native GLM5-next rejected workspace geometry')
        # Include native selection stacks, in addition to the geometry-sized heap.
        self._workspace_bytes = int(workspace_floats)*4 + 2*1024*1024
        self._resident_bytes += max(0, self._workspace_bytes - 8*1024*1024)
        self._schedule = (C.c_int32*(3*c.num_hidden_layers))(*[
            value for i in range(c.num_hidden_layers) for value in (
                int(c.layer_types[i] == "linear_attention"),
                int(c.mlp_layer_types[i] == "sparse"), int(c.indexer_types[i] == "shared"),
            )
        ])
        self._weight_buffers = (Buffer*(42*c.num_hidden_layers))(*[
            buffer(self.weights[name]) if name is not None else buffer(None)
            for slots in self._slots for name in slots
        ])
        self._embedding = buffer(self.weights["embed_tokens.weight"])
        self._norm = buffer(self.weights["norm.weight"])
        self._head = buffer(self.weights.get("lm_head.weight",
            self.weights["embed_tokens.weight"] if c.tie_word_embeddings else None))
        self._metadata_bytes = (C.sizeof(self._geometry)+C.sizeof(self._floats)+C.sizeof(self._schedule)+
            C.sizeof(self._weight_buffers)+C.sizeof(Buffer)*(3+2*c.num_hidden_layers))
        if self._resident_bytes+self._metadata_bytes>self.max_bytes:
            raise MemoryError('native GLM5-next descriptor budget exceeded')

    def _check_open(self):
        if self._closed or self._executing:
            raise RuntimeError('native GLM5-next decoder is closed or executing')
        self._library_identity.assert_unchanged()

    def close(self):
        with self._lock:
            if self._executing: raise RuntimeError('GLM native callback owns resources')
            if not self._closed:
                self._closed = True
                super().reset()
                self._snapshots.clear()
                self._snapshot_bytes = 0
                self._weight_buffers = None
                self._embedding = self._norm = self._head = Buffer()
                self.weights = {}
                self._expert_provider=None;self._expert_callback=None
            release_finalizer(self, '_release')
            self.library = None

    @property
    def cache(self):
        with self._lock:
            self._check_open()
            return super().cache

    def reset(self):
        with self._lock:
            self._check_open()
            super().reset()

    def snapshot(self):
        """Capture all KDA/DSA state before a later continuation mutates it."""
        with self._lock:
            self._check_open()
            if len(self._snapshots) >= self.max_snapshots:
                raise MemoryError("native GLM5 snapshot inventory exhausted")
            cache = super().cache
            extent = (0 if cache is None else
                      sum(array.nbytes for pair in cache for array in pair))
            if self._resident_bytes + self._snapshot_bytes + extent > self.max_bytes:
                raise MemoryError("native GLM5 snapshot exceeds aggregate decoder budget")
            receipt = NativeGlm5NextSnapshot(self._snapshot_owner, uuid4().hex, self.position)
            self._snapshots[receipt.key] = (receipt, cache, extent, receipt.position)
            self._snapshot_bytes += extent
            return receipt

    def _snapshot_entry(self, receipt):
        if (type(receipt) is not NativeGlm5NextSnapshot or
                receipt.owner is not self._snapshot_owner):
            raise ValueError("foreign native GLM5 snapshot")
        entry = self._snapshots.get(receipt.key)
        if entry is None or entry[0] is not receipt or receipt.position != entry[3]:
            raise ValueError("released or forged native GLM5 snapshot")
        return entry

    def restore(self, receipt):
        with self._lock:
            self._check_open()
            _, cache, extent, saved_position = self._snapshot_entry(receipt)
            current = (0 if self._cache is None else
                       sum(array.nbytes for pair in self._cache for array in pair))
            if self._resident_bytes + self._snapshot_bytes + current + extent > self.max_bytes:
                raise MemoryError("native GLM5 restore staging exceeds aggregate decoder budget")
            staged = None if cache is None else tuple(
                tuple(array.copy() for array in pair) for pair in cache)
            self._cache = staged
            self.position = saved_position

    def release_snapshot(self, receipt):
        with self._lock:
            self._check_open()
            entry = self._snapshot_entry(receipt)
            del self._snapshots[receipt.key]
            self._snapshot_bytes -= entry[2]

    def _step(self, token, position, cache_buffers, hidden, logits,*,cancelled=None,embedding=None,
              routing_logits=None,routing_weights=None):
        self._check_open()
        function=self.library.oa_glm5_next_step if self._expert_provider is None else self.library.oa_glm5_next_step_streamed
        extra=() if self._expert_provider is None else (self._expert_callback,None)
        if embedding is not None:
            version=self.library.oa_glm5_next_embeddings_version
            version.argtypes=[];version.restype=C.c_uint32
            if version()!=1: raise ValueError('unsupported GLM external embedding ABI')
            function=self.library.oa_glm5_next_step_embeddings
            function.argtypes=self.library.oa_glm5_next_step.argtypes+[Buffer,C.c_void_p,C.c_void_p]
            function.restype=C.c_int
            callback=None if self._expert_provider is None else C.cast(self._expert_callback,C.c_void_p)
            extra=(buffer(embedding),callback,None)
        if routing_logits is not None:
            version=self.library.oa_glm5_next_observation_version
            version.argtypes=[];version.restype=C.c_uint32
            if version()!=1:raise ValueError('unsupported GLM routing observation ABI')
            function=self.library.oa_glm5_next_step_observed
            function.argtypes=self.library.oa_glm5_next_step.argtypes+[Buffer,C.c_void_p,C.c_void_p,Buffer,Buffer]
            function.restype=C.c_int
            callback=None if self._expert_provider is None else C.cast(self._expert_callback,C.c_void_p)
            extra=(buffer(embedding),callback,None,buffer(routing_logits),buffer(routing_weights))
        self._expert_error=None;self._active_cancel=cancelled;self._executing=True
        try:
            code = function(
                self._geometry, self._floats, self._schedule, self._weight_buffers,
                self._embedding, self._norm, self._head, cache_buffers, position,
                token, buffer(hidden), buffer(logits),*extra,
            )
        finally:
            self._executing=False;self._active_cancel=None
        if self._expert_error is not None:
            error=self._expert_error;self._expert_error=None;raise error
        if code:
            reason = {1: "invalid native buffer/geometry", 2: "workspace allocation failed",
                      3: "invalid selection", 4: "ambiguous selection-boundary tie",
                      5: "non-finite arithmetic/state"}.get(code, "unknown native error")
            raise ValueError(f"native GLM5-next failed ({code}): {reason}")

    def forward(self, input_ids, *, use_cache=True, attention_mask=None, position_ids=None, cancelled=None,inputs_embeds=None,
                capture_routing=False):
        """Return hidden states and optional logits; commit only after the entire chunk succeeds.

        A cache-free call always starts at zero and leaves committed state alone.
        No padding/masking or caller-specified positions are accepted in this subset.
        """
        with self._lock:
            self._check_open()
            if cancelled is not None and not callable(cancelled):
                raise TypeError('callable GLM cancellation required')
            if cancelled is not None and cancelled():
                raise InterruptedError('GLM generation cancelled before staging')
            if type(use_cache) is not bool:
                raise ValueError("use_cache must be boolean")
            if type(capture_routing) is not bool:raise ValueError('boolean routing observation policy required')
            if attention_mask is not None or position_ids is not None:
                raise ValueError("mask/position overrides are unsupported; unpadded NoPE text only")
            ids = np.asarray(input_ids)
            if ids.dtype.kind not in "iu" or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
                raise ValueError("input_ids must be a nonempty integer [1, tokens] array")
            if (ids < 0).any() or (ids >= self.config.vocab_size).any():
                raise ValueError("token outside vocabulary")
            start = self.position if use_cache else 0
            count = ids.shape[1]
            if start+count > self.max_tokens:
                raise ValueError("context capacity exceeded")
            output_bytes = count*(self.config.hidden_size + (self.config.vocab_size if self._head.p else 0))*4
            if capture_routing:
                output_bytes+=count*self.config.num_hidden_layers*self.config.n_routed_experts*8
            embeddings=None
            if inputs_embeds is not None:
                source=np.asarray(inputs_embeds)
                if source.shape!=(1,count,self.config.hidden_size) or source.dtype!=np.float32:
                    raise ValueError('expanded batch-one FP32 GLM external embeddings required')
                output_bytes+=2*source.nbytes
            if self._resident_bytes + self._metadata_bytes + output_bytes + ids.size*8 > self.max_bytes:
                raise ValueError("output allocation exceeds bounded transaction budget")
            if inputs_embeds is not None:
                if not np.isfinite(source).all(): raise ValueError('nonfinite GLM external embeddings')
                embeddings=np.array(source,order='C',copy=True)
            ids = np.array(ids,dtype=np.int64,order='C',copy=True)
            if use_cache and self._cache is not None:
                candidate = tuple(tuple(a.copy() for a in pair) for pair in self._cache)
            else:
                candidate = tuple(tuple(np.zeros(shape,np.float32) for shape in pair) for pair in self._cache_shapes)
            cache_buffers = (Buffer*(2*self.config.num_hidden_layers))(*[
                buffer(a) for pair in candidate for a in pair
            ])
            hidden = np.empty((1,count,self.config.hidden_size),np.float32)
            logits = np.empty((1,count,self.config.vocab_size),np.float32) if self._head.p else None
            routing_shape=(1,count,self.config.num_hidden_layers,self.config.n_routed_experts)
            routing_logits=np.empty(routing_shape,np.float32) if capture_routing else None
            routing_weights=np.empty(routing_shape,np.float32) if capture_routing else None
            for i in range(count):
                if cancelled is not None and cancelled():
                    raise InterruptedError('GLM generation cancelled; continuation unchanged')
                self._step(int(ids[0,i]), start+i, cache_buffers, hidden[0,i], None if logits is None else logits[0,i],
                    cancelled=cancelled,embedding=None if embeddings is None else embeddings[0,i],
                    routing_logits=None if routing_logits is None else routing_logits[0,i],
                    routing_weights=None if routing_weights is None else routing_weights[0,i])
            result = {"last_hidden_state": hidden, "position": start+count}
            if logits is not None:
                result["logits"] = logits
            if capture_routing:
                if not np.isfinite(routing_logits).all() or not np.isfinite(routing_weights).all():
                    raise ValueError('nonfinite GLM routing observation')
                result['router_logits']=routing_logits
                result['routing_weights']=routing_weights
            if use_cache:
                if cancelled is not None and cancelled():
                    raise InterruptedError('GLM generation cancelled; continuation unchanged')
                self._cache = candidate
                self.position = start+count
            return result

    def status(self):
        with self._lock:
            return dict(super().status(), implementation=self.implementation,
                        native_range_execution=True, native_streamed_range_execution=False,
                        closed=self._closed,
                        supported_contract="batch1-unpadded-NoPE-KDA-KPool-MHC-dense-MoE-FP32")
