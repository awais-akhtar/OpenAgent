"""Checkpoint-backed native OLMoE ranges with admitted resident FP32 experts.

Packed INT8-row is decoded at load time, not by C during execution. No Python
range math, fallback, JIT or hidden disk runtime survives successful loading.
"""
import ctypes as C
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np

from .expert_kernels import _decode_payload
from .expert_packing import _scan_directory
from .expert_runtime import SparseExpertManifest
from .native_segment_executor import (
    NativeSegmentExecutor, NativeRangeSession, NativeExecutionError,
    RangeConfig, RangeLayer, VERSION, NUMERIC_CLASS, _tensor, _array)
from .olmoe import OlmoeConfig, OlmoeModel, dense_execution_fingerprint, validate_olmoe_tensors
from .olmoe_batch import _token_vector
from .olmoe_segments import OlmoeSegmentEngine, _integer, _cancel


_CHUNK = 65536


def _budget(value, name):
    return _integer(value, 1, int(np.iinfo(np.intp).max), name)


def _finite(array):
    chunks = np.nditer(array,flags=["external_loop","buffered","zerosize_ok"],
                       op_flags=["readonly"],order="C",buffersize=_CHUNK)
    return all(np.isfinite(chunk).all() for chunk in chunks)


def _shapes(c):
    h, i, e = c.hidden_size, c.intermediate_size, c.num_experts
    w = c.num_key_value_heads * (h // c.num_attention_heads)
    return dict(input_norm=(h,), q=(h,h), k=(w,h), v=(w,h), o=(h,h),
                q_norm=(h,), k_norm=(w,), post_norm=(h,), router=(e,h),
                gate=(e,i,h), up=(e,i,h), down=(e,h,i))


def _bytes(shape):
    return int(np.prod(shape, dtype=object)) * 4


def _admit(sizes, resident_limit, tensor_limit):
    required = sum(sizes)
    if required > resident_limit:
        raise MemoryError(f"resident FP32 weights require {required} bytes; budget is {resident_limit}")
    if any(size > tensor_limit for size in sizes):
        raise MemoryError("weight tensor exceeds native max_bytes")
    return required


def _decode_expert(payload, banks, expert, h, i):
    # No view escapes the lease. Multiplication writes directly into owned banks.
    # Copy packed bytes so exception tracebacks cannot pin mmap-exported arrays
    # while loader cleanup closes the disk store.
    g, u, d, gs, us, ds = _decode_payload(memoryview(bytes(payload)), h, i, "int8-row", len(payload))
    for bank, matrix, scales in zip(banks, (g,u,d), (gs,us,ds)):
        for row in range(matrix.shape[0]):
            for offset in range(0, matrix.shape[1], _CHUNK):
                np.multiply(matrix[row,offset:offset + _CHUNK], scales[row],
                            out=bank[expert,row,offset:offset + _CHUNK], dtype=np.float32)
        if not _finite(bank[expert]):
            raise ValueError("nonfinite decoded expert")


class _OwnedRange:
    """Register already privately owned loader arrays without duplicating banks."""
    def __init__(self, executor, c, layers, capacity):
        self.executor = executor
        self.config = RangeConfig(VERSION,c.num_attention_heads,c.num_key_value_heads,
            c.num_experts_per_tok,int(c.norm_topk_prob),0,c.hidden_size,c.intermediate_size,
            c.num_experts,len(layers),capacity,c.rms_norm_eps,c.rope_theta)
        shapes = _shapes(c)
        self._layers = (RangeLayer * len(layers))()
        self._weights = []
        for index, layer in enumerate(layers):
            if set(layer) != set(shapes):
                raise ValueError("native layer fields mismatch")
            for name, shape in shapes.items():
                value = layer[name]
                if (value.dtype != np.dtype("float32") or value.shape != shape or
                        not value.flags.c_contiguous or not _finite(value)):
                    raise ValueError(f"invalid owned native tensor {name}")
                value.flags.writeable = False
                self._weights.append(value)
                setattr(self._layers[index],name,_tensor(value))


class NativeOlmoeSegmentEngine:
    """Requested-layer checkpoint loader; all selected experts become resident.

    max_resident_bytes bounds exact final weight bytes only, not KV, C scratch,
    disk cache, conversion temporaries or process RSS. INT8-row manifests only.
    """
    native_range_execution = True
    range_resident_weights = True
    sparse_streamed_experts = False
    numeric_class = NUMERIC_CLASS
    upstream_abi_compatible = False

    def __init__(self, model_directory, manifest, begin, end, library,
                 max_resident_bytes, context_tokens=None, *, max_bytes=256 * 1024 * 1024,
                 max_conversion_bytes=None, max_snapshot_bytes=256 * 1024 * 1024):
        self.root = Path(model_directory).resolve(strict=True)
        self.config = c = OlmoeConfig.read(self.root)
        self.begin = _integer(begin,0,c.num_hidden_layers - 1,"begin")
        self.end = _integer(end,self.begin + 1,c.num_hidden_layers,"end")
        self.context_tokens = _integer(c.max_position_embeddings if context_tokens is None else context_tokens,
                                       1,c.max_position_embeddings,"context_tokens")
        self.max_resident_bytes = _budget(max_resident_bytes,"max_resident_bytes")
        self.max_bytes = _budget(max_bytes,"max_bytes")
        self.max_snapshot_bytes = _budget(max_snapshot_bytes,"max_snapshot_bytes")
        self.manifest = SparseExpertManifest.read(manifest) if isinstance(manifest,(str,Path)) else manifest
        self.manifest.validate()
        shapes = _shapes(c)
        sizes = [_bytes(s) for s in shapes.values()]
        self.resident_weight_bytes = _admit(sizes * (end - begin),self.max_resident_bytes,self.max_bytes)
        h,i = c.hidden_size,c.intermediate_size
        payload_bytes = 3*h*i + 4*(2*i+h)
        selected = self.manifest.layers[begin:end]
        if len(selected) != end - begin or any(layer.dtype != "int8-row" or
                any(region.size != payload_bytes for region in layer.regions.values()) for layer in selected):
            raise ValueError("native checkpoint loader requires exact INT8-row payload lengths")
        self.expert_cache_bytes = payload_bytes
        # Conservative array/byte temporary allowance beyond final owned weights.
        # Includes packed cache/copy, dense reads or endian fingerprint conversion
        # and bounded finite checks/ufunc work; excludes metadata, mmap VA and RSS.
        self.conversion_temporary_bound_bytes = 2*max(sizes) + 2*payload_bytes + 8*(2*i+h) + 16*_CHUNK
        if max_conversion_bytes is not None and self.conversion_temporary_bound_bytes > _budget(
                max_conversion_bytes,"max_conversion_bytes"):
            raise MemoryError("checkpoint conversion exceeds max_conversion_bytes")
        self.executor = NativeSegmentExecutor(library,max_bytes=self.max_bytes)
        self._lock = threading.RLock()
        self._sessions = set()
        self._closed = False
        loader = None
        try:
            loader = OlmoeSegmentEngine(self.root,self.manifest,begin,end,self.context_tokens,
                expert_cache_bytes=payload_bytes)
            layers = [dict(layer) for layer in loader.layers]
            for local,layer in enumerate(layers):
                banks = [np.empty(shapes[name],np.float32) for name in ("gate","up","down")]
                for expert in range(c.num_experts):
                    key = loader.experts.key(f"model.layers.{begin + local}",str(expert))
                    with loader.experts.store.acquire(key) as payload:
                        _decode_expert(payload,banks,expert,h,i)
                layer.update(zip(("gate","up","down"),banks))
            self._range = _OwnedRange(self.executor,c,layers,self.context_tokens)
            if sum(a.nbytes for a in self._range._weights) != self.resident_weight_bytes:
                raise RuntimeError("resident weight admission mismatch")
            content = dense_execution_fingerprint((f"layer.{begin + layer_index}.{name}",layer[name])
                for layer_index,layer in enumerate(layers) for name in sorted(layer))
            identity = dict(numeric_class=self.numeric_class,weights=content,
                range=[begin,end],hidden=h,intermediate=i,experts=c.num_experts,
                heads=c.num_attention_heads,kv_heads=c.num_key_value_heads,
                topk=c.num_experts_per_tok,normalize_topk=c.norm_topk_prob,
                epsilon=c.rms_norm_eps,theta=c.rope_theta)
            self.execution_identity = hashlib.sha256(json.dumps(identity,sort_keys=True,
                allow_nan=False,separators=(",",":")).encode("ascii")).hexdigest()
        finally:
            if loader is not None:
                loader.close()

    def _check(self):
        if self._closed:
            raise RuntimeError("native engine is closed")

    @classmethod
    def compose(cls,engines,*,max_resident_bytes=None,max_bytes=None,context_tokens=None):
        """Register adjacent loaded ranges as one weight-sharing native engine.

        Shares immutable FP32 banks, not source sessions/KV. The composed engine
        pins banks independently, so source engines may close after composition.
        It has its own identity and new KV sessions; no existing KV is migrated.
        """
        engines = tuple(engines)
        if not engines or any(not isinstance(e,cls) for e in engines):
            raise TypeError("expected a nonempty sequence of NativeOlmoeSegmentEngine")
        if len({id(e) for e in engines}) != len(engines):
            raise ValueError("duplicate source ranges")
        if any(e.sparse_streamed_experts for e in engines):
            raise ValueError("resident composition cannot accept leased expert ranges")
        with ExitStack() as locks:
            for engine in sorted(engines,key=id):
                locks.enter_context(engine._lock)
                engine._check()
            first = engines[0]
            if any(e.config != first.config or e.manifest.fingerprint != first.manifest.fingerprint
                   for e in engines):
                raise ValueError("source range checkpoint configs or fingerprints differ")
            if any(left.end != right.begin for left,right in zip(engines,engines[1:])):
                raise ValueError("source ranges must be adjacent and ordered")
            resident = sum(e.resident_weight_bytes for e in engines)
            budget = resident if max_resident_bytes is None else _budget(max_resident_bytes,"max_resident_bytes")
            limit = min(e.max_bytes for e in engines) if max_bytes is None else _budget(max_bytes,"max_bytes")
            owned_weights = [a for e in engines for a in e._range._weights]
            _admit([a.nbytes for a in owned_weights],budget,limit)
            capacity_limit = min(e.context_tokens for e in engines)
            capacity = capacity_limit if context_tokens is None else _integer(
                context_tokens,1,capacity_limit,"context_tokens")
            merged = cls.__new__(cls)
            merged.root,merged.config,merged.manifest = first.root,first.config,first.manifest
            merged.begin,merged.end = first.begin,engines[-1].end
            merged.context_tokens = capacity
            merged.max_resident_bytes,merged.max_bytes = budget,limit
            merged.max_snapshot_bytes = min(e.max_snapshot_bytes for e in engines)
            merged.resident_weight_bytes = resident
            merged.conversion_temporary_bound_bytes = 0
            merged.expert_cache_bytes = 0
            merged.executor = copy.copy(first.executor)
            merged.executor.max_bytes = limit
            merged._lock,merged._sessions,merged._closed = threading.RLock(),set(),False
            merged._shares_resident_weights = True
            config = RangeConfig.from_buffer_copy(first._range.config)
            config.layers = sum(e.end-e.begin for e in engines)
            config.capacity = capacity
            descriptors = (RangeLayer * config.layers)()
            offset = 0
            for engine in engines:
                for layer in engine._range._layers:
                    descriptors[offset] = layer
                    offset += 1
            merged._range = SimpleNamespace(executor=merged.executor,config=config,
                _layers=descriptors,_weights=owned_weights)
            identity = dict(numeric_class=merged.numeric_class,range=[merged.begin,merged.end],
                            constituent_identities=[e.execution_identity for e in engines])
            merged.execution_identity = hashlib.sha256(json.dumps(identity,sort_keys=True,
                separators=(",",":")).encode("ascii")).hexdigest()
            return merged

    def create_session(self, *, context_tokens=None):
        with self._lock:
            self._check()
            capacity = _integer(self.context_tokens if context_tokens is None else context_tokens,
                                1,self.context_tokens,"session context_tokens")
            session = NativeOlmoeSegmentSession(self,capacity)
            self._sessions.add(session)
            return session

    def forward(self, session, hidden, *, cancelled=None):
        if not isinstance(session,NativeOlmoeSegmentSession) or session._engine is not self:
            raise ValueError("session belongs to another native engine")
        return session.forward(hidden,cancelled=cancelled)

    def status(self):
        with self._lock:
            return dict(range=(self.begin,self.end),numeric_class=self.numeric_class,
                native_range_execution=True,range_resident_weights=True,
                sparse_streamed_experts=False,expert_storage="resident-fp32-decoded-int8-row",
                resident_weight_bytes=0 if self._closed else self.resident_weight_bytes,
                admitted_weight_bytes=self.resident_weight_bytes,max_resident_bytes=self.max_resident_bytes,
                conversion_temporary_bound_bytes=self.conversion_temporary_bound_bytes,
                shares_resident_weights=getattr(self,"_shares_resident_weights",False),
                disk_runtime_open=False,live_sessions=len(self._sessions),closed=self._closed)

    def close(self):
        with self._lock:
            if self._sessions:
                raise RuntimeError("close live sessions before closing native engine")
            if not self._closed:
                self._range = None
                self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self,*exc):
        self.close()


class NativeOlmoeSegmentSession:
    native_range_execution = True
    numeric_class = NUMERIC_CLASS

    def __init__(self,engine,capacity):
        self._engine = engine
        self.context_tokens = capacity
        self._lock = threading.RLock()
        self._closed = False
        self._active = False
        owned = engine._range
        config = RangeConfig.from_buffer_copy(owned.config)
        config.capacity = capacity
        view = SimpleNamespace(executor=engine.executor,config=config,_layers=owned._layers)
        self._native = NativeRangeSession(view)
        self.kv_bytes = self._native.kv_bytes

    def _check(self):
        self._engine._check()
        if self._closed:
            raise RuntimeError("native session is closed")
        if self._active:
            raise RuntimeError("native session is executing")

    @property
    def position(self):
        with self._lock:
            return self._native.position

    def forward(self,hidden,*,cancelled=None):
        with self._engine._lock,self._lock:
            self._check()
            self._active = True
            try:
                return self._forward(hidden,cancelled)
            finally:
                self._active = False

    def _forward(self,hidden,cancelled):
        _cancel(cancelled)
        if cancelled is None:
            return self._native.forward(hidden)
        # C already stages internally. A second Python-owned KV pair delays
        # publication until the post-call cancellation boundary succeeds.
        native = self._native
        c = native.engine.config
        x = np.asarray(hidden)
        if x.ndim != 2 or not x.shape[0]:
            raise ValueError("hidden must be nonempty FP32 rows x hidden")
        x = _array(x,(x.shape[0],c.hidden),self._engine.max_bytes,"hidden")
        if x.shape[0] > self.context_tokens - native.position:
            raise NativeExecutionError(5)
        keys,values = native._keys.copy(),native._values.copy()
        position = C.c_uint64(native.position)
        out = np.empty_like(x)
        self._engine.executor._check(self._engine.executor.lib.oa_executor_range(C.byref(c),
            native.engine._layers,c.layers,_tensor(x),len(x),out.ctypes.data,out.size,
            keys.ctypes.data,values.ctypes.data,keys.size,C.byref(position),self._engine.max_bytes))
        _cancel(cancelled)
        native._keys,native._values = keys,values
        native._position.value = position.value
        return out

    def reset(self):
        with self._engine._lock,self._lock:
            self._check()
            self._native.reset()

    def snapshot_metadata(self):
        with self._engine._lock,self._lock:
            self._check()
            e = self._engine
            return dict(format="openagent.native-olmoe-segment.v1",fingerprint=e.manifest.fingerprint,
                execution_identity=e.execution_identity,range=(e.begin,e.end),
                numeric_class=self.numeric_class,context_tokens=self.context_tokens)

    def snapshot_state(self):
        with self._engine._lock,self._lock:
            metadata = self.snapshot_metadata()
            pos = self.position
            n = self._native
            active_bytes = 2*n._keys.shape[0]*n._keys.shape[1]*pos*n._keys.shape[3]*4
            if active_bytes > self._engine.max_snapshot_bytes:
                raise MemoryError("snapshot arrays exceed max_snapshot_bytes")
            return dict(metadata,position=pos,
                keys=[a[:,:pos].copy() for a in n._keys],
                values=[a[:,:pos].copy() for a in n._values])

    def restore_state(self,state):
        with self._engine._lock,self._lock:
            expected = self.snapshot_metadata()
            if type(state) is not dict or set(state) != set(expected) | {"position","keys","values"}:
                raise ValueError("invalid native snapshot fields")
            for name,value in expected.items():
                if type(state[name]) is not type(value) or state[name] != value:
                    raise ValueError(f"native snapshot {name} mismatch")
            if any(type(i) is not int for i in state["range"]):
                raise ValueError("invalid native snapshot range")
            pos = _integer(state["position"],0,self.context_tokens,"snapshot position")
            n = self._native
            if n.kv_bytes > self._engine.max_snapshot_bytes:
                raise MemoryError("restore staging exceeds max_snapshot_bytes")
            for field,targets in (("keys",n._keys),("values",n._values)):
                if type(state[field]) is not list or len(state[field]) != len(targets):
                    raise ValueError("invalid native snapshot layer count")
                for source,target in zip(state[field],targets):
                    if (not isinstance(source,np.ndarray) or source.dtype != np.dtype("float32") or
                            source.shape != target[:,:pos].shape or not _finite(source)):
                        raise ValueError("invalid native snapshot KV")
            keys,values = np.zeros_like(n._keys),np.zeros_like(n._values)
            for destination,field in ((keys,"keys"),(values,"values")):
                for target,source in zip(destination,state[field]):
                    target[:,:pos] = source
            n._keys,n._values = keys,values
            n._position.value = pos

    def close(self):
        with self._engine._lock,self._lock:
            if self._active:
                raise RuntimeError("native session is executing")
            if not self._closed:
                self._native.close()
                self._engine._sessions.discard(self)
                self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self,*exc):
        self.close()


class NativeOlmoeEdge:
    """Load only requested embedding/head payloads; no layer or expert payloads."""
    _tensor = OlmoeModel._tensor
    native_range_execution = True
    numeric_class = NUMERIC_CLASS

    def __init__(self,model_directory,library,max_resident_bytes,*,mode="both",
                 max_bytes=256 * 1024 * 1024,max_conversion_bytes=None):
        if mode not in ("embedding","head","both"):
            raise ValueError("mode must be embedding, head or both")
        self.root = Path(model_directory).resolve(strict=True)
        # The edge loads dense checkpoint tensors independently of the leased
        # range engine. Guard this public loader boundary itself so direct
        # callers cannot rely on a separate engine/CLI admission having run.
        from .source_scope import require_checkpoint_inventory_admission
        require_checkpoint_inventory_admission(
            self.root,family='olmoe',model_type='olmoe',backend='native',
            arithmetic='fp32',max_metadata_bytes=max_bytes,
            persisted_format='float32')
        self.config = c = OlmoeConfig.read(self.root)
        self.mode = mode
        self.max_resident_bytes = _budget(max_resident_bytes,"max_resident_bytes")
        self.max_bytes = _budget(max_bytes,"max_bytes")
        sizes = ([c.vocab_size*c.hidden_size*4] if mode in ("embedding","both") else []) + (
            [c.vocab_size*c.hidden_size*4,c.hidden_size*4] if mode in ("head","both") else [])
        self.resident_weight_bytes = _admit(sizes,self.max_resident_bytes,self.max_bytes)
        self.conversion_temporary_bound_bytes = 2*max(sizes) + 16*_CHUNK
        if max_conversion_bytes is not None and self.conversion_temporary_bound_bytes > _budget(
                max_conversion_bytes,"max_conversion_bytes"):
            raise MemoryError("edge conversion exceeds max_conversion_bytes")
        self.executor = NativeSegmentExecutor(library,max_bytes=self.max_bytes)
        self._lock = threading.RLock()
        self._closed = False
        self._index = _scan_directory(self.root)
        validate_olmoe_tensors(c,self._index)
        self.embed = self.lm_head = self.final_norm = None
        shape = (c.vocab_size,c.hidden_size)
        if mode in ("embedding","both"):
            self.embed = self._tensor("model.embed_tokens.weight",shape)
        if mode in ("head","both"):
            self.lm_head = self._tensor("lm_head.weight",shape)
            self.final_norm = self._tensor("model.norm.weight",(c.hidden_size,))
        for array in (self.embed,self.lm_head,self.final_norm):
            if array is not None:
                if not _finite(array):
                    raise ValueError("nonfinite native edge weights")
                array.flags.writeable = False
        self._index.clear()
        self._decoder_sessions = set()
        tensors = [(name,array) for name,array in (("embedding",self.embed),
                   ("head",self.lm_head),("norm",self.final_norm)) if array is not None]
        identity = dict(weights=dense_execution_fingerprint(tensors),mode=mode,
                        hidden=c.hidden_size,vocab=c.vocab_size,epsilon=c.rms_norm_eps,
                        numeric_class=self.numeric_class)
        self.execution_identity = hashlib.sha256(json.dumps(identity,sort_keys=True,
            separators=(",",":"),allow_nan=False).encode("ascii")).hexdigest()

    def _check(self,required):
        if self._closed:
            raise RuntimeError("native edge is closed")
        if self.mode not in (required,"both"):
            raise RuntimeError(f"native edge did not load {required} weights")

    def embeddings(self,token_ids):
        with self._lock:
            self._check("embedding")
            ids = np.ascontiguousarray(_token_vector(token_ids),dtype=np.int64)
            h = self.config.hidden_size
            if ids.nbytes > self.max_bytes or len(ids)*h*4 > self.max_bytes:
                raise NativeExecutionError(5)
            out = np.empty((len(ids),h),np.float32)
            self.executor._check(self.executor.lib.oa_executor_embed(_tensor(self.embed),
                self.config.vocab_size,h,ids.ctypes.data,len(ids),out.ctypes.data,out.size,self.max_bytes))
            return out

    def final_head(self,hidden):
        with self._lock:
            self._check("head")
            x = np.asarray(hidden)
            h,v = self.config.hidden_size,self.config.vocab_size
            if x.ndim != 2 or not x.shape[0]:
                raise ValueError("hidden must be nonempty FP32 rows x hidden")
            x = _array(x,(len(x),h),self.max_bytes,"hidden")
            if len(x)*v*4 > self.max_bytes:
                raise NativeExecutionError(5)
            out = np.empty((len(x),v),np.float32)
            self.executor._check(self.executor.lib.oa_executor_head(_tensor(x),len(x),h,
                _tensor(self.final_norm),_tensor(self.lm_head),v,self.config.rms_norm_eps,
                out.ctypes.data,out.size,self.max_bytes))
            return out

    def status(self):
        with self._lock:
            return dict(mode=self.mode,native_range_execution=True,numeric_class=self.numeric_class,
                resident_weight_bytes=0 if self._closed else self.resident_weight_bytes,
                admitted_weight_bytes=self.resident_weight_bytes,max_resident_bytes=self.max_resident_bytes,
                conversion_temporary_bound_bytes=self.conversion_temporary_bound_bytes,
                live_decoder_sessions=len(self._decoder_sessions),closed=self._closed)

    def create_session(self,engine,*,context_tokens=None):
        """Full-checkpoint token-to-logit session with atomic C KV/head execution."""
        if not isinstance(engine,NativeOlmoeSegmentEngine):
            raise TypeError("expected NativeOlmoeSegmentEngine")
        with engine._lock,self._lock:
            engine._check()
            self._check("both")
            if engine.sparse_streamed_experts:
                raise RuntimeError("leased ranges require engine.create_decoder_session(edge)")
            if engine.begin != 0 or engine.end != engine.config.num_hidden_layers:
                raise ValueError("decoder session requires the full layer range")
            if engine.config != self.config:
                raise ValueError("edge and range checkpoint configs differ")
            if engine.executor._decode is None:
                raise RuntimeError("native library lacks atomic decode; rebuild segment_executor.c")
            session = NativeOlmoeDecoderSession(self,engine,context_tokens)
            self._decoder_sessions.add(session)
            return session

    def close(self):
        with self._lock:
            if self._decoder_sessions:
                raise RuntimeError("close live decoder sessions before closing native edge")
            self.embed = self.lm_head = self.final_norm = None
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("native edge is closed")
        return self

    def __exit__(self,*exc):
        self.close()


class NativeOlmoeDecoderSession:
    """Token IDs -> all-row logits, with one C transaction through final head.

    Owns a range session and pins edge weights. Snapshot identity includes both
    edges and range. This does not provide sampling or packed expert streaming.
    """
    native_range_execution = True
    numeric_class = NUMERIC_CLASS

    def __init__(self,edge,engine,capacity):
        self._edge,self._engine = edge,engine
        self._session = engine.create_session(context_tokens=capacity)
        self.context_tokens = self._session.context_tokens
        self.kv_bytes = self._session.kv_bytes
        self._closed = False

    @property
    def position(self):
        return self._session.position

    def _check(self):
        if self._closed:
            raise RuntimeError("native decoder session is closed")
        self._edge._check("both")
        self._session._check()

    def forward(self,token_ids,*,cancelled=None):
        with self._engine._lock,self._edge._lock,self._session._lock:
            self._check()
            session = self._session
            session._active = True
            try:
                _cancel(cancelled)
                ids = np.ascontiguousarray(_token_vector(token_ids),dtype=np.int64)
                native = session._native
                c = native.engine.config
                limit = min(self._engine.max_bytes,self._edge.max_bytes)
                vocab = self._edge.config.vocab_size
                if ids.nbytes > limit or len(ids)*vocab*4 > limit:
                    raise NativeExecutionError(5)
                if len(ids) > self.context_tokens-native.position:
                    raise NativeExecutionError(5)
                keys,values = native._keys,native._values
                if cancelled is not None:
                    keys,values = keys.copy(),values.copy()
                pos = C.c_uint64(native.position)
                out = np.empty((len(ids),vocab),np.float32)
                self._engine.executor._check(self._engine.executor._decode(C.byref(c),
                    native.engine._layers,c.layers,_tensor(self._edge.embed),
                    _tensor(self._edge.final_norm),_tensor(self._edge.lm_head),vocab,
                    ids.ctypes.data,len(ids),out.ctypes.data,out.size,keys.ctypes.data,
                    values.ctypes.data,keys.size,C.byref(pos),limit))
                _cancel(cancelled)
                native._keys,native._values = keys,values
                native._position.value = pos.value
                return out
            finally:
                session._active = False

    def reset(self):
        with self._engine._lock,self._edge._lock,self._session._lock:
            self._check()
            self._session.reset()

    def snapshot_metadata(self):
        with self._engine._lock,self._edge._lock,self._session._lock:
            self._check()
            return dict(format="openagent.native-olmoe-decoder.v1",
                edge_execution_identity=self._edge.execution_identity,
                range_metadata=self._session.snapshot_metadata())

    def snapshot_state(self):
        with self._engine._lock,self._edge._lock,self._session._lock:
            self._check()
            return dict(format="openagent.native-olmoe-decoder.v1",
                edge_execution_identity=self._edge.execution_identity,
                range_state=self._session.snapshot_state())

    def restore_state(self,state):
        with self._engine._lock,self._edge._lock,self._session._lock:
            self._check()
            if type(state) is not dict or set(state) != {"format","edge_execution_identity","range_state"}:
                raise ValueError("invalid native decoder snapshot fields")
            if type(state["format"]) is not str or state["format"] != "openagent.native-olmoe-decoder.v1":
                raise ValueError("native decoder snapshot format mismatch")
            if (type(state["edge_execution_identity"]) is not str or
                    state["edge_execution_identity"] != self._edge.execution_identity):
                raise ValueError("native decoder snapshot edge_execution_identity mismatch")
            self._session.restore_state(state["range_state"])

    def close(self):
        with self._engine._lock,self._edge._lock,self._session._lock:
            if not self._closed:
                self._session.close()
                self._edge._decoder_sessions.discard(self)
                self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self,*exc):
        self.close()
