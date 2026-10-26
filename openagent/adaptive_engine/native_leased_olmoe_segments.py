"""C transformer phases and INT8-row/Q8_0 experts with bounded leases.

Python coordinates storage, checksums and transactions only. No decoded expert
banks, Python attention/router/SwiGLU/residual math, or math callbacks.
"""
import ctypes as C
import copy
from dataclasses import replace
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from .expert_runtime import SparseExpertManifest, SparseExpertRuntime
from .expert_packing import _scan_directory
from .olmoe import OlmoeModel, validate_olmoe_tensors, olmoe_metadata_fingerprint, dense_execution_fingerprint
from .native_segment_executor import NativeSegmentExecutor, RangeConfig, RangeLayer, VERSION, _tensor, _array
from .native_olmoe_segments import (
    NativeOlmoeSegmentEngine, NativeOlmoeSegmentSession, NativeOlmoeDecoderSession,
    NativeOlmoeEdge, _budget, _bytes, _shapes, _admit, _finite)
from .olmoe import OlmoeConfig
from .olmoe_batch import _token_vector
from .olmoe_segments import OlmoeSegmentEngine, _integer, _cancel


NUMERIC_CLASS = "c-fp32-dense-double-dot-native-int8-row-leased-v1"
Q8_NUMERIC_CLASS = "c-fp32-dense-double-block-dot-native-q8_0-leased-v1"
Q4_NUMERIC_CLASS = "c-fp32-dense-double-block-dot-native-q4_0-leased-v1"
_BLOCK_BYTES = {'q4_0':18, 'q4_1':20, 'q5_0':22, 'q5_1':24, 'q8_0':34, 'q8_1':36}
_APPLY_ATTRIBUTES = {
    'int8-row':'_apply_packed', 'q4_0':'_apply_q4', 'q8_0':'_apply_q8',
    'q4_1':'_apply_q4_1', 'q5_0':'_apply_q5_0', 'q5_1':'_apply_q5_1', 'q8_1':'_apply_q8_1',
}
_COMPOSITION_LOCK = threading.RLock()


class _Q8DenseLoader:
    _tensor = OlmoeModel._tensor

    def __init__(self,root,manifest,begin,end,capacity,cache,storage_roots,*,dtype='q8_0'):
        self.root,self.config,self.manifest = root,OlmoeConfig.read(root),manifest
        c = self.config
        if dtype not in _BLOCK_BYTES:
            raise ValueError("unsupported native block-packed loader dtype")
        if c.hidden_size%32 or c.intermediate_size%32:
            raise ValueError("native block-packed row widths must be multiples of 32")
        payload_bytes = 3*c.hidden_size*c.intermediate_size//32*_BLOCK_BYTES[dtype]
        if manifest.architecture!='olmoe' or manifest.fingerprint!=olmoe_metadata_fingerprint(root) or len(manifest.layers)!=c.num_hidden_layers:
            raise ValueError(f"{dtype} manifest does not match checkpoint")
        for index,layer in enumerate(manifest.layers):
            if (layer.id!=f'model.layers.{index}' or layer.dtype!=dtype or
                layer.hidden_size!=c.hidden_size or layer.intermediate_size!=c.intermediate_size or
                layer.experts_used!=c.num_experts_per_tok or
                set(layer.regions)!={str(i) for i in range(c.num_experts)} or
                any(not r.sha256 or r.size!=payload_bytes for r in layer.regions.values())):
                raise ValueError(f"{dtype} manifest layer contract mismatch")
        self._index,self.layers,self.experts = {},[],None
        try:
            self._load_dense(root,manifest,begin,end,cache,storage_roots,dtype)
        except BaseException:
            self.close()
            raise

    def _load_dense(self,root,manifest,begin,end,cache,storage_roots,dtype):
        c = self.config
        self._index = _scan_directory(root)
        validate_olmoe_tensors(c,self._index)
        self.layers = [OlmoeModel._load_layer(self,index) for index in range(begin,end)]
        selected = replace(manifest,layers=manifest.layers[begin:end])
        dense = dense_execution_fingerprint((f'layer.{begin+i}.{name}',value)
            for i,layer in enumerate(self.layers) for name,value in sorted(layer.items()))
        identity = dict(dense=dense,experts=[layer.to_dict() for layer in selected.layers])
        self.execution_identity = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.experts = SparseExpertRuntime(root,selected,capacity_bytes=cache,
            io_mode='buffered',storage_roots=storage_roots,
            preferred_kernel='q8_0-numpy' if dtype=='q8_0' else dtype+'-reference')

    def close(self):
        if self.experts is not None:
            self.experts.close()
            self.experts = None
        self.layers.clear()
        self._index.clear()


class NativeLeasedOlmoeSegmentEngine(NativeOlmoeSegmentEngine):
    """Selected dense layers resident; selected packed experts leased on demand.

    expert_cache_bytes bounds packed cache storage, not OS cache or process RSS.
    max_staging_bytes admits conservative Python+C transaction scratch bytes.
    Buffered INT8-row or Q8_0 storage; no prefetch/math API.
    """
    sparse_streamed_experts = True
    range_resident_weights = False
    numeric_class = NUMERIC_CLASS

    def __init__(self,model_directory,manifest,begin,end,library,max_resident_bytes,
                 expert_cache_bytes,context_tokens=None,*,max_bytes=256*1024*1024,
                 max_staging_bytes=256*1024*1024,max_snapshot_bytes=256*1024*1024,
                 storage_roots=None):
        self.root = Path(model_directory).resolve(strict=True)
        self.config = c = OlmoeConfig.read(self.root)
        self.begin = _integer(begin,0,c.num_hidden_layers-1,"begin")
        self.end = _integer(end,self.begin+1,c.num_hidden_layers,"end")
        self.context_tokens = _integer(c.max_position_embeddings if context_tokens is None else context_tokens,
                                       1,c.max_position_embeddings,"context_tokens")
        self.max_resident_bytes = _budget(max_resident_bytes,"max_resident_bytes")
        self.max_bytes = _budget(max_bytes,"max_bytes")
        self.max_staging_bytes = _budget(max_staging_bytes,"max_staging_bytes")
        self.max_snapshot_bytes = _budget(max_snapshot_bytes,"max_snapshot_bytes")
        self.expert_cache_bytes = _budget(expert_cache_bytes,"expert_cache_bytes")
        self.manifest = SparseExpertManifest.read(manifest) if isinstance(manifest,(str,Path)) else copy.deepcopy(manifest)
        if not isinstance(self.manifest,SparseExpertManifest):
            raise TypeError("expected SparseExpertManifest or manifest path")
        self.manifest.validate()
        if len(self.manifest.layers)!=c.num_hidden_layers:
            raise ValueError("leased manifest layer count does not match checkpoint")
        dtype = self.manifest.layers[begin].dtype
        if dtype not in _APPLY_ATTRIBUTES:
            raise ValueError("unsupported native leased dtype: "+dtype)
        # Bind the original checkpoint inventory to the exact expert format
        # selected by this leased executor before loading the native ABI or
        # reading any tensor payload.  The manifest format is not inferable
        # from the dense SafeTensors dtype.
        from .source_scope import require_checkpoint_inventory_admission
        source_inventory = require_checkpoint_inventory_admission(
            self.root,family='olmoe',model_type='olmoe',backend='native',
            arithmetic='fp32',max_metadata_bytes=self.max_snapshot_bytes,
            persisted_format=dtype)
        self.dtype = dtype
        self.numeric_class = (f"c-fp32-dense-double-block-dot-native-{dtype}-leased-v1"
                              if dtype in _BLOCK_BYTES else NUMERIC_CLASS)
        shapes = {name:shape for name,shape in _shapes(c).items() if name not in ("gate","up","down")}
        self.resident_weight_bytes = _admit([_bytes(s) for s in shapes.values()]*(end-begin),
                                          self.max_resident_bytes,self.max_bytes)
        if dtype in _BLOCK_BYTES and (c.hidden_size%32 or c.intermediate_size%32):
            raise ValueError("native block-packed row widths must be multiples of 32")
        self.payload_bytes = (3*c.hidden_size*c.intermediate_size//32*_BLOCK_BYTES[dtype]
            if dtype in _BLOCK_BYTES else
            3*c.hidden_size*c.intermediate_size+4*(2*c.intermediate_size+c.hidden_size))
        selected = self.manifest.layers[begin:end]
        if len(selected)!=end-begin or any(layer.dtype!=dtype or
                any(region.size!=self.payload_bytes for region in layer.regions.values()) for layer in selected):
            raise ValueError("leased native engine requires exact single-format regions")
        if self.payload_bytes>self.expert_cache_bytes or self.payload_bytes>self.max_bytes:
            raise MemoryError("one packed expert exceeds cache or native transfer budget")
        self.executor = NativeSegmentExecutor(library,max_bytes=self.max_bytes)
        self._lock,self._sessions,self._closed = threading.RLock(),set(),False
        self._compositions = set()
        self.conversion_temporary_bound_bytes = 2*max(_bytes(s) for s in shapes.values())+16*65536
        loader = None
        try:
            self.executor.require_leased()
            if dtype in _BLOCK_BYTES:
                self.executor.require_quantized(dtype)
            if dtype in _BLOCK_BYTES:
                loader = _Q8DenseLoader(self.root,self.manifest,begin,end,self.context_tokens,
                    self.expert_cache_bytes,storage_roots,dtype=dtype)
            else:
                loader = OlmoeSegmentEngine(self.root,self.manifest,begin,end,self.context_tokens,
                    expert_cache_bytes=self.expert_cache_bytes,io_mode="buffered",storage_roots=storage_roots)
            source_inventory.assert_unchanged()
            descriptors = (RangeLayer*(end-begin))()
            owned = []
            for index,layer in enumerate(loader.layers):
                if set(layer)!=set(shapes):
                    raise ValueError("leased native dense layer fields mismatch")
                for name,shape in shapes.items():
                    value = layer[name]
                    if (value.shape!=shape or value.dtype!=np.dtype("float32") or
                            not value.flags.c_contiguous or not value.flags.aligned or not _finite(value)):
                        raise ValueError(f"invalid leased dense tensor {name}")
                    value.flags.writeable = False
                    owned.append(value)
                    setattr(descriptors[index],name,_tensor(value))
            cfg = RangeConfig(VERSION,c.num_attention_heads,c.num_key_value_heads,c.num_experts_per_tok,
                int(c.norm_topk_prob),0,c.hidden_size,c.intermediate_size,c.num_experts,
                end-begin,self.context_tokens,c.rms_norm_eps,c.rope_theta)
            self._range = SimpleNamespace(executor=self.executor,config=cfg,_layers=descriptors,_weights=owned)
            identity = dict(numeric_class=self.numeric_class,loader_identity=loader.execution_identity,
                range=[begin,end],heads=c.num_attention_heads,kv_heads=c.num_key_value_heads,
                epsilon=c.rms_norm_eps,theta=c.rope_theta,topk=c.num_experts_per_tok,
                normalize_topk=c.norm_topk_prob)
            self.execution_identity = hashlib.sha256(json.dumps(identity,sort_keys=True,
                separators=(",",":"),allow_nan=False).encode("ascii")).hexdigest()
            self._loader = loader
        except BaseException:
            try:
                if loader is not None:
                    loader.close()
            finally:
                self._range = None
                self.executor._release()
                self.executor = None
            raise

    @classmethod
    def compose(cls,engines,*,max_resident_bytes=None,max_bytes=None,context_tokens=None,
                max_staging_bytes=None,max_snapshot_bytes=None):
        """Share adjacent immutable dense ranges and pin their leased stores.

        Source sessions are never adopted. Closing sources is rejected until
        this composition closes; new sessions own one range-wide KV transaction.
        """
        engines = tuple(engines)
        if not engines or any(type(e) is not cls for e in engines):
            raise TypeError("expected nonempty leased engine sequence")
        if len({id(e) for e in engines}) != len(engines):
            raise ValueError("duplicate source ranges")
        with _COMPOSITION_LOCK, ExitStack() as locks:
            for e in sorted(engines,key=id):
                locks.enter_context(e._lock)
                e._check()
            first = engines[0]
            if any(e.config != first.config or e.manifest != first.manifest for e in engines):
                raise ValueError("source checkpoint configs or exact manifests differ")
            if any(e.dtype != first.dtype or e.numeric_class != first.numeric_class for e in engines):
                raise ValueError("source leased formats or numeric classes differ")
            if any(a.end != b.begin for a,b in zip(engines,engines[1:])):
                raise ValueError("source ranges must be adjacent and ordered")
            resident = sum(e.resident_weight_bytes for e in engines)
            budget = resident if max_resident_bytes is None else _budget(max_resident_bytes,"max_resident_bytes")
            limit = min(e.max_bytes for e in engines) if max_bytes is None else _budget(max_bytes,"max_bytes")
            weights = [a for e in engines for a in e._range._weights]
            _admit([a.nbytes for a in weights],budget,limit)
            if first.payload_bytes > limit:
                raise MemoryError("one packed expert exceeds composed native transfer budget")
            capacity_limit = min(e.context_tokens for e in engines)
            capacity = capacity_limit if context_tokens is None else _integer(context_tokens,1,capacity_limit,"context_tokens")
            merged = cls.__new__(cls)
            merged.root,merged.config,merged.manifest = first.root,first.config,first.manifest
            merged.begin,merged.end = first.begin,engines[-1].end
            merged.context_tokens = capacity
            merged.max_bytes,merged.max_resident_bytes = limit,budget
            merged.max_staging_bytes = min(e.max_staging_bytes for e in engines) if max_staging_bytes is None else _budget(max_staging_bytes,"max_staging_bytes")
            merged.max_snapshot_bytes = min(e.max_snapshot_bytes for e in engines) if max_snapshot_bytes is None else _budget(max_snapshot_bytes,"max_snapshot_bytes")
            merged.resident_weight_bytes = resident
            merged.payload_bytes = first.payload_bytes
            merged.dtype,merged.numeric_class = first.dtype,first.numeric_class
            merged.conversion_temporary_bound_bytes = 0
            merged._sources = engines
            merged._source_layers = tuple(source for e in engines for source in
                getattr(e,"_source_layers",(e,)*(e.end-e.begin)))
            merged.expert_cache_bytes = sum(e.expert_cache_bytes for e in engines)
            merged.executor = copy.copy(first.executor)
            merged.executor.max_bytes = limit
            cfg = RangeConfig.from_buffer_copy(first._range.config)
            cfg.layers,cfg.capacity = merged.end-merged.begin,capacity
            descriptors = (RangeLayer*cfg.layers)()
            offset = 0
            for e in engines:
                for layer in e._range._layers:
                    descriptors[offset] = layer
                    offset += 1
            merged._range = SimpleNamespace(executor=merged.executor,config=cfg,_layers=descriptors,_weights=weights)
            merged._lock,merged._sessions,merged._closed = threading.RLock(),set(),False
            merged._compositions = set()
            identity = dict(numeric_class=merged.numeric_class,range=[merged.begin,merged.end],
                            constituent_identities=[e.execution_identity for e in engines])
            merged.execution_identity = hashlib.sha256(json.dumps(identity,sort_keys=True,
                separators=(",",":")).encode("ascii")).hexdigest()
            try:
                for e in engines:
                    e._compositions.add(merged)
            except BaseException:
                for e in engines:
                    e._compositions.discard(merged)
                raise
            return merged

    def create_session(self,*,context_tokens=None):
        with self._lock:
            self._check()
            capacity = _integer(self.context_tokens if context_tokens is None else context_tokens,
                                1,self.context_tokens,"context_tokens")
            session = NativeLeasedOlmoeSegmentSession(self,capacity)
            try:
                self._sessions.add(session)
            except BaseException:
                session.close()
                raise
            return session

    def create_decoder_session(self,edge,*,context_tokens=None):
        if not isinstance(edge,NativeOlmoeEdge):
            raise TypeError("expected NativeOlmoeEdge")
        with self._lock,edge._lock:
            self._check()
            edge._check("both")
            if self.begin!=0 or self.end!=self.config.num_hidden_layers or self.config!=edge.config:
                raise ValueError("leased decoder requires full matching range and edge configs")
            decoder = NativeLeasedOlmoeDecoderSession(edge,self,context_tokens)
            try:
                edge._decoder_sessions.add(decoder)
            except BaseException:
                decoder.close()
                raise
            return decoder

    def staging_required(self,rows,capacity,*,vocab=None):
        c = self.config
        n = rows*c.hidden_size
        kv = c.num_key_value_heads*(c.hidden_size//c.num_attention_heads)
        cache = capacity*kv
        routes = rows*c.num_experts_per_tok
        prepare = 4*(2*n+2*cache+3*c.hidden_size+2*kv+capacity+c.num_experts+routes)+8*routes
        apply = 4*(n+c.intermediate_size+c.hidden_size)
        amount = 8*(self.end-self.begin)*cache+28*n+12*routes+self.payload_bytes+max(prepare,apply)
        if vocab is not None:
            amount += 4*(2*n+2*rows*vocab+c.hidden_size)+16*rows
        return amount

    def _admit_call(self,rows,capacity,*,vocab=None):
        if self.staging_required(rows,capacity,vocab=vocab)>self.max_staging_bytes:
            raise MemoryError("leased transaction exceeds max_staging_bytes")
        c = self.config
        n = rows*c.hidden_size
        kv = c.num_key_value_heads*(c.hidden_size//c.num_attention_heads)
        routes = rows*c.num_experts_per_tok
        prepare = 4*(2*n+2*capacity*kv+3*c.hidden_size+2*kv+capacity+c.num_experts+routes)+8*routes
        apply = 4*(n+c.intermediate_size+c.hidden_size)
        if max(prepare,apply,8*routes,self.payload_bytes)>self.max_bytes:
            raise MemoryError("leased native phase exceeds max_bytes")

    def status(self):
        with self._lock:
            result = super().status()
            result.update(expert_storage=f"leased-packed-{self.dtype}",disk_runtime_open=not self._closed,
                expert_cache_bytes=self.expert_cache_bytes,max_staging_bytes=self.max_staging_bytes,
                sparse_streamed_experts=True,resident_expert_weight_bytes=0,
                range_resident_weights=False,dense_resident_weights=True,
                checksum_policy="sha256-every-lease")
            if not self._closed:
                sources = set(getattr(self,"_source_layers",(self,)))
                result["expert_cache_used_bytes"] = sum(e._loader.experts.store._used() for e in sources)
            result["live_compositions"] = len(self._compositions)
            return result

    def close(self):
        with _COMPOSITION_LOCK,self._lock:
            if self._sessions:
                raise RuntimeError("close live sessions before closing native leased engine")
            if self._compositions:
                raise RuntimeError("close live compositions before closing native leased engine")
            if not self._closed:
                if hasattr(self,"_sources"):
                    with ExitStack() as locks:
                        for source in sorted(self._sources,key=id):
                            locks.enter_context(source._lock)
                        for source in self._sources:
                            source._compositions.discard(self)
                    self._sources = self._source_layers = ()
                else:
                    self._loader.close()
                    self._loader = None
                self._range = None
                self.executor = None
                self._closed = True


class NativeLeasedOlmoeSegmentSession(NativeOlmoeSegmentSession):
    numeric_class = NUMERIC_CLASS

    def __init__(self,engine,capacity):
        super().__init__(engine,capacity)
        self.numeric_class = engine.numeric_class

    def _layer(self,x,keys,values,position,index,cancelled=None):
        engine = self._engine
        executor = engine.executor
        c = RangeConfig.from_buffer_copy(self._native.engine.config)
        c.layers = 1
        residual,normalized = np.empty_like(x),np.empty_like(x)
        ids = np.empty((len(x),c.topk),np.uint64)
        weights = np.empty(ids.shape,np.float32)
        executor._check(executor._prepare(C.byref(c),C.byref(engine._range._layers[index]),
            _tensor(x),len(x),residual.ctypes.data,normalized.ctypes.data,x.size,
            ids.ctypes.data,weights.ctypes.data,ids.size,keys.ctypes.data,values.ctypes.data,
            keys.size,position,engine.max_bytes))
        accumulator = np.zeros_like(x)
        layer_id = f"model.layers.{engine.begin+index}"
        source = engine._source_layers[index] if hasattr(engine,"_source_layers") else engine
        started = time.perf_counter()
        # Group order follows first occurrence in the native route table. Only
        # selection/lease coordination happens here; weighting is performed in C.
        # Each distinct expert is leased once for every row (batch-union).
        for expert in dict.fromkeys(map(int,ids.flat)):
            _cancel(cancelled)
            region = source.manifest.layers[engine.begin+index].regions[str(expert)]
            key = source._loader.experts.key(layer_id,str(expert))
            with source._loader.experts.store.acquire(key) as payload:
                if len(payload)!=engine.payload_bytes:
                    raise ValueError("leased expert length mismatch")
                packed = bytes(payload)
                if len(packed)!=engine.payload_bytes:
                    raise ValueError("owned leased expert length mismatch")
                if hashlib.sha256(packed).hexdigest()!=region.sha256:
                    raise ValueError("leased expert checksum mismatch")
                updated = np.empty_like(x)
                apply = getattr(executor,_APPLY_ATTRIBUTES[engine.dtype])
                executor._check(apply(_tensor(normalized),len(x),c.hidden,c.intermediate,
                    ids.ctypes.data,weights.ctypes.data,ids.size,c.topk,c.experts,expert,
                    C.c_char_p(packed),len(packed),_tensor(accumulator),updated.ctypes.data,x.size,engine.max_bytes))
                accumulator = updated
            del packed
        # Feed the exact native selections back to the runtime: route learning,
        # bounded lookahead read-ahead and batch-union counters.
        source._loader.experts.record_external_routes(
            layer_id, [[str(expert) for expert in row] for row in ids.tolist()],
            elapsed=time.perf_counter()-started)
        out = np.empty_like(x)
        executor._check(executor._residual(_tensor(residual),_tensor(accumulator),
            out.ctypes.data,out.size,engine.max_bytes))
        return out

    def _execute(self,hidden,keys,values,position,cancelled=None):
        x = np.asarray(hidden)
        if x.ndim!=2 or not x.shape[0]:
            raise ValueError("hidden must be a nonempty FP32 matrix")
        x = _array(x,(len(x),self._engine.config.hidden_size),self._engine.max_bytes,"hidden")
        for index in range(self._engine.end-self._engine.begin):
            _cancel(cancelled)
            x = self._layer(x,keys[index],values[index],position,index,cancelled)
        return x

    def _forward(self,hidden,cancelled):
        _cancel(cancelled)
        x = np.asarray(hidden)
        if x.ndim!=2 or not x.shape[0]:
            raise ValueError("hidden must be a nonempty FP32 matrix")
        if len(x)>self.context_tokens-self.position:
            raise ValueError("context capacity exceeded")
        self._engine._admit_call(len(x),self.context_tokens)
        n = self._native
        keys,values = n._keys.copy(),n._values.copy()
        out = self._execute(hidden,keys,values,n.position,cancelled)
        _cancel(cancelled)
        position = n.position+len(x)
        n._keys,n._values = keys,values
        n._position.value = position
        return out


class NativeLeasedOlmoeDecoderSession(NativeOlmoeDecoderSession):
    """Storage-coordinated C phases through head, with range-wide KV rollback."""
    numeric_class = NUMERIC_CLASS

    def __init__(self,edge,engine,capacity):
        super().__init__(edge,engine,capacity)
        self.numeric_class = engine.numeric_class

    def forward(self,token_ids,*,cancelled=None):
        with self._engine._lock,self._edge._lock,self._session._lock:
            self._check()
            session = self._session
            session._active = True
            try:
                _cancel(cancelled)
                ids = _token_vector(token_ids)
                if len(ids)>session.context_tokens-session.position:
                    raise ValueError("context capacity exceeded")
                self._engine._admit_call(len(ids),session.context_tokens,vocab=self._edge.config.vocab_size)
                native = session._native
                keys,values = native._keys.copy(),native._values.copy()
                hidden = self._edge.embeddings(ids)
                hidden = session._execute(hidden,keys,values,native.position,cancelled)
                _cancel(cancelled)
                logits = self._edge.final_head(hidden)
                _cancel(cancelled)
                position = native.position+len(ids)
                native._keys,native._values = keys,values
                native._position.value = position
                return logits
            finally:
                session._active = False
