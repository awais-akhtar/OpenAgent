"""Qwen3.5 native text ranges registered through the public Segment/Edge ABI.

The inherited registration, tokenizer and resource ledger are transport only.
Every neural operation uses the separate Qwen3.5 native kernel and every session
owns its KV, delta recurrence, convolution and occupied-token mask. The public
text wire carries contiguous unmasked positions; wrapper media uses its explicit
three-axis composition API.
"""
import ctypes as C
import json
import math
import threading
import numpy as np
from .mixtral_public_adapters import NativeMixtralPublicAdapters
from .segment_edge_adapters import _SegmentAdapter,_SOPEN,_CREATE,_RUN,_SNAPSHOT,_WriteStream,_ReadStream
from .segment_edge_runtime import _SegmentCaps,_integer
from .native_qwen35_ranges import NativeQwen35Range,NativeQwen35Edge
from .qwen35_cache_recovery import write_qwen35_cache,restore_qwen35_cache


class _RangeEngine:
    def __init__(self,template,limit):
        self.template=template;self.limit=limit;self.sessions=set();self.lock=threading.RLock();self.closed=False

    def create(self,context,budget):
        with self.lock:
            if self.closed or len(self.sessions)>=self.limit:raise RuntimeError('Qwen3.5 engine closed or session limit reached')
            t=self.template
            if not 0<context<=t.capacity:raise ValueError('Qwen3.5 session context exceeds admitted range')
            # Keep the admitted source config unchanged: expert-store pins bind
            # its capacity too. The public dispatcher enforces the session's
            # smaller logical context, while this owner budgets physical storage.
            config=dict(vars(t.config))
            model=NativeQwen35Range(config,t.weights,begin=t.layer_begin,end=t.layer_end,
                library=t.library._name,max_bytes=budget,expert_provider=t._expert_provider,context_tokens=t.capacity)
            session=_RangeSession(self,model);self.sessions.add(session);return session

    def close(self):
        with self.lock:
            if self.sessions:raise RuntimeError('close Qwen3.5 sessions before engine')
            self.template.close();self.closed=True


class _RangeSession:
    def __init__(self,engine,model):self.engine=engine;self.model=model
    def close(self):
        with self.engine.lock:
            self.model.close();self.engine.sessions.discard(self)


class _CacheReadStream(_ReadStream):
    def read(self,count):
        if self.extent is not None and self.count==self.extent:return b''
        if type(count) is not int or not 0<count<=self.limit-self.count:
            raise MemoryError('Qwen3.5 public cache transfer exceeds admission')
        if self.extent is not None and count>self.extent-self.count:
            raise ValueError('Qwen3.5 cache transfer exceeds declared frame')
        buffer=C.create_string_buffer(count)
        if self.callback(self.user,buffer,count):raise OSError('Qwen3.5 public cache reader rejected bytes')
        raw=buffer.raw
        if self.count==0:
            if count!=12 or raw[:8]!=b'OAQ35K01':raise ValueError('Qwen3.5 public cache magic differs')
            self.header_bytes=int.from_bytes(raw[8:],'little')
            if not 0<self.header_bytes<=min(65536,self.limit-44):raise ValueError('Qwen3.5 cache header exceeds admission')
        elif self.count==12:
            from .segment_snapshot import _pairs
            if count!=self.header_bytes:raise ValueError('Qwen3.5 cache header transfer differs')
            header=json.loads(raw,object_pairs_hook=_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            schema=header.get('schema') if type(header) is dict else None
            if type(schema) is not list or len(schema)>8193:
                raise ValueError('complete bounded Qwen3.5 public cache schema required')
            payload=0
            for row in schema:
                if (type(row) is not list or len(row)!=3 or row[1] not in ('<f4','|u1')
                        or type(row[2]) is not list or not 1<=len(row[2])<=4
                        or any(type(size) is not int or size<0 for size in row[2])):
                    raise ValueError('Qwen3.5 public cache leaf declaration differs')
                payload+=math.prod(row[2])*(4 if row[1]=='<f4' else 1)
                if payload>self.limit-44-self.header_bytes:raise MemoryError('Qwen3.5 public cache payload exceeds admission')
            self.extent=44+self.header_bytes+payload
        self.count+=count;return raw


class NativeQwen35PublicAdapters(NativeMixtralPublicAdapters):
    wire_schema=b'openagent.qwen35.hidden.f32.v1'
    numeric_class=b'openagent.qwen35.native-fp32-boundary-double-dot.v1'

    def __init__(self,runtime_library,**options):
        self.context_tokens=options.pop('context_tokens',None)
        if self.context_tokens is not None and (type(self.context_tokens) is not int or not 1<=self.context_tokens<=65536):
            raise ValueError('explicit bounded native Qwen3.5 adapter context required')
        options.setdefault('segment_id','openagent-qwen35-native')
        options.setdefault('edge_id','openagent-qwen35-edge')
        super().__init__(runtime_library,**options)

    def _create_edge(self,budget):
        return NativeQwen35Edge.from_directory(self.root,library=self.kernel,
            max_weight_bytes=self.staging_bytes,max_bytes=budget,context_tokens=self.context_tokens)

    def _segment_record(self):
        def open_engine(output,caps,raw_options):
            options=raw_options.contents
            self._allowed(options.model,options.plan,options.plan_size,options.backend)
            budget=_integer(int(options.memory),'Qwen3.5 engine memory',maximum=2**63-1)
            template=NativeQwen35Range.from_directory(self.root,begin=int(options.begin),end=int(options.end),
                library=self.kernel,max_weight_bytes=self.staging_bytes,max_bytes=budget,expert_provider=self.expert_provider,
                context_tokens=int(options.context) or self.context_tokens)
            try:
                c=template.config
                if int(options.context) and not 0<int(options.context)<=template.capacity:
                    raise ValueError('public Qwen3.5 context exceeds admitted model')
                engine=_RangeEngine(template,self.max_sessions)
                record=_SegmentCaps(version=1,flags=(1<<8)|2|4|8,engine=self.segment_id,
                    schema=self.wire_schema,numeric=self.numeric_class,dtype=1,width=c.hidden_size,
                    rows=self.rows,context=template.capacity,layers=c.num_hidden_layers)
                C.memmove(caps,C.byref(record),C.sizeof(record))
            except BaseException:
                template.close();raise
            self._publish(engine,'engine',output)
        def create(engine_id,output,raw_options):
            engine=self._get(engine_id,'engine');options=raw_options.contents
            budget=_integer(int(options.memory),'Qwen3.5 session memory',maximum=2**63-1)
            session=engine.create(int(options.context) or engine.template.capacity,budget)
            self._publish(session,'session',output)
        def run(session_id,raw_request):
            session=self._get(session_id,'session');model=session.model;request=raw_request.contents
            with model._lock:
                model._check_open();rows=int(request.rows);width=model.config.hidden_size
                if int(request.position)!=model.position or not 0<rows<=self.rows:
                    raise ValueError('Qwen3.5 public position or row admission differs')
                if not request.input or not request.output or request.input_bytes<rows*width*4 or request.output_bytes<rows*width*4:
                    raise ValueError('Qwen3.5 public activation capacity differs')
                hidden=np.ctypeslib.as_array(C.cast(request.input,C.POINTER(C.c_float)),shape=(rows*width,)).reshape(1,rows,width)
                cancelled=(lambda:bool(request.cancel(request.cancel_data))) if request.cancel else None
                result=model.forward(hidden,cancelled=cancelled).hidden[0]
                C.memmove(request.output,result.ctypes.data,result.nbytes)
        def snapshot(session_id,callback,user):
            model=self._get(session_id,'session').model
            write_qwen35_cache(model,_WriteStream(callback,user,self.snapshot_bytes),max_bytes=self.snapshot_bytes)
        def restore(session_id,callback,user):
            model=self._get(session_id,'session').model
            restore_qwen35_cache(model,_CacheReadStream(callback,user,self.snapshot_bytes),max_bytes=self.snapshot_bytes)
        return _SegmentAdapter(version=1,engine=self.segment_id,
            open=self._guard(_SOPEN,open_engine),destroy=self._destroy('engine'),
            create=self._guard(_CREATE,create),destroy_session=self._destroy('session'),
            run=self._guard(_RUN,run),snapshot=self._guard(_SNAPSHOT,snapshot),restore=self._guard(_SNAPSHOT,restore))

    def create_execution_service(self,config,*,checkpoint_identity,context_tokens,native_memory_bytes,
            layers_per_range,max_transfer_bytes,max_new_tokens=256,max_sessions=1,
            prefill_chunk_tokens=32,max_snapshots_per_session=4):
        from .native_qwen35_ranges import qwen35_tensor_inventory
        from .segment_edge_runtime import SegmentRange
        from .execution_factories import create_segment_edge_execution_service
        if not self.registered_segment or not self.registered_edge:raise RuntimeError('register both Qwen3.5 adapters first')
        qwen35_tensor_inventory(config)
        if type(context_tokens) is not int or not 1<=context_tokens<=config['max_position_embeddings']:
            raise ValueError('bounded Qwen3.5 public serving context required')
        if context_tokens>65536 or (self.context_tokens is not None and context_tokens>self.context_tokens):
            raise ValueError('serving context exceeds the admitted native physical owner')
        stride=_integer(layers_per_range,'layers_per_range',maximum=config['num_hidden_layers'])
        if (config['num_hidden_layers']+stride-1)//stride>4096:raise ValueError('Qwen3.5 public range inventory exceeds admission')
        ranges=tuple(SegmentRange(self.segment_id.decode(),str(self.root),begin,min(begin+stride,config['num_hidden_layers']))
            for begin in range(0,config['num_hidden_layers'],stride))
        return create_segment_edge_execution_service(self.runtime_library,edge_adapter=self.edge_id.decode(),
            model_directory=str(self.root),checkpoint_identity=checkpoint_identity,ranges=ranges,
            vocabulary=config['vocab_size'],context_tokens=context_tokens,native_memory_bytes=native_memory_bytes,
            max_transfer_bytes=max_transfer_bytes,max_snapshot_bytes=self.snapshot_bytes,max_new_tokens=max_new_tokens,
            max_sessions=max_sessions,prefill_chunk_tokens=prefill_chunk_tokens,max_snapshots_per_session=max_snapshots_per_session,
            backend_mask=1<<8)


__all__=['NativeQwen35PublicAdapters']
