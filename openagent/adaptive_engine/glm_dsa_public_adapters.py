"""Native DSA layer/edge kernels exposed through the existing public ABI."""
import ctypes as C
import json
import threading
from contextlib import contextmanager

import numpy as np

from .mixtral_public_adapters import NativeMixtralPublicAdapters
from .native_glm_dsa_ranges import NativeGlmDsaRange,NativeGlmDsaEdge,partitions
from .native_glm_dsa_cache import write_cache,restore_cache,_extent,MAGIC
from .segment_edge_adapters import (_SegmentAdapter,_SOPEN,_CREATE,_RUN,_SNAPSHOT,
                                    _WriteStream,_ReadStream)
from .segment_edge_runtime import _SegmentCaps,_integer
from .segment_snapshot import _pairs


class _Cancel:
    def __init__(self,request):self.request=request
    def is_set(self):
        if not self.request.cancel:return False
        value=self.request.cancel(self.request.cancel_data)
        if value not in (0,1):
            raise ValueError('DSA cancellation callback must return exactly 0 or 1')
        return value==1


class _Read(_ReadStream):
    def read(self,count):
        if self.extent is not None and self.count==self.extent:return b''
        if type(count) is not int or count<0 or count>self.limit-self.count:
            raise MemoryError('DSA snapshot reader exceeds admission')
        if self.extent is not None and count>self.extent-self.count:
            raise ValueError('DSA snapshot reader exceeds declared frame')
        buffer=C.create_string_buffer(max(1,count))
        if self.callback(self.user,buffer,count):raise OSError('DSA snapshot reader rejected transfer')
        raw=buffer.raw[:count]
        if self.count==0:
            if count!=12 or raw[:8]!=MAGIC:raise ValueError('invalid DSA snapshot frame')
            self.header_bytes=int.from_bytes(raw[8:],'little')
            if not 0<self.header_bytes<=min(65536,self.limit-44):
                raise ValueError('DSA snapshot header exceeds admission')
        elif self.count==12:
            if count!=self.header_bytes:raise ValueError('DSA snapshot header transfer differs')
            header=json.loads(raw,object_pairs_hook=_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if (type(header) is not dict or set(header)!={'identity','batch','position','widths','mask_shape'}
                    or type(header['batch']) is not int or not 0<=header['batch']<=128
                    or type(header['position']) is not int or not 0<=header['position']<=1048576
                    or type(header['widths']) is not list or len(header['widths'])>4096
                    or any(type(row) is not list or len(row)!=3 or any(type(width) is not int
                        or not 0<=width<=2**31-1 for width in row) for row in header['widths'])):
                raise ValueError('invalid DSA snapshot physical extent')
            self.extent=_extent(header,self.header_bytes)
            if self.extent>self.limit:raise MemoryError('DSA snapshot physical extent exceeds admission')
        self.count+=count
        return raw


class _Session:
    def __init__(self,engine,model):
        self.engine,self.model=engine,model
        # The ABI may call run, snapshot, restore and destroy_session from
        # different host threads.  The model lock protects individual model
        # mutations, but cache codecs read/write several related physical
        # fields and must not interleave with a forward or session close.
        self.lock=threading.RLock()
        self.operation_owner=None

    @contextmanager
    def operation(self):
        with self.lock:
            owner=threading.get_ident()
            if self.operation_owner==owner:
                raise RuntimeError('reentrant GLM-DSA public session operation rejected')
            self.operation_owner=owner
            try:
                yield
            finally:
                self.operation_owner=None

    def close(self):
        with self.engine.lock:
            with self.operation():
                self.model.close()
                self.engine.sessions.discard(self)


class _Engine:
    def __init__(self,template,limit):
        self.template,self.limit=template,limit
        self.lock,self.sessions,self.closed=threading.RLock(),set(),False
    def create(self,context,budget):
        with self.lock:
            if self.closed or len(self.sessions)>=self.limit:
                raise RuntimeError('DSA range engine closed or session inventory full')
            t=self.template
            t._check_open()
            if type(context) is not int or not 0<context<=t.capacity:
                raise ValueError('DSA session context exceeds engine admission')
            model=NativeGlmDsaRange(t.source_config,t.source_tensors,begin=t.begin,end=t.end,
                library=t.library,max_bytes=budget,context_tokens=context,expert_provider=t._expert_provider)
            session=_Session(self,model);self.sessions.add(session);return session
    def close(self):
        with self.lock:
            if self.sessions:raise RuntimeError('close DSA sessions before their range engine')
            self.template.close();self.closed=True


class NativeGlmDsaPublicAdapters(NativeMixtralPublicAdapters):
    _source_route_identity=('glm-dsa','glm_moe_dsa')
    wire_schema=b'openagent.glm-dsa.hidden.f32.v1'
    numeric_class=b'openagent.glm-dsa.native-fp32-double-dot.v1'

    def __init__(self,runtime_library,*,kernel_library,model_directory,max_staging_bytes,
                 max_snapshot_bytes,max_batch_rows=32,max_sessions=64,tokenizer_path=None,
                 max_resource_bytes=None,segment_id='openagent-glm-dsa-native',edge_id='openagent-glm-dsa-edge',
                 expert_provider=None):
        if expert_provider is not None:
            raise ValueError(
                'GLM-DSA packed experts are restricted until source-content identity is bound')
        self.expert_provider=None
        super().__init__(runtime_library,kernel_library=kernel_library,model_directory=model_directory,
            max_staging_bytes=max_staging_bytes,max_snapshot_bytes=max_snapshot_bytes,
            max_batch_rows=max_batch_rows,max_sessions=max_sessions,tokenizer_path=tokenizer_path,
            max_resource_bytes=max_resource_bytes,segment_id=segment_id,edge_id=edge_id)

    def _create_edge(self,budget):
        return self._load_source_owner(lambda:NativeGlmDsaEdge.from_directory(
            self.root,library=self.library,max_weight_bytes=self.staging_bytes,max_bytes=budget),
            kind='edge')

    def create_execution_service(self,config,*,checkpoint_identity,context_tokens,native_memory_bytes,
            layers_per_range,max_transfer_bytes,max_new_tokens=256,max_sessions=1,
            prefill_chunk_tokens=32,max_snapshots_per_session=4):
        from .independent_glm_dsa import IndependentGlmDsaDecoder
        from .segment_edge_runtime import SegmentRange
        from .execution_factories import create_segment_edge_execution_service
        if not self.registered_segment or not self.registered_edge:
            raise RuntimeError('register both DSA adapters before creating a service')
        self._source_snapshot.assert_unchanged()
        c=IndependentGlmDsaDecoder(config,{},_inventory_only=True).config
        if type(context_tokens) is not int or not 0<context_tokens<=c.max_position_embeddings:
            raise ValueError('bounded DSA service context required')
        ranges=tuple(SegmentRange(self.segment_id.decode(),str(self.root),begin,end)
            for begin,end in partitions(config,layers_per_range))
        return create_segment_edge_execution_service(self.runtime_library,
            edge_adapter=self.edge_id.decode(),model_directory=str(self.root),
            checkpoint_identity=checkpoint_identity,ranges=ranges,vocabulary=c.vocab_size,
            source_family='glm-dsa',model_type='glm_moe_dsa',arithmetic='fp32',
            context_tokens=context_tokens,native_memory_bytes=native_memory_bytes,
            max_transfer_bytes=max_transfer_bytes,max_snapshot_bytes=self.snapshot_bytes,
            max_new_tokens=max_new_tokens,max_sessions=max_sessions,prefill_chunk_tokens=prefill_chunk_tokens,
            max_snapshots_per_session=max_snapshots_per_session,backend_mask=1<<8)

    def _segment_record(self):
        def open_engine(output,caps,raw_options):
            options=raw_options.contents
            self._allowed(options)
            begin,end=int(options.begin),int(options.end)
            if not 0<=begin<end<=4096:
                raise ValueError('DSA public layer range must be nonempty and within the admitted layer ceiling')
            budget=_integer(int(options.memory),'explicit DSA engine budget',maximum=(1<<63)-1)
            template=self._load_source_owner(lambda:NativeGlmDsaRange.from_directory(
                self.root,begin=begin,end=end,
                library=self.library,context_tokens=int(options.context) or None,
                max_weight_bytes=self.staging_bytes,max_bytes=budget,expert_provider=self.expert_provider),
                kind='engine')
            engine=_Engine(template,self.max_sessions)
            c=template.config
            try:
                record=_SegmentCaps(version=1,flags=(1<<8)|2|4|8,engine=self.segment_id,
                    schema=self.wire_schema,numeric=self.numeric_class,dtype=1,width=c.hidden_size,
                    rows=self.rows,context=template.capacity,layers=template.source_config['num_hidden_layers'])
                C.memmove(caps,C.byref(record),C.sizeof(record))
            except BaseException as error:
                # The public handle has not adopted this engine yet. Keep the
                # template's mapped weights and provider lease from leaking if
                # capability construction fails before _publish.
                try:
                    engine.close()
                except BaseException as cleanup:
                    with self.lock:
                        orphan=-id(engine)
                        self.objects[orphan]=('engine',engine)
                        self.pending[orphan]=cleanup
                    error.add_note('DSA range construction cleanup pending: '+str(cleanup))
                raise
            self._publish(engine,'engine',output)
        def create(engine_id,output,raw_options):
            engine=self._get(engine_id,'engine');options=raw_options.contents
            budget=_integer(int(options.memory),'explicit DSA session budget',maximum=(1<<63)-1)
            self._publish(engine.create(int(options.context) or engine.template.capacity,budget),'session',output)
        def run(session_id,raw_request):
            session=self._get(session_id,'session');request=raw_request.contents
            with session.operation():
                model=session.model
                with model._lock:
                    model._check_open()
                    rows,width=int(request.rows),model.config.hidden_size
                    if (int(request.size)!=C.sizeof(type(request)) or
                            int(request.position)!=model.position or not 0<rows<=self.rows):
                        raise ValueError('DSA range position or row admission differs')
                    if request.tokens or request.token_count or any(request.reserved):
                        raise ValueError('DSA hidden-activation ABI does not accept token or reserved fields')
                    # This ABI describes one exact hidden-state tensor, not a
                    # capacity-bearing buffer. Reject trailing bytes as well
                    # as truncation so callers cannot accidentally pair a
                    # wider allocation/schema with the DSA kernel contract.
                    expected_bytes=rows*width*4
                    if (not request.input or not request.output or
                            int(request.input_bytes)!=expected_bytes or
                            int(request.output_bytes)!=expected_bytes):
                        raise ValueError('DSA activation transfer extent differs')
                    hidden=np.ctypeslib.as_array(C.cast(request.input,C.POINTER(C.c_float)),
                        shape=(rows*width,)).reshape(1,rows,width)
                    if not np.isfinite(hidden).all():
                        raise ValueError('DSA activation input must be finite')
                    with model.cancellation_scope(_Cancel(request)):
                        result=model.forward_hidden(hidden)[0]
                    if not np.isfinite(result).all():
                        raise ValueError('DSA activation output must be finite')
                    C.memmove(request.output,result.ctypes.data,result.nbytes)
        def snapshot(session_id,callback,user):
            session=self._get(session_id,'session')
            with session.operation():
                write_cache(session.model,_WriteStream(callback,user,self.snapshot_bytes),
                    max_bytes=self.snapshot_bytes)
        def restore(session_id,callback,user):
            session=self._get(session_id,'session')
            with session.operation():
                restore_cache(session.model,_Read(callback,user,self.snapshot_bytes),
                    max_bytes=self.snapshot_bytes)
        return _SegmentAdapter(version=1,engine=self.segment_id,open=self._guard(_SOPEN,open_engine),
            destroy=self._destroy('engine'),create=self._guard(_CREATE,create),
            destroy_session=self._destroy('session'),run=self._guard(_RUN,run),
            snapshot=self._guard(_SNAPSHOT,snapshot),restore=self._guard(_SNAPSHOT,restore))


__all__=['NativeGlmDsaPublicAdapters']
