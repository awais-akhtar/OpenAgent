"""Public native Qwen4 ranges preserve HC channels and PLE token inputs."""
import ctypes as C
import json
import threading
from types import SimpleNamespace

import numpy as np

from .mixtral_public_adapters import NativeMixtralPublicAdapters, _RangeSession
from .segment_edge_adapters import _SegmentAdapter, _SOPEN, _CREATE, _RUN, _SNAPSHOT, _WriteStream, _ReadStream
from .segment_edge_runtime import _integer, _SegmentCaps, SegmentRange, _cancel_requested
from .native_qwen4_ranges import NativeQwen4Range, qwen4_tensor_inventory
from .native_qwen4_edge import NativeQwen4Edge


class _QwenRangeSession(_RangeSession):
    """Retry-safe Qwen4 session close for repeated native destroy callbacks."""
    def __init__(self, engine, model):
        super().__init__(engine, model)
        self.closed = False

    def close(self):
        with self.engine.lock:
            if self.closed:
                return
            # Match the base public-range ownership ordering: serialize close
            # with run/snapshot/restore and reject same-thread callback
            # reentry instead of closing a model from inside its active ABI
            # operation. Retain idempotence after successful destruction.
            with self.operation():
                self.model.close()
                self.engine.sessions.discard(self)
                self.closed = True


class _QwenEngine:
    def __init__(self, template, library, session_limit):
        self.template,self.library,self.session_limit = template,library,session_limit
        self.sessions,self.lock,self.closed = set(),threading.RLock(),False

    def create(self, context, budget):
        with self.lock:
            t = self.template
            if self.closed or len(self.sessions)>=self.session_limit:
                raise RuntimeError('Qwen4 range engine closed or session inventory full')
            if context!=t.config.max_position_embeddings:
                raise ValueError('Qwen4 session context must match admitted range capacity')
            model = NativeQwen4Range(t._source_config,t.weights,begin=t.layer_begin,end=t.layer_end,
                library=self.library,max_bytes=budget,max_snapshots=1,expert_provider=t._expert_provider,
                ple_provider=t._ple_provider)
            session = _QwenRangeSession(self,model)
            self.sessions.add(session)
            return session

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.sessions:
                raise RuntimeError('close Qwen4 sessions before their engine')
            self.template.close()
            self.closed = True


class _QwenEdge:
    def __init__(self, edge):
        self.edge = edge
        self.config = SimpleNamespace(**vars(edge.config))
        self.config.hidden_size *= edge.config.hc_count
        self.resident_weight_bytes = edge.weight_bytes

    def embeddings(self, ids):
        return self.edge.embeddings(ids)[0]

    def final_head(self, hidden):
        return self.edge.final_head(hidden)

    def close(self):
        self.edge.close()


class _QwenReadStream(_ReadStream):
    def read(self,count):
        if self.extent is not None and self.count==self.extent:
            return b''
        if type(count) is not int or count<0 or count>self.limit-self.count:
            raise MemoryError('Qwen4 snapshot read exceeds admission')
        if self.extent is not None and count>self.extent-self.count:
            raise ValueError('Qwen4 snapshot read exceeds frame extent')
        buffer = C.create_string_buffer(max(1,count))
        if self.callback(self.user,buffer,count):
            raise OSError('Qwen4 snapshot reader rejected data')
        raw = buffer.raw[:count]
        if self.count==0:
            if count!=12 or raw[:8]!=b'OAQ4KV01':
                raise ValueError('invalid Qwen4 snapshot frame')
            self.header_bytes = int.from_bytes(raw[8:],'little')
            if not 0<self.header_bytes<=min(65536,self.limit-44):
                raise ValueError('invalid Qwen4 snapshot header extent')
        elif self.count==12:
            if count!=self.header_bytes:
                raise ValueError('Qwen4 snapshot header transfer mismatch')
            document = json.loads(raw)
            schema = document.get('schema') if type(document) is dict else None
            if type(schema) is not list or len(schema)>8192:
                raise ValueError('invalid Qwen4 snapshot inventory')
            payload = 0
            for item in schema:
                if type(item) is not list or len(item)!=3 or item[1] not in ('<f4','<i8','|b1'):
                    raise ValueError('invalid Qwen4 snapshot array schema')
                shape = item[2]
                if type(shape) is not list or not 1<=len(shape)<=4:
                    raise ValueError('invalid Qwen4 snapshot shape')
                size = {'<f4':4,'<i8':8,'|b1':1}[item[1]]
                for dimension in shape:
                    if type(dimension) is not int or dimension<0:
                        raise ValueError('invalid Qwen4 snapshot dimension')
                    size *= dimension
                    if size>self.limit:
                        raise MemoryError('Qwen4 snapshot array exceeds admission')
                payload += size
            self.extent = 44+self.header_bytes+payload
            if self.extent>self.limit:
                raise MemoryError('Qwen4 snapshot frame exceeds admission')
        self.count += count
        return raw


class NativeQwen4PublicAdapters(NativeMixtralPublicAdapters):
    # Inherit the ABI transport and lifecycle, never Mixtral's source identity.
    # This gate runs before native-library loading and must bind the checkpoint
    # to the exact Qwen4 public-range row.
    _source_route_identity=('qwen4','qwen4_exp_text')
    wire_schema = b'openagent.qwen4.hyper-hidden.f32.v1'
    numeric_class = b'openagent.qwen4.native-fp32-double-dot.v1'

    def __init__(self, runtime_library, *, kernel_library, model_directory,
                 max_staging_bytes,max_snapshot_bytes,max_batch_rows=32,max_sessions=64,
                 tokenizer_path=None,segment_id='openagent-qwen4-native',edge_id='openagent-qwen4-edge',
                 expert_provider=None,max_resource_bytes=None,ple_provider=None):
        self.ple_provider=ple_provider
        super().__init__(runtime_library,kernel_library=kernel_library,model_directory=model_directory,
            max_staging_bytes=max_staging_bytes,max_snapshot_bytes=max_snapshot_bytes,
            max_batch_rows=max_batch_rows,max_sessions=max_sessions,tokenizer_path=tokenizer_path,
            segment_id=segment_id,edge_id=edge_id,expert_provider=expert_provider,
            max_resource_bytes=max_resource_bytes)

    def _create_edge(self,budget):
        return _QwenEdge(NativeQwen4Edge.from_directory(self.root,library=self.kernel,
            max_weight_bytes=self.staging_bytes,max_bytes=budget,ple_provider=self.ple_provider))

    def _segment_record(self):
        def open_engine(output,caps,raw_options):
            options = raw_options.contents
            self._allowed(options)
            budget = _integer(int(options.memory),'explicit Qwen4 engine budget',maximum=(1<<63)-1)
            template = NativeQwen4Range.from_directory(self.root,begin=int(options.begin),end=int(options.end),
                library=self.kernel,max_weight_bytes=self.staging_bytes,max_bytes=budget,max_snapshots=1,
                expert_provider=self.expert_provider,ple_provider=self.ple_provider)
            engine = _QwenEngine(template,self.kernel,self.max_sessions)
            if int(options.context) not in (0,template.config.max_position_embeddings):
                template.close()
                raise ValueError('Qwen4 range context must match checkpoint capacity')
            c = template.config
            record = _SegmentCaps(version=1,flags=(1<<8)|1|2|4|8,engine=self.segment_id,
                schema=self.wire_schema,numeric=self.numeric_class,dtype=1,
                width=c.hc_count*c.hidden_size,rows=self.rows,
                context=c.max_position_embeddings,layers=c.num_hidden_layers)
            C.memmove(caps,C.byref(record),C.sizeof(record))
            self._publish(engine,'engine',output)

        def create(engine_id,output,raw_options):
            engine = self._get(engine_id,'engine')
            options = raw_options.contents
            budget = _integer(int(options.memory),'explicit Qwen4 session budget',maximum=(1<<63)-1)
            self._publish(engine.create(int(options.context) or engine.template.config.max_position_embeddings,budget),
                          'session',output)

        def run(session_id,raw_request):
            session = self._get(session_id,'session')
            model = session.model
            request = raw_request.contents
            with session.operation():
                with model._lock:
                    model._check_open()
                    rows = request.rows
                    if (request.size != C.sizeof(type(request)) or
                            type(rows) is not int or not 0 < rows <= self.rows or
                            any(request.reserved) or int(request.position) != model.position):
                        raise ValueError('Qwen4 run position/rows differ from continuation admission')
                    if not request.tokens or request.token_count != rows:
                        raise ValueError('Qwen4 range requires matching PLE token-side rows')
                    model._transport = True
                    try:
                        if _cancel_requested(request.cancel,request.cancel_data):
                            raise RuntimeError('Qwen4 range execution cancelled')
                    finally:
                        model._transport = False
                    width = model.config.hc_count*model.config.hidden_size
                    expected_bytes = rows*width*C.sizeof(C.c_float)
                    if (not request.input or not request.output or
                            request.input_bytes != expected_bytes or
                            request.output_bytes != expected_bytes):
                        raise ValueError('Qwen4 range activation extent mismatch')
                    hidden = np.ctypeslib.as_array(C.cast(request.input,C.POINTER(C.c_float)),
                        shape=(rows*width,)).reshape(1,rows,width)
                    ids = np.ctypeslib.as_array(C.cast(request.tokens,C.POINTER(C.c_int32)),shape=(rows,))
                    cancelled = (lambda: _cancel_requested(request.cancel,request.cancel_data)) if request.cancel else None
                    result = model.forward(hidden,ids.astype(np.int64),cancelled=cancelled).hidden
                    C.memmove(request.output,result.ctypes.data,result.nbytes)

        def snapshot(session_id,callback,user):
            session = self._get(session_id,'session')
            with session.operation():
                session.model.write_cache(_WriteStream(callback,user,self.snapshot_bytes),
                                          max_bytes=self.snapshot_bytes)

        def restore(session_id,callback,user):
            session = self._get(session_id,'session')
            with session.operation():
                session.model.restore_cache(_QwenReadStream(callback,user,self.snapshot_bytes),
                                            max_bytes=self.snapshot_bytes)

        return _SegmentAdapter(version=1,engine=self.segment_id,
            open=self._guard(_SOPEN,open_engine),destroy=self._destroy('engine'),
            create=self._guard(_CREATE,create),destroy_session=self._destroy('session'),
            run=self._guard(_RUN,run),snapshot=self._guard(_SNAPSHOT,snapshot),restore=self._guard(_SNAPSHOT,restore))

    def create_execution_service(self,config,*,checkpoint_identity,context_tokens,
                                 native_memory_bytes,layers_per_range,max_transfer_bytes,
                                 max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=32,
                                 max_snapshots_per_session=4):
        from .execution_factories import create_segment_edge_execution_service
        qwen4_tensor_inventory(config,max_bytes=self.staging_bytes)
        if not self.registered_segment or not self.registered_edge:
            raise RuntimeError('register both Qwen4 adapters before service creation')
        if context_tokens!=config['max_position_embeddings']:
            raise ValueError('Qwen4 service context must match checkpoint capacity')
        layers = config['num_hidden_layers']
        stride = _integer(layers_per_range,'layers_per_range',maximum=layers)
        ranges = tuple(SegmentRange(self.segment_id.decode(),str(self.root),begin,min(layers,begin+stride))
                       for begin in range(0,layers,stride))
        return create_segment_edge_execution_service(self.runtime_library,
            edge_adapter=self.edge_id.decode(),model_directory=str(self.root),
            checkpoint_identity=checkpoint_identity,ranges=ranges,vocabulary=config['vocab_size'],
            source_family='qwen4',model_type='qwen4_exp_text',arithmetic='fp32',
            context_tokens=context_tokens,native_memory_bytes=native_memory_bytes,
            max_transfer_bytes=max_transfer_bytes,max_snapshot_bytes=self.snapshot_bytes,
            max_new_tokens=max_new_tokens,max_sessions=max_sessions,prefill_chunk_tokens=prefill_chunk_tokens,
            max_snapshots_per_session=max_snapshots_per_session,backend_mask=1<<8)


__all__ = ['NativeQwen4PublicAdapters']
