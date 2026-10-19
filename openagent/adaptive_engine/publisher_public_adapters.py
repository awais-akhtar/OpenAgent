"""Counted public native composition for complete K3 and V4.1 backbones.

The complete range retains AttnRes or mHC/Engram/index-sharing dependencies in
one owner. Hidden rows cross the public wire between actual embedding and head
operations; snapshot callbacks transport the full physical family codec.
Factories are the concrete source factories of a supplied execution service.
"""
import ctypes as C
import hashlib
import struct
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .failure_reporting import add_failure_note, failure_summary
from .segment_edge_adapters import (NativeOlmoePublicAdapters, _SegmentAdapter,
    _EdgeAdapter, _SOPEN, _CREATE, _RUN, _SNAPSHOT, _EOPEN, _EMBED, _LOGITS,
    _WriteStream)
from .segment_edge_runtime import _Functions, _SegmentCaps, _EdgeCaps, _identifier, _integer, _cancel_requested
from .segment_edge_runtime import _Embed, _Logits
from .native_library_identity import NativeLibraryIdentity
from .session_recovery import _path
from .executor_budget import _ReservedExecutor, _ConstructionCleanup
from .resource_ledger import LedgerResource

_PUBLIC_REGISTRY={}
_PUBLIC_REGISTRY_LOCK=threading.RLock()


def _position(model):
    length=getattr(model,'length',None)
    return model.position if length is None else length


class _Event:
    def __init__(self, request):
        self.request=request

    def is_set(self):
        return _cancel_requested(self.request.cancel,self.request.cancel_data)


class _FrameReader:
    """Read an exact-count public callback until the family frame's own EOF."""
    def __init__(self, callback, user, limit):
        if not callback:
            raise ValueError('public cache reader callback required')
        self.callback,self.user,self.limit=callback,user,limit
        self.prefix=None
        self.extent=self.count=0

    def _read(self, count):
        buffer=C.create_string_buffer(max(1,count))
        if self.callback(self.user,buffer,count):
            raise OSError('public publisher cache reader failed')
        return buffer.raw[:count]

    def read(self, count):
        if type(count) is not int or not 0<=count<=1024**2:
            raise ValueError('bounded counted public cache read required')
        if not count:
            return b''
        if self.prefix is None:
            self.prefix=self._read(20)
            magic,header,payload=struct.unpack('<8sIQ',self.prefix)
            if magic not in (b'OAK3S001',b'OAK3S002',b'OAK3S003',b'OAP41S03') or not 0<header<=65536:
                raise ValueError('complete native publisher cache frame required')
            self.extent=20+header+payload+32
            if self.extent>self.limit:
                raise MemoryError('public publisher cache exceeds transport admission')
        remaining=min(count,self.extent-self.count)
        if not remaining:
            return b''
        result=b''
        if self.count<20:
            take=min(remaining,20-self.count)
            result=self.prefix[self.count:self.count+take]
            self.count+=take
            remaining-=take
        if remaining:
            result+=self._read(remaining)
            self.count+=remaining
        return result


class _Engine:
    def __init__(self, owner, memory):
        self.owner,self.memory,self.children,self.closed=owner,memory,0,False

    def close(self):
        with self.owner.lock:
            if self.children:
                raise RuntimeError('public publisher sessions must close before their engine')
            self.closed=True


class _Session:
    def __init__(self, engine, model):
        self.engine,self.model,self.closed=engine,model,False
        self.lock=threading.RLock()
        self._operation_thread=None
        with engine.owner.lock:
            if engine.closed:
                raise RuntimeError('public publisher engine closed during session construction')
            engine.children+=1

    @contextmanager
    def operation(self):
        """Serialize ABI operations and reject same-thread callback reentry."""
        thread=threading.get_ident()
        if self._operation_thread==thread:
            raise RuntimeError('reentrant public publisher session operation rejected')
        with self.lock:
            if self._operation_thread==thread:
                raise RuntimeError('reentrant public publisher session operation rejected')
            self._operation_thread=thread
            try:
                yield
            finally:
                self._operation_thread=None

    def close(self):
        with self.operation():
            if not self.closed:
                self.model.close()
                with self.engine.owner.lock:
                    self.engine.children-=1
                    self.closed=True


class NativePublisherPublicAdapters(NativeOlmoePublicAdapters):
    """Register owned native K3 or V4.1 token-to-hidden-to-logit composition.

    The backing service, its checkpoint providers and libraries must remain
    alive until every public session and Edge is released. The public registry
    pins callbacks for the runtime library's process lifetime.
    """
    def __init__(self, runtime_library, backing_service, *, family, config,
                 model_directory, checkpoint_identity, executor_bytes,
                 max_resource_bytes, max_snapshot_bytes, max_work_bytes,
                 max_batch_rows=32):
        from .execution_service import NativeExecutionService
        from .k3_execution.executor import K3Config
        from .publisher_checkpoint.v41_profile import V41PublisherProfile
        if type(backing_service) is not NativeExecutionService:
            raise TypeError('concrete native execution service required')
        if family=='kimi-k3' and type(config) is K3Config:
            width,vocab,context,layers=config.hidden,config.vocab,config.capacity,len(config.layer_types)
        elif family=='deepseek-v41' and type(config) is V41PublisherProfile:
            width,vocab,context,layers=config.hidden_size,config.vocabulary,config.context,len(config.layers)
        else:
            raise ValueError('matching exact native publisher family and geometry required')
        # This class is also a public constructor, not merely a factory
        # implementation detail. Keep the P02 gate here so direct callers
        # cannot load an ABI library around compose_public_publisher_service.
        route = {
            'kimi-k3': ('kimi-k3', 'kimi_k3', 'fp32-approximate'),
            'deepseek-v41': ('deepseek-v41', 'deepseek_v41',
                             'publisher-bf16-fp8-fp4'),
        }[family]
        from .source_scope import (require_checkpoint_inventory_admission,
                                   require_source_family_backend_admission)
        require_source_family_backend_admission(
            family=route[0], model_type=route[1], backend='public-range',
            arithmetic=route[2])
        self.root=_path(model_directory).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError('original local publisher model directory required')
        self.inventory = require_checkpoint_inventory_admission(
            self.root, family=route[0], model_type=route[1],
            backend='public-range', arithmetic=route[2],
            max_metadata_bytes=max_work_bytes)
        payload=getattr(backing_service,'publisher_payload',None)
        if family=='kimi-k3':
            from .publisher_checkpoint.k3_adapter import require_k3_payload_admission
            require_k3_payload_admission(payload)
        else:
            from .publisher_checkpoint.v41_adapter import require_v41_source_admission
            require_v41_source_admission(payload)
        if (payload._source_root!=str(self.inventory.root) or
                payload._source_files!=tuple(sorted(self.inventory.files.items())) or
                payload.config_sha256!=self.inventory.config_sha256):
            raise ValueError('public publisher payload and model directory are not the same checkpoint snapshot')
        if ((family=='kimi-k3' and payload.config!=config) or
                (family=='deepseek-v41' and payload.profile!=config)):
            raise ValueError('public publisher config differs from the checkpoint-bound payload')
        # Provenance and checkpoint-bound config checks above remain the
        # authoritative admission errors.  Once those pass, ensure the
        # already-constructed service cannot disagree with the geometry this
        # ABI is about to advertise; keep this before loading the ABI library.
        limits=getattr(backing_service,'limits',None)
        if (limits is None or getattr(limits,'vocabulary',None)!=vocab or
                getattr(limits,'context',None)!=context):
            raise ValueError('backing service limits differ from original publisher geometry')
        if type(checkpoint_identity) is not str or not 0<len(checkpoint_identity)<=4096:
            raise ValueError('pinned public publisher checkpoint identity required')
        self.width,self.vocabulary,self.context,self.layers=width,vocab,context,layers
        self.executor_bytes=_integer(executor_bytes,'complete publisher executor reservation',maximum=2**63-1)
        self.snapshot_bytes=_integer(max_snapshot_bytes,'publisher snapshot admission',maximum=2**63-1)
        self.staging_bytes=_integer(max_work_bytes,'publisher transfer work admission',maximum=2**63-1)
        self.rows=_integer(max_batch_rows,'publisher public rows',maximum=4096)
        if self.rows*(2*width+vocab+1)*4>self.staging_bytes:
            raise MemoryError('public publisher transfer rows exceed work admission')
        self.family,self.config,self.service=family,config,backing_service
        self.inventory.assert_unchanged()
        self._runtime_identity=NativeLibraryIdentity.read(runtime_library)
        self.native=_Functions(runtime_library)
        self._runtime_identity.assert_unchanged()
        self.inventory.assert_unchanged()
        self.lock=threading.RLock()
        self.objects,self.pending,self.next_id={}, {}, 1
        self._initialize_resource_ledger(max_resource_bytes)
        self._callbacks=[]
        self.registered_segment=self.registered_edge=False
        self.tokenizer=None
        self._model_identity=None
        pin=hashlib.sha256(repr((family,checkpoint_identity,self._runtime_identity.sha256,config,
            executor_bytes,max_resource_bytes,max_snapshot_bytes,max_work_bytes,max_batch_rows)).encode()).hexdigest()
        self.identity='openagent-public-publisher-v1:'+pin
        self.segment_id=_identifier('openagent-'+family+'-full-'+pin[:16])
        self.edge_id=_identifier('openagent-'+family+'-edge-'+pin[:16])
        self.wire_schema=('openagent.'+family+'.normalized-hidden.f32.v1').encode()
        self.numeric_class=('openagent.'+family+'.native-publisher-boundaries.v1').encode()
        self.segment,self.edge=self._segment_record(),self._edge_record()
        self.native.check_layouts({9:_SegmentAdapter,10:_EdgeAdapter})

    def register(self):
        key=(str(self._runtime_identity.path),self.segment_id)
        with _PUBLIC_REGISTRY_LOCK:
            previous=_PUBLIC_REGISTRY.get(key)
            if previous is not None and previous is not self:
                with previous.lock:
                    previous.retry_cleanup()
                    if previous.objects or not previous.service._closed:
                        raise RuntimeError('the same public publisher registration still owns a live backing service')
                    previous._runtime_identity.assert_unchanged()
                    self.native.close()
                    previous.service=self.service
                    previous._initialize_resource_ledger(self.resource_ledger.max_bytes)
                    NativeOlmoePublicAdapters.register(previous)
                    return previous
            try:
                super().register()
            finally:
                if self.registered_segment:
                    _PUBLIC_REGISTRY[key]=self
            return self

    def _allowed(self, options):
        self._runtime_identity.assert_unchanged()
        if self.service._closed:
            raise RuntimeError('backing publisher service is closed')
        return super()._allowed(options)

    def _retain_construction(self, primary):
        cleanup=getattr(self.service.factory,'retry_cleanup',None)
        if not callable(cleanup):
            cleanup=getattr(primary,'cleanup',None)
        if not callable(cleanup):
            return
        resource=_ConstructionCleanup(cleanup)
        try:
            resource.close()
        except BaseException as failure:
            frame=getattr(self._construction,'frame',None)
            if frame is not None:
                resource=self.resource_ledger.adopt(frame[0],resource)
                frame[1]=True
            orphan=-id(resource)
            with self.lock:
                self.objects[orphan]=('session',resource)
                self.pending[orphan]=failure
            add_failure_note(primary, 'public publisher construction cleanup retained: '+failure_summary(failure))

    def _new(self, memory):
        if memory<self.executor_bytes:
            raise MemoryError('complete publisher executor reservation required')
        if self.pending:
            raise RuntimeError('public publisher cleanup is pending')
        try:
            model=self.service.factory()
        except BaseException as primary:
            self._retain_construction(primary)
            raise
        try:
            from .k3_execution.executor import K3Execution
            from .v41_execution.publisher import PublisherV41Execution
            from .v41_execution.publisher_arithmetic import NativePublisherV41Arithmetic
            from .execution_factories import _OwnedPublisherExecutor
            concrete=model._executor if type(model) is _ReservedExecutor else model
            concrete=concrete.model if type(concrete) is _OwnedPublisherExecutor else concrete
            if self.family=='kimi-k3':
                valid=type(concrete) is K3Execution and concrete.config==self.config and concrete._arithmetic is None
                identity=concrete._identity
            else:
                arithmetic=getattr(concrete.provider,'arithmetic',None)
                valid=(type(concrete) is PublisherV41Execution and concrete.profile==self.config and
                       type(arithmetic) is NativePublisherV41Arithmetic)
                identity=concrete.identity
            if not valid or getattr(model,'declared_executor_bytes',model.max_bytes)>memory or _position(model):
                raise ValueError('factory must create the exact empty native publisher owner')
            with self.lock:
                if self._model_identity is not None and self._model_identity!=identity:
                    raise ValueError('public publisher factory changed its source or arithmetic identity')
                self._model_identity=identity
            return model
        except BaseException as primary:
            try:
                model.close()
            except BaseException as failure:
                frame=getattr(self._construction,'frame',None)
                if frame is not None:
                    model=self.resource_ledger.adopt(frame[0],model)
                    frame[1]=True
                orphan=-id(model)
                with self.lock:
                    self.objects[orphan]=('session',model)
                    self.pending[orphan]=failure
                add_failure_note(primary, 'public publisher admission cleanup retained: '+failure_summary(failure))
            raise

    def _scope(self, model, request):
        event=_Event(request)
        if event.is_set():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('public publisher request cancelled')
        scope=getattr(model,'cancellation_scope',None)
        if not callable(scope):
            raise TypeError('concrete native publisher cancellation scope required')
        return scope(event)

    def _segment_record(self):
        def open_engine(output,caps,raw_options):
            options=raw_options.contents
            self._allowed(options)
            if options.begin!=0 or options.end!=self.layers or options.context not in (0,self.context):
                raise ValueError('complete publisher residual/shared-index layer range and original capacity required')
            engine=_Engine(self,int(options.memory))
            record=_SegmentCaps(version=1,flags=(1<<8)|1|2|4,
                engine=self.segment_id,schema=self.wire_schema,numeric=self.numeric_class,
                dtype=1,width=self.width,rows=self.rows,context=self.context,layers=self.layers)
            C.memmove(caps,C.byref(record),C.sizeof(record))
            self._publish(engine,'engine',output)
        def create(engine_id,output,raw_options):
            engine=self._get(engine_id,'engine')
            engine=engine._owner if type(engine) is LedgerResource else engine
            options=raw_options.contents
            if engine.closed or options.context not in (0,self.context):
                raise ValueError('live complete publisher engine and original session capacity required')
            model=self._new(int(options.memory))
            try:
                session=_Session(engine,model)
            except BaseException as primary:
                try:
                    model.close()
                except BaseException as failure:
                    frame=getattr(self._construction,'frame',None)
                    if frame is not None:
                        model=self.resource_ledger.adopt(frame[0],model)
                        frame[1]=True
                    orphan=-id(model)
                    with self.lock:
                        self.objects[orphan]=('session',model)
                        self.pending[orphan]=failure
                    add_failure_note(primary, 'public session construction cleanup retained: '+failure_summary(failure))
                raise
            self._publish(session,'session',output)
        def run(session_id,raw_request):
            session=self._get(session_id,'session')
            request=raw_request.contents
            if (request.size != C.sizeof(type(request)) or
                    type(request.rows) is not int or not 1<=request.rows<=self.rows or
                    type(request.position) is not int or not 0<=request.position<=self.context or
                    any(request.reserved) or request.token_count!=request.rows or
                    not request.tokens or not request.input or not request.output or
                    request.input_bytes != request.rows*self.width*4 or
                    request.output_bytes != request.rows*self.width*4 or
                    request.position+request.rows > self.context):
                raise ValueError('public publisher position or occupied token inventory differs')
            ids=np.ctypeslib.as_array(C.cast(request.tokens,C.POINTER(C.c_int32)),shape=(request.rows,))
            if np.any(ids < 0) or np.any(ids >= self.vocabulary):
                raise ValueError('public publisher token IDs are outside the vocabulary')
            rows=np.ctypeslib.as_array(C.cast(request.input,C.POINTER(C.c_float)),
                shape=(request.rows*self.width,)).reshape(request.rows,self.width)
            with session.operation():
                if session.closed:
                    raise ValueError('public publisher session is closed')
                # Position admission must be serialized with continuation
                # mutation. Checking it before acquiring the operation lease
                # lets callbacks validate the same old boundary and commit
                # sequentially from stale state.
                if _position(session.model)!=request.position:
                    raise ValueError('public publisher position changed before operation admission')
                with self._scope(session.model,request):
                    output=session.model.forward_hidden(tuple(int(token) for token in ids),rows)
            if output.shape!=rows.shape or output.dtype!=np.float32 or not np.isfinite(output).all():
                raise ArithmeticError('invalid public publisher normalized hidden output')
            C.memmove(request.output,output.ctypes.data,output.nbytes)
        def snapshot(session_id,writer,user):
            if not writer:
                raise ValueError('public publisher cache writer required')
            session=self._get(session_id,'session')
            with session.operation():
                if session.closed:
                    raise ValueError('public publisher session is closed')
                session.model.write_cache(_WriteStream(writer,user,self.snapshot_bytes),max_bytes=self.snapshot_bytes)
        def restore(session_id,reader,user):
            session=self._get(session_id,'session')
            with session.operation():
                if session.closed:
                    raise ValueError('public publisher session is closed')
                session.model.restore_cache(_FrameReader(reader,user,self.snapshot_bytes),max_bytes=self.snapshot_bytes)
        return _SegmentAdapter(version=1,engine=self.segment_id,
            open=self._guard(_SOPEN,open_engine),destroy=self._destroy('engine'),
            create=self._guard(_CREATE,create),destroy_session=self._destroy('session'),
            run=self._guard(_RUN,run),snapshot=self._guard(_SNAPSHOT,snapshot),restore=self._guard(_SNAPSHOT,restore))

    def _edge_record(self):
        def open_edge(output,caps,raw_options):
            options=raw_options.contents
            self._allowed(options)
            model=self._new(int(options.memory))
            record=_EdgeCaps(version=2,flags=(1<<8)|8,engine=self.edge_id,
                schema=self.wire_schema,numeric=self.numeric_class,tokenizer=b'',dtype=1,
                width=self.width,vocab=self.vocabulary,rows=self.rows,context=self.context,
                layers=self.layers,bos=-1,eos=-1,resident_bytes=self.executor_bytes)
            C.memmove(caps,C.byref(record),C.sizeof(record))
            self._publish(model,'edge',output)
        def embed(edge_id,raw_request):
            model=self._get(edge_id,'edge')
            request=raw_request.contents
            if (request.size != C.sizeof(_Embed) or any(request.reserved) or
                    type(request.rows) is not int or not 1<=request.rows<=self.rows or
                    request.token_count != request.rows or not request.tokens or
                    not request.output or
                    request.output_bytes != request.rows*self.width*4):
                raise ValueError('public publisher embedding row budget exceeded')
            ids=np.ctypeslib.as_array(C.cast(request.tokens,C.POINTER(C.c_int32)),shape=(request.rows,))
            if np.any(ids < 0) or np.any(ids >= self.vocabulary):
                raise ValueError('public publisher embedding token IDs are outside the vocabulary')
            with self._scope(model,request):
                output=model.embedding_rows(tuple(int(token) for token in ids))
            if output.shape!=(request.rows,self.width) or output.dtype!=np.float32 or not np.isfinite(output).all():
                raise ArithmeticError('invalid native publisher embedding output')
            C.memmove(request.output,output.ctypes.data,output.nbytes)
        def logits(edge_id,raw_request):
            model=self._get(edge_id,'edge')
            request=raw_request.contents
            if (request.size != C.sizeof(_Logits) or any(request.reserved) or
                    type(request.rows) is not int or not 1<=request.rows<=self.rows or
                    not request.input or not request.output or
                    request.input_bytes != request.rows*self.width*4 or
                    request.capacity != request.rows*self.vocabulary):
                raise ValueError('public publisher logits row budget exceeded')
            rows=np.ctypeslib.as_array(C.cast(request.input,C.POINTER(C.c_float)),
                shape=(request.rows*self.width,)).reshape(request.rows,self.width)
            with self._scope(model,request):
                output=model.head_rows(rows)
            if output.shape!=(request.rows,self.vocabulary) or output.dtype!=np.float32 or not np.isfinite(output).all():
                raise ArithmeticError('invalid native publisher vocabulary output')
            C.memmove(request.output,output.ctypes.data,output.nbytes)
        return _EdgeAdapter(version=2,engine=self.edge_id,open=self._guard(_EOPEN,open_edge),
            destroy=self._destroy('edge'),embed=self._guard(_EMBED,embed),logits=self._guard(_LOGITS,logits))


__all__=['NativePublisherPublicAdapters']
