"""Mixtral native execution registered through the public Segment/Edge ABI."""
import ctypes as C
import _ctypes
import os
from pathlib import Path
import threading
import weakref
from types import SimpleNamespace

import numpy as np

from .segment_edge_adapters import (
    NativeOlmoePublicAdapters, _SegmentAdapter, _EdgeAdapter, _SOPEN, _CREATE,
    _RUN, _SNAPSHOT, _WriteStream, _ReadStream,
)
from .segment_edge_runtime import _Functions, _identifier, _integer, _SegmentCaps
from .native_mixtral_ranges import NativeMixtralRange
from .native_mixtral_edge import NativeMixtralEdge
from .mixtral_range_cache import write_range_cache, restore_range_cache


class _RangeEngine:
    def __init__(self, template, session_limit):
        self.template = template
        self.sessions = set()
        self.lock = threading.RLock()
        self.closed = False
        self.session_limit = session_limit

    def create(self, context, budget):
        with self.lock:
            if self.closed or len(self.sessions) >= self.session_limit:
                raise RuntimeError('Mixtral range engine closed or session inventory full')
            t = self.template
            if not 0 < context <= t.capacity:
                raise ValueError('session context exceeds opened Mixtral range')
            model = NativeMixtralRange(t.config,t.weights,begin=t.begin,end=t.end,
                library=t.library,context_tokens=context,max_bytes=budget,
                expert_provider=t._expert_provider)
            session = _RangeSession(self,model)
            self.sessions.add(session)
            return session

    def close(self):
        with self.lock:
            if self.sessions:
                raise RuntimeError('close Mixtral sessions before their engine')
            self.template.close()
            self.closed = True


class _RangeSession:
    def __init__(self, engine, model):
        self.engine,self.model = engine,model

    def close(self):
        with self.engine.lock:
            self.model.close()
            self.engine.sessions.discard(self)


class _EdgeFacade:
    def __init__(self, edge):
        self.edge = edge
        self.config = SimpleNamespace(**dict(edge.config))
        self.resident_weight_bytes = edge._weight_bytes

    def embeddings(self, ids):
        return self.edge.embed(ids)

    def final_head(self, hidden):
        return self.edge.logits(hidden)

    def close(self):
        self.edge.close()


class _MixtralReadStream(_ReadStream):
    def read(self, count):
        # Public transfer callbacks require exact byte counts; the frame carries
        # its complete extent, allowing the codec's EOF probe without a callback.
        if self.extent is not None and self.count == self.extent:
            return b''
        if type(count) is not int or count < 0 or count > self.limit-self.count:
            raise MemoryError('Mixtral snapshot read exceeds admission')
        if self.extent is not None and count > self.extent-self.count:
            raise ValueError('Mixtral snapshot read exceeds declared frame')
        buffer = C.create_string_buffer(max(1,count))
        if self.callback(self.user,buffer,count):
            raise OSError('Mixtral snapshot reader rejected data')
        raw = buffer.raw[:count]
        if self.count == 0:
            if count != 12 or raw[:8] != b'OAMXR\x00\x01\x00':
                raise ValueError('invalid Mixtral snapshot frame')
            self.header_bytes = int.from_bytes(raw[8:],'little')
            if not 0 < self.header_bytes <= min(65536,self.limit-44):
                raise ValueError('invalid Mixtral snapshot header extent')
        elif self.count == 12:
            import json
            if count != self.header_bytes:
                raise ValueError('invalid Mixtral snapshot header transfer')
            header = json.loads(raw)
            if type(header) is not dict:
                raise ValueError('invalid Mixtral snapshot header')
            payload = header.get('payload')
            if type(payload) is not int or not 0 <= payload <= self.limit-44-self.header_bytes:
                raise MemoryError('Mixtral snapshot payload exceeds admission')
            self.extent = 44+self.header_bytes+payload
        self.count += count
        return raw


class NativeMixtralPublicAdapters(NativeOlmoePublicAdapters):
    """Dense Mixtral ranges and edge kernels; registry pins this owner forever.

    Each session owns independent KV and immutable weights with an explicit
    budget. The native kernel library is retained by the registered owner.
    """
    wire_schema = b'openagent.mixtral.hidden.f32.v1'
    numeric_class = b'openagent.mixtral.native-fp32-boundary-double-dot.v1'

    def __init__(self, runtime_library, *, kernel_library, model_directory,
                 max_staging_bytes, max_snapshot_bytes, max_batch_rows=32,
                 max_sessions=64, tokenizer_path=None,
                 expert_provider=None, max_resource_bytes=None,
                 segment_id='openagent-mixtral-native', edge_id='openagent-mixtral-edge'):
        from .native_library_identity import NativeLibraryIdentity
        self._runtime_identity=NativeLibraryIdentity.read(runtime_library)
        self._kernel_identity=NativeLibraryIdentity.read(kernel_library)
        self.native = _Functions(runtime_library)
        self._runtime_identity.assert_unchanged()
        self.runtime_library = str(Path(runtime_library).resolve(strict=True))
        self.root = Path(model_directory).resolve(strict=True)
        path = Path(kernel_library).resolve(strict=True)
        if not self.root.is_dir() or not path.is_file():
            raise ValueError('explicit local model directory and kernel library required')
        self.kernel = str(path)
        self.staging_bytes = _integer(max_staging_bytes,'max_staging_bytes',maximum=(1<<63)-1)
        self.snapshot_bytes = _integer(max_snapshot_bytes,'max_snapshot_bytes',maximum=(1<<63)-1)
        self.rows = _integer(max_batch_rows,'max_batch_rows',maximum=4096)
        self.max_sessions = _integer(max_sessions,'max_sessions',maximum=65536)
        self.segment_id,self.edge_id = _identifier(segment_id),_identifier(edge_id)
        self.lock = threading.RLock()
        self.objects,self.pending,self.next_id = {},{},1
        self._initialize_resource_ledger(max_resource_bytes)
        self.registered_segment = self.registered_edge = False
        self.tokenizer = None
        if tokenizer_path is not None:
            from tokenizers import Tokenizer
            tokenizer = Path(tokenizer_path).resolve(strict=True)
            if not tokenizer.is_relative_to(self.root) or not tokenizer.is_file():
                raise ValueError('tokenizer must be inside admitted model root')
            if tokenizer.stat().st_size > self.staging_bytes:
                raise MemoryError('tokenizer exceeds metadata admission')
            self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.library = C.CDLL(self.kernel)
        # Public registrations pin this owner for the registry's lifetime.
        # Unregistered construction failures still release an unused library;
        # borrowed range references keep the CDLL itself alive until disposal.
        release=_ctypes.FreeLibrary if os.name=='nt' else _ctypes.dlclose
        self._kernel_release=weakref.finalize(self.library,release,self.library._handle)
        self._kernel_identity.assert_unchanged()
        self.expert_provider = expert_provider
        self._callbacks = []
        self.segment,self.edge = self._segment_record(),self._edge_record()
        self.native.check_layouts({9:_SegmentAdapter,10:_EdgeAdapter})

    def _allowed(self,path,plan,plan_size,backend):
        self._runtime_identity.assert_unchanged();self._kernel_identity.assert_unchanged()
        return super()._allowed(path,plan,plan_size,backend)

    def create_execution_service(self, config, *, checkpoint_identity,
                                 context_tokens, native_memory_bytes,
                                 layers_per_range, max_transfer_bytes,
                                 max_new_tokens=256, max_sessions=1,
                                 prefill_chunk_tokens=32,
                                 max_snapshots_per_session=4):
        """Compose registered Mixtral ranges into the native serving lifecycle.

        The service borrows this registered owner and any packed expert provider.
        Explicit checkpoint identity participates in every executor recovery key.
        Runtime edge/range capabilities must agree with the admitted config.
        """
        from .native_mixtral_decoder import configuration
        from .segment_edge_runtime import SegmentRange
        from .execution_factories import create_segment_edge_execution_service
        if not self.registered_segment or not self.registered_edge:
            raise RuntimeError('register both Mixtral adapters before creating a service')
        settings,context = configuration(config,context_tokens)
        stride = _integer(layers_per_range,'layers_per_range',maximum=settings['num_hidden_layers'])
        count = (settings['num_hidden_layers']+stride-1)//stride
        if count > 4096:
            raise ValueError('Mixtral range inventory exceeds public admission')
        ranges = tuple(SegmentRange(self.segment_id.decode('utf-8'),str(self.root),
            begin,min(settings['num_hidden_layers'],begin+stride))
            for begin in range(0,settings['num_hidden_layers'],stride))
        return create_segment_edge_execution_service(self.runtime_library,
            edge_adapter=self.edge_id.decode('utf-8'),model_directory=str(self.root),
            checkpoint_identity=checkpoint_identity,ranges=ranges,
            vocabulary=settings['vocab_size'],context_tokens=context,
            native_memory_bytes=native_memory_bytes,max_transfer_bytes=max_transfer_bytes,
            max_snapshot_bytes=self.snapshot_bytes,max_new_tokens=max_new_tokens,
            max_sessions=max_sessions,prefill_chunk_tokens=prefill_chunk_tokens,
            max_snapshots_per_session=max_snapshots_per_session,backend_mask=1<<8)

    def _create_edge(self, budget):
        return _EdgeFacade(NativeMixtralEdge.from_directory(self.root,library=self.library,
            max_weight_bytes=self.staging_bytes,max_bytes=budget,tokenizer=self.tokenizer))

    def _segment_record(self):
        def open_engine(output,caps,raw_options):
            options = raw_options.contents
            self._allowed(options.model,options.plan,options.plan_size,options.backend)
            budget = _integer(int(options.memory),'explicit Mixtral engine budget',maximum=(1<<63)-1)
            template = NativeMixtralRange.from_directory(self.root,begin=int(options.begin),
                end=int(options.end),library=self.library,context_tokens=int(options.context) or None,
                max_weight_bytes=self.staging_bytes,max_bytes=budget,
                expert_provider=self.expert_provider)
            engine = _RangeEngine(template,self.max_sessions)
            c = template.config
            record = _SegmentCaps(version=1,flags=(1<<8)|2|4|8,
                engine=self.segment_id,schema=self.wire_schema,numeric=self.numeric_class,
                dtype=1,width=c['hidden_size'],rows=self.rows,context=template.capacity,
                layers=c['num_hidden_layers'])
            C.memmove(caps,C.byref(record),C.sizeof(record))
            self._publish(engine,'engine',output)

        def create(engine_id,output,raw_options):
            engine = self._get(engine_id,'engine')
            options = raw_options.contents
            budget = _integer(int(options.memory),'explicit Mixtral session budget',maximum=(1<<63)-1)
            session = engine.create(int(options.context) or engine.template.capacity,budget)
            self._publish(session,'session',output)

        def run(session_id,raw_request):
            session = self._get(session_id,'session')
            model,request = session.model,raw_request.contents
            with model._lock:
                model._check()
                if int(request.position) != model.position or not 0 < int(request.rows) <= self.rows:
                    raise ValueError('Mixtral run position/row admission mismatch')
                model._busy = True
                try:
                    if request.cancel and request.cancel(request.cancel_data):
                        raise RuntimeError('Mixtral range execution cancelled')
                finally:
                    model._busy = False
                width = model.config['hidden_size']
                if (not request.input or not request.output or request.input_bytes<int(request.rows)*width*4 or
                        request.output_bytes<int(request.rows)*width*4):
                    raise ValueError('Mixtral range activation capacity mismatch')
                hidden = np.ctypeslib.as_array(C.cast(request.input,C.POINTER(C.c_float)),
                    shape=(int(request.rows)*width,)).reshape(1,int(request.rows),width)
                cancelled = (lambda: bool(request.cancel(request.cancel_data))) if request.cancel else None
                result = model.forward(hidden,cancelled=cancelled)[0]
                # Native execution is atomic; cancellation is admitted at the
                # operation boundary before committing a subsequent call.
                C.memmove(request.output,result.ctypes.data,result.nbytes)

        def snapshot(session_id,callback,user):
            session = self._get(session_id,'session')
            write_range_cache(session.model,_WriteStream(callback,user,self.snapshot_bytes),
                              max_bytes=self.snapshot_bytes)

        def restore(session_id,callback,user):
            session = self._get(session_id,'session')
            restore_range_cache(session.model,_MixtralReadStream(callback,user,self.snapshot_bytes),
                                max_bytes=self.snapshot_bytes)

        return _SegmentAdapter(version=1,engine=self.segment_id,
            open=self._guard(_SOPEN,open_engine),destroy=self._destroy('engine'),
            create=self._guard(_CREATE,create),destroy_session=self._destroy('session'),
            run=self._guard(_RUN,run),snapshot=self._guard(_SNAPSHOT,snapshot),
            restore=self._guard(_SNAPSHOT,restore))


__all__ = ['NativeMixtralPublicAdapters']
