"""Own a native Qwen4 model registered in the actual Segment/Edge dispatcher.

No Python execution callback is registered. Resident weights are copied by the
native owner; expert/PLE contexts are retained by native reference counts.
Kernel and runtime libraries remain pinned until every public child closes.
"""
import ctypes as C
import _ctypes
import hashlib
import os
from pathlib import Path
import threading
import weakref
from contextlib import ExitStack,contextmanager
import numpy as np
from .native_qwen35 import Tensor,tensor
from .native_qwen4 import NativeQwen4TextDecoder
from .native_qwen4_composition import ComposeConfig,ComposeLayer,NativeQwen4Composition
from .native_library_identity import NativeLibraryIdentity
from .segment_edge_runtime import _Functions
from .session_recovery import _path as _safe_path


class Qwen4PublicDescriptor(C.Structure):
    _fields_=[('size',C.c_uint32),('version',C.c_uint32),('rows',C.c_uint32),('reserved',C.c_uint32),
        ('root',C.c_char_p),('identity',C.c_char_p),('max_resource_bytes',C.c_uint64),
        ('max_work_bytes',C.c_uint64),('external_store_bytes',C.c_uint64),('config',ComposeConfig),
        ('layers',C.POINTER(ComposeLayer)),('embedding',Tensor),('wire',C.c_void_p),('head',C.c_void_p),
        ('experts',C.c_void_p),('ple',C.POINTER(C.c_void_p)),('hidden',C.c_void_p)]


_PINNED=set()
_PIN_LOCK=threading.RLock()


def _register_checkpoint_bound_model(snapshot,register,*args):
    """Recheck the admitted headers immediately before native publication."""
    snapshot.assert_unchanged()
    return register(*args)


class NativeQwen4PublicAdapters:
    segment_id=edge_id='openagent-qwen4-native-model'
    family='qwen4_exp_text'
    layer_ranges_enabled=True

    def __init__(self,runtime_library,decoder,*,model_directory,max_resource_bytes,max_work_bytes,
                 max_batch_rows=32,own_decoder=False):
        if type(decoder) is not NativeQwen4TextDecoder or decoder.range_mode:
            raise TypeError('exact complete native Qwen4 decoder required')
        if decoder.position or decoder._snapshots:raise ValueError('fresh Qwen4 model without cache/snapshots required')
        if any(type(v) is not int or not 0<v<2**64 for v in (max_resource_bytes,max_work_bytes,max_batch_rows)) or max_batch_rows>4096:
            raise ValueError('bounded aggregate native model/work and row admissions required')
        if type(own_decoder) is not bool:raise ValueError('boolean source decoder ownership required')
        root=_safe_path(model_directory).resolve(strict=True)
        if not root.is_dir():raise ValueError('admitted local publisher checkpoint root required')
        # Do not let a preloaded Python decoder bypass exact public-route
        # admission. Header-only inventory runs before runtime registration.
        from .source_scope import require_checkpoint_inventory_admission
        inventory=require_checkpoint_inventory_admission(root,family='qwen4',
            model_type='qwen4_exp_text',backend='public-range',arithmetic='fp32',
            max_metadata_bytes=max_resource_bytes)
        from .family_source_binding import require_decoder_snapshot
        require_decoder_snapshot(decoder,inventory,family='qwen4',model_type='qwen4_exp_text')
        self._lock=threading.RLock();self._closed=False;self._model_id=None
        self._active_operation=False
        self._observing=False
        self._pending_observation=None
        self.native=self.kernel=self._kernel_release=None
        self._decoder=decoder if own_decoder else None
        self.root=root;self.rows=max_batch_rows;self.max_resource_bytes=max_resource_bytes;self.max_work_bytes=max_work_bytes
        c=decoder.config;self.layers=c.num_hidden_layers;self.context=c.max_position_embeddings
        self.hidden_size=c.hidden_size;self.streams=c.hc_count;self.transport_width=c.hidden_size*c.hc_count+5
        self.vocabulary=c.vocab_size
        source_bytes=sum(value.nbytes for value in decoder.weights.values())
        experts=getattr(decoder,'_composition_experts',None);ple=getattr(decoder,'_composition_ple',None)
        if (decoder._expert_provider is not None and experts is None) or (decoder._ple_provider is not None and ple is None):
            raise ValueError('public native Qwen4 requires native original streamed stores')
        external=sum(owner.max_bytes for owner in (experts,ple) if owner is not None)
        source_stores=sum(getattr(provider,'max_bytes',0) for provider in
            (decoder._expert_provider,decoder._ple_provider) if provider is not None)
        descriptor_bytes=c.num_hidden_layers*(C.sizeof(ComposeLayer)+C.sizeof(C.c_void_p))+C.sizeof(Qwen4PublicDescriptor)
        if source_bytes+source_stores+descriptor_bytes+external+max_work_bytes>=max_resource_bytes:
            raise MemoryError('source/native Qwen4 construction exceeds aggregate admission')
        try:
            runtime_identity=NativeLibraryIdentity.read(runtime_library)
            kernel_identity=NativeLibraryIdentity.read(decoder.library._name)
            self.native=_Functions(runtime_identity.path)
            self.kernel=C.CDLL(str(kernel_identity.path))
            release=_ctypes.FreeLibrary if os.name=='nt' else _ctypes.dlclose
            self._kernel_release=weakref.finalize(self,release,self.kernel._handle)
            NativeQwen4Composition(self.kernel)
            with decoder._lock,ExitStack() as borrowed:
                decoder._check_open()
                for bank in (experts,ple):
                    if bank is None:continue
                    borrowed.enter_context(bank._lock)
                    bank_runtime=NativeLibraryIdentity.read(bank._runtime.library._name)
                    if bank_runtime.sha256!=runtime_identity.sha256:
                        raise ValueError('streamed native stores must use the same admitted runtime image')
                descriptor=Qwen4PublicDescriptor();descriptor.size=C.sizeof(descriptor);descriptor.version=2
                descriptor.rows=max_batch_rows;descriptor.root=str(root).encode('utf-8')
                descriptor.max_resource_bytes=max_resource_bytes-source_bytes-source_stores-descriptor_bytes
                descriptor.max_work_bytes=max_work_bytes;descriptor.external_store_bytes=external
                descriptor.config=NativeQwen4Composition(decoder.library).configure(decoder,max_work_bytes=max_work_bytes)
                descriptor.embedding=tensor(decoder.weights['model.embed_tokens.weight'])
                descriptor.wire=C.cast(self.kernel.oa_qwen4_compose_wire,C.c_void_p)
                descriptor.head=C.cast(self.kernel.oa_qwen4_compose_head_wire,C.c_void_p)
                descriptor.hidden=C.cast(self.kernel.oa_qwen4_compose_hidden_wire,C.c_void_p)
                layers=(ComposeLayer*c.num_hidden_layers)();contexts=(C.c_void_p*c.num_hidden_layers)()
                def hyper(prefix):return (Tensor*4)(*[tensor(decoder.weights[prefix+name]) for name in
                    ('hc_norm.weight','input_mix_weight_down.weight','input_mix_weight_up.weight','block_inject_weight.weight')])
                for index in range(c.num_hidden_layers):
                    p=f'model.layers.{index}.';layer=layers[index]
                    layer.attention_hyper=hyper(p+'attn_hyper_connection.');layer.mlp_hyper=hyper(p+'mlp_hyper_connection.')
                    layer.kind=int(c.layer_types[index]!='linear_attention')
                    if layer.kind:
                        bank='self_attn.';names=('q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight','q_norm.weight',
                            'k_norm.weight','indexer.index_qk_proj.weight','indexer.q_layernorm.weight','indexer.k_layernorm.weight')
                    else:
                        bank='linear_attn.';names=('in_proj_qkv.weight','in_proj_z.weight','in_proj_b.weight','in_proj_a.weight',
                            'conv1d.weight','norm.weight','dt_bias','A_log','out_proj.weight')
                    layer.attention=(Tensor*9)(*[tensor(decoder.weights[p+bank+name]) for name in names])
                    layer.mlp=(Tensor*7)(*[Tensor() if experts is not None and slot in (1,2) else tensor(decoder.weights[p+'mlp.'+name])
                        for slot,name in enumerate(('gate.weight','experts.gate_up_proj','experts.down_proj','shared_expert.gate_proj.weight',
                            'shared_expert.up_proj.weight','shared_expert.down_proj.weight','shared_expert_gate.weight'))])
                    if index+1 in c.ple_layer_ids:
                        layer.ple_enabled=1;prefix=p+'ple.'
                        layer.ple_weights=(Tensor*7)(*[Tensor() if ple is not None and slot==0 else tensor(decoder.weights[prefix+name])
                            for slot,name in enumerate(('ple_embedding.ngram_embedding.weight','key_proj.weight','value_proj.weight',
                                'norm_key.weight','norm_query.weight','norm_conv.weight','conv1d.weight'))])
                        for field,name in (('multipliers','layer_multipliers'),('sizes','ngram_heads_vocab_sizes'),('offsets','ngram_heads_offsets')):
                            value=decoder.weights[prefix+'ple_embedding.'+name]
                            setattr(layer,field,value.ctypes.data_as(C.POINTER(C.c_int64)))
                        layer.multiplier_count=c.ngram_size;layer.lane_count=(c.ngram_size-1)*c.heads_per_ngram
                        if ple is not None:contexts[index]=ple.context(index);layer.ple_table_rows=ple.shapes[index][0]
                descriptor.layers=layers;descriptor.ple=contexts
                if experts is not None:descriptor.experts=experts.handle
                identity=(decoder.identity,runtime_identity.sha256,kernel_identity.sha256,
                    max_resource_bytes,max_work_bytes,max_batch_rows,
                    None if experts is None else experts.identity,None if ple is None else ple.identity)
                self.identity=hashlib.sha256(repr(identity).encode()).hexdigest();descriptor.identity=self.identity.encode('ascii')
                probe=self.native.library.oa_qwen4_public_model_layout
                probe.argtypes=[C.POINTER(C.c_uint64),C.c_size_t];probe.restype=C.c_size_t
                expected=[C.sizeof(Qwen4PublicDescriptor),len(Qwen4PublicDescriptor._fields_)]+[
                    getattr(Qwen4PublicDescriptor,name).offset for name,_ in Qwen4PublicDescriptor._fields_]
                actual=(C.c_uint64*len(expected))()
                if probe(actual,len(actual))!=len(expected) or list(actual)!=expected:raise ValueError('public Qwen4 model descriptor ABI differs')
                register=self.native.library.oa_qwen4_public_register
                register.argtypes=[C.POINTER(Qwen4PublicDescriptor),C.POINTER(C.c_uint64),C.c_void_p,C.c_size_t];register.restype=C.c_int
                result=C.c_uint64();error=C.create_string_buffer(4096)
                status=_register_checkpoint_bound_model(inventory,register,
                    C.byref(descriptor),C.byref(result),error,len(error))
                if status:raise RuntimeError(error.value.decode(errors='replace'))
                self._model_id=int(result.value)
                with _PIN_LOCK:_PINNED.add(self)
                runtime_identity.assert_unchanged();kernel_identity.assert_unchanged()
            if own_decoder:decoder.close();self._decoder=None
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:
                from .failure_reporting import retain_cleanup_failure
                retain_cleanup_failure(primary,self.close,cleanup,
                    'public Qwen4 ownership cleanup retained: ')
            raise

    @contextmanager
    def routing_scope(self, observer, *, max_host_bytes):
        """Copy executed native routing traces inside a bounded request scope."""
        if not callable(observer) or type(max_host_bytes) is not int or max_host_bytes <= 0:
            raise ValueError('bounded routing observer required')
        with self._lock:
            if (self._closed or self._model_id is None or self._observing or
                    self._active_operation or self._pending_observation is not None):
                raise RuntimeError('idle open Qwen4 observation owner required')
            signature=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_uint32,C.c_uint64,C.c_uint64,C.c_uint64,
                C.POINTER(C.c_float),C.POINTER(C.c_int64),C.POINTER(C.c_float))
            failures=[]
            def capture(context,layer,rows,experts,topk,raw,chosen,scales):
                try:
                    if (not 0 <= layer < self.layers or not 0 < rows <= self.rows or
                            not 0 < topk <= experts or
                            rows*(experts*12+topk*40)>max_host_bytes):
                        raise MemoryError('native routing export exceeds scope admission')
                    if not raw or not chosen or not scales:
                        raise ValueError('native routing callback supplied null array storage')
                    logits=np.ctypeslib.as_array(raw,shape=(rows*experts,)).copy().reshape(rows,experts)
                    selected=np.ctypeslib.as_array(chosen,shape=(rows*topk,)).copy().reshape(rows,topk)
                    weights=np.ctypeslib.as_array(scales,shape=(rows*topk,)).copy().reshape(rows,topk)
                    observer(layer,logits,selected,weights)
                    return 0
                except BaseException as failure:
                    failures.append(failure);return 1
            callback=signature(capture)
            begin=self.native.library.oa_qwen4_public_observation_begin
            end=self.native.library.oa_qwen4_public_observation_end
            for operation in (begin,end):
                operation.argtypes=[C.c_char_p,C.c_void_p,C.c_void_p];operation.restype=C.c_int
            identity=self.identity.encode('ascii');pointer=C.cast(callback,C.c_void_p)
            if begin(identity,pointer,None):raise RuntimeError('native routing scope already active')
            self._observing=True
            def finish():
                status=end(identity,pointer,None)
                if status:
                    self._pending_observation=(threading.get_ident(),identity,callback,pointer,end)
                    raise RuntimeError('native routing scope release failed; callback retained')
                self._observing=False
            try:
                yield
                if failures:raise RuntimeError('native routing observation failed') from failures[0]
            except BaseException as primary:
                try:
                    finish()
                except BaseException as cleanup:
                    from .failure_reporting import add_failure_note, failure_summary
                    add_failure_note(primary,'native routing observer cleanup failed: '+
                        failure_summary(cleanup))
                raise
            else:
                finish()

    def final_hidden(self,wire,*,cancelled=None,max_host_bytes):
        with self._lock:
            # routing_scope invokes user code synchronously from a native
            # callback while this RLock is held.  RLock recursion would let
            # that callback re-enter the native model through this method,
            # which the public ABI does not support.
            if self._observing or self._active_operation:
                raise RuntimeError('native Qwen4 model operation reentry from a native callback is unsupported')
            if self._closed or self._model_id is None:raise RuntimeError('closed public Qwen4 owner')
            if cancelled is not None and not callable(cancelled):
                raise ValueError('Qwen4 cancellation observer must be callable')
            value=np.asarray(wire)
            if (value.dtype!=np.float32 or value.ndim!=2 or value.shape[1]!=self.transport_width
                    or not 0<value.shape[0]<=self.rows or not np.isfinite(value).all()):
                raise ValueError('occupied complete native Qwen4 residual wire required')
            required=2*value.nbytes+value.shape[0]*self.hidden_size*4
            if type(max_host_bytes) is not int or required>max_host_bytes:raise MemoryError('hidden export exceeds host admission')
            value=np.array(value,dtype=np.float32,order='C',copy=True)
            output=np.empty((len(value),self.hidden_size),np.float32);failures=[]
            def check(user):
                try:return 1 if cancelled is not None and bool(cancelled()) else 0
                except BaseException as failure:failures.append(failure);return 1
            callback=C.CFUNCTYPE(C.c_int,C.c_void_p)(check)
            operation=self.native.library.oa_qwen4_public_hidden
            operation.argtypes=[C.c_uint64,C.c_uint32,C.POINTER(C.c_float),C.c_uint64,
                C.POINTER(C.c_float),C.c_uint64,C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t]
            operation.restype=C.c_int;error=C.create_string_buffer(4096)
            self._active_operation=True
            try:
                status=operation(self._model_id,len(value),value.ctypes.data_as(C.POINTER(C.c_float)),value.size,
                    output.ctypes.data_as(C.POINTER(C.c_float)),output.size,C.cast(callback,C.c_void_p),None,error,len(error))
            finally:
                self._active_operation=False
            if failures:raise InterruptedError('hidden export cancellation failed') from failures[0]
            if status:raise RuntimeError(error.value.decode(errors='replace'))
            if not np.isfinite(output).all():raise ValueError('nonfinite native final hidden output')
            return output

    def close(self):
        with self._lock:
            if self._closed:return
            if self._active_operation:
                raise RuntimeError('cannot close Qwen4 owner during an active native operation')
            if self._pending_observation is not None:
                thread,identity,callback,pointer,end=self._pending_observation
                if thread!=threading.get_ident() or end(identity,pointer,None):
                    raise RuntimeError('native routing scope requires cleanup on its owning thread')
                self._pending_observation=None;self._observing=False
            if self._observing:raise RuntimeError('cannot close active Qwen4 routing observer')
            if self._model_id is not None:
                operation=self.native.library.oa_qwen4_public_unregister
                operation.argtypes=[C.c_uint64,C.c_void_p,C.c_size_t];operation.restype=C.c_int
                error=C.create_string_buffer(4096)
                if operation(self._model_id,error,len(error)):raise RuntimeError(error.value.decode(errors='replace'))
                self._model_id=None
            if self._decoder is not None:self._decoder.close();self._decoder=None
            from .native_library_identity import release_finalizer
            release_finalizer(self,'_kernel_release')
            if self.native is not None:self.native.close();self.native=None
            self.kernel=None;self._closed=True
            with _PIN_LOCK:_PINNED.discard(self)


__all__=['NativeQwen4PublicAdapters','Qwen4PublicDescriptor']
