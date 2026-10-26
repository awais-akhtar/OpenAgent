"""Register owned native GLM/Inkling models in the public Segment/Edge ABI.

Registration copies immutable tensor storage into the native runtime. Native
callbacks execute the decoder library directly, without Python run callbacks.
The owner retains both libraries until every public handle has been closed.
"""
import ctypes as C
import _ctypes
import hashlib
import os
from pathlib import Path
import threading
import weakref
import numpy as np
from .segment_edge_runtime import _Functions
from .native_glm5_next import NativeGlm5NextTextDecoder
from .native_inkling import NativeInklingDecoder,Tensor,Config,Layer,tensor
from .independent_inkling import GLOBAL_NAMES,ALL_LAYER_NAMES
from .session_recovery import _path as _safe_path
from .native_library_identity import release_finalizer

class FamilyModelDescriptor(C.Structure):
    _fields_=[('size',C.c_uint32),('version',C.c_uint32),('family',C.c_uint32),('rows',C.c_uint32),
        ('model',C.c_char_p),('identity',C.c_char_p),('max_resource_bytes',C.c_uint64),('max_work_bytes',C.c_uint64),
        ('glm_config',C.c_int32*26),('glm_floats',C.c_float*6),('glm_schedule',C.POINTER(C.c_int32)),
        ('glm_weights',C.POINTER(Tensor)),('glm_step',C.c_void_p),('inkling_config',Config),
        ('inkling_layers',C.POINTER(Layer)),('inkling_run',C.c_void_p),('globals',Tensor*4)]


def _verify_family_abi_layouts(probe):
    """Fail closed unless every public family record matches the C ABI.

    Kept separate from registration so the full record contract (including
    nested Tensor/Config/Layer layouts) can be exercised without constructing
    a model or allocating native state.
    """
    probe.argtypes=[C.c_uint32,C.POINTER(C.c_uint64),C.c_size_t]
    probe.restype=C.c_size_t
    records=((1,FamilyModelDescriptor),(2,Tensor),(3,Config),(4,Layer))
    for number,record in records:
        expected=[C.sizeof(record),len(record._fields_)]+[
            getattr(record,name).offset for name,_ in record._fields_]
        observed=(C.c_uint64*len(expected))()
        count=probe(number,observed,len(observed))
        if count!=len(expected) or list(observed)!=expected:
            raise ValueError('native family descriptor ABI layout differs: '+str(number))

_PINNED=set()
_GUARD=threading.RLock()

def _update_checkpoint_provenance(digest,inventory):
    """Bind public ABI cache identity to the exact admitted checkpoint root/config."""
    root=Path(inventory.root).resolve(strict=True)
    config_digest=inventory.config_sha256
    if (type(config_digest) is not str or len(config_digest)!=64 or
            any(char not in '0123456789abcdef' for char in config_digest)):
        raise ValueError('verified checkpoint configuration digest required for public ABI identity')
    inventory.assert_unchanged()
    digest.update(b'openagent-public-family-checkpoint-v1\0')
    digest.update(str(root).encode('utf-8'))
    digest.update(b'\0'+config_digest.encode('ascii'))

class NativeFamilyPublicAdapters:
    def __init__(self,runtime_library,decoder,*,model_directory,max_resource_bytes,max_work_bytes,
                 max_batch_rows=32,own_decoder=False,enable_layer_ranges=False):
        if type(decoder) not in (NativeGlm5NextTextDecoder,NativeInklingDecoder):
            raise TypeError('exact admitted native GLM or Inkling decoder required')
        if type(enable_layer_ranges) is not bool or type(own_decoder) is not bool:
            raise ValueError('boolean native layer-range extension policy required')
        provider=getattr(decoder,'_expert_provider',None)
        if decoder.position:
            raise ValueError('public native model registration requires a fresh decoder')
        if any(type(value) is not int or value<=0 for value in (max_resource_bytes,max_work_bytes,max_batch_rows)) or max_batch_rows>4096:
            raise ValueError('explicit aggregate native resource/work and row limits required')
        root=_safe_path(model_directory).resolve(strict=True)
        if not root.is_dir(): raise ValueError('admitted local checkpoint root required')
        source_family,model_type=(('glm5-next','glm5_next_text')
            if type(decoder) is NativeGlm5NextTextDecoder else ('inkling','inkling_text'))
        # An already-materialized decoder does not prove that its source
        # checkpoint belongs to an admitted public-ABI route. Reconcile the
        # persisted headers before opening either native library.
        from .source_scope import require_checkpoint_inventory_admission
        inventory=require_checkpoint_inventory_admission(root,family=source_family,
            model_type=model_type,backend='public-range',arithmetic='fp32',
            max_metadata_bytes=max_resource_bytes)
        from .family_source_binding import require_decoder_snapshot
        require_decoder_snapshot(decoder,inventory,family=source_family,model_type=model_type)
        self._lock=threading.RLock();self._closed=False;self._model_id=None
        self._decoder=decoder if own_decoder else None
        self.native=self.kernel=self._kernel_release=None
        self._expert_bank=None
        expert_budget=0 if provider is None else getattr(provider,'max_bytes',0)
        if provider is not None and (type(expert_budget) is not int or expert_budget<=0):
            raise ValueError('explicit original expert owner admission required')
        source_bytes=sum(value.nbytes for value in decoder.weights.values())
        if source_bytes+max_work_bytes+expert_budget>=max_resource_bytes:
            raise MemoryError('source and owned native model construction exceed aggregate admission')
        self.root=root;self.family='glm5_next_text' if type(decoder) is NativeGlm5NextTextDecoder else 'inkling_text'
        self.segment_id=self.edge_id='openagent-glm5-native-model' if self.family=='glm5_next_text' else 'openagent-inkling-native-model'
        self.max_resource_bytes=max_resource_bytes;self.max_work_bytes=max_work_bytes
        self.rows=max_batch_rows
        try:
            from .native_library_identity import NativeLibraryIdentity
            decoder._library_identity.assert_unchanged()
            runtime_identity=NativeLibraryIdentity.read(runtime_library)
            self.native=_Functions(runtime_library)
            runtime_identity.assert_unchanged()
            kernel_identity=NativeLibraryIdentity.read(decoder.library._name)
            if kernel_identity.sha256 != decoder._library_identity.sha256:
                raise ValueError('public family kernel differs from the admitted decoder library')
            library=kernel_identity.path;digest=hashlib.sha256()
            digest.update(bytes.fromhex(kernel_identity.sha256))
            self.kernel=C.CDLL(str(library));release=_ctypes.FreeLibrary if os.name=='nt' else _ctypes.dlclose
            self._kernel_release=weakref.finalize(self,release,self.kernel._handle)
            kernel_identity.assert_unchanged()
            self._runtime_identity,self._kernel_identity=runtime_identity,kernel_identity
            _update_checkpoint_provenance(digest,inventory)
            if provider is not None:
                from .native_original_experts import NativeOriginalExperts
                self._expert_bank=NativeOriginalExperts(self.native,provider,max_bytes=expert_budget)
                digest.update(self._expert_bank.identity.encode())
            runtime_identity.assert_unchanged();digest.update(runtime_identity.sha256.encode())
            descriptor=FamilyModelDescriptor();descriptor.size=C.sizeof(descriptor);descriptor.version=1
            descriptor.rows=max_batch_rows;descriptor.model=str(root).encode('utf-8')
            descriptor.max_resource_bytes=max_resource_bytes-source_bytes-expert_budget
            descriptor.max_work_bytes=max_work_bytes
            digest.update(repr((self.family,decoder.config,max_batch_rows,max_work_bytes)).encode())
            for name,value in sorted(decoder.weights.items()):
                digest.update(name.encode());digest.update(memoryview(value).cast('B'))
            layers=None
            if self.family=='glm5_next_text':
                descriptor.family=1
                descriptor.glm_config=decoder._geometry;descriptor.glm_floats=decoder._floats
                descriptor.glm_schedule=decoder._schedule
                descriptor.glm_weights=C.cast(decoder._weight_buffers,C.POINTER(Tensor))
                descriptor.glm_step=C.cast(self.kernel.oa_glm5_next_step_embeddings,C.c_void_p)
                descriptor.globals[0]=Tensor(decoder._embedding.p,decoder._embedding.n)
                descriptor.globals[2]=Tensor(decoder._norm.p,decoder._norm.n)
                descriptor.globals[3]=Tensor(decoder._head.p,decoder._head.n)
                self.layers=decoder.config.num_hidden_layers;self.context=decoder.max_tokens
                self.vocabulary=decoder.config.vocab_size
                self.hidden_size=decoder.config.hidden_size;self.streams=int(decoder._geometry[3])
            else:
                descriptor.family=2;descriptor.inkling_config=decoder._c
                layers=(Layer*len(decoder.geometry))()
                for index,g in enumerate(decoder.geometry):
                    weights=(Tensor*26)(*[tensor(decoder.weights[f'model.layers.{index}.{name}'])
                        if f'model.layers.{index}.{name}' in decoder.weights else Tensor() for name in ALL_LAYER_NAMES])
                    layers[index]=Layer(g.heads,g.kv_heads,g.dim,g.extent,g.window,
                        int(decoder.config['mlp_layer_types'][index]=='sparse'),weights,Tensor())
                descriptor.inkling_layers=layers
                descriptor.inkling_run=C.cast(self.kernel.oa_inkling_run_embeddings,C.c_void_p)
                descriptor.globals=(Tensor*4)(*[tensor(decoder.weights[name]) for name in GLOBAL_NAMES])
                self.layers=decoder.config['num_hidden_layers'];self.context=decoder.config['max_position_embeddings']
                self.vocabulary=decoder.config['unpadded_vocab_size'] or decoder.config['vocab_size']
                self.hidden_size=decoder.config['hidden_size'];self.streams=1
            self.identity=digest.hexdigest();descriptor.identity=self.identity.encode('ascii')
            register=self.native.library.oa_segment_family_register
            register.argtypes=[C.POINTER(FamilyModelDescriptor),C.POINTER(C.c_uint64),C.c_void_p,C.c_size_t]
            register.restype=C.c_int
            version=self.native.library.oa_segment_family_version;version.argtypes=[];version.restype=C.c_uint32
            if version()!=1:raise ValueError('unsupported native public family model ABI')
            probe=self.native.library.oa_segment_family_layout
            _verify_family_abi_layouts(probe)
            result=C.c_uint64();error=C.create_string_buffer(4096)
            runtime_identity.assert_unchanged();kernel_identity.assert_unchanged()
            if register(C.byref(descriptor),C.byref(result),error,len(error)):
                raise RuntimeError(error.value.decode('utf-8',errors='replace'))
            self._model_id=int(result.value)
            with _GUARD:_PINNED.add(self)
            if enable_layer_ranges:
                glm=self.family=='glm5_next_text'
                capability=(self.kernel.oa_glm5_next_range_version if glm else self.kernel.oa_inkling_range_version)
                capability.argtypes=[];capability.restype=C.c_uint32 if glm else C.c_uint64
                if capability()!=1:raise ValueError('unsupported native publisher layer-range ABI')
                enable=(self.native.library.oa_segment_family_enable_glm_ranges if glm
                        else self.native.library.oa_segment_family_enable_inkling_ranges)
                enable.argtypes=[C.c_uint64,C.c_void_p,C.c_void_p,C.c_size_t];enable.restype=C.c_int
                pointer=C.cast(self.kernel.oa_glm5_next_step_range if glm else self.kernel.oa_inkling_run_range,C.c_void_p)
                if enable(self._model_id,pointer,error,len(error)):
                    raise RuntimeError(error.value.decode('utf-8',errors='replace'))
            self.layer_ranges_enabled=enable_layer_ranges
            self.routing_observation_supported=self.family=='inkling_text'
            if enable_layer_ranges and self.family=='glm5_next_text':
                enable_observer=getattr(self.native.library,'oa_segment_family_enable_glm_route_observer',None)
                observe=getattr(self.kernel,'oa_glm5_next_step_range_observed',None)
                if enable_observer is not None and observe is not None:
                    enable_observer.argtypes=[C.c_uint64,C.c_void_p,C.c_void_p,C.c_size_t]
                    enable_observer.restype=C.c_int
                    if enable_observer(self._model_id,C.cast(observe,C.c_void_p),error,len(error)):
                        raise RuntimeError(error.value.decode('utf-8',errors='replace'))
                    self.routing_observation_supported=True
            self.transport_width=(self.hidden_size*self.streams+self.context
                if enable_layer_ranges and self.family=='glm5_next_text' else self.hidden_size)
            if self._expert_bank is not None:
                attach=self.native.library.oa_segment_family_attach_original_experts
                attach.argtypes=[C.c_uint64,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t]
                attach.restype=C.c_int
                glm=inkling=inkling_range=None
                if self.family=='glm5_next_text':
                    if enable_layer_ranges:glm=C.cast(self.kernel.oa_glm5_next_step_range_streamed,C.c_void_p)
                else:
                    inkling=C.cast(self.kernel.oa_inkling_run_streamed,C.c_void_p)
                    if enable_layer_ranges:inkling_range=C.cast(self.kernel.oa_inkling_run_range_streamed,C.c_void_p)
                if attach(self._model_id,self._expert_bank.handle,glm,inkling,inkling_range,error,len(error)):
                    raise RuntimeError(error.value.decode('utf-8',errors='replace'))
            if own_decoder:
                decoder.close();self._decoder=None
        except BaseException as error:
            try:self.close()
            except BaseException as failure:
                from .failure_reporting import retain_cleanup_failure
                retain_cleanup_failure(error,self.close,failure,
                    'native public model cleanup retained: ')
            raise

    def final_hidden(self,wire):
        """Use the native GLM head's stream mean and final RMS boundary."""
        with self._lock:
            if self._closed or self._model_id is None or not self.layer_ranges_enabled or self.family!='glm5_next_text':
                raise RuntimeError('open native GLM stream owner required')
            self._runtime_identity.assert_unchanged();self._kernel_identity.assert_unchanged()
            value=np.asarray(wire)
            if value.dtype!=np.float32 or value.ndim!=2 or not 0<value.shape[0]<=self.rows or value.shape[1]!=self.transport_width:
                raise ValueError('bounded native GLM stream rows required')
            # This method is a public activation boundary. Reject non-finite
            # values before crossing into native code, and do not publish a
            # non-finite result as a valid hidden-state tensor.
            if not np.isfinite(value).all():
                raise ValueError('native GLM stream activations must be finite')
            if value.nbytes*2+value.shape[0]*self.hidden_size*8>self.max_work_bytes:
                raise MemoryError('native GLM final hidden transport exceeds admission')
            value=np.ascontiguousarray(value);output=np.empty((value.shape[0],self.hidden_size),np.float32)
            operation=self.native.library.oa_segment_family_glm_final_hidden
            operation.argtypes=[C.c_uint64,C.c_void_p,C.c_uint32,C.c_uint64,C.c_void_p,C.c_uint64,C.c_void_p,C.c_size_t]
            operation.restype=C.c_int;error=C.create_string_buffer(4096)
            if operation(self._model_id,value.ctypes.data,value.shape[0],value.size,
                    output.ctypes.data,output.size,error,len(error)):
                raise RuntimeError(error.value.decode('utf-8',errors='replace'))
            if not np.isfinite(output).all():
                raise FloatingPointError('native GLM final hidden output is non-finite')
            return output

    def close(self):
        with self._lock:
            if self._closed:return
            if self._model_id is not None:
                release=self.native.library.oa_segment_family_release
                release.argtypes=[C.c_uint64,C.c_void_p,C.c_size_t];release.restype=C.c_int
                error=C.create_string_buffer(4096)
                if release(self._model_id,error,len(error)):
                    raise RuntimeError(error.value.decode('utf-8',errors='replace'))
                self._model_id=None
            if self._expert_bank is not None:
                self._expert_bank.close();self._expert_bank=None
            if self._decoder is not None:
                self._decoder.close();self._decoder=None
            release_finalizer(self,'_kernel_release')
            if self.native is not None:self.native.close()
            self.kernel=None;self._closed=True
            with _GUARD:_PINNED.discard(self)

__all__=['NativeFamilyPublicAdapters','FamilyModelDescriptor']
