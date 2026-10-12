"""Owned native dense projections from original publisher shard ranges."""
import ctypes as C
import hashlib
import math
import threading
from types import MappingProxyType
import numpy as np
from .native_original_experts import OriginalFile,OriginalMatrix
from .publisher_tensor_reader import PublisherTensorReader,TensorSlice


class NativeOriginalProjections:
    policy='original-row-stream-native-selected-dot-v1'

    @classmethod
    def from_k3(cls,runtime,checkpoint,*,max_bytes,include_vision=False):
        """Select mapped matrix operations, excluding routers and row tables."""
        from .publisher_checkpoint.service import PublisherCheckpoint
        if type(checkpoint) is not PublisherCheckpoint or checkpoint.report.family!='kimi_k3' or type(include_vision) is not bool:
            raise ValueError('exact K3 checkpoint and media selection required')
        allowed={'q_proj','k_proj','v_proj','q_a_proj','q_b_proj','kv_a_proj_with_mqa','kv_b_proj',
                 'o_proj','f_a_proj','f_b_proj','b_proj','g_proj','gate_proj','up_proj','down_proj',
                 'lm_head','routed_expert_down_proj','routed_expert_up_proj','wqkv','wo','fc0','fc1'}
        bindings={}
        for rule in checkpoint.report.contracts:
            if (rule.encoding!='float32' or len(rule.shape)!=2 or len(rule.sources)!=1
                    or (rule.target.startswith('k3.vision.') and not include_vision)):
                continue
            name=rule.sources[0]
            if '.experts.' in name or not name.endswith('.weight'):continue
            if name.split('.')[-2] not in allowed and name not in ('mm_projector.proj.0.weight','mm_projector.proj.2.weight'):continue
            if name in bindings:raise ValueError('duplicate original native K3 projection')
            bindings[name]=TensorSlice(name,rule.shape)
        reader=PublisherTensorReader(checkpoint._metadata,max_work_bytes=max_bytes)
        try:return cls(runtime,reader,bindings,max_bytes=max_bytes,reduction='fp64')
        finally:reader.close()

    def __init__(self,runtime,reader,bindings,*,max_bytes,reduction='fp32',rows_only=False):
        if type(rows_only) is not bool:raise ValueError('boolean original row-storage policy required')
        self.rows_only=rows_only
        if type(reader) is not PublisherTensorReader or type(bindings) is not dict or not bindings:
            raise ValueError('admitted original reader and named dense slices required')
        if type(max_bytes) is not int or max_bytes<=0 or len(bindings)>1048576:
            raise ValueError('bounded native dense inventory required')
        self._lock=threading.RLock();self._runtime=runtime;self._handle=None;self._closed=False
        self.max_bytes=max_bytes;self.sources=MappingProxyType(dict(bindings))
        if reduction not in ('fp32','fp64'):raise ValueError('explicit native dense reduction required')
        self.reduction=reduction
        snapshot=reader.snapshot;snapshot.assert_unchanged()
        dtype={'F32':1,'F16':2,'BF16':3,'F8_E4M3':4,'F8_E8M0':5}
        records={};shards=set()
        for name,binding in sorted(bindings.items()):
            if type(name) is not str or not name or type(binding) is not TensorSlice:
                raise ValueError('unique named original tensor slices required')
            record=snapshot.tensors[binding.source];scale=None
            if (len(binding.shape)!=2 or len(record.shape) not in (2,3) or
                    any(type(v) is not int or not 0<v<=limit for v,limit in zip(binding.shape,
                        (2**32-1 if rows_only else 1048576,1048576))) or
                    binding.shape[1]!=record.shape[-1] or record.dtype not in ('F32','F16','BF16','F8_E4M3') or
                    type(binding.element_offset) is not int or binding.element_offset<0 or
                    binding.element_offset%binding.shape[1] or
                    binding.element_offset+math.prod(binding.shape)>math.prod(record.shape)):
                raise ValueError('original dense matrix layout differs: '+name)
            if record.dtype=='F8_E4M3':
                blocks=reader.block_shape
                if blocks is None:raise ValueError('original FP8 block geometry required')
                scale_name=reader.scale_names.get(binding.source,
                    binding.source[:-7]+'.weight_scale_inv' if binding.source.endswith('.weight') else binding.source+'_scale_inv')
                scale=snapshot.tensors[scale_name]
                shape=record.shape[:-2]+tuple((v+b-1)//b for v,b in zip(record.shape[-2:],blocks))
                if scale.shape!=shape or scale.dtype not in ('F32','F16','BF16','F8_E8M0'):
                    raise ValueError('original dense FP8 scales differ')
                shards.add(scale.shard)
            records[name]=(record,scale);shards.add(record.shard)
        shards=tuple(sorted(shards))
        staging=(len(bindings)*C.sizeof(OriginalMatrix)+len(shards)*C.sizeof(OriginalFile))
        if len(shards)>4096 or 2*staging>=max_bytes:
            raise MemoryError('native dense descriptors exceed aggregate admission')
        files=(OriginalFile*len(shards))();paths=[];file_ids={name:i for i,name in enumerate(shards)}
        for index,name in enumerate(shards):
            path=(snapshot.root/name).resolve(strict=True);stamp=snapshot.files[name]
            if not path.is_relative_to(snapshot.root) or not 0<=stamp[2]<2**64 or not -2**63<=stamp[3]<2**63:
                raise ValueError('native dense file identity exceeds admitted root or integer bounds')
            paths.append(str(path).encode());files[index]=OriginalFile(paths[-1],stamp[2],stamp[3])
        entries=(OriginalMatrix*len(bindings))();self._indices={}
        for index,(name,binding) in enumerate(sorted(bindings.items())):
            record,scale=records[name];blocks=reader.block_shape or (0,0)
            values=(record.base+record.offsets[0],math.prod(record.shape),binding.element_offset,
                *binding.shape,record.shape[-2],0 if scale is None else scale.base+scale.offsets[0],
                0 if scale is None else math.prod(scale.shape),0 if scale is None else blocks[0],
                0 if scale is None else blocks[1])
            if any(type(value) is not int or not 0<=value<2**64 for value in values):
                raise ValueError('native dense descriptor exceeds unsigned integer bounds')
            entries[index]=OriginalMatrix(file_ids[record.shard],dtype[record.dtype],
                0 if scale is None else file_ids[scale.shard],0 if scale is None else dtype[scale.dtype],*values)
            self._indices[name]=index
        lib=runtime.library
        version=lib.oa_original_projections_version;version.argtypes=[];version.restype=C.c_uint32
        if version()!=1:raise ValueError('unsupported original dense projection ABI')
        # Require the complete additive execution ABI before opening shard
        # handles, including the explicitly selected dot reduction boundary.
        self._cancel_type=C.CFUNCTYPE(C.c_int,C.c_void_p)
        self._operation=lib.oa_original_projection_evaluate_policy
        self._operation.argtypes=[C.c_void_p,C.c_uint32,C.c_void_p,C.c_uint64,C.c_void_p,C.c_uint64,
                                  self._cancel_type,C.c_void_p,C.c_uint32]
        self._operation.restype=C.c_int
        layout=lib.oa_original_experts_layout
        layout.argtypes=[C.c_uint32,C.POINTER(C.c_uint64),C.c_size_t];layout.restype=C.c_size_t
        for kind,record_type in ((1,OriginalFile),(2,OriginalMatrix)):
            expected=[C.sizeof(record_type),len(record_type._fields_)]+[getattr(record_type,n).offset for n,_ in record_type._fields_]
            actual=(C.c_uint64*len(expected))()
            if layout(kind,actual,len(actual))!=len(expected) or list(actual)!=expected:
                raise ValueError('native dense descriptor ABI layout differs')
        native_budget=max_bytes-staging-sum(len(path)+1 for path in paths)
        if native_budget<=0:raise MemoryError('native dense path staging exceeds admission')
        create=lib.oa_original_projections_create_rows if rows_only else lib.oa_original_projections_create
        create.argtypes=[C.POINTER(OriginalFile),C.c_uint32,C.POINTER(OriginalMatrix),C.c_uint32,
                         C.c_uint64,C.POINTER(C.c_void_p),C.c_void_p,C.c_size_t];create.restype=C.c_int
        handle=C.c_void_p();error=C.create_string_buffer(4096)
        if create(files,len(files),entries,len(entries),native_budget,C.byref(handle),error,len(error)):
            raise RuntimeError(error.value.decode(errors='replace'))
        self._handle=handle
        try:
            self.identity=hashlib.sha256(repr((self.policy,reduction,rows_only,snapshot.config_sha256,
                tuple(sorted(snapshot.files.items())),tuple(sorted(bindings.items())),max_bytes)).encode()).hexdigest()
            snapshot.assert_unchanged()
        except BaseException as failure:
            try:self.close()
            except BaseException as cleanup:
                failure.cleanup=self.close;failure.add_note('native dense cleanup retained: '+str(cleanup))
            raise

    @property
    def handle(self):
        if self._closed or self._handle is None:raise RuntimeError('native dense owner is closed')
        return self._handle

    def shape(self,name):return self.sources[name].shape

    def linear(self,name,values,*,max_work_bytes,cancelled=None):
        """Project complete input axes in C, returning only committed rows.

        Native file/scratch ownership and caller input/output staging have
        separate admissions. Cancellation callbacks contain no model math.
        """
        with self._lock:
            if self.rows_only:raise ValueError('row-only checkpoint owner has no dense projection output buffer')
            handle=self.handle;rows,columns=self.shape(name)
            source=np.asarray(values)
            if (source.dtype!=np.float32 or source.ndim<1 or source.shape[-1]!=columns
                    or not source.size or type(max_work_bytes) is not int or max_work_bytes<=0
                    or (cancelled is not None and not callable(cancelled))):
                raise ValueError('bounded matching original native projection input required')
            count=source.size//columns
            if source.nbytes*2+count*rows*8+4096>max_work_bytes:
                raise MemoryError('complete native projection input/output staging exceeds admission')
            if not np.isfinite(source).all():raise ValueError('finite original native input required')
            incoming=np.array(source,dtype=np.float32,order='C',copy=True).reshape(count,columns)
            result=np.empty((count,rows),np.float32);errors=[]
            def cancel(_):
                try:return int(cancelled is not None and bool(cancelled()))
                except BaseException as failure:
                    errors.append(failure);return 1
            callback=self._cancel_type(cancel)
            operation=self._operation
            for index in range(count):
                if operation(handle,self._indices[name],incoming[index].ctypes.data,columns,
                             result[index].ctypes.data,rows,callback,None,int(self.reduction=='fp32')):
                    if errors:raise errors[0]
                    if cancelled is not None and cancelled():raise InterruptedError('original native projection cancelled')
                    raise RuntimeError('original native projection rejected source, arithmetic or ownership')
            if errors:raise errors[0]
            if cancelled is not None and cancelled():raise InterruptedError('original native projection cancelled before publication')
            return result.reshape((*source.shape[:-1],rows))

    def handles(self,name):return name in self.sources

    def close(self):
        with self._lock:
            if self._handle is not None:
                release=self._runtime.library.oa_original_projections_release
                release.argtypes=[C.c_void_p];release.restype=C.c_int
                if release(self._handle):raise RuntimeError('native dense release remains pending')
                self._handle=None
            self._closed=True


__all__=['NativeOriginalProjections']
