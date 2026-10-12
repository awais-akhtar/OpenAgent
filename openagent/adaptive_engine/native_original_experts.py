"""Native original-checkpoint expert files with no Python evaluation callback."""
import ctypes as C
import hashlib
import math
import threading

from .glm5_publisher_experts import Glm5PublisherExperts
from .inkling_publisher_experts import InklingPublisherExperts
from .qwen4_publisher_experts import Qwen4PublisherExperts


class OriginalFile(C.Structure):
    _fields_=[('path',C.c_char_p),('size',C.c_uint64),('mtime_ns',C.c_int64)]


class OriginalMatrix(C.Structure):
    _fields_=[(name,C.c_uint32) for name in ('file','dtype','scale_file','scale_dtype')]+[
        (name,C.c_uint64) for name in ('offset','elements','element_offset','rows','columns','parent_rows',
                                      'scale_offset','scale_elements','block_rows','block_columns')]


class OriginalExpert(C.Structure):
    _fields_=[('layer',C.c_uint32),('expert',C.c_uint32),('gate',OriginalMatrix),
              ('up',OriginalMatrix),('down',OriginalMatrix)]


class NativeOriginalExperts:
    """Borrow a loaded Segment runtime; own native files and streamed scratch.

    Tensor mappings come from the original publisher provider's admitted slices.
    The provider is needed only during registration; native evaluation owns its
    own file descriptors and never invokes the provider's Python evaluate method.
    """
    def __init__(self,runtime,provider,*,max_bytes):
        if type(provider) not in (Glm5PublisherExperts,InklingPublisherExperts,Qwen4PublisherExperts):
            raise TypeError('exact original publisher expert inventory required')
        if type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('positive native original expert admission required')
        self._lock=threading.RLock();self._runtime=runtime;self._handle=None
        self.max_bytes=max_bytes;self._closed=False
        self.provider_identity=provider.identity
        self.family=('glm5_next_text' if type(provider) is Glm5PublisherExperts else
                     'qwen4_exp_text' if type(provider) is Qwen4PublisherExperts else 'inkling_text')
        snapshot=provider.reader.snapshot;snapshot.assert_unchanged()
        if not 0<len(provider.bindings)<=1048576 or 2*len(provider.bindings)*C.sizeof(OriginalExpert)>max_bytes:
            raise MemoryError('original expert descriptor inventory exceeds aggregate admission')
        dtype={'F32':1,'F16':2,'BF16':3,'F8_E4M3':4,'F8_E8M0':5}
        matrices={};names=set()
        for pair,slices in sorted(provider.bindings.items()):
            if type(pair) is not tuple or len(pair)!=2 or any(type(v) is not int or v<0 or v>2147483647 for v in pair):
                raise ValueError('bounded original expert layer and index required')
            for binding in slices:
                record=snapshot.tensors[binding.source]
                if len(binding.shape)!=2 or len(record.shape) not in (2,3) or record.dtype not in dtype or dtype[record.dtype]==5:
                    raise ValueError('native original expert matrix encoding differs')
                if binding.shape[1]!=record.shape[-1]:raise ValueError('native original expert source columns differ')
                scale=None
                if record.dtype=='F8_E4M3':
                    blocks=provider.reader.block_shape
                    if blocks is None:raise ValueError('original FP8 expert block geometry required')
                    scale_name=provider.reader.scale_names.get(binding.source,
                        binding.source[:-7]+'.weight_scale_inv' if binding.source.endswith('.weight') else binding.source+'_scale_inv')
                    scale=snapshot.tensors[scale_name]
                    shape=record.shape[:-2]+tuple((v+b-1)//b for v,b in zip(record.shape[-2:],blocks))
                    if scale.shape!=shape or scale.dtype not in ('F32','F16','BF16','F8_E8M0'):
                        raise ValueError('native original FP8 expert scale geometry differs')
                    names.add(scale.shard)
                names.add(record.shard);matrices[binding]=(record,scale)
        shards=tuple(sorted(names));indices={name:index for index,name in enumerate(shards)}
        if not 0<len(shards)<=4096:raise ValueError('bounded original expert shard inventory required')
        descriptor_bytes=2*len(provider.bindings)*C.sizeof(OriginalExpert)+len(shards)*C.sizeof(OriginalFile)
        if descriptor_bytes>=max_bytes:raise MemoryError('original native expert descriptor copies exceed admission')
        files=(OriginalFile*len(shards))()
        paths=[]
        for index,name in enumerate(shards):
            path=(snapshot.root/name).resolve(strict=True)
            if not path.is_relative_to(snapshot.root):raise ValueError('original expert shard escapes checkpoint root')
            paths.append(str(path).encode('utf-8'));stamp=snapshot.files[name]
            if not 0<=stamp[2]<2**64 or not -2**63<=stamp[3]<2**63:
                raise ValueError('original expert file guards exceed native bounds')
            files[index]=OriginalFile(paths[-1],stamp[2],stamp[3])
        def matrix(binding):
            record,scale=matrices[binding];blocks=provider.reader.block_shape or (0,0)
            return OriginalMatrix(indices[record.shard],dtype[record.dtype],0 if scale is None else indices[scale.shard],
                0 if scale is None else dtype[scale.dtype],record.base+record.offsets[0],math.prod(record.shape),
                binding.element_offset,*binding.shape,record.shape[-2],
                0 if scale is None else scale.base+scale.offsets[0],0 if scale is None else math.prod(scale.shape),
                0 if scale is None else blocks[0],0 if scale is None else blocks[1])
        entries=(OriginalExpert*len(provider.bindings))()
        for index,((layer,expert),slices) in enumerate(sorted(provider.bindings.items())):
            if len(slices)!=3:raise ValueError('exact native gate/up/down expert slices required')
            entries[index]=OriginalExpert(layer,expert,*(matrix(binding) for binding in slices))
        if type(provider) is Glm5PublisherExperts:
            hidden=provider.contract.hidden_size;width=provider.contract.moe_intermediate_size
            limit=provider.contract.swiglu_limit
        else:
            hidden=provider.config['hidden_size'];width=provider.config['moe_intermediate_size']
            limit=float.fromhex('0x1.fffffep+127')
        self.hidden_size=hidden;self.intermediate_size=width
        self.bindings=frozenset(provider.bindings)
        self.identity=hashlib.sha256(repr((provider.identity,max_bytes,
            'native-original-row-fp64-projection-fp32-swiglu-v1')).encode()).hexdigest()
        library=runtime.library
        version=library.oa_original_experts_version;version.argtypes=[];version.restype=C.c_uint32
        if version()!=1:raise ValueError('unsupported native original expert ABI')
        layout=library.oa_original_experts_layout
        layout.argtypes=[C.c_uint32,C.POINTER(C.c_uint64),C.c_size_t];layout.restype=C.c_size_t
        for kind,record_type in ((1,OriginalFile),(2,OriginalMatrix),(3,OriginalExpert)):
            expected=[C.sizeof(record_type),len(record_type._fields_)]+[getattr(record_type,name).offset for name,_ in record_type._fields_]
            observed=(C.c_uint64*len(expected))()
            if layout(kind,observed,len(observed))!=len(expected) or list(observed)!=expected:
                raise ValueError('native original expert descriptor layout differs')
        create=library.oa_original_experts_create
        create.argtypes=[C.POINTER(OriginalFile),C.c_uint32,C.POINTER(OriginalExpert),C.c_uint32,
                        C.c_uint64,C.c_uint64,C.c_float,C.c_uint64,C.POINTER(C.c_void_p),C.c_void_p,C.c_size_t]
        create.restype=C.c_int;handle=C.c_void_p();error=C.create_string_buffer(4096)
        staging=C.sizeof(files)+C.sizeof(entries)+sum(len(path)+1 for path in paths)
        native_budget=max_bytes-staging
        if native_budget<=0:raise MemoryError('original native expert descriptor staging exceeds aggregate admission')
        if create(files,len(files),entries,len(entries),hidden,width,limit,native_budget,C.byref(handle),error,len(error)):
            raise RuntimeError(error.value.decode('utf-8',errors='replace'))
        self._handle=handle
        try:snapshot.assert_unchanged()
        except BaseException as failure:
            try:self.close()
            except BaseException as cleanup:
                failure.cleanup=self.close;failure.add_note('native original expert cleanup retained: '+str(cleanup))
            raise

    @property
    def handle(self):
        if self._closed or self._handle is None:raise RuntimeError('native original expert owner closed')
        return self._handle

    def close(self):
        with self._lock:
            if self._closed:return
            if self._handle is not None:
                release=self._runtime.library.oa_original_experts_release
                release.argtypes=[C.c_void_p];release.restype=C.c_int
                if release(self._handle):raise RuntimeError('native original expert callback remains active')
                self._handle=None
            self._closed=True


__all__=['NativeOriginalExperts']
