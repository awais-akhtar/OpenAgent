"""Qwen4 native Segment wire: residual channels and exact lexical/position lanes.

This is a counted FP32 transport contract. Side lanes encode only bounded
integers representable exactly in FP32; callers cannot round token IDs or
positions into admission. Native composition exports consume the same layout.
"""
import ctypes as C
from dataclasses import dataclass
import numpy as np
from .native_qwen35 import Tensor,tensor
from .native_qwen4_composition import NativeQwen4Composition,ComposeConfig


@dataclass(frozen=True)
class Qwen4SegmentWire:
    hidden:int
    streams:int
    vocabulary:int
    context:int
    mrope:bool
    schema='openagent.qwen4.hyper-token-mask-mrope.f32.v1'

    def __post_init__(self):
        if (type(self.hidden) is not int or not 0<self.hidden<=4096
                or type(self.streams) is not int or not 2<=self.streams<=64
                or type(self.vocabulary) is not int or not 0<self.vocabulary<=1048576
                or type(self.context) is not int or not 0<self.context<=65536 or type(self.mrope) is not bool):
            raise ValueError('bounded exact native Qwen4 wire geometry required')

    @property
    def activation_width(self):return self.hidden*self.streams
    @property
    def width(self):return self.activation_width+5

    def admit(self,value,*,max_bytes):
        if (type(max_bytes) is not int or max_bytes<=0 or type(value) is not np.ndarray
                or value.dtype!=np.float32 or value.ndim!=2 or value.shape[1]!=self.width
                or not 0<value.shape[0]<=4096 or not value.flags.c_contiguous):
            raise ValueError('occupied contiguous complete Qwen4 wire rows required')
        if value.nbytes*3>max_bytes:raise MemoryError('complete wire staging exceeds admission')
        if not np.isfinite(value).all():raise ValueError('nonfinite activation or side lane')
        side=value[:,self.activation_width:]
        if (np.any(side!=np.floor(side)) or np.any(side[:,0]<0) or np.any(side[:,0]>=self.vocabulary)
                or np.any((side[:,1]!=0)&(side[:,1]!=1)) or np.any(side[:,2:]<0)
                or np.any(side[:,2:]>=self.context)):
            raise ValueError('exact bounded token, mask and three-axis position lanes required')
        if not self.mrope and not np.array_equal(side[:,2:],np.repeat(side[:,2:3],3,axis=1)):
            raise ValueError('text RoPE wire requires identical position axes')

    def pack(self,hidden,token_ids,mask,positions,*,max_bytes):
        hidden=np.asarray(hidden);ids=np.asarray(token_ids);mask=np.asarray(mask);positions=np.asarray(positions)
        rows=hidden.shape[0] if hidden.ndim==2 else 0
        if (hidden.dtype!=np.float32 or hidden.shape!=(rows,self.activation_width) or not 0<rows<=4096
                or ids.shape!=(rows,) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=self.vocabulary)
                or mask.shape!=(rows,) or mask.dtype.kind not in 'biu' or not np.isin(mask,[0,1]).all()):
            raise ValueError('original activation, lexical and occupied-mask inventories required')
        if positions.shape==(rows,) and not self.mrope:positions=np.broadcast_to(positions,(3,rows))
        if (positions.shape!=(3,rows) or positions.dtype.kind not in 'iu'
                or np.any(positions<0) or np.any(positions>=self.context)):
            raise ValueError('three explicit native position axes required')
        if type(max_bytes) is not int or max_bytes<=0 or rows*self.width*12>max_bytes:
            raise MemoryError('wire construction exceeds complete staging admission')
        wire=np.empty((rows,self.width),np.float32)
        wire[:,:self.activation_width]=hidden
        wire[:,self.activation_width]=ids;wire[:,self.activation_width+1]=mask
        wire[:,self.activation_width+2:]=positions.T
        self.admit(wire,max_bytes=max_bytes)
        return np.frombuffer(wire.tobytes(),np.float32).reshape(wire.shape)


class NativeQwen4WireEdge:
    """Borrow one admitted decoder for the native final mixer/head operation."""
    def __init__(self,decoder):
        self.decoder=decoder;self.composition=NativeQwen4Composition(decoder.library)
        c=decoder.config
        if decoder.range_mode:raise ValueError('complete native Edge weight owner required')
        self.contract=Qwen4SegmentWire(c.hidden_size,c.hc_count,c.vocab_size,c.max_position_embeddings,
            decoder.mrope_section is not None)
        version=decoder.library.oa_qwen4_compose_wire_version;version.argtypes=[];version.restype=C.c_uint64
        if version()!=1:raise ValueError('unsupported native Qwen4 wire ABI')
        self.operation=decoder.library.oa_qwen4_compose_head_wire
        self.operation.argtypes=[C.POINTER(ComposeConfig),C.c_uint64,Tensor,Tensor,C.c_void_p,C.c_void_p]
        self.operation.restype=C.c_int

    def logits(self,wire,*,max_work_bytes,cancelled=None):
        decoder=self.decoder
        with decoder._lock:
            decoder._check_open();self.contract.admit(wire,max_bytes=max_work_bytes)
            if cancelled is not None and not callable(cancelled):raise ValueError('callable cancellation required')
            required=wire.nbytes*2+wire.shape[0]*self.contract.vocabulary*4
            available=decoder.max_bytes-decoder.weight_bytes-decoder._snapshot_bytes-decoder._state_bytes()
            available-=decoder._expert_bytes+decoder._ple_bytes+required
            if required>=max_work_bytes or available<=0:raise MemoryError('native Edge aggregate admission exhausted')
            config=self.composition.configure(decoder,max_work_bytes=min(max_work_bytes-required,available))
            wire=np.array(wire,dtype=np.float32,order='C',copy=True)
            output=np.empty((wire.shape[0],self.contract.vocabulary),np.float32)
            failures=[]
            def check(user):
                try:return int(cancelled is not None and cancelled())
                except BaseException as error:failures.append(error);return 1
            callback=C.CFUNCTYPE(C.c_int,C.c_void_p)(check)
            decoder._executing=True
            try:status=self.operation(C.byref(config),len(wire),tensor(wire),tensor(output),C.cast(callback,C.c_void_p),None)
            finally:decoder._executing=False
            if failures:raise InterruptedError('native Edge cancellation failed') from failures[0]
            if status==4:raise InterruptedError('native Edge cancelled')
            decoder._result(status)
            if not np.isfinite(output).all():raise ValueError('nonfinite native Edge logits')
            return output


__all__=['Qwen4SegmentWire','NativeQwen4WireEdge']
