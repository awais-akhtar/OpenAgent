"""Owned, explicit CPU media arithmetic library; no framework execution."""
import ctypes as C
import _ctypes
import os
import threading
import weakref
import numpy as np
from .native_library_identity import NativeLibraryIdentity

class Tensor(C.Structure):
    _fields_ = [('data',C.POINTER(C.c_float)),('count',C.c_uint64)]

def tensor(value):
    if value is None:
        return Tensor()
    if value.dtype != np.float32 or not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError('owned aligned contiguous FP32 media tensor required')
    return Tensor(value.ctypes.data_as(C.POINTER(C.c_float)),value.size)

class NativeMediaOps:
    def __init__(self,library):
        self._source=NativeLibraryIdentity.read(library)
        self.identity='openagent-native-media-arithmetic-v1:'+self._source.sha256
        self.library=C.CDLL(str(self._source.path))
        self._release=weakref.finalize(self,_ctypes.FreeLibrary if os.name=='nt' else _ctypes.dlclose,self.library._handle)
        self._lock=threading.RLock();self._closing=False
        try:
            self._source.assert_unchanged()
            self.library.oa_media_ops_version.argtypes=[]
            self.library.oa_media_ops_version.restype=C.c_uint64
            if self.library.oa_media_ops_version()!=1:
                raise ValueError('unsupported media operations ABI')
            self.library.oa_media_linear.argtypes=[C.c_uint64]*3+[Tensor]*4
            self.library.oa_media_norm.argtypes=[C.c_uint64]*2+[C.c_double,C.c_int]+[Tensor]*4
            self.library.oa_media_activation.argtypes=[C.c_int,Tensor,Tensor]
            self.library.oa_media_audio_embed.argtypes=[C.c_uint64]*4+[C.POINTER(C.c_int64),Tensor,Tensor]
            self.library.oa_media_vision_attention.argtypes=[C.c_uint64]*3+[Tensor,C.POINTER(C.c_int64),
                C.POINTER(C.c_uint64),C.c_uint64,Tensor]
            for name in ('linear','norm','activation','vision_attention','audio_embed'):
                getattr(self.library,'oa_media_'+name).restype=C.c_int
            self._source.assert_unchanged()
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup:
                error.cleanup=self.close
                error.add_note('native media library cleanup remains pending: '+repr(cleanup))
            raise

    def _call(self,name,*arguments):
        if self.library is None or self._closing:
            raise RuntimeError('media library is closed')
        self._source.assert_unchanged()
        status=getattr(self.library,'oa_media_'+name)(*arguments)
        if status:
            raise RuntimeError(f'native media {name} rejected arithmetic ({status})')

    def linear(self,x,weight,bias=None):
        with self._lock:
            if x.ndim<2 or weight.ndim!=2 or x.shape[-1]!=weight.shape[1]:
                raise ValueError('media projection geometry differs')
            output=np.empty((*x.shape[:-1],weight.shape[0]),np.float32)
            self._call('linear',x.size//x.shape[-1],x.shape[-1],weight.shape[0],tensor(x),tensor(weight),tensor(bias),tensor(output))
            return output

    def norm(self,x,weight,bias=None,*,rms=False,epsilon=1e-6):
        with self._lock:
            output=np.empty_like(x)
            self._call('norm',x.size//x.shape[-1],x.shape[-1],epsilon,int(rms),tensor(x),tensor(weight),tensor(bias),tensor(output))
            return output

    def activation(self,x,*,kind='gelu'):
        with self._lock:
            output=np.empty_like(x)
            self._call('activation',{'gelu':0,'gelu_pytorch_tanh':1,'silu':2}[kind],tensor(x),tensor(output))
            return output

    def clamped_swiglu(self,gate,up,*,limit):
        with self._lock:
            if self.library is None or self._closing: raise RuntimeError('media library is closed')
            self._source.assert_unchanged()
            self.library.oa_media_swiglu_version.argtypes=[]
            self.library.oa_media_swiglu_version.restype=C.c_uint64
            if self.library.oa_media_swiglu_version()!=1: raise ValueError('unsupported clamped SwiGLU ABI')
            function=self.library.oa_media_swiglu_clamped
            function.argtypes=[C.c_float,Tensor,Tensor,Tensor];function.restype=C.c_int
            if gate.shape!=up.shape: raise ValueError('SwiGLU gate/up geometry differs')
            output=np.empty_like(gate)
            self._call('swiglu_clamped',limit,tensor(gate),tensor(up),tensor(output))
            return output

    def vision_attention(self,qkv,positions,offsets,*,heads):
        with self._lock:
            if qkv.ndim!=2 or type(heads) is not int or heads<=0 or qkv.shape[1]%(3*heads):
                raise ValueError('invalid vision QKV geometry')
            positions=np.asarray(positions)
            offsets=np.asarray(offsets)
            if positions.shape!=(qkv.shape[0],2) or positions.dtype!=np.int64 or offsets.dtype!=np.uint64 or offsets.ndim!=1 or offsets.size<2:
                raise ValueError('owned vision position and frame offset geometry required')
            positions=np.ascontiguousarray(positions)
            offsets=np.ascontiguousarray(offsets)
            output=np.empty((qkv.shape[0],qkv.shape[1]//3),np.float32)
            self._call('vision_attention',qkv.shape[0],heads,qkv.shape[1]//(3*heads),tensor(qkv),
                positions.ctypes.data_as(C.POINTER(C.c_int64)),offsets.ctypes.data_as(C.POINTER(C.c_uint64)),
                offsets.size-1,tensor(output))
            return output

    def audio_embed(self,ids,table,*,vocabulary):
        with self._lock:
            ids=np.asarray(ids)
            if ids.ndim!=2 or ids.dtype!=np.int64 or table.ndim!=2 or table.shape[0]!=ids.shape[1]*vocabulary:
                raise ValueError('audio codebooks and embedding table differ')
            ids=np.ascontiguousarray(ids)
            output=np.empty((ids.shape[0],table.shape[1]),np.float32)
            self._call('audio_embed',ids.shape[0],ids.shape[1],vocabulary,table.shape[1],
                ids.ctypes.data_as(C.POINTER(C.c_int64)),tensor(table),tensor(output))
            return output

    def close(self):
        with self._lock:
            if self.library is not None:
                self._closing=True
                if self._release.alive:
                    info=self._release.detach()
                    try:info[1](*info[2],**info[3])
                    except BaseException:
                        self._release=weakref.finalize(self,info[1],*info[2],**info[3])
                        raise
                self.library=None

__all__=['NativeMediaOps']
