"""Selected GLM5-next experts over original BF16/FP8 checkpoint rows."""
from pathlib import Path
import hashlib
import threading
import numpy as np
from .independent_glm5_next import Glm5NextTextContract
from .publisher_tensor_reader import PublisherTensorReader,TensorSlice
from .native_media_ops import NativeMediaOps

class Glm5PublisherExperts:
    def __init__(self,snapshot,config,*,library,max_bytes,block_shape=None,prefix='model.language_model.'):
        self.contract=Glm5NextTextContract.parse(config);c=self.contract
        if type(max_bytes) is not int or max_bytes<=32*(c.hidden_size+4*c.moe_intermediate_size):
            raise MemoryError('GLM selected expert staging exceeds admission')
        self.max_bytes=max_bytes;self._lock=threading.RLock();self._closed=False;self._busy=False
        self.reader=PublisherTensorReader(snapshot,max_work_bytes=max_bytes//2,block_shape=block_shape)
        self.bindings={};digest=hashlib.sha256(repr((snapshot.config_sha256,c,block_shape)).encode())
        digest.update(repr(tuple(sorted(snapshot.files.items()))).encode())
        for layer,kind in enumerate(c.mlp_layer_types):
            if kind!='sparse': continue
            p=f'{prefix}layers.{layer}.mlp.experts.';h,m,e=c.hidden_size,c.moe_intermediate_size,c.n_routed_experts
            fused=p+'gate_up_proj';down=p+'down_proj'
            for expert in range(e):
                if fused in snapshot.tensors:
                    if snapshot.tensors[fused].shape!=(e,2*m,h) or down not in snapshot.tensors or snapshot.tensors[down].shape!=(e,h,m):
                        raise ValueError('GLM fused publisher expert geometry differs')
                    values=(TensorSlice(fused,(m,h),expert*2*m*h),TensorSlice(fused,(m,h),(expert*2+1)*m*h),
                        TensorSlice(down,(h,m),expert*h*m))
                else:
                    values=tuple(TensorSlice(p+f'{expert}.{name}_proj.weight',shape)
                        for name,shape in (('gate',(m,h)),('up',(m,h)),('down',(h,m))))
                    for value in values:
                        if value.source not in snapshot.tensors or snapshot.tensors[value.source].shape!=value.shape:
                            raise ValueError('GLM split publisher expert geometry differs')
                self.bindings[layer,expert]=values
                digest.update(repr(values).encode())
        if not self.bindings: raise ValueError('GLM checkpoint has no configured routed expert layers')
        for name in sorted({value.source for values in self.bindings.values() for value in values}):
            digest.update(repr(snapshot.tensors[name]).encode())
        with Path(library).resolve(strict=True).open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''): digest.update(block)
        snapshot.assert_unchanged()
        self.identity='openagent-glm5-publisher-experts-v1:'+digest.hexdigest()
        self.ops=NativeMediaOps(library)

    def evaluate(self,layer,expert,inputs,*,cancelled=None):
        with self._lock:
            if self._closed or self._busy: raise RuntimeError('GLM expert provider is closed or executing')
            if (layer,expert) not in self.bindings: raise ValueError('GLM expert outside admitted layer inventory')
            self._busy=True
            try:
                self.reader.snapshot.assert_unchanged()
                source=np.asarray(inputs)
                if source.shape!=(self.contract.hidden_size,) or source.dtype!=np.float32 or not np.isfinite(source).all():
                    raise ValueError('invalid GLM expert input')
                x=np.array(source,order='C',copy=True).reshape(1,-1)
                gate,up,down=self.bindings[layer,expert]
                block=max(1,min(128,self.reader.max_bytes//(256*max(self.contract.hidden_size,self.contract.moe_intermediate_size))))
                g=self.reader.linear(self.ops,gate,x,row_block=block,cancelled=cancelled)
                u=self.reader.linear(self.ops,up,x,row_block=block,cancelled=cancelled)
                z=self.ops.clamped_swiglu(g,u,limit=self.contract.swiglu_limit)
                return self.reader.linear(self.ops,down,z,row_block=block,cancelled=cancelled)[0]
            finally: self._busy=False

    def close(self):
        with self._lock:
            if self._busy: raise RuntimeError('GLM expert callback owns resources')
            if not self._closed:
                self.ops.close();self.reader.close();self._closed=True

__all__=['Glm5PublisherExperts']
