"""Selected Inkling routed experts from immutable original publisher storage."""
import hashlib
from pathlib import Path
import threading
import numpy as np
from .independent_inkling import configuration
from .publisher_weight_mapping import expert_bindings
from .publisher_tensor_reader import PublisherTensorReader,TensorSlice
from .native_media_ops import NativeMediaOps

class InklingPublisherExperts:
    @staticmethod
    def _configuration(config): return configuration(config)[0]

    def __init__(self,snapshot,config,*,library,max_bytes,block_shape=None,prefix='model.language_model.'):
        self.config=self._configuration(config);c=self.config
        h,m,e=c['hidden_size'],c['moe_intermediate_size'],c['n_routed_experts']
        if type(max_bytes) is not int or max_bytes<=max(64*(h+4*m),512*max(h,m)):
            raise MemoryError('one original Inkling expert row transaction exceeds admission')
        self.max_bytes=max_bytes;self.reader=PublisherTensorReader(snapshot,max_work_bytes=max_bytes//2,block_shape=block_shape)
        self._lock=threading.RLock();self._closed=False;self._busy=False;self.bindings={};self.source_names=set()
        self.ops=None;digest=hashlib.sha256(repr((snapshot.config_sha256,c,block_shape)).encode())
        digest.update(repr(tuple(sorted(snapshot.files.items()))).encode())
        for layer,kind in enumerate(c['mlp_layer_types']):
            if kind!='sparse': continue
            bindings=expert_bindings(snapshot,prefix=f'{prefix}layers.{layer}.mlp.experts.',
                expert_count=e,hidden_size=h,intermediate_size=m)
            gate,down=bindings
            self.source_names.update(source.source for binding in bindings for source in binding.sources)
            for expert in range(e):
                if gate.operation=='identity':
                    source=gate.sources[0];g=TensorSlice(source.source,(m,h),source.element_offset+expert*2*m*h)
                    u=TensorSlice(source.source,(m,h),source.element_offset+(expert*2+1)*m*h)
                else: g,u=gate.sources[expert*2:expert*2+2]
                if down.operation=='identity':
                    source=down.sources[0];d=TensorSlice(source.source,(h,m),source.element_offset+expert*h*m)
                else: d=down.sources[expert]
                self.bindings[layer,expert]=(g,u,d);digest.update(repr((g,u,d)).encode())
        if not self.bindings: raise ValueError('Inkling selected expert streaming requires sparse layers')
        for name in sorted(self.source_names): digest.update(repr(snapshot.tensors[name]).encode())
        path=Path(library).resolve(strict=True);before=path.stat()
        with path.open('rb') as stream:
            for value in iter(lambda:stream.read(1024**2),b''): digest.update(value)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('Inkling expert arithmetic library changed during admission')
        snapshot.assert_unchanged();self.identity='openagent-inkling-publisher-experts-v1:'+digest.hexdigest()
        self.ops=NativeMediaOps(library)

    def evaluate(self,layer,expert,inputs,*,cancelled=None):
        with self._lock:
            if self._closed or self._busy: raise RuntimeError('Inkling expert owner is closed or executing')
            if (layer,expert) not in self.bindings: raise ValueError('Inkling expert outside admitted source inventory')
            source=np.asarray(inputs);h=self.config['hidden_size'];m=self.config['moe_intermediate_size']
            if source.shape!=(h,) or source.dtype!=np.float32 or not np.isfinite(source).all():
                raise ValueError('one finite FP32 Inkling routed expert input required')
            self._busy=True
            try:
                self.reader.snapshot.assert_unchanged();x=np.array(source,order='C',copy=True).reshape(1,h)
                block=max(1,min(128,self.reader.max_bytes//(256*max(h,m))))
                gate,up,down=self.bindings[layer,expert]
                g=self.reader.linear(self.ops,gate,x,row_block=block,cancelled=cancelled)
                u=self.reader.linear(self.ops,up,x,row_block=block,cancelled=cancelled)
                z=self.ops.clamped_swiglu(g,u,limit=float(np.finfo(np.float32).max))
                return self.reader.linear(self.ops,down,z,row_block=block,cancelled=cancelled)[0]
            finally: self._busy=False

    def close(self):
        with self._lock:
            if self._busy: raise RuntimeError('Inkling expert callback owns resources')
            if self._closed: return
            if self.ops is not None: self.ops.close();self.ops=None
            self.reader.close();self._closed=True

__all__=['InklingPublisherExperts']
