"""Original K3 dense checkpoint rows with bounded BF16 device projection."""
from contextlib import contextmanager
import hashlib
import threading
from types import MappingProxyType
import numpy as np
from ..publisher_tensor_reader import PublisherTensorReader,TensorSlice


class K3PublisherProjections:
    dtype_name='bfloat16'
    boundary_observation_supported=True
    policy='k3-original-dense-output-row-stream-bf16-v1'

    def __init__(self,checkpoint,*,device,torch_version,max_bytes,include_vision=False):
        import torch
        from ..publisher_checkpoint.service import PublisherCheckpoint
        if (type(checkpoint) is not PublisherCheckpoint or checkpoint.report.family!='kimi_k3'
                or str(torch.__version__)!=torch_version or type(max_bytes) is not int or max_bytes<=0
                or type(include_vision) is not bool):
            raise ValueError('exact K3 checkpoint, Torch pin and dense-row admission required')
        from .device_policy import admit_device
        target=admit_device(torch,device,allow_cpu=True)
        allowed={'q_proj','k_proj','v_proj','q_a_proj','q_b_proj','kv_a_proj_with_mqa','kv_b_proj',
                 'o_proj','f_a_proj','f_b_proj','b_proj','g_proj','gate_proj','up_proj','down_proj',
                 'w1','w2','w3','lm_head','embed_tokens','routed_expert_down_proj',
                 'routed_expert_up_proj','wqkv','wo','fc0','fc1'}
        bindings={}
        for rule in checkpoint.report.contracts:
            if rule.encoding!='float32' or len(rule.shape)!=2 or len(rule.sources)!=1:continue
            name=rule.sources[0]
            vision=rule.target.startswith('k3.vision.')
            if vision and not include_vision:continue
            if (not name.endswith('.weight') or
                    (name.split('.')[-2] not in allowed and name not in
                     ('mm_projector.proj.0.weight','mm_projector.proj.2.weight'))):continue
            if '.experts.' in name:continue
            source=checkpoint._metadata.tensors.get(name)
            if source is None or source.shape!=rule.shape or source.dtype not in ('BF16','F16','F32'):
                raise ValueError('original dense K3 source layout differs: '+name)
            if name in bindings:raise ValueError('duplicate original dense K3 projection')
            bindings[name]=TensorSlice(name,rule.shape)
        if not bindings:raise ValueError('nonempty original K3 dense projection inventory required')
        self.sources=MappingProxyType(bindings);self.device=target;self.max_bytes=max_bytes
        self.reader=PublisherTensorReader(checkpoint._metadata,max_work_bytes=max_bytes)
        self._lock=threading.RLock();self._local=threading.local();self._closed=False;self._busy=False
        self._arithmetic=None
        snapshot=checkpoint._metadata
        self.identity=hashlib.sha256(repr((self.policy,snapshot.config_sha256,
            tuple(sorted(snapshot.files.items())),tuple(sorted(bindings.items())),
            str(target),torch_version,max_bytes)).encode()).hexdigest()

    def bind(self,arithmetic):
        from .bf16_arithmetic import K3BF16Arithmetic
        with self._lock:
            if (self._closed or self._arithmetic is not None or type(arithmetic) is not K3BF16Arithmetic
                    or arithmetic._projection_backend is not self or arithmetic.device!=self.device):
                raise ValueError('one matching arithmetic owner must bind the original K3 projection bank')
            self._arithmetic=arithmetic

    def handles(self,name):return name in self.sources
    def shape(self,name):return self.sources[name].shape

    def _check(self):
        if self._closed:raise RuntimeError('original K3 projection bank is closed')
        event=getattr(self._local,'event',None)
        if event is not None and event.is_set():
            from ..execution_service import ExecutionCancelled
            raise ExecutionCancelled('original K3 dense projection cancelled')

    @contextmanager
    def cancellation_scope(self,event):
        if event is not None and not callable(getattr(event,'is_set',None)):
            raise ValueError('dense projection cancellation must provide is_set')
        previous=getattr(self._local,'event',None);self._local.event=event
        try:yield
        finally:self._local.event=previous

    def linear(self,name,values):
        with self._lock:
            self._check()
            if self._busy or self._arithmetic is None:raise RuntimeError('bound idle K3 projection owner required')
            source=self.sources[name];rows,columns=source.shape
            incoming=np.asarray(values)
            if (incoming.dtype!=np.float32 or incoming.ndim not in (1,2)
                    or incoming.shape[-1]!=columns or not incoming.size):
                raise ValueError('matching finite K3 dense input required')
            batch=1 if incoming.ndim==1 else incoming.shape[0]
            fixed=incoming.size*8+batch*rows*8+4096
            tile_rows=min(rows,(self.max_bytes-fixed)//(columns*128+batch*8))
            if tile_rows<=0:raise MemoryError('one original K3 row and complete output exceed dense admission')
            if not np.isfinite(incoming).all():raise ValueError('nonfinite original K3 projection input')
            result=np.empty((rows,) if incoming.ndim==1 else (batch,rows),np.float32)
            self.reader.snapshot.assert_unchanged();self._busy=True
            try:
                for begin in range(0,rows,tile_rows):
                    self._check();end=min(rows,begin+tile_rows)
                    weight=self.reader.read(TensorSlice(source.source,(end-begin,columns),begin*columns))
                    projected=self._arithmetic.matrix_batch(weight,incoming[None] if incoming.ndim==1 else incoming,
                        check_cancelled=self._check)
                    result[...,begin:end]=projected[0] if incoming.ndim==1 else projected
                    del weight,projected
                self.reader.snapshot.assert_unchanged();self._check();return result
            finally:self._busy=False

    def gather_rows(self,name,indices):
        """Read token rows directly; never expand the vocabulary table."""
        with self._lock:
            self._check()
            if self._busy or self._arithmetic is None:raise RuntimeError('bound idle K3 row owner required')
            if name!='language_model.model.embed_tokens.weight':
                raise ValueError('only original K3 token embedding rows may be gathered')
            source=self.sources[name];rows,columns=source.shape
            ids=np.asarray(indices)
            if (ids.ndim!=1 or not ids.size or ids.dtype.kind not in 'iu'
                    or np.any(ids<0) or np.any(ids>=rows)):
                raise ValueError('occupied one-dimensional vocabulary row IDs required')
            if ids.size*(columns*8+16)+columns*128+4096>self.max_bytes:
                raise MemoryError('original embedding rows exceed complete staging admission')
            output=np.empty((ids.size,columns),np.float32)
            self.reader.snapshot.assert_unchanged();self._busy=True
            try:
                for index,token in enumerate(ids):
                    self._check()
                    value=self.reader.read(TensorSlice(source.source,(1,columns),int(token)*columns))
                    output[index]=value[0]
                self.reader.snapshot.assert_unchanged();self._check();return output
            finally:self._busy=False

    def close(self):
        with self._lock:
            if self._busy:raise RuntimeError('original K3 projection owns execution resources')
            if self._closed:return
            self.reader.close();self._arithmetic=None;self._closed=True


__all__=['K3PublisherProjections']
