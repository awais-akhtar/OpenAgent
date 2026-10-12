"""Streamed BF16 and logical BF16 grouped publisher row projections."""
from contextlib import contextmanager
import hashlib
import json
import threading
from types import MappingProxyType

import numpy as np

from ..publisher_checkpoint.v41_adapter import _shard_rule
from ..publisher_tensor_reader import PublisherTensorReader,TensorSlice
from .execution import V41Execution


class PublisherBF16Projections:
    def __init__(self,checkpoint,*,max_bytes,row_block=128):
        if checkpoint.report.family!='deepseek_v41' or type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('explicit V4.1 BF16 projection checkpoint and admission required')
        if type(row_block) is not int or not 1<=row_block<=65536:
            raise ValueError('bounded BF16 publisher row tile required')
        self.snapshot=checkpoint._metadata;self.max_bytes=max_bytes;self.row_block=row_block
        selected=tuple(rule for rule in checkpoint._select(None) if _shard_rule(rule) and
            ((len(rule.sources)==1 and self.snapshot.tensors[rule.sources[0]].dtype=='BF16') or
             (len(rule.sources)==2 and rule.sources[0].endswith('.wo_a.weight') and
              rule.encoding=='fp8-e4m3-g32x32')))
        self.sources=MappingProxyType({rule.sources[0]:tuple(rule.shape) for rule in selected})
        self._decoded_bf16={rule.sources[0] for rule in selected if len(rule.sources)==2}
        scale_names={rule.sources[0]:rule.sources[1] for rule in selected if len(rule.sources)==2}
        if not self.sources:raise ValueError('original BF16 publisher projection inventory required')
        digest=hashlib.sha256(self.snapshot.config_sha256.encode())
        digest.update(repr((str(self.snapshot.root),tuple(sorted(self.snapshot.files.items())))).encode())
        digest.update(json.dumps(sorted(self.sources.items()),separators=(',',':')).encode())
        for name in sorted(self.sources):
            record=self.snapshot.tensors[name]
            digest.update(repr((name,record)).encode())
            if name in scale_names:digest.update(repr(self.snapshot.tensors[scale_names[name]]).encode())
        digest.update(repr((max_bytes,row_block,'native-bf16-logical-grouped-row-stream-v2')).encode())
        self.identity=digest.hexdigest()
        self.reader=PublisherTensorReader(self.snapshot,max_work_bytes=max_bytes,
            block_shape=(32,32),scale_names=scale_names)
        self._lock=threading.RLock();self._local=threading.local()
        self._arithmetic=None;self._closed=self._busy=False

    def bind_arithmetic(self,arithmetic):
        with self._lock:
            if self._closed or self._busy or self._arithmetic is not None:
                raise RuntimeError('idle unbound BF16 projection owner required')
            backend=getattr(arithmetic,'_projection_backend',None)
            contains=getattr(backend,'contains_projection_owner',None)
            if backend is not self and (not callable(contains) or not contains(self)):
                raise ValueError('BF16 projection must bind its admitted arithmetic owner')
            self._arithmetic=arithmetic

    def handles(self,name):return not self._closed and name in self.sources
    def shape(self,name):return self.sources[name]

    def contains_projection_owner(self,owner):return owner is self

    def project_rows(self,name,value,*,row_start,row_count,groups=1):
        """Evaluate an original contiguous row interval for a collective rank."""
        with self._lock:
            self._boundary()
            if self._busy:raise RuntimeError('BF16 projection owner is already executing')
            rows,columns=self.sources[name];x=np.asarray(value)
            if type(row_start) is not int or type(row_count) is not int or row_start<0 or row_count<0 or row_start+row_count>rows:
                raise ValueError('bounded BF16 projection row interval required')
            if type(groups) is not int or groups<=0 or rows%groups or x.dtype!=np.float32 or x.ndim!=3 or x.shape[-2:]!=(groups,columns) or not x.shape[0]:
                raise ValueError('explicit batched group-local BF16 projection input required')
            transport=x.nbytes*3+x.shape[0]*row_count*12
            available=self.max_bytes-transport
            tile=min(self.row_block,max(0,available//(columns*128)),max(0,self._arithmetic.max_bytes//(columns*16)))
            if row_count and tile<1:raise MemoryError('BF16 rank projection row tile exceeds admission')
            if not np.isfinite(x).all():raise ValueError('nonfinite BF16 rank input')
            output=np.empty((x.shape[0],row_count),np.float32);per_group=rows//groups
            self._busy=True
            try:
                begin=row_start;stop=row_start+row_count
                while begin<stop:
                    self._boundary();group=begin//per_group
                    end=min(stop,begin+tile,(group+1)*per_group)
                    weight=self.reader.read(TensorSlice(name,(end-begin,columns),begin*columns))
                    if name in self._decoded_bf16:weight=self._arithmetic.cast(weight.reshape(-1)).reshape(weight.shape)
                    for batch in range(x.shape[0]):
                        self._boundary()
                        device_product=getattr(self._arithmetic,'bf16_matvec',None)
                        output[batch,begin-row_start:end-row_start]=(device_product(weight,x[batch,group]) if callable(device_product)
                            else V41Execution._linear(self._arithmetic,weight,x[batch,group]))
                    begin=end
                if not np.isfinite(output).all():raise ArithmeticError('nonfinite BF16 rank projection')
                self._boundary();return output
            finally:self._busy=False

    @contextmanager
    def cancellation_scope(self,event):
        if event is not None and not callable(getattr(event,'is_set',None)):
            raise TypeError('event-like BF16 projection cancellation required')
        old=getattr(self._local,'cancel',None);self._local.cancel=event
        try:yield self
        finally:self._local.cancel=old

    def _boundary(self):
        event=getattr(self._local,'cancel',None)
        if event is not None and event.is_set():raise InterruptedError('BF16 publisher projection cancelled')
        if self._closed or self._arithmetic is None:raise RuntimeError('closed or unbound BF16 projection bank')
        self._arithmetic._ensure_open();self.snapshot.assert_unchanged()

    def linear(self,name,value):
        rows,columns=self.sources[name];x=np.asarray(value)
        if x.dtype!=np.float32 or x.shape!=(columns,):
            raise ValueError('complete FP32 publisher projection vector required')
        return self._execute(name,x[None,None,:],groups=1)[0,0]

    def grouped_linear(self,name,value,*,groups):
        x=np.asarray(value);rows,columns=self.sources[name]
        if type(groups) is not int or groups<=0 or rows%groups or x.dtype!=np.float32 or x.ndim not in (2,3) or x.shape[-2:]!=(groups,columns):
            raise ValueError('exact group-local BF16 publisher projection geometry required')
        return self._execute(name,x[None] if x.ndim==2 else x,groups=groups)[0] if x.ndim==2 else self._execute(name,x,groups=groups)

    def _execute(self,name,x,*,groups):
        with self._lock:
            self._boundary()
            if self._busy:raise RuntimeError('BF16 projection owner is already executing')
            rows,columns=self.sources[name];batch=x.shape[0];per_group=rows//groups
            transport=x.nbytes*3+batch*rows*12
            available=self.max_bytes-transport
            tile=min(self.row_block,per_group,max(0,available//(columns*128)),max(0,self._arithmetic.max_bytes//(columns*16)))
            if not batch or tile<1:raise MemoryError('BF16 row conversion and output exceed admission')
            if not np.isfinite(x).all():raise ValueError('nonfinite BF16 publisher projection input')
            output=np.empty((batch,groups,per_group),np.float32);self._busy=True
            try:
                for group in range(groups):
                    for begin in range(0,per_group,tile):
                        self._boundary();count=min(tile,per_group-begin)
                        weight=self.reader.read(TensorSlice(name,(count,columns),(group*per_group+begin)*columns))
                        if name in self._decoded_bf16:weight=self._arithmetic.cast(weight.reshape(-1)).reshape(weight.shape)
                        for row in range(batch):
                            self._boundary()
                            device_product=getattr(self._arithmetic,'bf16_matvec',None)
                            output[row,group,begin:begin+count]=(device_product(weight,x[row,group]) if callable(device_product)
                                else V41Execution._linear(self._arithmetic,weight,x[row,group]))
                if not np.isfinite(output).all():raise ArithmeticError('nonfinite streamed BF16 publisher projection')
                self._boundary();return output
            finally:self._busy=False

    def close(self):
        with self._lock:
            if self._busy:raise RuntimeError('BF16 projection owns an active transaction')
            if self._closed:return
            self.reader.close();self._arithmetic=None;self._closed=True


__all__=['PublisherBF16Projections']
