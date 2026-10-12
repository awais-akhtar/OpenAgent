"""Bounded original-weight projections on CUDA, HIP and Metal devices.

Only selected row blocks cross the host/device boundary. This owner never
keeps a complete decoded publisher matrix resident on either side.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import math
import threading
import numpy as np
from .publisher_tensor_reader import PublisherTensorReader,TensorSlice
from .device_execution_policy import admit_fp32

class DevicePublisherMatrix:
    def __init__(self,owner,name):
        self.owner,self.name=owner,name;self.shape=owner.bindings[name].shape
        self.ndim=2;self.dtype=owner.torch.float32;self.device=owner.device
    def linear(self,inputs,*,cancel=None): return self.owner.linear(self.name,inputs,cancel=cancel)
    def __getitem__(self,indices): return self.owner.gather(self.name,indices)
    def numel(self): return math.prod(self.shape)
    def element_size(self): return 4

class DevicePublisherTable:
    def __init__(self,owner,sources):
        self.owner=owner;self.sources=tuple(sources);self.ndim=2
        widths={owner.bindings[name].shape[1] for name in sources}
        if len(widths)!=1: raise ValueError('lexical shard widths differ')
        self.shape=(sum(owner.bindings[name].shape[0] for name in sources),widths.pop())
        self.dtype=owner.torch.float32;self.device=owner.device
    def numel(self): return math.prod(self.shape)
    def element_size(self): return 4
    def __getitem__(self,indices):
        bank=self.owner;torch=bank.torch
        with bank._lock:
            bank._check();count=indices.numel();columns=self.shape[1]
            if indices.dtype!=torch.int64 or indices.device!=self.device or not count:
                raise ValueError('device integer lexical indices required')
            if bank.max_cache_bytes+count*(640+16*columns)+8*columns>bank.max_bytes or count*640>bank.reader.max_bytes:
                raise MemoryError('sharded lexical output exceeds device bank admission')
            if bool((indices<0).any()) or bool((indices>=self.shape[0]).any()):
                raise ValueError('sharded lexical index outside source extent')
            with torch.inference_mode():
                output=torch.empty((*indices.shape,columns),device=self.device,dtype=torch.float32)
                offset=0
                for name in self.sources:
                    rows=bank.bindings[name].shape[0]
                    selected=(indices>=offset)&(indices<offset+rows)
                    if bool(selected.any()):
                        output[selected]=bank.gather(name,indices[selected]-offset,reserved_bytes=count*columns*4)
                    offset+=rows
                return output

class DevicePublisherProjections:
    @classmethod
    def from_snapshot(cls,snapshot,names,**options):
        """Bind private checkpoint matrices without decoding their payloads."""
        selected=tuple(names)
        if not selected or len(set(selected))!=len(selected):
            raise ValueError('nonempty unique projection tensor selection required')
        bindings={}
        for name in selected:
            record=snapshot.tensors.get(name)
            if record is None or len(record.shape)!=2:
                raise ValueError('selected checkpoint projection is not a matrix: '+name)
            bindings[name]=TensorSlice(name,tuple(record.shape))
        return cls(snapshot,bindings,**options)

    def __init__(self,snapshot,bindings,*,torch,device,max_bytes,max_host_bytes,
                 max_cache_bytes=0,row_block=128,block_shape=None,table_bindings=None):
        if any(type(v) is not int or v<=0 for v in (max_bytes,max_host_bytes,row_block)) or type(max_cache_bytes) is not int or not 0<=max_cache_bytes<max_bytes:
            raise ValueError('explicit host/device projection and row-cache admission required')
        target=torch.device(device)
        admit_fp32(torch,target)
        if target.type not in ('cuda','mps'): raise ValueError('explicit CUDA/HIP or Metal projection device required')
        if target.type=='cuda' and target.index is None:
            target=torch.device('cuda:'+str(torch.cuda.current_device()))
        self.torch,self.device=torch,target;self.max_bytes=max_bytes;self.max_cache_bytes=max_cache_bytes;self.row_block=row_block
        self.reader=PublisherTensorReader(snapshot,max_work_bytes=max_host_bytes,block_shape=block_shape)
        self.bindings=dict(bindings);matrix_names=tuple(self.bindings);tables={}
        for name,sources in ({} if table_bindings is None else table_bindings).items():
            if name in self.bindings or not sources: raise ValueError('unique nonempty lexical table bindings required')
            keys=[]
            for index,source in enumerate(sources):
                key=name+'#publisher-shard-'+str(index)
                if key in self.bindings: raise ValueError('lexical shard binding collision')
                self.bindings[key]=source;keys.append(key)
            tables[name]=tuple(keys)
        self._lock=threading.RLock();self._closed=False;self._busy=False
        self._cache=OrderedDict();self._cache_bytes=0;self._local=threading.local()
        digest=hashlib.sha256(repr((snapshot.config_sha256,target,torch.__version__,block_shape,row_block,max_bytes,max_host_bytes,max_cache_bytes)).encode())
        digest.update(repr(tuple(sorted(snapshot.files.items()))).encode())
        for name,binding in sorted(self.bindings.items()):
            if type(name) is not str or type(binding) is not TensorSlice or len(binding.shape)!=2 or any(type(v) is not int or v<=0 for v in binding.shape):
                raise ValueError('explicit original matrix slice bindings required')
            record=snapshot.tensors.get(binding.source)
            if record is None or binding.element_offset<0 or binding.element_offset+math.prod(binding.shape)>math.prod(record.shape):
                raise ValueError('device projection exceeds original tensor inventory')
            if record.dtype not in ('F32','BF16','F16','F8_E4M3'): raise ValueError('unsupported original device projection encoding')
            if binding.shape[1]*128>max_host_bytes: raise MemoryError('one decoded publisher row exceeds host admission')
            digest.update(repr((name,binding,record)).encode())
        snapshot.assert_unchanged();self.identity='openagent-device-publisher-projections-v1:'+digest.hexdigest()
        self.matrices={name:DevicePublisherMatrix(self,name) for name in matrix_names}
        self.matrices.update({name:DevicePublisherTable(self,sources) for name,sources in tables.items()})

    def _check(self):
        if self._closed or self._busy: raise RuntimeError('device publisher projection owner is closed or executing')
        admit_fp32(self.torch,self.device)

    def alias(self,name,source,*,expected_binding=None,cancel=None):
        """Share one original row source after bounded duplicate-weight comparison."""
        with self._lock:
            self._check()
            if name in self.matrices or source not in self.bindings or source not in self.matrices:
                raise ValueError('unique alias and owned original matrix source required')
            binding=self.bindings[source];rows,columns=binding.shape
            if expected_binding is not None:
                if type(expected_binding) is not TensorSlice or expected_binding.shape!=binding.shape:
                    raise ValueError('tied publisher matrix geometry differs')
                block=min(self.row_block,self.reader.max_bytes//(256*columns))
                if block<1:raise MemoryError('one tied-weight comparison row exceeds host admission')
                for begin in range(0,rows,block):
                    self._cancel(cancel);end=min(rows,begin+block)
                    left=self.reader.read(TensorSlice(binding.source,(end-begin,columns),binding.element_offset+begin*columns))
                    right=self.reader.read(TensorSlice(expected_binding.source,(end-begin,columns),expected_binding.element_offset+begin*columns))
                    if not np.array_equal(left,right):raise ValueError('publisher tied embedding/head values differ')
                    del left,right
            self.reader.snapshot.assert_unchanged();self._cancel(cancel)
            self.bindings[name]=binding;self.matrices[name]=self.matrices[source]
            self.identity=hashlib.sha256((self.identity+repr(('tied-alias',name,source,expected_binding))).encode()).hexdigest()

    def _cancel(self,cancel):
        if cancel is None: cancel=getattr(self._local,'cancel',None)
        if cancel is None: return
        callback=cancel if callable(cancel) else getattr(cancel,'is_set',None)
        if not callable(callback): raise TypeError('callable or event-like projection cancellation required')
        if callback():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('original device projection cancelled')

    @contextmanager
    def cancellation_scope(self,cancel):
        old=getattr(self._local,'cancel',None);self._local.cancel=cancel
        try:
            yield self
        finally:
            self._local.cancel=old

    def _block(self,name,begin,end):
        self.reader.snapshot.assert_unchanged();key=(name,begin,end)
        if key in self._cache:
            value=self._cache.pop(key);self._cache[key]=value;return value
        binding=self.bindings[name];columns=binding.shape[1]
        source=self.reader.read(TensorSlice(binding.source,(end-begin,columns),binding.element_offset+begin*columns))
        charge=source.nbytes
        while self._cache and self._cache_bytes+charge>self.max_cache_bytes:
            _,victim=self._cache.popitem(last=False);self._cache_bytes-=victim.numel()*victim.element_size()
        value=self.torch.tensor(source,device=self.device,dtype=self.torch.float32)
        if charge<=self.max_cache_bytes:
            self._cache[key]=value;self._cache_bytes+=charge
        return value

    def linear(self,name,inputs,*,cancel=None):
        with self._lock:
            self._check();torch=self.torch;binding=self.bindings[name];rows,columns=binding.shape
            if inputs.ndim<2 or inputs.dtype!=torch.float32 or inputs.device!=self.device or inputs.shape[-1]!=columns:
                raise ValueError('matching FP32 publisher projection activations required')
            items=inputs.numel()//columns
            if not items: raise ValueError('nonempty device projection activations required')
            host_block=max(1,min(self.row_block,self.reader.max_bytes//(128*columns)))
            fixed=self.max_cache_bytes+8*(inputs.numel()+items*rows)
            available=self.max_bytes-fixed
            device_block=available//(8*(columns+items))
            block=min(host_block,device_block)
            if block<1: raise MemoryError('device row transaction exceeds projection admission')
            self._cancel(cancel)
            if not bool(torch.isfinite(inputs).all()): raise ValueError('nonfinite device projection inputs')
            self._busy=True
            try:
                with torch.inference_mode():
                    output=torch.empty((*inputs.shape[:-1],rows),device=self.device,dtype=torch.float32)
                    for begin in range(0,rows,block):
                        self._cancel(cancel);end=min(rows,begin+block)
                        weight=self._block(name,begin,end)
                        value=inputs@weight.T
                        if not bool(torch.isfinite(value).all()): raise ArithmeticError('nonfinite original device projection')
                        output[...,begin:end]=value;del value,weight
                    self._cancel(cancel);return output
            finally: self._busy=False

    def gather(self,name,indices,*,cancel=None,reserved_bytes=0):
        """Fetch lexical rows without materializing a full embedding table."""
        with self._lock:
            self._check();torch=self.torch;rows,columns=self.bindings[name].shape
            if indices.dtype!=torch.int64 or indices.device!=self.device or not indices.numel():
                raise ValueError('nonempty device integer embedding indices required')
            count=indices.numel()
            if type(reserved_bytes) is not int or reserved_bytes<0:
                raise ValueError('nonnegative outer gather reservation required')
            fixed=reserved_bytes+self.max_cache_bytes+count*(512+8*columns)+8*columns
            if fixed>self.max_bytes or count*512>self.reader.max_bytes:
                raise MemoryError('lexical gather exceeds host/device projection admission')
            block=min(self.row_block,(self.reader.max_bytes-count*512)//(128*columns),
                (self.max_bytes-fixed)//(8*columns))
            if block<1:raise MemoryError('one selected original embedding row exceeds admission')
            if bool((indices<0).any()) or bool((indices>=rows).any()):
                raise ValueError('lexical row index exceeds publisher table')
            self._cancel(cancel);self._busy=True
            try:
                with torch.inference_mode():
                    selected=indices.reshape(-1).detach().cpu().tolist()
                    output=torch.empty((count,columns),dtype=torch.float32,device=self.device)
                    locations={}
                    for index,row in enumerate(selected):locations.setdefault(row,[]).append(index)
                    unique=sorted(locations);cursor=0
                    while cursor<len(unique):
                        self._cancel(cancel);stop=cursor+1
                        while stop<len(unique) and stop-cursor<block and unique[stop]==unique[stop-1]+1:stop+=1
                        first=unique[cursor];last=unique[stop-1]+1
                        values=self._block(name,first,last)
                        for row in unique[cursor:stop]:
                            self._cancel(cancel)
                            destination=torch.tensor(locations[row],device=self.device,dtype=torch.int64)
                            output[destination]=values[row-first]
                            del destination
                        del values;cursor=stop
                    self._cancel(cancel)
                    return output.reshape(*indices.shape,columns)
            finally:
                self._busy=False

    def clear_cache(self):
        with self._lock:
            self._check();self._cache.clear();self._cache_bytes=0

    def close(self):
        with self._lock:
            if self._busy: raise RuntimeError('device projection owns an active transaction')
            if self._closed: return
            self._cache.clear();self._cache_bytes=0;self.reader.close();self._closed=True

def device_projection(torch,inputs,weight,*,cancel=None):
    admit_fp32(torch,inputs.device)
    if type(weight) is DevicePublisherMatrix: return weight.linear(inputs,cancel=cancel)
    return inputs@weight.T

__all__=['DevicePublisherProjections','DevicePublisherMatrix','DevicePublisherTable','device_projection']
