"""Bounded row reads from original Qwen4 PLE tables, including logical shards."""
from bisect import bisect_right
from copy import deepcopy
from collections import OrderedDict
from types import MappingProxyType,SimpleNamespace
import hashlib
import json
import threading
import numpy as np
from .publisher_checkpoint.storage import Snapshot
from .qwen4_ple import metadata

class Qwen4PLEStore:
    def __init__(self,snapshot,config,bindings,*,max_bytes,max_rows=4096):
        if type(snapshot) is not Snapshot or config.get('model_type')!='qwen4_exp_text':
            raise ValueError('immutable local snapshot and exact Qwen4 text config required')
        if type(max_bytes) is not int or max_bytes<=0 or type(max_rows) is not int or max_rows<=0:
            raise ValueError('positive PLE row cache budgets required')
        self.config=MappingProxyType(deepcopy(dict(config)));self.snapshot=snapshot
        self._lock=threading.RLock();self._closed=False;self._cache=OrderedDict();self._used=0
        self.max_bytes,self.max_rows=max_bytes,max_rows;self.bindings={};self.shapes={}
        digest=hashlib.sha256(snapshot.config_sha256.encode())
        c=SimpleNamespace(**dict(config));expected_layers=tuple(i-1 for i in c.ple_layer_ids)
        if any(type(layer) is not int for layer in bindings) or set(bindings)!=set(expected_layers): raise ValueError('each configured PLE layer must own an explicit table binding')
        for ordinal,layer in enumerate(expected_layers):
            names=bindings[layer]
            if type(names) not in (list,tuple) or not names or len(set(names))!=len(names):
                raise ValueError('ordered unique original PLE source shards required')
            _,sizes,_=metadata(c,ordinal);part=c.ple_embed_dim//sizes.size
            rows=(int(sizes.sum())+c.make_ngram_vocab_size_divisible_by-1)//c.make_ngram_vocab_size_divisible_by*c.make_ngram_vocab_size_divisible_by
            starts=[0]
            for name in names:
                record=snapshot.tensors[name]
                if record.dtype not in ('F32','BF16','F16') or len(record.shape)!=2 or record.shape[1]!=part or record.shape[0]<=0:
                    raise ValueError('PLE source shard geometry or dtype differs')
                starts.append(starts[-1]+record.shape[0]);digest.update(repr(record).encode())
            if starts[-1]!=rows: raise ValueError('PLE logical shard rows do not cover the exact padded vocabulary')
            if part*32>max_bytes: raise MemoryError('one PLE row transaction exceeds cache admission')
            self.bindings[layer]=(tuple(names),tuple(starts));self.shapes[layer]=(rows,part)
        snapshot.assert_unchanged()
        digest.update(json.dumps(dict(config),sort_keys=True,allow_nan=False).encode())
        self.identity='openagent-qwen4-streamed-ple-v1:'+digest.hexdigest()

    def row(self,layer,index):
        with self._lock:
            if self._closed: raise RuntimeError('PLE store is closed')
            if type(layer) is not int or layer not in self.shapes or type(index) is not int or not 0<=index<self.shapes[layer][0]:
                raise ValueError('PLE row outside admitted logical table')
            key=(layer,index)
            if key in self._cache:
                # Cache hits also reject a checkpoint replaced after admission.
                self.snapshot.assert_unchanged();value=self._cache.pop(key);self._cache[key]=value;return value
            names,starts=self.bindings[layer];part=self.shapes[layer][1]
            slot=bisect_right(starts,index)-1;record=self.snapshot.tensors[names[slot]]
            width={'F32':4,'BF16':2,'F16':2}[record.dtype]
            while self._cache and (self._used+part*32>self.max_bytes or len(self._cache)>=self.max_rows):
                _,evicted=self._cache.popitem(last=False);self._used-=evicted.nbytes
            raw=self.snapshot.read_range(names[slot],(index-starts[slot])*part*width,part*width)
            if record.dtype=='BF16':
                value=(np.frombuffer(raw,'<u2').astype(np.uint32)<<np.uint32(16)).view('<f4')
            else: value=np.frombuffer(raw,'<f4' if record.dtype=='F32' else '<f2').astype(np.float32)
            if not np.isfinite(value).all(): raise ValueError('nonfinite PLE checkpoint row')
            value=np.frombuffer(value.tobytes(),np.float32)
            self._cache[key]=value;self._used+=value.nbytes;return value

    def status(self):
        with self._lock:
            return dict(identity=self.identity,resident_rows=len(self._cache),resident_bytes=self._used,max_bytes=self.max_bytes,closed=self._closed)

    def close(self):
        with self._lock:
            self._cache.clear();self._used=0;self._closed=True

__all__=['Qwen4PLEStore']
