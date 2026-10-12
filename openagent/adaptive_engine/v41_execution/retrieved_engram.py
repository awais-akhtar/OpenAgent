"""Selected-row I/O adapter; hashing, projections and gating execute in C.

The provider must expose immutable contents for its declared revision. It is
never asked to execute hashing, compression, projection or gating math.
"""
import ctypes as C
from dataclasses import dataclass
from itertools import islice

import numpy as np


@dataclass(frozen=True)
class EngramRowRequest:
    head: int
    table_id: str
    row: int
    width: int
    row_format: str


@dataclass(frozen=True)
class PackedEngramRow:
    """Private local FP4: eight nibble bytes and one E4M3 scale per 16 values.

    Codes are low-nibble-first E2M1; scales are unsigned E4M3 codes 0..126.
    This is not an assertion about a publisher packed embedding format.
    """
    codes: bytes
    scales: bytes


@dataclass(frozen=True)
class RetrievedEngramRow:
    request: EngramRowRequest
    payload: object


class _Head(C.Structure):
    _fields_ = [('order',C.c_uint32),('width',C.c_uint32),('rows',C.c_uint32),
                ('reserved',C.c_uint32),('seed',C.c_uint64),('multiplier',C.c_uint64*4)]


def _int(value, lo, hi, name):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError('invalid retrieved Engram '+name)
    return value


def _bytes(value, count, name):
    view = memoryview(value)
    try:
        if view.nbytes != count:
            raise ValueError('retrieved row '+name+' extent mismatch')
        return view.tobytes()
    finally:
        view.release()


class RetrievedEngram:
    def __init__(self, owner, spec, *, available_bytes):
        self.owner = owner
        self.dim = _int(spec['dim'],16,16384,'dim')
        self.branches = _int(spec['branches'],1,8,'branches')
        if (self.dim,self.branches) != (owner.config.hidden,owner.config.streams):
            raise ValueError('retrieved Engram residual geometry mismatch')
        self.vocabulary = _int(spec['vocabulary'],1,2**32-1,'vocabulary')
        self.pad = _int(spec['pad'],0,self.vocabulary-1,'pad')
        self.epsilon = spec['epsilon']
        if type(self.epsilon) not in (int,float) or not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError('invalid retrieved Engram epsilon')
        self.revision = spec['revision']
        if type(self.revision) is not str or not self.revision or len(self.revision)>4096:
            raise ValueError('immutable provider revision is required')
        self.provider = spec['provider']
        if not callable(getattr(self.provider,'fetch_rows',None)):
            raise ValueError('row provider must expose fetch_rows(requests)')
        allowed = {'kind','dim','branches','vocabulary','pad','epsilon','heads','key','value',
                   'query_scale','key_scale','provider','revision','weight_budget'}
        if set(spec)-allowed:
            raise ValueError('unknown retrieved Engram descriptor fields')
        descriptors = tuple(islice(iter(spec['heads']),25))
        if not 1 <= len(descriptors) <= 24:
            raise ValueError('retrieved Engram requires 1..24 heads')
        metadata = []
        self._native_heads = (_Head*len(descriptors))()
        memory = 0
        for i, head in enumerate(descriptors):
            if set(head) != {'order','width','rows','seed','multipliers','table_id','row_format'}:
                raise ValueError('exact retrieved head metadata required; no resident table')
            order = _int(head['order'],2,4,'order')
            width = _int(head['width'],1,16384,'row width')
            rows = _int(head['rows'],1,2**32-1,'rows')
            seed = _int(head['seed'],0,2**64-1,'seed')
            multipliers = tuple(islice(iter(head['multipliers']),5))
            if len(multipliers)!=4:
                raise ValueError('four explicit hash multipliers required')
            multipliers = tuple(_int(v,0,2**64-1,'multiplier') for v in multipliers)
            table_id, row_format = head['table_id'],head['row_format']
            if type(table_id) is not str or not table_id or len(table_id)>4096:
                raise ValueError('explicit table identity required')
            if row_format not in ('fp32','local-fp4-block16-v1'):
                raise ValueError('unresolved retrieved row encoding')
            if row_format=='local-fp4-block16-v1' and width%16:
                raise ValueError('packed retrieved row width must be divisible by 16')
            self._native_heads[i] = _Head(order,width,rows,0,seed,(C.c_uint64*4)(*multipliers))
            metadata.append((order,width,rows,seed,multipliers,table_id,row_format))
            memory += width
        if {m[0] for m in metadata} != {2,3,4} or memory>65536:
            raise ValueError('all n-gram orders 2/3/4 and admitted memory width required')
        self.metadata = tuple(metadata)
        self.memory = memory
        self._weights = {}
        shapes = {'key': (self.branches,self.dim,memory), 'value': (self.dim,memory),
                  'query_scale': (self.branches,self.dim), 'key_scale': (self.branches,self.dim)}
        self.weight_bytes = sum(int(np.prod(s))*4 for s in shapes.values())
        budget = spec.get('weight_budget',available_bytes)
        if type(budget) is not int or budget<0 or self.weight_bytes>min(budget,available_bytes):
            raise ValueError('retrieved Engram projections exceed admission before copying')
        for name, shape in shapes.items():
            x = np.asarray(spec[name])
            if x.dtype!=np.float32 or x.shape!=shape or not x.flags.c_contiguous or not np.isfinite(x).all():
                raise ValueError('invalid retrieved Engram FP32 tensor: '+name)
            self._weights[name] = np.frombuffer(x.tobytes(),dtype=np.float32).reshape(shape)
        self._closed = False

    def update_digest(self, digest):
        digest.update(repr(('retrieved-engram-v1',self.dim,self.branches,self.vocabulary,
                            self.pad,self.epsilon,self.revision,self.metadata)).encode('utf8'))
        for name, value in self._weights.items():
            digest.update(name.encode('ascii')); digest.update(value.tobytes())

    def session(self, capacity):
        if self._closed: raise RuntimeError('retrieved Engram is closed')
        return _Session(self,capacity)

    def close(self):
        self._closed = True


class _Session:
    def __init__(self, model, capacity):
        self.model = model
        self.capacity = capacity
        self.length = 0
        self._tail = ()

    def snapshot(self):
        return (self.length,self._tail)

    def restore(self, snapshot):
        if (type(snapshot) is not tuple or len(snapshot)!=2 or type(snapshot[0]) is not int or
            not 0<=snapshot[0]<=self.capacity or type(snapshot[1]) is not tuple or
            len(snapshot[1])!=min(snapshot[0],3) or any(type(t) is not int or not 0<=t<self.model.vocabulary for t in snapshot[1])):
            raise ValueError('invalid retrieved Engram session snapshot')
        self.length,self._tail = snapshot

    def forward(self, compressed_ids, hidden):
        model = self.model
        if model._closed: raise RuntimeError('retrieved Engram is closed')
        ids = tuple(islice(iter(compressed_ids),2))
        if len(ids)!=1 or self.length>=self.capacity:
            raise ValueError('retrieved session accepts one admitted token per scheduling step')
        token = _int(ids[0],0,model.vocabulary-1,'compressed token')
        history = (C.c_uint32*(len(self._tail)+1))(*(self._tail+(token,)))
        addresses = (C.c_uint32*len(model.metadata))()
        status = model.owner._hash(model._native_heads,len(model.metadata),history,len(history),
                                   model.vocabulary,model.pad,addresses,len(addresses))
        if status: raise ValueError(f'native retrieved Engram hash failure {status}')
        requests = tuple(EngramRowRequest(i,meta[5],int(addresses[i]),meta[1],meta[6])
                         for i,meta in enumerate(model.metadata))
        # Provider sees only selected addresses. Bound response consumption before copying.
        responses = tuple(islice(iter(model.provider.fetch_rows(requests)),len(requests)+1))
        if len(responses)!=len(requests): raise ValueError('provider returned wrong selected-row count')
        rows = []
        for request, response in zip(requests,responses):
            if (type(response) is not RetrievedEngramRow or
                type(response.request) is not EngramRowRequest or response.request!=request):
                raise ValueError('provider response does not identify the requested row')
            if request.row_format=='fp32':
                x = np.asarray(response.payload)
                if x.dtype!=np.float32 or x.shape!=(request.width,) or not x.flags.c_contiguous or not np.isfinite(x).all():
                    raise ValueError('provider returned invalid FP32 row')
                # Detach provider-owned mutable memory before native arithmetic.
                rows.append(np.frombuffer(x.tobytes(),dtype=np.float32))
            else:
                if type(response.payload) is not PackedEngramRow:
                    raise ValueError('packed row requires explicit codes and scales')
                codes = _bytes(response.payload.codes,request.width//2,'codes')
                scales = _bytes(response.payload.scales,request.width//16,'scales')
                packed = b''.join(scales[i:i+1]+codes[i*8:i*8+8] for i in range(len(scales)))
                rows.append(model.owner._decode_width(packed,request.width))
        memory = np.concatenate(rows)
        keys = np.concatenate([model.owner._linear(w,memory) for w in model._weights['key']])
        value = model.owner._linear(model._weights['value'],memory)
        z = np.concatenate((value,model._weights['query_scale'].reshape(-1),model._weights['key_scale'].reshape(-1)))
        delta = model.owner._kernel(15,hidden,keys,z,a=model.branches,b=model.dim,p=model.epsilon,
                                    n=model.branches*model.dim)
        self._tail = (self._tail+(token,))[-3:]
        self.length += 1
        return delta,tuple(int(v) for v in addresses)
