"""Complete service generation boundaries, including prefix-stop replay."""
from dataclasses import dataclass,asdict
import hashlib
import hmac
import json
import struct
import tempfile
import numpy as np
from .sampling import SamplingPolicy,TokenSampler
from .segment_snapshot import _read,_write,_pairs

_PREFIX=struct.Struct('<8sIQQ')
_MAGIC=b'OASVGN01'


@dataclass(frozen=True)
class ServiceGenerationBoundary:
    identity:str
    tokens:tuple
    logits:bytes
    dtype:str
    vocabulary:int
    context:int
    policy:SamplingPolicy|None
    sampler:bytes
    stops:tuple
    stopped:bool=False
    application_identity:str=''
    stop_prefixes:tuple=()

    @property
    def position(self):return len(self.tokens)

    @classmethod
    def capture(cls,session,tokens,logits,sampler,stops,limits,stopped=False,application_identity='',stop_prefixes=()):
        if (type(tokens) not in (tuple, list) or not tokens or len(tokens) > limits.context or
                any(type(token) is not int or not 0 <= token < limits.vocabulary for token in tokens) or
                type(stops) not in (tuple, list, frozenset) or
                any(type(token) is not int or not 0 <= token < limits.vocabulary for token in stops)):
            raise ValueError('complete bounded generation token inventory required')
        if (type(stop_prefixes) not in (tuple, list) or len(stop_prefixes) > 64 or
                any(type(prefix) not in (tuple, list) or not prefix or len(prefix) > 128 or
                    any(type(token) is not int or not 0 <= token < limits.vocabulary
                        for token in prefix) for prefix in stop_prefixes) or
                len({tuple(prefix) for prefix in stop_prefixes}) != len(stop_prefixes)):
            raise ValueError('complete bounded stop-prefix inventory required')
        if type(stopped) is not bool:
            raise ValueError('explicit generation stop-state required')
        if (type(application_identity) is not str or
                not 0 < len(application_identity) <= 8192 or '\x00' in application_identity):
            raise ValueError('bounded generation application identity required')
        if (len(tokens)*128+len(stops)*128+sum(len(prefix)*16 for prefix in stop_prefixes)+
                limits.vocabulary*32+1024**2>limits.max_checkpoint_buffer_bytes):
            raise MemoryError('generation boundary construction exceeds scheduler host reservation')
        row=np.asarray(logits)
        if row.shape!=(limits.vocabulary,) or row.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(row).all():
            raise ValueError('complete finite next-token row required')
        identity=getattr(session.model,'identity',None)
        if type(identity) is not str or not identity:raise ValueError('explicit generation execution identity required')
        dtype='<f4' if row.dtype.itemsize==4 else '<f8'
        return cls(identity,tuple(tokens),row.astype(dtype,copy=False).tobytes(),dtype,limits.vocabulary,
            limits.context,None if sampler is None else sampler.policy,
            b'' if sampler is None else sampler.snapshot(),tuple(sorted(stops)),stopped,
            application_identity,tuple(tuple(prefix) for prefix in stop_prefixes))

    def admit(self,limits,max_bytes):
        if (type(max_bytes) is not int or max_bytes<=0 or type(self.identity) is not str or not 0<len(self.identity)<=8192
                or type(self.vocabulary) is not int or self.vocabulary!=limits.vocabulary
                or type(self.context) is not int or self.context!=limits.context
                or self.dtype not in ('<f4','<f8') or type(self.logits) is not bytes
                or len(self.logits)!=self.vocabulary*np.dtype(self.dtype).itemsize):
            raise ValueError('complete matching generation boundary required')
        for tokens in (self.tokens,self.stops):
            if type(tokens) is not tuple or len(tokens)>self.context or any(type(t) is not int or not 0<=t<self.vocabulary for t in tokens):
                raise ValueError('bounded exact generation token history required')
        if not self.tokens or len(set(self.stops))!=len(self.stops):raise ValueError('occupied unique-stop boundary required')
        if (type(self.stop_prefixes) is not tuple or len(self.stop_prefixes)>64 or
                any(type(prefix) is not tuple or not prefix or len(prefix)>128 or
                    any(type(t) is not int or not 0 <= t < self.vocabulary for t in prefix)
                    for prefix in self.stop_prefixes) or
                len(set(self.stop_prefixes)) != len(self.stop_prefixes)):
            raise ValueError('bounded stop-prefix boundary required')
        prefix_stopped = any(len(self.tokens) >= len(prefix) and
                             tuple(self.tokens[-len(prefix):]) == prefix
                             for prefix in self.stop_prefixes)
        if (type(self.stopped) is not bool or
                (self.stopped and self.tokens[-1] not in self.stops and not prefix_stopped)):
            raise ValueError('generation stopping boundary differs')
        if (type(self.application_identity) is not str or
                not 0 < len(self.application_identity) <= 8192 or
                '\x00' in self.application_identity):
            raise ValueError('bounded generation application identity required')
        if not np.isfinite(self.row()).all():raise ValueError('nonfinite checkpoint next logits')
        if self.policy is None:
            if self.sampler!=b'':raise ValueError('greedy boundary has unexpected RNG state')
        else:
            if type(self.policy) is not SamplingPolicy:raise ValueError('exact sampling policy required')
            owner=TokenSampler(self.policy,max_buffer_bytes=limits.max_sampling_buffer_bytes,max_history_tokens=limits.context)
            owner.restore(self.sampler)
        host=(len(self.logits)*3+len(self.sampler)*4+(len(self.tokens)+len(self.stops))*128+
              sum(len(prefix)*16 for prefix in self.stop_prefixes)+1024**2)
        if host>min(max_bytes,limits.max_checkpoint_buffer_bytes):
            raise MemoryError('generation boundary host staging exceeds admission')

    def row(self):return np.frombuffer(self.logits,dtype=self.dtype)


def write_boundary(model,boundary,output,*,limits,max_bytes):
    boundary.admit(limits,max_bytes)
    if getattr(model,'identity',None)!=boundary.identity or getattr(model,'position',getattr(model,'length',None))!=len(boundary.tokens):
        raise ValueError('generation boundary differs from live cache owner')
    metadata=dict(version=1,identity=boundary.identity,tokens=list(boundary.tokens),dtype=boundary.dtype,
        vocabulary=boundary.vocabulary,context=boundary.context,policy=None if boundary.policy is None else asdict(boundary.policy),
        sampler=boundary.sampler.hex(),stops=list(boundary.stops),stopped=boundary.stopped,
        stop_prefixes=[list(prefix) for prefix in boundary.stop_prefixes],
        application_identity=boundary.application_identity)
    encoded=json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
    if len(encoded)*16+len(boundary.logits)*3+65536>limits.max_checkpoint_buffer_bytes:
        raise MemoryError('generation encoding exceeds scheduler host reservation')
    if len(encoded)>min(1024**2,max_bytes//2):raise MemoryError('generation metadata exceeds admission')
    overhead=_PREFIX.size+len(encoded)+len(boundary.logits)+32
    if overhead>=max_bytes:raise MemoryError('generation frame exhausted before cache')
    with tempfile.TemporaryFile('w+b') as cache:
        size=model.write_cache(cache,max_bytes=max_bytes-overhead)
        if type(size) is not int or size<=0 or size!=cache.tell() or size>max_bytes-overhead:
            raise ValueError('generation cache writer extent differs')
        prefix=_PREFIX.pack(_MAGIC,len(encoded),len(boundary.logits),size)
        digest=hashlib.sha256()
        for data in (prefix,encoded,boundary.logits):digest.update(data);_write(output,data)
        cache.seek(0);remaining=size
        while remaining:
            data=cache.read(min(65536,remaining))
            if not data:raise ValueError('generation cache staging truncated')
            digest.update(data);_write(output,data);remaining-=len(data)
        _write(output,digest.digest());return overhead+size


def read_boundary(source,*,limits,max_bytes):
    """Verify the entire frame before exposing staged cache for live restore."""
    if type(max_bytes) is not int or max_bytes<128:raise ValueError('bounded generation recovery admission required')
    prefix=_read(source,_PREFIX.size);magic,header_bytes,logit_bytes,cache_bytes=_PREFIX.unpack(prefix)
    if (magic!=_MAGIC or not 0<header_bytes<=min(1024**2,max_bytes//2)
            or logit_bytes not in (limits.vocabulary*4,limits.vocabulary*8) or not cache_bytes
            or _PREFIX.size+header_bytes+logit_bytes+cache_bytes+32>max_bytes):
        raise ValueError('generation recovery frame extent differs')
    if header_bytes*32+logit_bytes*3+65536>limits.max_checkpoint_buffer_bytes:
        raise MemoryError('generation metadata parsing exceeds scheduler host reservation')
    encoded=_read(source,header_bytes);header=json.loads(encoded,object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    fields={'version','identity','tokens','dtype','vocabulary','context','policy','sampler','stops','stopped','application_identity','stop_prefixes'}
    if type(header) is not dict or set(header)!=fields or type(header['version']) is not int or header['version']!=1:
        raise ValueError('generation recovery metadata differs')
    if (type(header['tokens']) is not list or type(header['stops']) is not list or
            type(header['stop_prefixes']) is not list or type(header['sampler']) is not str):
        raise ValueError('generation recovery inventories differ')
    policy=None if header['policy'] is None else SamplingPolicy(**header['policy'])
    logits=_read(source,logit_bytes)
    boundary=ServiceGenerationBoundary(header['identity'],tuple(header['tokens']),bytes(logits),header['dtype'],
        header['vocabulary'],header['context'],policy,bytes.fromhex(header['sampler']),tuple(header['stops']),
        header['stopped'],header['application_identity'],
        tuple(tuple(prefix) for prefix in header['stop_prefixes']))
    boundary.admit(limits,max_bytes);cache=tempfile.TemporaryFile('w+b')
    try:
        digest=hashlib.sha256(prefix+encoded+logits);remaining=cache_bytes
        while remaining:
            data=_read(source,min(65536,remaining));digest.update(data);_write(cache,data);remaining-=len(data)
        if not hmac.compare_digest(bytes(_read(source,32)),digest.digest()):raise ValueError('generation recovery checksum differs')
        if source.read(1):raise ValueError('generation recovery frame has trailing data')
        cache.seek(0);return boundary,cache,cache_bytes
    except BaseException:cache.close();raise


__all__=['ServiceGenerationBoundary','write_boundary','read_boundary']
