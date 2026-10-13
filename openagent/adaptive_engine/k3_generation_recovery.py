"""Durable generation boundary: complete cache, next logits and sampler state.

The cache codec owns model identity and physical-state validation. This outer
frame binds the application state to that exact cache, without token replay.
Writes target caller-owned binary streams; publication/retention belongs to
the serving application's checkpoint store.
"""
import copy
import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MAGIC=b'OAK3G001'
_PREFIX=struct.Struct('<8sIQQ')
_CHUNK=1024*1024
_HEADER_LIMIT=1024*1024


@dataclass(frozen=True)
class GenerationBoundary:
    config_sha256:str
    position:int
    prompt_ids:tuple
    generated_ids:tuple
    expanded_prompt_tokens:int
    next_logits:np.ndarray
    rng_state:dict
    temperature:float
    stop_token_ids:tuple
    execution_scope:str
    stop_prefixes:tuple=()
    media_metadata:tuple=()
    tokenizer_sha256:str=''

    def admit(self,*,vocabulary,capacity,max_bytes):
        if (type(max_bytes) is not int or max_bytes<=0 or type(self.position) is not int
                or not 0<self.position<=capacity or type(self.expanded_prompt_tokens) is not int
                or self.expanded_prompt_tokens<=0
                or self.position!=self.expanded_prompt_tokens+len(self.generated_ids)):
            raise ValueError('occupied complete generation boundary required')
        for tokens in (self.prompt_ids,self.generated_ids,self.stop_token_ids):
            if type(tokens) is not tuple or any(type(v) is not int or not 0<=v<vocabulary for v in tokens):
                raise ValueError('exact vocabulary token inventories required')
        if not self.prompt_ids or len(self.prompt_ids)>self.expanded_prompt_tokens:
            raise ValueError('original and expanded prompt inventories differ')
        if (type(self.stop_prefixes) is not tuple or len(self.stop_prefixes)>64 or
                any(type(prefix) is not tuple or not prefix or len(prefix)>128 or
                    any(type(v) is not int or not 0<=v<vocabulary for v in prefix)
                    for prefix in self.stop_prefixes) or
                len(set(self.stop_prefixes)) != len(self.stop_prefixes)):
            raise ValueError('bounded unique generation stop-prefix inventory required')
        if (type(self.config_sha256) is not str or len(self.config_sha256)!=64
                or any(c not in '0123456789abcdef' for c in self.config_sha256)
                or self.execution_scope not in ('text-only','image-text','mixed-media-text')):
            raise ValueError('exact checkpoint configuration and media scope required')
        if (type(self.tokenizer_sha256) is not str or len(self.tokenizer_sha256)!=64
                or any(c not in '0123456789abcdef' for c in self.tokenizer_sha256)):
            raise ValueError('exact tokenizer identity required for generation recovery')
        if (isinstance(self.temperature,bool) or not isinstance(self.temperature,(int,float))
                or not math.isfinite(self.temperature) or self.temperature<0):
            raise ValueError('finite sampling temperature required')
        if (type(self.next_logits) is not np.ndarray or self.next_logits.dtype!=np.float32
                or self.next_logits.shape!=(vocabulary,) or not np.isfinite(self.next_logits).all()):
            raise ValueError('complete finite FP32 next-token logits required')
        if type(self.rng_state) is not dict or self.rng_state.get('bit_generator')!='PCG64':
            raise ValueError('original PCG64 sampler boundary required')
        if type(self.media_metadata) is not tuple:
            raise ValueError('detached media provenance tuple required')
        header=self.header()
        encoded=json.dumps(header,allow_nan=False,separators=(',',':')).encode('utf-8')
        if len(encoded)>min(max_bytes,_HEADER_LIMIT) or len(encoded)+self.next_logits.nbytes>max_bytes:
            raise MemoryError('generation boundary metadata exceeds admission')
        # NumPy admits the complete nested PCG64 state without modifying live RNG.
        temporary=np.random.PCG64();temporary.state=copy.deepcopy(self.rng_state)

    def header(self):
        return dict(schema='openagent.k3-generation-boundary.v2',config_sha256=self.config_sha256,
            position=self.position,prompt_ids=list(self.prompt_ids),generated_ids=list(self.generated_ids),
            expanded_prompt_tokens=self.expanded_prompt_tokens,rng_state=copy.deepcopy(self.rng_state),
            temperature=self.temperature,stop_token_ids=list(self.stop_token_ids),
            stop_prefixes=[list(prefix) for prefix in self.stop_prefixes],
            execution_scope=self.execution_scope,media_metadata=copy.deepcopy(list(self.media_metadata)),
            tokenizer_sha256=self.tokenizer_sha256)


def _write(stream,value):
    view=memoryview(value)
    while view:
        count=stream.write(view)
        if type(count) is not int or not 0<count<=len(view):raise OSError('checkpoint sink made no progress')
        view=view[count:]


def _read(stream,size):
    parts=[];remaining=size
    while remaining:
        requested=min(_CHUNK,remaining)
        value=stream.read(requested)
        if type(value) is not bytes or not value or len(value)>requested:
            raise ValueError('truncated binary generation checkpoint')
        parts.append(value);remaining-=len(value)
    return b''.join(parts)


def write_generation_boundary(model,boundary,output,*,max_bytes,scratch_directory=None,cancelled=None):
    """Capture a consistent boundary under the model owner lock."""
    if type(boundary) is not GenerationBoundary:raise ValueError('exact generation boundary required')
    if cancelled is not None and not callable(cancelled):raise TypeError('cancellation callback required')
    boundary.admit(vocabulary=model.config.vocab,capacity=model.config.capacity,max_bytes=max_bytes)
    def check_cancelled():
        if cancelled is not None and cancelled():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('K3 checkpoint publication cancelled')
    # Bind sampler history and physical cache serialization to one owner lease.
    # The model lock is reentrant, so write_cache may retain its own protection.
    with model._lock:
        check_cancelled()
        if model.position!=boundary.position:raise ValueError('sampler boundary and live cache position differ')
        return _write_generation_boundary_locked(model,boundary,output,max_bytes=max_bytes,
            scratch_directory=scratch_directory,check_cancelled=check_cancelled)


def _write_generation_boundary_locked(model,boundary,output,*,max_bytes,scratch_directory,check_cancelled):
    descriptor,name=tempfile.mkstemp(prefix='openagent-generation-',dir=scratch_directory)
    os.close(descriptor);path=Path(name)
    try:
        with path.open('wb') as cache:
            check_cancelled()
            model.write_cache(cache,max_bytes=max_bytes,scratch_directory=scratch_directory)
        check_cancelled()
        header=json.dumps(boundary.header(),allow_nan=False,separators=(',',':')).encode('utf-8')
        logits=boundary.next_logits.astype('<f4',copy=False).tobytes()
        size=path.stat().st_size
        prefix=_PREFIX.pack(_MAGIC,len(header),len(logits),size)
        if len(prefix)+len(header)+len(logits)+size+32>max_bytes:
            raise MemoryError('complete generation recovery frame exceeds admission')
        digest=hashlib.sha256()
        for value in (prefix,header,logits):digest.update(value);_write(output,value)
        with path.open('rb') as source:
            for value in iter(lambda:source.read(_CHUNK),b''):
                check_cancelled()
                digest.update(value);_write(output,value)
        check_cancelled()
        _write(output,digest.digest())
        return len(prefix)+len(header)+len(logits)+size+32
    finally:path.unlink(missing_ok=True)


def restore_generation_boundary(model,source,*,config_sha256,tokenizer_sha256,max_bytes,scratch_directory=None,cancelled=None):
    """Verify the entire envelope before restoring the identity-bound cache."""
    if cancelled is not None and not callable(cancelled):raise TypeError('cancellation callback required')
    def check_cancelled():
        if cancelled is not None and cancelled():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('K3 checkpoint recovery cancelled')
    check_cancelled()
    if type(max_bytes) is not int or max_bytes<=_PREFIX.size+32:raise ValueError('positive recovery budget required')
    check_cancelled()
    prefix=_read(source,_PREFIX.size)
    magic,header_size,logit_size,cache_size=_PREFIX.unpack(prefix)
    if (magic!=_MAGIC or not 0<header_size<=_HEADER_LIMIT or logit_size!=model.config.vocab*4
            or not cache_size or _PREFIX.size+header_size+logit_size+cache_size+32>max_bytes):
        raise ValueError('invalid complete generation checkpoint geometry')
    if 3*(header_size+logit_size)+min(_CHUNK,cache_size)>max_bytes:
        raise MemoryError('recovery metadata, logits and transport scratch exceed admission')
    check_cancelled()
    header=_read(source,header_size);check_cancelled();logits=_read(source,logit_size)
    from .native_checkpoint import _pairs
    data=json.loads(header.decode('utf-8'),object_pairs_hook=_pairs)
    fields={'schema','config_sha256','position','prompt_ids','generated_ids','expanded_prompt_tokens',
            'rng_state','temperature','stop_token_ids','stop_prefixes','execution_scope','media_metadata','tokenizer_sha256'}
    if type(data) is not dict or set(data)!=fields or data['schema']!='openagent.k3-generation-boundary.v2':
        raise ValueError('exact generation boundary schema required')
    if data['config_sha256']!=config_sha256:raise ValueError('foreign generation checkpoint configuration')
    if data['tokenizer_sha256']!=tokenizer_sha256:raise ValueError('foreign generation checkpoint tokenizer')
    if any(type(data[key]) is not list for key in ('prompt_ids','generated_ids','stop_token_ids','stop_prefixes','media_metadata')):
        raise ValueError('ordered complete generation inventories required')
    boundary=GenerationBoundary(data['config_sha256'],data['position'],tuple(data['prompt_ids']),
        tuple(data['generated_ids']),data['expanded_prompt_tokens'],
        np.frombuffer(logits,dtype='<f4'),data['rng_state'],data['temperature'],
        tuple(data['stop_token_ids']),data['execution_scope'],
        tuple(tuple(prefix) for prefix in data['stop_prefixes']),
        tuple(data['media_metadata']),data['tokenizer_sha256'])
    boundary.admit(vocabulary=model.config.vocab,capacity=model.config.capacity,max_bytes=max_bytes)
    descriptor,name=tempfile.mkstemp(prefix='openagent-generation-',dir=scratch_directory)
    os.close(descriptor);path=Path(name)
    digest=hashlib.sha256(prefix+header+logits)
    try:
        with path.open('wb') as output:
            remaining=cache_size
            while remaining:
                check_cancelled()
                value=_read(source,min(_CHUNK,remaining));digest.update(value)
                output.write(value);remaining-=len(value)
            output.flush();os.fsync(output.fileno())
        import hmac
        check_cancelled()
        if not hmac.compare_digest(digest.digest(),_read(source,32)) or source.read(1)!=b'':
            raise ValueError('corrupt or trailing generation checkpoint bytes')
        check_cancelled()
        with model._lock:
            check_cancelled()
            before=model.snapshot()
            try:
                with path.open('rb') as cache:
                    check_cancelled()
                    model.restore_cache(cache,max_bytes=max_bytes,scratch_directory=scratch_directory)
                check_cancelled()
                if model.position!=boundary.position:raise ValueError('cache and generation boundary positions differ')
            except BaseException as primary:
                try:model.restore(before)
                except BaseException as rollback:primary.add_note('generation restore rollback failed: '+repr(rollback))
                raise
        return boundary
    finally:path.unlink(missing_ok=True)


__all__=['GenerationBoundary','write_generation_boundary','restore_generation_boundary']
