"""Authenticated bounded Inkling device KV/FIR transport without replay."""
import hashlib
import hmac
import json
import math
import struct
import numpy as np
from .segment_snapshot import _read,_write,_pairs

def _header(model,batch,position):
    return dict(identity=model.identity,batch=batch,position=position,
        shapes=[[batch,model._state_size(g)] for g in model.geometry] if batch else [])

def write_cache(model,output,*,max_bytes):
    with model._lock:
        model._check();model._cancel()
        if type(max_bytes) is not int or max_bytes<44: raise ValueError('explicit Inkling transport budget required')
        batch=model._cache[0].shape[0] if model._cache else 0
        header=_header(model,batch,model.position);encoded=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
        extent=44+len(encoded)+model._state_bytes()
        if extent>max_bytes or len(encoded)>65536: raise MemoryError('Inkling cache output exceeds admission')
        model._admit(3*1024**2);model._busy=True
        try:
            digest=hashlib.sha256();prefix=getattr(model,'cache_magic',b'OATINK01')+struct.pack('<I',len(encoded))+encoded
            digest.update(prefix);_write(output,prefix)
            for bank,shape in zip(model._cache,header['shapes']):
                if tuple(bank.shape)!=tuple(shape): raise ValueError('Inkling physical state geometry differs')
                flat=bank.reshape(-1)
                for begin in range(0,flat.numel(),262144):
                    model._cancel();host=np.ascontiguousarray(flat[begin:begin+262144].detach().cpu().numpy(),dtype='<f4')
                    if not np.isfinite(host).all(): raise ValueError('nonfinite Inkling physical continuation')
                    raw=memoryview(host).cast('B');digest.update(raw);_write(output,raw)
            model._cancel();_write(output,digest.digest());return extent
        finally:
            model._busy=False

def restore_cache(model,source,*,max_bytes):
    with model._lock:
        model._check();model._cancel()
        if type(max_bytes) is not int or max_bytes<44: raise ValueError('explicit Inkling transport budget required')
        # Stream callbacks are caller code, including the first header read.
        # Exclude mutation before allowing any callback to observe this owner.
        model._admit(3*1024**2);model._busy=True
        try:
            prefix=_read(source,12)
            if prefix[:8]!=getattr(model,'cache_magic',b'OATINK01'): raise ValueError('invalid device cache magic')
            length=struct.unpack('<I',prefix[8:])[0]
            if not 0<length<=min(65536,max_bytes-44): raise ValueError('Inkling cache header exceeds admission')
            encoded=_read(source,length);header=json.loads(encoded,object_pairs_hook=_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if type(header) is not dict or set(header)!={'identity','batch','position','shapes'}:
                raise ValueError('invalid Inkling cache fields')
            batch,position=header['batch'],header['position']
            if (type(batch) is not int or not 0<=batch<=128 or type(position) is not int
                    or not 0<=position<=model.config['max_position_embeddings'] or bool(batch)!=bool(position)
                    or header!=_header(model,batch,position)):
                raise ValueError('Inkling cache identity or geometry differs')
            size=sum(math.prod(shape)*4 for shape in header['shapes'])
            if 44+length+size>max_bytes: raise MemoryError('Inkling cache payload exceeds transport admission')
            model._admit(2*size+3*1024**2)
            digest=hashlib.sha256(prefix+encoded);host=[];pending=();array=flat=raw=values=None
            committed=False
            try:
                for shape in header['shapes']:
                    array=np.empty(shape,dtype='<f4');flat=array.reshape(-1)
                    for begin in range(0,flat.size,262144):
                        model._cancel();end=min(flat.size,begin+262144);raw=_read(source,(end-begin)*4)
                        digest.update(raw);values=np.frombuffer(raw,'<f4');flat[begin:end]=values
                    if not np.isfinite(array).all(): raise ValueError('nonfinite Inkling cache payload')
                    host.append(array);array=flat=raw=values=None
                if not hmac.compare_digest(_read(source,32),digest.digest()) or source.read(1):
                    raise ValueError('Inkling cache digest or trailing bytes differ')
                validator=getattr(model,'_validate_recovered_state',None)
                if callable(validator):validator(tuple(host),position)
                model._cancel();pending=tuple(model.torch.tensor(value,device=model.device) for value in host)
                model._cancel();model._cache,model.position=pending,position;committed=True
            finally:
                if not committed:
                    host.clear();pending=()
                array=flat=raw=values=None
        finally:
            model._busy=False

__all__=['write_cache','restore_cache']
