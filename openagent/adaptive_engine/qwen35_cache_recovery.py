"""Complete native Qwen3.5 KV/recurrent/mask continuation transport."""
from contextlib import contextmanager
import hashlib
import hmac
import json
import struct

import numpy as np

from .segment_snapshot import _read,_write,_pairs

_MAGIC = b'OAQ35K01'
_CHUNK = 1024*1024


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')


@contextmanager
def _access(model,limit):
    from .native_qwen35 import NativeQwen35TextDecoder
    if not isinstance(model,NativeQwen35TextDecoder) or model.edge_only:
        raise TypeError('native Qwen3.5 layer owner required')
    if type(limit) is not int or limit<44:
        raise ValueError('explicit Qwen3.5 continuation budget required')
    with model._lock:
        model._check_open()
        model._transport = True
        try:
            yield
        finally:
            model._transport = False


def _schema(model,batch,position):
    if not batch:
        return []
    c = model.config
    width = 2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
    schema = []
    for index in range(model.layer_begin,model.layer_end):
        if c.layer_types[index]=='full_attention':
            shapes = ((batch,c.num_key_value_heads,position,c.head_dim),)*2
        else:
            shapes = ((batch,width,c.linear_conv_kernel_dim),
                      (batch,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim))
        schema.extend((f'cache/{index}/{slot}','<f4',list(shape)) for slot,shape in enumerate(shapes))
    schema.append(('mask','|u1',[batch,position]))
    return schema


def _state_bytes(model):
    return sum(value.nbytes for bank in model.cache if bank for value in bank)+(0 if model.mask is None else model.mask.nbytes)


def write_qwen35_cache(model,output,*,max_bytes):
    with _access(model,max_bytes):
        batch = 0 if model.mask is None else model.mask.shape[0]
        position = model.position
        if (not batch and position) or (batch and model.mask.shape!=(batch,position)):
            raise ValueError('inconsistent Qwen3.5 committed mask/position')
        schema = _schema(model,batch,position)
        header = _canonical(dict(identity=model.identity,position=position,batch=batch,schema=schema))
        payload = sum(int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize for _,dtype,shape in schema)
        extent = 44+len(header)+payload
        if len(header)>65536 or extent>max_bytes:
            raise MemoryError('Qwen3.5 continuation frame exceeds admission')
        if sum(value.nbytes for value in model.weights.values())+_state_bytes(model)+3*_CHUNK+model._expert_bytes+model._snapshot_bytes>model.max_bytes:
            raise MemoryError('Qwen3.5 continuation transport staging exceeds admission')
        digest = hashlib.sha256()
        for data in (_MAGIC,struct.pack('<I',len(header)),header):
            digest.update(data);_write(output,data)
        for name,dtype,shape in schema:
            if name=='mask':
                array = model.mask
            else:
                _,index,slot = name.split('/')
                array = model.cache[int(index)][int(slot)]
                if model.config.layer_types[int(index)]=='full_attention':
                    array = array[:,:,:position]
            if list(array.shape)!=shape or array.dtype!=np.dtype(dtype):
                raise ValueError('Qwen3.5 committed continuation inventory differs')
            iterator = np.nditer(array,flags=['external_loop','buffered','zerosize_ok'],
                op_flags=['readonly'],op_dtypes=[dtype],order='C',buffersize=_CHUNK//np.dtype(dtype).itemsize)
            for values in iterator:
                if (dtype=='<f4' and not np.isfinite(values).all()) or (name=='mask' and np.any(values>1)):
                    raise ValueError('invalid Qwen3.5 committed continuation values')
                data = values.tobytes();digest.update(data);_write(output,data)
        _write(output,digest.digest())
        return extent


def restore_qwen35_cache(model,source,*,max_bytes):
    with _access(model,max_bytes):
        prefix = _read(source,12)
        if prefix[:8]!=_MAGIC:
            raise ValueError('invalid Qwen3.5 continuation magic/version')
        length = struct.unpack('<I',prefix[8:])[0]
        if not 0<length<=min(65536,max_bytes-44):
            raise ValueError('invalid Qwen3.5 continuation header extent')
        encoded = _read(source,length)
        header = json.loads(encoded,object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if type(header) is not dict or set(header)!={'identity','position','batch','schema'}:
            raise ValueError('invalid Qwen3.5 continuation header')
        batch,position = header['batch'],header['position']
        if (header['identity']!=model.identity or type(batch) is not int or not 0<=batch<=128 or
                type(position) is not int or not 0<=position<=model.capacity or
                (not batch and position)):
            raise ValueError('Qwen3.5 continuation identity/geometry mismatch')
        schema = _schema(model,batch,position)
        if _canonical(header['schema'])!=_canonical(schema):
            raise ValueError('Qwen3.5 continuation inventory mismatch')
        payload = sum(int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize for _,dtype,shape in schema)
        full_bytes = batch*position
        c = model.config
        for index in range(model.layer_begin,model.layer_end):
            if c.layer_types[index]=='full_attention':
                full_bytes+=8*batch*c.num_key_value_heads*model.capacity*c.head_dim
            else:
                full_bytes += sum(int(np.prod(shape,dtype=object))*4 for name,_,shape in schema
                                  if name.startswith(f'cache/{index}/'))
        weights = sum(value.nbytes for value in model.weights.values())
        if 44+length+payload>max_bytes or weights+_state_bytes(model)+full_bytes+3*_CHUNK+model._expert_bytes+model._snapshot_bytes>model.max_bytes:
            raise MemoryError('Qwen3.5 continuation restore staging exceeds admission')
        digest = hashlib.sha256(prefix+encoded)
        banks = [None]*c.num_hidden_layers
        for index in range(model.layer_begin,model.layer_end):
            if not batch:
                break
            shapes = [shape for name,_,shape in schema if name.startswith(f'cache/{index}/')]
            if c.layer_types[index]=='full_attention':
                shapes=[[batch,c.num_key_value_heads,model.capacity,c.head_dim]]*2
            banks[index] = tuple(np.zeros(shape,np.float32) for shape in shapes)
        mask = np.empty((batch,position),np.uint8) if batch else None
        for name,dtype,shape in schema:
            if name=='mask':
                array = mask
            else:
                _,index,slot = name.split('/')
                array = banks[int(index)][int(slot)]
                if c.layer_types[int(index)]=='full_attention':
                    array = array[:,:,:position]
            # Strided active KV prefixes are filled through a bounded iterator.
            iterator = np.nditer(array,flags=['external_loop','buffered','zerosize_ok'],
                op_flags=['writeonly'],order='C',buffersize=_CHUNK//np.dtype(dtype).itemsize)
            for values in iterator:
                data = _read(source,values.size*np.dtype(dtype).itemsize)
                digest.update(data)
                incoming = np.frombuffer(data,dtype=dtype)
                if (dtype=='<f4' and not np.isfinite(incoming).all()) or (name=='mask' and np.any(incoming>1)):
                    raise ValueError('invalid Qwen3.5 continuation values')
                values[:] = incoming
            iterator.close()
        if not hmac.compare_digest(digest.digest(),_read(source,32)) or source.read(1)!=b'':
            raise ValueError('Qwen3.5 continuation checksum/EOF mismatch')
        model.cache,model.mask,model.position = banks,mask,position
        return position


__all__ = ['write_qwen35_cache','restore_qwen35_cache']
