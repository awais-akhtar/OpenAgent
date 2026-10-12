"""Complete Qwen3.5 device hybrid-state transport without token replay."""
import hashlib
import hmac
import json
import math
import struct
import numpy as np

from .segment_snapshot import _read,_write,_pairs


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _schema(model,batch,position):
    c = model.config
    shapes = []
    if batch:
        for kind in c.layer_types:
            if kind=='full_attention':
                shape = [batch,c.num_key_value_heads,position,c.head_dim]
                shapes.append([shape,shape])
            else:
                width = 2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
                shapes.append([[batch,width,c.linear_conv_kernel_dim],
                    [batch,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim]])
    return dict(identity=model.identity,batch=batch,position=position,shapes=shapes)


def _payload(header):
    return sum(math.prod(shape)*4 for bank in header['shapes'] for shape in bank)+header['batch']*header['position']


def write_device_cache(model,output,*,max_bytes):
    with model._lock:
        model._check();model._cancel()
        if type(max_bytes) is not int or max_bytes<44:
            raise ValueError('explicit Qwen3.5 cache byte admission required')
        batch = 0 if model._mask is None else model._mask.shape[0]
        header = _schema(model,batch,model.position)
        encoded = _json(header)
        payload = _payload(header)
        extent = 44+len(encoded)+payload
        if len(encoded)>65536 or extent>max_bytes or model.weight_bytes+model._expert_bytes+getattr(model,'_projection_bytes',0)+model._snapshot_bytes+2*payload+3*1024**2>model.max_bytes:
            raise MemoryError('Qwen3.5 cache transport exceeds admission')
        if len(model._cache)!=len(header['shapes']):
            raise ValueError('incomplete Qwen3.5 continuation')
        model._busy = True
        try:
            digest = hashlib.sha256()
            for data in (b'OATQ3501',struct.pack('<I',len(encoded)),encoded):
                digest.update(data);_write(output,data)
            for bank,shapes in zip(model._cache,header['shapes']):
                if len(bank)!=2:
                    raise ValueError('incomplete hybrid continuation bank')
                for value,shape in zip(bank,shapes):
                    model._cancel()
                    if tuple(value.shape)!=tuple(shape):
                        raise ValueError('invalid hybrid continuation geometry')
                    host = np.ascontiguousarray(value.detach().cpu().numpy(),dtype='<f4')
                    if not np.isfinite(host).all():
                        raise ValueError('nonfinite hybrid continuation')
                    raw = memoryview(host).cast('B')
                    for offset in range(0,len(raw),1024**2):
                        model._cancel()
                        data = raw[offset:offset+1024**2]
                        digest.update(data);_write(output,data)
                    del host,raw
            mask = b'' if model._mask is None else model._mask.tobytes()
            digest.update(mask);_write(output,mask)
            model._cancel();_write(output,digest.digest())
            return extent
        finally:
            model._busy = False


def restore_device_cache(model,source,*,max_bytes):
    with model._lock:
        model._check();model._cancel()
        if type(max_bytes) is not int or max_bytes<44:
            raise ValueError('explicit Qwen3.5 cache byte admission required')
        model._busy = True
        try:
            prefix = _read(source,12)
            if prefix[:8]!=b'OATQ3501':
                raise ValueError('invalid Qwen3.5 device cache magic/version')
            length = struct.unpack('<I',prefix[8:])[0]
            if not 0<length<=min(65536,max_bytes-44):
                raise ValueError('invalid Qwen3.5 device header extent')
            encoded = _read(source,length)
            header = json.loads(encoded,object_pairs_hook=_pairs)
            if type(header) is not dict or set(header)!={'identity','batch','position','shapes'}:
                raise ValueError('invalid Qwen3.5 device cache fields')
            batch,position = header['batch'],header['position']
            if (type(batch) is not int or not 0<=batch<=128 or type(position) is not int or
                    not 0<=position<=model.capacity or bool(batch)!=bool(position)):
                raise ValueError('invalid Qwen3.5 device cache batch/position')
            expected = _schema(model,batch,position)
            if _json(header)!=_json(expected):
                raise ValueError('Qwen3.5 device cache identity/geometry mismatch')
            payload = _payload(expected)
            if 44+length+payload>max_bytes or model.weight_bytes+model._expert_bytes+getattr(model,'_projection_bytes',0)+model._snapshot_bytes+model._state_bytes()+2*payload+3*1024**2>model.max_bytes:
                raise MemoryError('Qwen3.5 device recovery staging exceeds admission')
            digest = hashlib.sha256(prefix+encoded)
            pending = []
            for shapes in expected['shapes']:
                bank = []
                for shape in shapes:
                    model._cancel()
                    host = np.empty(shape,dtype='<f4')
                    flat = host.reshape(-1)
                    for offset in range(0,flat.size,262144):
                        model._cancel()
                        count = min(262144,flat.size-offset)
                        data = _read(source,count*4);digest.update(data)
                        value = np.frombuffer(data,dtype='<f4')
                        if not np.isfinite(value).all():
                            raise ValueError('nonfinite Qwen3.5 continuation')
                        flat[offset:offset+count] = value
                    bank.append(model.torch.tensor(host,device=model.device,dtype=model.torch.float32))
                    del host,flat
                pending.append(tuple(bank))
            mask_data = _read(source,batch*position);digest.update(mask_data)
            if any(value>1 for value in mask_data):
                raise ValueError('invalid Qwen3.5 occupancy mask')
            mask = np.frombuffer(mask_data,dtype=np.uint8).reshape(batch,position).copy() if batch else None
            if not hmac.compare_digest(digest.digest(),_read(source,32)) or source.read(1)!=b'':
                raise ValueError('Qwen3.5 cache checksum/EOF mismatch')
            model._cancel()
            model._cache,model._mask,model.position = tuple(pending),mask,position
            return position
        finally:
            model._busy = False
