"""Complete bounded device Mixtral KV/mask transport; no token replay."""
import hashlib
import hmac
import json
import struct
import numpy as np

from .segment_snapshot import _read,_write,_pairs


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _schema(model,batch,position):
    c = model.config
    extent = position if c['sliding_window'] is None else min(position,c['sliding_window'])
    return dict(identity=model.identity,batch=batch,position=position,
                shape=[batch,c['num_key_value_heads'],extent,c['head_dim']],
                layers=c['num_hidden_layers'] if batch else 0)


def write_device_cache(model,output,*,max_bytes):
    with model._lock:
        model._check()
        if type(max_bytes) is not int or max_bytes<44:
            raise ValueError('explicit device cache budget required')
        batch = 0 if model._mask is None else model._mask.shape[0]
        header = _json(_schema(model,batch,model.position))
        size = sum(value.numel()*4 for bank in model._cache for value in bank)
        payload = size+batch*model.position
        extent = 44+len(header)+payload
        if len(header)>65536 or extent>max_bytes or model.weight_bytes+model._snapshot_bytes+2*payload+3*1024**2+model._expert_bytes>model.max_bytes:
            raise MemoryError('device cache transport exceeds admission')
        model._busy = True
        try:
            digest = hashlib.sha256()
            for data in (b'OATMXR01',struct.pack('<I',len(header)),header):
                digest.update(data);_write(output,data)
            for bank in model._cache:
                for value in bank:
                    host = value.cpu().numpy()
                    if not np.isfinite(host).all():
                        raise ValueError('nonfinite device cache')
                    raw = memoryview(np.ascontiguousarray(host,dtype='<f4')).cast('B')
                    for start in range(0,len(raw),1024**2):
                        data = raw[start:start+1024**2]
                        digest.update(data);_write(output,data)
            mask = b'' if model._mask is None else model._mask.tobytes()
            digest.update(mask);_write(output,mask);_write(output,digest.digest())
            return extent
        finally:
            model._busy = False


def restore_device_cache(model,source,*,max_bytes):
    with model._lock:
        model._check()
        if type(max_bytes) is not int or max_bytes<44:
            raise ValueError('explicit device cache budget required')
        model._busy = True
        try:
            prefix = _read(source,12)
            if prefix[:8]!=b'OATMXR01':
                raise ValueError('invalid device Mixtral cache magic/version')
            length = struct.unpack('<I',prefix[8:])[0]
            if not 0<length<=min(65536,max_bytes-44):
                raise ValueError('invalid device cache header extent')
            encoded = _read(source,length)
            header = json.loads(encoded, object_pairs_hook=_pairs,
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if type(header) is not dict or set(header)!={'identity','batch','position','shape','layers'}:
                raise ValueError('invalid device cache fields')
            batch,position = header['batch'],header['position']
            if type(batch) is not int or not 0<=batch<=128 or type(position) is not int or not 0<=position<=model.capacity or (not batch and position):
                raise ValueError('invalid device cache batch/position')
            if _json(header)!=_json(_schema(model,batch,position)):
                raise ValueError('device cache execution identity/geometry mismatch')
            shape = header['shape']
            elements = int(np.prod(shape,dtype=object))
            payload = elements*8*header['layers']+batch*position
            old = sum(value.numel()*4 for bank in model._cache for value in bank)
            if 44+length+payload>max_bytes or model.weight_bytes+model._snapshot_bytes+old+2*payload+3*1024**2+model._expert_bytes>model.max_bytes:
                raise MemoryError('device cache recovery staging exceeds admission')
            digest = hashlib.sha256(prefix+encoded)
            banks = []
            bank=[];host=flat=value=data=mask_data=mask=None;committed=False
            try:
                for _ in range(header['layers']):
                    bank = []
                    for _ in range(2):
                        host = np.empty(shape,dtype='<f4')
                        flat = host.reshape(-1)
                        for start in range(0,elements,262144):
                            count = min(262144,elements-start)
                            data = _read(source,count*4);digest.update(data)
                            value = np.frombuffer(data,dtype='<f4')
                            if not np.isfinite(value).all():
                                raise ValueError('nonfinite device continuation cache')
                            flat[start:start+count] = value
                        bank.append(model.torch.tensor(host,device=model.device,dtype=model.torch.float32))
                        host=flat=value=data=None
                    banks.append(tuple(bank));bank=[]
                mask_data = _read(source,batch*position);digest.update(mask_data)
                if any(value>1 for value in mask_data):
                    raise ValueError('invalid device continuation occupancy mask')
                mask = np.frombuffer(mask_data,dtype=np.uint8).reshape(batch,position).copy() if batch else None
                if not hmac.compare_digest(digest.digest(),_read(source,32)) or source.read(1)!=b'':
                    raise ValueError('device cache checksum/EOF mismatch')
                model._cache,model._mask,model.position = tuple(banks),mask,position
                committed=True
                return position
            finally:
                if not committed:
                    bank.clear();banks.clear();mask=None
                host=flat=value=data=mask_data=None
        finally:
            model._busy = False


__all__ = ['write_device_cache','restore_device_cache']
