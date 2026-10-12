"""Versioned complete device Qwen4 continuation transport."""
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
    banks,lexical = [],[]
    if batch:
        width = 2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
        for index,kind in enumerate(c.layer_types):
            if kind=='linear_attention':
                banks.append([[batch,width,c.linear_conv_kernel_dim],
                    [batch,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim]])
            else:
                banks.append([[batch,c.num_key_value_heads,position,c.head_dim],
                    [batch,c.num_key_value_heads,position,c.head_dim],[batch,position,c.indexer_head_dim]])
            lexical.append([[batch,c.ngram_size-1],
                [batch,c.hc_count*c.hidden_size,(c.ple_conv_kernel_size-1)*c.ngram_size]] if index+1 in c.ple_layer_ids else None)
    return dict(identity=model.identity,batch=batch,position=position,banks=banks,lexical=lexical,
        position_axes=3 if model.mrope_section is not None else 1)


def _entries(header):
    for index,bank in enumerate(header['banks']):
        for slot,shape in enumerate(bank):
            yield ('layer',index,slot,shape,'<f4')
    for index,bank in enumerate(header['lexical']):
        if bank is not None:
            for slot,shape in enumerate(bank):
                yield ('lexical',index,slot,shape,'<i8' if slot==0 else '<f4')
    if header['batch']:
        shape=[header['batch'],header['position']]
        if header['position_axes']==3: shape=[3,*shape]
        yield ('positions',0,0,shape,'<i8')


def _size(header):
    return sum(math.prod(shape)*np.dtype(dtype).itemsize for _,_,_,shape,dtype in _entries(header))+header['batch']*header['position']


def write_device_cache(model,output,*,max_bytes):
    with model._lock:
        model._check();model._cancel()
        if type(max_bytes) is not int or max_bytes<44:
            raise ValueError('explicit Qwen4 cache byte admission required')
        batch = 0 if model._mask is None else model._mask.shape[0]
        header = _schema(model,batch,model.position)
        encoded,payload = _json(header),_size(header)
        extent = 44+len(encoded)+payload
        if len(encoded)>65536 or extent>max_bytes or model.weight_bytes+model._expert_bytes+model._snapshot_bytes+2*payload+3*1024**2>model.max_bytes:
            raise MemoryError('Qwen4 cache transport exceeds admission')
        model._busy = True
        try:
            digest = hashlib.sha256()
            for data in (b'OATQW401',struct.pack('<I',len(encoded)),encoded):
                digest.update(data);_write(output,data)
            for kind,index,slot,shape,dtype in _entries(header):
                model._cancel()
                value = model._positions if kind=='positions' else (model._cache if kind=='layer' else model._ple)[index][slot]
                if tuple(value.shape)!=tuple(shape):
                    raise ValueError('incomplete Qwen4 continuation geometry')
                host = np.ascontiguousarray(value.detach().cpu().numpy(),dtype=dtype)
                if not np.isfinite(host).all():
                    raise ValueError('nonfinite Qwen4 continuation')
                raw = memoryview(host).cast('B')
                for offset in range(0,len(raw),1024**2):
                    model._cancel()
                    data = raw[offset:offset+1024**2];digest.update(data);_write(output,data)
                del raw,host
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
            raise ValueError('explicit Qwen4 cache byte admission required')
        model._busy = True
        try:
            prefix = _read(source,12)
            if prefix[:8]!=b'OATQW401':
                raise ValueError('invalid Qwen4 device cache magic/version')
            length = struct.unpack('<I',prefix[8:])[0]
            if not 0<length<=min(65536,max_bytes-44):
                raise ValueError('invalid Qwen4 cache header extent')
            encoded = _read(source,length)
            header = json.loads(encoded,object_pairs_hook=_pairs)
            if type(header) is not dict or set(header)!={'identity','batch','position','banks','lexical','position_axes'}:
                raise ValueError('invalid Qwen4 cache fields')
            batch,position = header['batch'],header['position']
            if type(batch) is not int or not 0<=batch<=128 or type(position) is not int or not 0<=position<=model.config.max_position_embeddings or bool(batch)!=bool(position):
                raise ValueError('invalid Qwen4 cache batch/position')
            expected = _schema(model,batch,position)
            if _json(header)!=_json(expected):
                raise ValueError('Qwen4 cache identity/geometry mismatch')
            payload = _size(expected)
            if 44+length+payload>max_bytes or model.weight_bytes+model._expert_bytes+model._snapshot_bytes+model._state_bytes()+2*payload+3*1024**2>model.max_bytes:
                raise MemoryError('Qwen4 cache restore staging exceeds admission')
            banks = [[] for _ in expected['banks']]
            lexical = [None if bank is None else [] for bank in expected['lexical']]
            positions = None
            digest = hashlib.sha256(prefix+encoded)
            for kind,index,slot,shape,dtype in _entries(expected):
                model._cancel()
                host = np.empty(shape,dtype=dtype)
                flat,item = host.reshape(-1),host.dtype.itemsize
                block = 1024**2//item
                for offset in range(0,flat.size,block):
                    model._cancel()
                    count = min(block,flat.size-offset)
                    data = _read(source,count*item);digest.update(data)
                    value = np.frombuffer(data,dtype=dtype)
                    if not np.isfinite(value).all():
                        raise ValueError('nonfinite Qwen4 continuation')
                    if dtype=='<i8' and (np.any(value<0) or np.any(value>=(model.config.max_position_embeddings if kind=='positions' else model.config.vocab_size))):
                        raise ValueError('invalid Qwen4 lexical/position state')
                    flat[offset:offset+count] = value
                value = model.torch.tensor(host,device=model.device,dtype=model.torch.int64 if dtype=='<i8' else model.torch.float32)
                if kind=='positions':
                    positions = value
                else:
                    (banks if kind=='layer' else lexical)[index].append(value)
                del flat,host
            mask_data = _read(source,batch*position);digest.update(mask_data)
            if any(value>1 for value in mask_data):
                raise ValueError('invalid Qwen4 occupancy mask')
            mask = np.frombuffer(mask_data,dtype=np.uint8).reshape(batch,position).copy() if batch else None
            if not hmac.compare_digest(digest.digest(),_read(source,32)) or source.read(1)!=b'':
                raise ValueError('Qwen4 cache checksum/EOF mismatch')
            model._cancel()
            model._cache,model._ple,model._positions,model._mask,model.position = tuple(tuple(bank) for bank in banks),tuple(None if bank is None else tuple(bank) for bank in lexical),positions,mask,position
            return position
        finally:
            model._busy = False
