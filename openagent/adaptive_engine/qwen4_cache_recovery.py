"""Complete bounded Qwen4 continuation transport; no token replay or pickle."""
import hashlib
import hmac
import json
import struct
from contextlib import contextmanager

import numpy as np

from .segment_snapshot import _read, _write, _pairs

_MAGIC = b'OAQ4KV01'
_CHUNK = 1024*1024
_HEADER = 65536


@contextmanager
def _access(model, limit):
    from .independent_qwen4 import IndependentQwen4TextDecoder
    from .native_qwen4 import NativeQwen4TextDecoder
    from .native_qwen4_ranges import NativeQwen4Range
    if type(model) not in (IndependentQwen4TextDecoder, NativeQwen4TextDecoder, NativeQwen4Range):
        raise TypeError('exact Qwen4 text decoder required')
    if type(limit) is not int or limit < 44:
        raise ValueError('explicit transport byte budget required')
    with model._lock:
        model._check_snapshot_access()
        model._transport = True
        try:
            yield
        finally:
            model._transport = False


def _schema(model, batch, position):
    if not batch:
        return []
    c = model.config
    width = 2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
    result = []
    for layer, kind in enumerate(c.layer_types):
        if not model.layer_begin<=layer<model.layer_end:
            continue
        shapes = ((batch,width,c.linear_conv_kernel_dim),
                  (batch,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim)) if kind == 'linear_attention' else (
                  (batch,c.num_key_value_heads,position,c.head_dim),
                  (batch,c.num_key_value_heads,position,c.head_dim),
                  (batch,position,c.indexer_head_dim))
        result.extend((f'cache/{layer}/{i}', '<f4', list(shape)) for i,shape in enumerate(shapes))
        if layer+1 in c.ple_layer_ids:
            result.extend(((f'ple/{layer}/0','<i8',[batch,c.ngram_size-1]),
                (f'ple/{layer}/1','<f4',[batch,c.hc_count*c.hidden_size,(c.ple_conv_kernel_size-1)*c.ngram_size])))
    position_shape = [3,batch,position] if model.mrope_section is not None else [batch,position]
    result.extend((('mask','|b1',[batch,position]),('positions','<i8',position_shape)))
    return result


def _array(model, name):
    if name in ('mask','positions'):
        return getattr(model,name)
    bank, layer, slot = name.split('/')
    return (model.cache if bank=='cache' else model.ple_cache)[int(layer)][int(slot)]


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')


def _validate(model, name, values):
    if values.dtype.kind == 'f' and not np.isfinite(values).all():
        raise ValueError('nonfinite Qwen4 continuation state')
    if name == 'positions' and (np.any(values<0) or np.any(values>=model.config.max_position_embeddings)):
        raise ValueError('invalid Qwen4 continuation positions')
    if name.startswith('ple/') and name.endswith('/0') and (np.any(values<0) or np.any(values>=model.config.vocab_size)):
        raise ValueError('invalid Qwen4 PLE token history')


def write_qwen4_cache(model, output, *, max_bytes):
    with _access(model,max_bytes):
        batch = 0 if model.mask is None else model.mask.shape[0]
        if (not batch and model.position) or (batch and model.mask.shape != (batch,model.position)):
            raise ValueError('inconsistent committed Qwen4 position and mask')
        schema = _schema(model,batch,model.position)
        header = dict(version=1,identity=model.identity,batch=batch,position=model.position,
                      context=model.config.max_position_embeddings,schema=schema)
        encoded = _canonical(header)
        payload = sum(int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize for _,dtype,shape in schema)
        extent = 44+len(encoded)+payload
        if len(encoded)>_HEADER or extent>max_bytes or (
                model.weight_bytes+model._snapshot_bytes+model._state_bytes()+3*_CHUNK+getattr(model,'_expert_bytes',0)+getattr(model,'_ple_bytes',0)>model.max_bytes):
            raise MemoryError('Qwen4 serialization exceeds admission')
        digest = hashlib.sha256()
        for data in (_MAGIC,struct.pack('<I',len(encoded)),encoded):
            digest.update(data);_write(output,data)
        for name,dtype,shape in schema:
            array = _array(model,name)
            if array is None or list(array.shape)!=shape or array.dtype.kind!=np.dtype(dtype).kind:
                raise ValueError('Qwen4 committed continuation schema differs')
            iterator = np.nditer(array,flags=['external_loop','buffered','zerosize_ok'],
                op_flags=['readonly'],op_dtypes=[dtype],order='C',buffersize=_CHUNK//np.dtype(dtype).itemsize)
            for values in iterator:
                _validate(model,name,values)
                data=values.tobytes();digest.update(data);_write(output,data)
        _write(output,digest.digest())
        return extent


def restore_qwen4_cache(model, source, *, max_bytes):
    with _access(model,max_bytes):
        prefix=_read(source,12)
        if prefix[:8]!=_MAGIC:
            raise ValueError('invalid Qwen4 continuation magic')
        length=struct.unpack('<I',prefix[8:])[0]
        if not 0<length<=min(_HEADER,max_bytes-44):
            raise ValueError('Qwen4 continuation header exceeds admission')
        encoded=_read(source,length)
        header=json.loads(encoded,object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if type(header) is not dict or set(header)!={'version','identity','batch','position','context','schema'}:
            raise ValueError('invalid Qwen4 continuation header')
        batch,position=header['batch'],header['position']
        if (type(header['version']) is not int or header['version']!=1 or
                header['identity']!=model.identity or type(batch) is not int or not 0<=batch<=128 or
                type(position) is not int or not 0<=position<=model.config.max_position_embeddings or
                (not batch and position) or type(header['context']) is not int or
                header['context']!=model.config.max_position_embeddings):
            raise ValueError('Qwen4 continuation identity or geometry differs')
        schema=_schema(model,batch,position)
        if _canonical(header['schema'])!=_canonical(schema):
            raise ValueError('Qwen4 continuation state inventory differs')
        payload=sum(int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize for _,dtype,shape in schema)
        if 44+length+payload>max_bytes or payload+3*_CHUNK>max_bytes or (
                model.weight_bytes+model._snapshot_bytes+model._state_bytes()+payload+3*_CHUNK+getattr(model,'_expert_bytes',0)+getattr(model,'_ple_bytes',0)>model.max_bytes):
            raise MemoryError('Qwen4 restore exceeds admission')
        digest=hashlib.sha256(prefix+encoded)
        arrays={}
        for name,dtype,shape in schema:
            array=np.empty(shape,dtype=dtype);flat=array.reshape(-1);width=array.dtype.itemsize
            for start in range(0,flat.size,_CHUNK//width):
                size=min(_CHUNK//width,flat.size-start);data=_read(source,size*width)
                digest.update(data)
                if name=='mask' and any(value>1 for value in data):
                    raise ValueError('invalid Qwen4 continuation mask bytes')
                values=np.frombuffer(data,dtype=dtype);_validate(model,name,values)
                flat[start:start+size]=values
            arrays[name]=array
        if not hmac.compare_digest(bytes(_read(source,32)),digest.digest()) or source.read(1)!=b'':
            raise ValueError('Qwen4 continuation checksum or EOF differs')
        cache=[None]*model.config.num_hidden_layers;ple=[None]*len(cache)
        if batch:
            for layer,kind in enumerate(model.config.layer_types):
                if not model.layer_begin<=layer<model.layer_end:
                    continue
                cache[layer]=tuple(arrays[f'cache/{layer}/{i}'] for i in range(2 if kind=='linear_attention' else 3))
                if layer+1 in model.config.ple_layer_ids:
                    ple[layer]=tuple(arrays[f'ple/{layer}/{i}'] for i in range(2))
        model.cache,model.ple_cache,model.mask,model.positions,model.position=(
            cache,ple,arrays.get('mask'),arrays.get('positions'),position)
        return position
