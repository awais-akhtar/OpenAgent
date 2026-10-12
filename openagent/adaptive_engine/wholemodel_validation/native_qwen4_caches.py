"""Bounded observations of actual Qwen4 public native continuation frames."""
import hashlib
import hmac
import struct
import numpy as np


def _inventory(config,position):
    width=2*config.linear_num_key_heads*config.linear_key_head_dim+config.linear_num_value_heads*config.linear_value_head_dim
    for layer,kind in enumerate(config.layer_types):
        if kind=='linear_attention':
            yield layer,'convolution','<f4',(1,width,config.linear_conv_kernel_dim)
            yield layer,'recurrent','<f4',(1,config.linear_num_value_heads,config.linear_key_head_dim,config.linear_value_head_dim)
        else:
            for name in ('key','value'):
                yield layer,name,'<f4',(1,config.num_key_value_heads,position,config.head_dim)
            yield layer,'indexer','<f4',(1,position,config.indexer_head_dim)
        if layer+1 in config.ple_layer_ids:
            yield layer,'ple_convolution','<f4',(1,config.hc_count*config.hidden_size,
                (config.ple_conv_kernel_size-1)*config.ngram_size)
            yield layer,'ple_tokens','<i8',(1,config.ngram_size-1)


def read_public_qwen4_cache(model,*,position,max_bytes,include_auxiliary=False):
    from ..native_qwen4_public_text import NativeQwen4PublicTextDecoder
    if type(model) is not NativeQwen4PublicTextDecoder or type(position) is not int or position<=0:
        raise ValueError('occupied exact public Qwen4 cache owner required')
    if type(include_auxiliary) is not bool:raise ValueError('explicit public auxiliary export policy required')
    config=model.config;inventory=tuple(_inventory(config,position))
    total=sum(int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize for _,_,dtype,shape in inventory)
    total+=position*(1+8*(3 if model.wire.mrope else 1))
    if type(max_bytes) is not int or 3*total+1024**2>max_bytes:
        raise MemoryError('complete public Qwen4 cache observation exceeds admission')
    result={'position':np.asarray(position,dtype=np.int64)}
    with model._lock,model.execution._lock:
        model._check();execution=model.execution;execution._check()
        if position!=execution.position or position>model.max_tokens:raise ValueError('Qwen4 observation position differs')
        states=execution._capture()
        try:
            for state,(begin,end) in zip(states,model.layer_ranges,strict=True):
                stream=state.file;stream.seek(0);header=stream.read(128)
                if len(header)!=128 or header[:8]!=b'OAQ4PUB1':raise ValueError('public Qwen4 cache header differs')
                observed,payload,first,last=struct.unpack_from('<QQQQ',header,8)
                capacity,width,mrope=struct.unpack_from('<QQQ',header,104)
                leaves=[item for item in inventory if begin<=item[0]<end]
                leaf_bytes=sum(int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize for _,_,dtype,shape in leaves)
                tail=position*(1+8*(3 if model.wire.mrope else 1))
                if ((observed,first,last)!=(position,begin,end) or header[40:104]!=model.owner.identity.encode('ascii')
                        or (capacity,width,mrope)!=(model.max_tokens,model.wire.width,int(model.wire.mrope))
                        or payload!=leaf_bytes+tail or state.size!=160+payload):
                    raise ValueError('public Qwen4 cache inventory differs')
                inner=hashlib.sha256(header);outer=hashlib.sha256(header);remaining=payload
                while remaining:
                    data=stream.read(min(remaining,256*1024))
                    if not data:raise ValueError('public Qwen4 cache truncated')
                    remaining-=len(data);inner.update(data);outer.update(data)
                trailer=stream.read(32);outer.update(trailer)
                if not hmac.compare_digest(trailer,inner.digest()) or not hmac.compare_digest(state.digest,outer.digest()) or stream.read(1):
                    raise ValueError('public Qwen4 cache ownership checksum differs')
                stream.seek(128)
                for layer,name,dtype,shape in leaves:
                    count=int(np.prod(shape,dtype=object));raw=stream.read(count*np.dtype(dtype).itemsize)
                    if len(raw)!=count*np.dtype(dtype).itemsize:raise ValueError('public Qwen4 cache leaf truncated')
                    value=np.frombuffer(raw,dtype=dtype).reshape(shape)
                    if dtype=='<f4' and not np.isfinite(value).all():raise ValueError('nonfinite public Qwen4 cache leaf')
                    if dtype=='<i8' and (np.any(value<0) or np.any(value>=config.vocab_size)):
                        raise ValueError('public Qwen4 lexical history outside vocabulary')
                    result[f'layer.{layer}.{name}']=value
                mask=np.frombuffer(stream.read(position),dtype=np.uint8)
                raw=stream.read(position*8*(3 if model.wire.mrope else 1))
                positions=np.frombuffer(raw,dtype='<i8')
                if len(mask)!=position or positions.size!=position*(3 if model.wire.mrope else 1):
                    raise ValueError('public Qwen4 auxiliary continuation truncated')
                if include_auxiliary:
                    if np.any(mask>1) or np.any(positions<0) or np.any(positions>=model.max_tokens):
                        raise ValueError('invalid occupied public Qwen4 mask/rotary positions')
                    mask=mask.reshape(1,position)
                    positions=positions.reshape(3,1,position) if model.wire.mrope else positions.reshape(1,position)
                    for name,value in (('mask',mask),('positions',positions)):
                        if name in result and not np.array_equal(result[name],value):
                            raise ValueError('public Qwen4 ranges disagree on auxiliary continuation')
                        result[name]=value
                elif not np.all(mask==1) or not np.all(positions.reshape(-1,position)==np.arange(position)):
                    raise ValueError('comparison declares unmasked contiguous Qwen4 text')
        finally:execution._dispose(states)
    return result


__all__=['read_public_qwen4_cache']
