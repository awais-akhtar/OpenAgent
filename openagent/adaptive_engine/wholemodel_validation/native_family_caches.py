"""Bounded logical observations of native public GLM/Inkling cache frames."""
import hashlib
import struct

import numpy as np


def _words(family,config,geometry,layer,capacity):
    if family=='inkling_text':
        width=geometry[layer].kv_width
        return 2*capacity*width+2*config['conv_kernel_size']*(width+config['hidden_size'])
    if config.layer_types[layer]=='linear_attention':
        return (3*config.linear_num_heads*config.linear_head_dim*config.linear_conv_kernel_dim+
                config.linear_num_heads*config.linear_head_dim**2)
    return capacity*(config.num_attention_heads*(config.qk_nope_head_dim+config.v_head_dim)+2*config.index_head_dim)


def _export_size(family,config,geometry,position):
    size=8
    if family=='inkling_text':
        for g in geometry:
            span=min(position,g.window-1) if g.window else position
            size+=4*(2*span*g.kv_width+2*config['conv_kernel_size']*(g.kv_width+config['hidden_size']))
    else:
        for layer,kind in enumerate(config.layer_types):
            if kind=='linear_attention':
                size+=4*(3*config.linear_num_heads*config.linear_head_dim*config.linear_conv_kernel_dim+
                         config.linear_num_heads*config.linear_head_dim**2)
            else:
                size+=4*position*config.num_attention_heads*(config.qk_nope_head_dim+config.v_head_dim)
                if config.indexer_types[layer]=='full':size+=4*position*(2*config.index_head_dim+1)
    return size


def _frame(state,*,family,identity,position,words):
    stream=state.file;stream.seek(0);header=stream.read(96)
    if len(header)!=96 or header[:8]!=b'OACFAM01':raise ValueError('native publisher cache frame header differs')
    observed_position,observed_words,observed_family=struct.unpack_from('<QQQ',header,8)
    if (observed_position,observed_words,observed_family)!=(position,words,1 if family=='glm5_next_text' else 2):
        raise ValueError('native publisher cache continuation or physical geometry differs')
    if header[32:]!=identity.encode('ascii') or state.size!=96+words*4+32:
        raise ValueError('native publisher cache frame identity or length differs')
    inner=hashlib.sha256(header);outer=hashlib.sha256(header);remaining=words*4
    while remaining:
        raw=stream.read(min(remaining,256*1024))
        if not raw:raise ValueError('native publisher cache payload truncated')
        inner.update(raw);outer.update(raw);remaining-=len(raw)
    trailer=stream.read(32)
    if trailer!=inner.digest() or stream.read(1):raise ValueError('native publisher cache payload checksum differs')
    outer.update(trailer)
    if state.digest!=outer.digest():raise ValueError('native publisher cache snapshot ownership checksum differs')


def _read(state,offset,count,shape):
    if type(offset) is not int or type(count) is not int or offset<0 or count<0 or 96+(offset+count)*4>state.size-32:
        raise ValueError('native publisher logical cache slice exceeds its physical frame')
    state.file.seek(96+offset*4);raw=state.file.read(count*4)
    if len(raw)!=count*4:raise ValueError('native publisher cache slice truncated')
    return np.frombuffer(raw,dtype='<f4').reshape(shape)


def read_public_cache(model,*,family,config,geometry,position,max_bytes):
    """Capture actual native state under ownership locks, then export live slices.

    Snapshot payloads remain disk-backed. The full declared logical export and
    its detached copies are admitted before any snapshot or array allocation.
    """
    if type(position) is not int or position<0 or type(max_bytes) is not int or max_bytes<=0:
        raise ValueError('bounded native publisher cache observation required')
    if 3*_export_size(family,config,geometry,position)+1024**2>max_bytes:
        raise MemoryError('native publisher logical cache copies exceed export admission')
    result={}
    with model._lock:
        model._check();execution=model.execution
        with execution._lock:
            execution._check()
            if model.family!=family or execution.position!=position or position>execution.context:
                raise ValueError('native publisher cache family or continuation differs')
            capacity=execution.context;layers=len(geometry) if geometry is not None else len(config.layer_types)
            partitions=model.layer_ranges
            if model.owner.layers!=layers or len(execution._segments)!=len(partitions):
                raise ValueError('complete native publisher layer ownership required')
            states=execution._capture();primary=None
            try:
                for state,(begin,end) in zip(states,partitions,strict=True):
                    identity=model.owner.identity
                    if begin or end!=layers:
                        identity=hashlib.sha256(identity.encode('ascii')+struct.pack('<QQ',begin,end)).hexdigest()
                    words=sum(_words(family,config,geometry,layer,capacity) for layer in range(begin,end))
                    _frame(state,family=family,identity=identity,position=position,words=words)
                    offset=0
                    for layer in range(begin,end):
                        if family=='inkling_text':
                            g=geometry[layer];width=g.kv_width;kernel=config['conv_kernel_size'];hidden=config['hidden_size']
                            span=min(position,g.window-1) if g.window else position;first=position-span
                            for name in ('key','value'):
                                value=_read(state,offset+first*width,span*width,(1,span,g.kv_heads,g.dim))
                                result[f'layer.{layer}.{name}']=value.transpose(0,2,1,3);offset+=capacity*width
                            for name,channels in (('key_fir',width),('value_fir',width),('attention_fir',hidden),('mlp_fir',hidden)):
                                value=_read(state,offset,kernel*channels,(1,kernel,channels))
                                result[f'layer.{layer}.{name}']=value.transpose(0,2,1);offset+=kernel*channels
                        elif config.layer_types[layer]=='linear_attention':
                            channels=3*config.linear_num_heads*config.linear_head_dim;kernel=config.linear_conv_kernel_dim
                            result[f'layer.{layer}.convolution']=_read(state,offset,channels*kernel,(1,channels,kernel));offset+=channels*kernel
                            count=config.linear_num_heads*config.linear_head_dim**2
                            result[f'layer.{layer}.recurrent']=_read(state,offset,count,
                                (1,config.linear_num_heads,config.linear_head_dim,config.linear_head_dim));offset+=count
                        else:
                            width=config.num_attention_heads*(config.qk_nope_head_dim+config.v_head_dim)
                            value=_read(state,offset,position*width,
                                (1,position,config.num_attention_heads,config.qk_nope_head_dim+config.v_head_dim))
                            result[f'layer.{layer}.key']=value[...,:config.qk_nope_head_dim].transpose(0,2,1,3)
                            result[f'layer.{layer}.value']=value[...,config.qk_nope_head_dim:].transpose(0,2,1,3)
                            offset+=capacity*width
                            if config.indexer_types[layer]=='full':
                                from .publisher_family_caches import _IndexerExport
                                result[f'layer.{layer}.indexer']=_IndexerExport(_read(state,offset,position*2*config.index_head_dim,
                                    (1,position,2*config.index_head_dim)))
                            offset+=capacity*2*config.index_head_dim
                    if offset!=words:raise ValueError('native publisher cache decoder did not consume its complete layer state')
            except BaseException as failure:
                primary=failure;raise
            finally:
                try:execution._dispose(states)
                except BaseException as cleanup:
                    if primary is not None:primary.add_note('native cache observation cleanup retained: '+str(cleanup))
                    else:raise
    return result


__all__=['read_public_cache']
