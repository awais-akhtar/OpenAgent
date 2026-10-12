"""Qwen4 device hyperconnections and lexical continuation operations."""
import math

from .device_qwen35_layers import _cancel,_weights
from .device_publisher_projections import device_projection


def qwen4_device_mix(torch,hidden,weights,c,*,prefix,inject,max_scratch_bytes,cancel=None):
    width = c.hc_count*c.hidden_size
    if hidden.ndim!=3 or hidden.shape[-1]!=width or hidden.dtype!=torch.float32:
        raise ValueError('exact Qwen4 device hyper-channel input required')
    shapes = {'hc_norm.weight':(width,),'input_mix_weight_down.weight':(c.hc_lowrank,width),
              'input_mix_weight_up.weight':(width,c.hc_lowrank)}
    if inject:
        shapes['block_inject_weight.weight'] = (c.hc_count,width)
    _weights(torch,weights,prefix,shapes,hidden.device)
    if type(max_scratch_bytes) is not int or hidden.shape[0]*hidden.shape[1]*(32*width+16*c.hc_lowrank)>max_scratch_bytes:
        raise MemoryError('Qwen4 device hyperconnection scratch exceeds admission')
    _cancel(cancel)
    with torch.inference_mode():
        grouped = hidden.reshape(*hidden.shape[:-1],c.hc_count,c.hidden_size)
        normalized = grouped*torch.rsqrt(grouped.square().mean(dim=-1,keepdim=True)+c.rms_norm_eps)
        normalized *= 1+weights[prefix+'hc_norm.weight'].reshape(c.hc_count,c.hidden_size)
        flat = normalized.reshape(hidden.shape)
        low = torch.nn.functional.silu(device_projection(torch,flat,weights[prefix+'input_mix_weight_down.weight'],cancel=cancel)/c.hc_count)
        gates = device_projection(torch,low,weights[prefix+'input_mix_weight_up.weight'],cancel=cancel).sigmoid().reshape(normalized.shape)
        mixed = (gates*normalized).mean(dim=-2)
        injection = 2*(device_projection(torch,flat,weights[prefix+'block_inject_weight.weight'],cancel=cancel)/c.hc_count).sigmoid() if inject else None
        _cancel(cancel)
        return mixed,injection


def qwen4_device_inject(hidden,value,gates,c):
    if value.shape!=(*hidden.shape[:-1],c.hidden_size) or gates.shape!=(*hidden.shape[:-1],c.hc_count):
        raise ValueError('matching Qwen4 hyperconnection injection required')
    return hidden+(value[...,None,:]*gates[...,:,None]).reshape(hidden.shape)


def qwen4_device_ple(torch,hidden,ids,weights,c,*,prefix,previous,mask,max_scratch_bytes,cancel=None):
    """Keep hashed token history and dilated convolution state on device."""
    batch,tokens,width = hidden.shape
    if hidden.dtype!=torch.float32 or width!=c.hc_count*c.hidden_size or ids.shape!=(batch,tokens) or mask.shape!=ids.shape:
        raise ValueError('exact Qwen4 lexical input geometry required')
    if ids.device!=hidden.device or ids.dtype!=torch.int64 or mask.device!=hidden.device or mask.dtype!=torch.bool:
        raise ValueError('matching device integer token IDs and occupancy required')
    p = prefix+'ple.'
    lane = p+'ple_embedding.'
    lanes = (c.ngram_size-1)*c.heads_per_ngram
    shapes = {'key_proj.weight':(width,c.ple_embed_dim),'value_proj.weight':(c.hidden_size,c.ple_embed_dim),
        'norm_key.weight':(width,),'norm_query.weight':(width,),'norm_conv.weight':(width,),
        'conv1d.weight':(width,1,c.ple_conv_kernel_size)}
    _weights(torch,weights,p,shapes,hidden.device)
    for name,shape in (('layer_multipliers',(c.ngram_size,)),('ngram_heads_vocab_sizes',(lanes,)),('ngram_heads_offsets',(lanes,))):
        value = weights.get(lane+name)
        if value is None or tuple(value.shape)!=shape or value.dtype!=torch.int64 or value.device!=hidden.device:
            raise ValueError('exact Qwen4 device hash metadata required')
    embedding = weights.get(lane+'ngram_embedding.weight')
    if embedding is None or embedding.ndim!=2 or embedding.shape[1]*lanes!=c.ple_embed_dim or embedding.dtype!=torch.float32 or embedding.device!=hidden.device:
        raise ValueError('exact Qwen4 lexical embedding required')
    span = (c.ple_conv_kernel_size-1)*c.ngram_size
    history_shape = (batch,width,span)
    if previous is not None and (type(previous) is not tuple or len(previous)!=2 or
            tuple(previous[0].shape)!=(batch,c.ngram_size-1) or previous[0].dtype!=torch.int64 or
            tuple(previous[1].shape)!=history_shape or previous[1].dtype!=torch.float32 or
            any(value.device!=hidden.device for value in previous)):
        raise ValueError('complete Qwen4 device lexical continuation required')
    scratch = batch*tokens*(48*width+16*c.ple_embed_dim+16*lanes)+batch*width*(span+1)*16+batch*(tokens+c.ngram_size)*16
    if type(max_scratch_bytes) is not int or scratch>max_scratch_bytes:
        raise MemoryError('Qwen4 device lexical scratch exceeds admission')
    _cancel(cancel)
    with torch.inference_mode():
        eos = c.eos_token_id[0] if isinstance(c.eos_token_id,list) else c.eos_token_id
        occupied = torch.where(mask,ids,eos)
        old_tokens = torch.full((batch,c.ngram_size-1),eos,dtype=torch.int64,device=hidden.device) if previous is None else previous[0]
        sequence = torch.cat((old_tokens,occupied),dim=1)
        hashed = torch.empty((batch,tokens,lanes),device=hidden.device,dtype=torch.int64)
        multipliers = weights[lane+'layer_multipliers']
        sizes,offsets = weights[lane+'ngram_heads_vocab_sizes'],weights[lane+'ngram_heads_offsets']
        for step in range(tokens):
            _cancel(cancel)
            here = c.ngram_size-1+step
            combined = torch.zeros(batch,device=hidden.device,dtype=torch.int64)
            for lag in range(c.ngram_size):
                source = here-lag
                valid = ~(sequence[:,source:here]==eos).any(dim=1)
                token = torch.where(valid,sequence[:,source],eos)
                combined = torch.bitwise_xor(combined,token*multipliers[lag])
                if lag:
                    start = (lag-1)*c.heads_per_ngram
                    end = start+c.heads_per_ngram
                    hashed[:,step,start:end] = combined[:,None].remainder(sizes[start:end])+offsets[start:end]
        embed = embedding[hashed].reshape(batch,tokens,c.ple_embed_dim)
        group_shape = (batch,tokens,c.hc_count,c.hidden_size)
        def norm(value,name):
            grouped = value.reshape(group_shape)
            return grouped*torch.rsqrt(grouped.square().mean(dim=-1,keepdim=True)+c.rms_norm_eps)*(1+weights[p+name+'.weight'].reshape(c.hc_count,c.hidden_size))
        key = norm(device_projection(torch,embed,weights[p+'key_proj.weight'],cancel=cancel),'norm_key')
        query = norm(hidden,'norm_query')
        score = (key*query).sum(dim=-1,keepdim=True)/math.sqrt(c.hidden_size)
        gate = score.sign()*score.abs().clamp_min(1e-6).sqrt()
        value = device_projection(torch,embed,weights[p+'value_proj.weight'],cancel=cancel)
        gated = (gate.sigmoid()*value[...,None,:]).reshape(hidden.shape)
        normalized = norm(gated,'norm_conv').reshape(hidden.shape)*mask[...,None]
        gated = gated*mask[...,None]
        history = torch.zeros(history_shape,device=hidden.device,dtype=torch.float32) if previous is None else previous[1].clone()
        output = torch.empty_like(hidden)
        convolution = weights[p+'conv1d.weight'][:,0]
        for step in range(tokens):
            _cancel(cancel)
            window = torch.cat((history,normalized[:,step,:,None]),dim=-1)
            output[:,step] = gated[:,step]+torch.nn.functional.silu((window[...,::c.ngram_size]*convolution).sum(dim=-1))
            history = window[...,1:].clone() if span else window[...,:0].clone()
        _cancel(cancel)
        if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(history).all()):
            raise ArithmeticError('nonfinite Qwen4 device lexical result')
        return output,(sequence[:,-(c.ngram_size-1):].clone(),history)
