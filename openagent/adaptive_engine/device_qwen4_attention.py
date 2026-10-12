from .device_publisher_projections import device_projection
"""Qwen4 compressed-indexer sparse attention on an explicit device."""
import math

from .device_attention import device_attention
from .device_qwen35_layers import _cancel,_weights


def qwen4_device_qsa(torch,hidden,weights,c,*,prefix,previous,positions,
                     full_positions,mask,start,rotary_dim,theta,rope_scale,
                     max_scratch_bytes,cancel=None,mrope_section=None):
    batch,tokens,width = hidden.shape
    heads,kv,dim = c.num_attention_heads,c.num_key_value_heads,c.head_dim
    end = start+tokens
    if hidden.dtype!=torch.float32 or width!=c.hidden_size or type(start) is not int or start<0:
        raise ValueError('exact Qwen4 device sparse input required')
    if mask.shape!=(batch,end) or mask.dtype!=torch.bool or mask.device!=hidden.device:
        raise ValueError('complete Qwen4 device sparse occupancy required')
    for value,shape in ((positions,(batch,tokens)),(full_positions,(batch,end))):
        if mrope_section is not None:
            shape=(3,*shape)
        if value.shape!=shape or value.device!=hidden.device or value.dtype!=torch.int64:
            raise ValueError('complete device text positions required')
    if type(rotary_dim) is not int or rotary_dim%2 or not 2<=rotary_dim<=min(dim,c.indexer_head_dim):
        raise ValueError('exact Qwen4 partial rotary geometry required')
    if not math.isfinite(theta) or theta<=0 or not math.isfinite(rope_scale) or rope_scale<=0:
        raise ValueError('positive Qwen4 rotary policy required')
    a = prefix+'self_attn.'
    _weights(torch,weights,a,{'q_proj.weight':(2*heads*dim,width),
        'k_proj.weight':(kv*dim,width),'v_proj.weight':(kv*dim,width),'o_proj.weight':(width,heads*dim),
        'q_norm.weight':(dim,),'k_norm.weight':(dim,),
        'indexer.index_qk_proj.weight':((c.indexer_n_heads+1)*c.indexer_head_dim,width),
        'indexer.q_layernorm.weight':(c.indexer_head_dim,),
        'indexer.k_layernorm.weight':(c.indexer_head_dim,)},hidden.device)
    if previous is not None:
        shapes = ((batch,kv,start,dim),(batch,kv,start,dim),(batch,start,c.indexer_head_dim))
        if type(previous) is not tuple or len(previous)!=3 or any(tuple(value.shape)!=shape or value.dtype!=torch.float32 or value.device!=hidden.device for value,shape in zip(previous,shapes)):
            raise ValueError('complete Qwen4 device KV/indexer continuation required')
    # Reserve full pending banks plus one bounded selected-key query operation.
    fixed = batch*end*(16*kv*dim+8*c.indexer_head_dim)+batch*tokens*(48*heads*dim+16*width+16*c.indexer_n_heads*c.indexer_head_dim)
    fixed += end*(32+24*c.indexer_head_dim+16*c.indexer_n_heads)+batch*tokens*heads*dim*4
    if type(max_scratch_bytes) is not int or fixed>=max_scratch_bytes:
        raise MemoryError('Qwen4 device sparse projection/indexer exceeds admission')
    def norm(value,name):
        return value*torch.rsqrt(value.square().mean(dim=-1,keepdim=True)+c.rms_norm_eps)*(1+weights[a+name])
    def rotate(value,pos):
        frequency = theta**(-torch.arange(0,rotary_dim,2,device=hidden.device,dtype=torch.float32)/rotary_dim)
        angle = pos.to(torch.float32)[...,None]/rope_scale*frequency
        if mrope_section is not None:
            mixed=angle[0].clone()
            for axis in (1,2):
                lanes=slice(axis,mrope_section[axis]*3,3)
                mixed[...,lanes]=angle[axis,...,lanes]
            angle=mixed
        cosine,sine = angle.cos()[...,None,:],angle.sin()[...,None,:]
        left,right = value[...,:rotary_dim//2],value[...,rotary_dim//2:rotary_dim]
        return torch.cat((left*cosine-right*sine,right*cosine+left*sine,value[...,rotary_dim:]),dim=-1)
    _cancel(cancel)
    with torch.inference_mode():
        indexed = device_projection(torch,hidden,weights[a+'indexer.index_qk_proj.weight'],cancel=cancel)
        query_width = c.indexer_n_heads*c.indexer_head_dim
        iq = indexed[...,:query_width].reshape(batch,tokens,c.indexer_n_heads,c.indexer_head_dim)
        iq = rotate(norm(iq,'indexer.q_layernorm.weight'),positions)
        raw = indexed[...,query_width:]
        if previous is not None:
            raw = torch.cat((previous[2],raw),dim=1)
        qg = (device_projection(torch,hidden,weights[a+'q_proj.weight'],cancel=cancel)).reshape(batch,tokens,heads,2*dim)
        q,gate = qg.chunk(2,dim=-1)
        q = rotate(norm(q,'q_norm.weight'),positions).transpose(1,2)
        k = (device_projection(torch,hidden,weights[a+'k_proj.weight'],cancel=cancel)).reshape(batch,tokens,kv,dim)
        k = rotate(norm(k,'k_norm.weight'),positions).transpose(1,2)
        v = (device_projection(torch,hidden,weights[a+'v_proj.weight'],cancel=cancel)).reshape(batch,tokens,kv,dim).transpose(1,2)
        if previous is not None:
            k,v = torch.cat((previous[0],k),dim=2),torch.cat((previous[1],v),dim=2)
        context = torch.empty((batch,heads,tokens,dim),device=hidden.device,dtype=torch.float32)
        ratio,budget = c.indexer_compress_ratio,c.indexer_budget//c.indexer_compress_ratio
        for row in range(batch):
            for step in range(tokens):
                _cancel(cancel)
                visible = torch.nonzero(mask[row,:start+step+1],as_tuple=False).flatten()
                if visible.numel()==0:
                    raise ValueError('Qwen4 sparse query has no visible keys')
                blocks = visible.numel()//ratio
                selected = visible[blocks*ratio:]
                if blocks:
                    groups = visible[:blocks*ratio].reshape(blocks,ratio)
                    pooled = raw[row,groups].mean(dim=1)
                    keys = norm(pooled,'indexer.k_layernorm.weight')
                    selected_positions=(full_positions[:,row:row+1,groups[:,0]] if mrope_section is not None
                        else full_positions[row,groups[:,0]][None])
                    keys = rotate(keys[None,:,None,:],selected_positions)[0,:,0]
                    scores = (iq[row,step]@keys.T).clamp_min(0).sum(dim=0)/math.sqrt(c.indexer_head_dim)
                    chosen = torch.argsort(scores,descending=True,stable=True)[:min(budget,blocks)]
                    selected = torch.cat((groups[chosen].flatten(),selected))
                if selected.numel()==0:
                    raise ValueError('Qwen4 sparse selection is empty')
                # Selection is causal already; compact keys use their own physical offsets.
                count = selected.numel()
                value = device_attention(torch,q[row:row+1,:,step:step+1],
                    k[row:row+1,:,selected],v[row:row+1,:,selected],
                    key_mask=torch.ones((1,count),device=hidden.device,dtype=torch.bool),
                    query_start=count-1,key_start=0,window=None,
                    max_scratch_bytes=max_scratch_bytes-fixed,cancel=cancel)
                context[row,:,step] = value[0,:,0]
        context = context.transpose(1,2)*gate.sigmoid()
        result = device_projection(torch,context.reshape(batch,tokens,heads*dim),weights[a+'o_proj.weight'],cancel=cancel)
        _cancel(cancel)
        if any(not bool(torch.isfinite(value).all()) for value in (result,k,v,raw)):
            raise ArithmeticError('nonfinite Qwen4 device sparse result/state')
        return result,(k,v,raw)
