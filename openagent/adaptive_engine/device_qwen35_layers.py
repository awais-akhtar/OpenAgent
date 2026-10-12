from .device_publisher_projections import device_projection
"""Family-specific Qwen3.5 device attention and selected/shared expert math."""
import math
from .device_attention import device_attention


def _cancel(cancel):
    if cancel is not None and cancel.is_set():
        from .execution_service import ExecutionCancelled
        raise ExecutionCancelled('Qwen3.5 device layer cancelled')


def _weights(torch,weights,prefix,shapes,device):
    for name,shape in shapes.items():
        value = weights.get(prefix+name)
        if value is None or tuple(value.shape)!=shape or value.dtype!=torch.float32 or value.device!=device:
            raise ValueError('exact Qwen3.5 device layer tensor required: '+prefix+name)


def qwen35_device_attention(torch,hidden,weights,c,*,prefix,previous,positions,mask,
                            start,rotary_dim,theta,max_scratch_bytes,cancel=None):
    batch,tokens,width = hidden.shape
    spatial=tuple(positions.shape)==(3,batch,tokens)
    section=c.rope_parameters.get('mrope_section')
    if (hidden.dtype!=torch.float32 or width!=c.hidden_size or
            (tuple(positions.shape)!=(batch,tokens) and not (spatial and section is not None)) or
            positions.dtype!=torch.int64 or positions.device!=hidden.device):
        raise ValueError('matching Qwen3.5 device hidden/rotary inputs required')
    if spatial and (type(section) not in (list,tuple) or len(section)!=3 or
            any(type(value) is not int or value<0 for value in section) or sum(section)!=rotary_dim//2):
        raise ValueError('complete interleaved Qwen3.5 MRoPE section contract required')
    if type(start) is not int or start<0 or mask.shape!=(batch,start+tokens):
        raise ValueError('complete Qwen3.5 logical attention mask required')
    if type(rotary_dim) is not int or rotary_dim%2 or not 2<=rotary_dim<=c.head_dim:
        raise ValueError('exact partial rotary dimension required')
    a = prefix+'self_attn.'
    heads,kv,dim = c.num_attention_heads,c.num_key_value_heads,c.head_dim
    _weights(torch,weights,a,{'q_proj.weight':(2*heads*dim,width),'k_proj.weight':(kv*dim,width),
        'v_proj.weight':(kv*dim,width),'o_proj.weight':(width,heads*dim),
        'q_norm.weight':(dim,),'k_norm.weight':(dim,)},hidden.device)
    if previous is not None and (type(previous) is not tuple or len(previous)!=2 or any(
            tuple(value.shape)!=(batch,kv,start,dim) or value.dtype!=torch.float32 or value.device!=hidden.device
            for value in previous)):
        raise ValueError('complete Qwen3.5 device KV continuation required')
    workspace = batch*tokens*(width+6*heads*dim+4*kv*dim+4*rotary_dim)*4+batch*kv*(start+tokens)*dim*8
    if type(max_scratch_bytes) is not int or workspace>=max_scratch_bytes:
        raise MemoryError('Qwen3.5 device projection/KV scratch exceeds admission')
    _cancel(cancel)
    with torch.inference_mode():
        query_gate = (device_projection(torch,hidden,weights[a+'q_proj.weight'],cancel=cancel)).reshape(batch,tokens,heads,2*dim)
        q,gate = query_gate.chunk(2,dim=-1)
        q = q*torch.rsqrt(q.square().mean(dim=-1,keepdim=True)+c.rms_norm_eps)*(1+weights[a+'q_norm.weight'])
        k = (device_projection(torch,hidden,weights[a+'k_proj.weight'],cancel=cancel)).reshape(batch,tokens,kv,dim)
        k = k*torch.rsqrt(k.square().mean(dim=-1,keepdim=True)+c.rms_norm_eps)*(1+weights[a+'k_norm.weight'])
        v = (device_projection(torch,hidden,weights[a+'v_proj.weight'],cancel=cancel)).reshape(batch,tokens,kv,dim).transpose(1,2)
        frequencies = theta**(-torch.arange(0,rotary_dim,2,device=hidden.device,dtype=torch.float32)/rotary_dim)
        if spatial:
            axes=torch.zeros(rotary_dim//2,dtype=torch.int64,device=hidden.device)
            axes[1:3*section[1]:3]=1
            axes[2:3*section[2]:3]=2
            selected=positions.permute(1,2,0).index_select(-1,axes)
            angles=selected.to(torch.float32)*frequencies
        else:
            angles = positions.to(torch.float32)[...,None]*frequencies
        cosine,sine = angles.cos()[:,:,None],angles.sin()[:,:,None]
        def rotate(value):
            left,right = value[...,:rotary_dim//2],value[...,rotary_dim//2:rotary_dim]
            return torch.cat((left*cosine-right*sine,right*cosine+left*sine,value[...,rotary_dim:]),dim=-1).transpose(1,2)
        q,k = rotate(q),rotate(k)
        if previous is not None:
            k,v = torch.cat((previous[0],k),dim=2),torch.cat((previous[1],v),dim=2)
        context = device_attention(torch,q,k,v,key_mask=mask,query_start=start,key_start=0,
            window=None,max_scratch_bytes=max_scratch_bytes-workspace,cancel=cancel)
        context = context.transpose(1,2)*gate.sigmoid()
        result = device_projection(torch,context.reshape(batch,tokens,heads*dim),weights[a+'o_proj.weight'],cancel=cancel)
        _cancel(cancel)
        return result,(k,v)


def qwen35_device_moe(torch,hidden,weights,c,*,prefix,max_scratch_bytes,cancel=None,
                      expert_provider=None,layer_index=None,normalize_routes=True,
                      routing_observer=None):
    if hidden.ndim!=3 or hidden.dtype!=torch.float32 or hidden.shape[-1]!=c.hidden_size:
        raise ValueError('FP32 device Qwen3.5 MoE rows required')
    h,e,m,s = c.hidden_size,c.num_experts,c.moe_intermediate_size,c.shared_expert_intermediate_size
    if type(normalize_routes) is not bool:
        raise ValueError('explicit device routing normalization policy required')
    if routing_observer is not None and not callable(routing_observer):
        raise TypeError('callable device routing observer required')
    a = prefix+'mlp.'
    shapes = {'gate.weight':(e,h),'experts.gate_up_proj':(e,2*m,h),
        'experts.down_proj':(e,h,m),'shared_expert.gate_proj.weight':(s,h),
        'shared_expert.up_proj.weight':(s,h),'shared_expert.down_proj.weight':(h,s),
        'shared_expert_gate.weight':(1,h)}
    if expert_provider is not None:
        shapes.pop('experts.gate_up_proj');shapes.pop('experts.down_proj')
        if type(layer_index) is not int or not 0<=layer_index<c.num_hidden_layers:
            raise ValueError('original Qwen3.5 provider layer required')
    _weights(torch,weights,a,shapes,hidden.device)
    rows = hidden.numel()//h
    scratch = rows*(24*h+24*m+24*s+16*e+24*c.num_experts_per_tok)
    if type(max_scratch_bytes) is not int or scratch>max_scratch_bytes:
        raise MemoryError('Qwen3.5 device routed/shared expert scratch exceeds admission')
    _cancel(cancel)
    with torch.inference_mode():
        flat = hidden.reshape(rows,h)
        logits = device_projection(torch,flat,weights[a+'gate.weight'],cancel=cancel)
        probabilities = logits.softmax(dim=-1)
        selected = torch.argsort(probabilities,descending=True,stable=True,dim=-1)[:,:c.num_experts_per_tok]
        scales = probabilities.gather(-1,selected)
        if normalize_routes:
            scales = scales/scales.sum(dim=-1,keepdim=True)
        if routing_observer is not None:
            _cancel(cancel)
            routing_observer(logits, selected, scales)
            _cancel(cancel)
        output = torch.zeros_like(flat)
        for expert in range(e):
            _cancel(cancel)
            locations = torch.nonzero(selected==expert,as_tuple=False)
            if not locations.numel():
                continue
            indices,ranks = locations[:,0],locations[:,1]
            if expert_provider is None:
                gate,up = (flat[indices]@weights[a+'experts.gate_up_proj'][expert].T).chunk(2,dim=-1)
                value = (torch.nn.functional.silu(gate)*up)@weights[a+'experts.down_proj'][expert].T
            else:
                from .device_qwen4_publisher_experts import DeviceQwen4PublisherExperts
                from .torch_mixtral_experts import TorchMixtralExperts
                options={'cancel':cancel} if isinstance(expert_provider,(DeviceQwen4PublisherExperts,TorchMixtralExperts)) else {}
                value = expert_provider.execute(layer_index,expert,flat[indices],torch=torch,device=str(hidden.device),**options)
            output.index_add_(0,indices,value*scales[indices,ranks,None])
        gate = device_projection(torch,flat,weights[a+'shared_expert.gate_proj.weight'],cancel=cancel)
        up = device_projection(torch,flat,weights[a+'shared_expert.up_proj.weight'],cancel=cancel)
        shared = device_projection(torch,torch.nn.functional.silu(gate)*up,weights[a+'shared_expert.down_proj.weight'],cancel=cancel)
        shared *= (device_projection(torch,flat,weights[a+'shared_expert_gate.weight'],cancel=cancel)).sigmoid()
        result = (output+shared).reshape(hidden.shape)
        _cancel(cancel)
        if not bool(torch.isfinite(result).all().item()):
            raise ArithmeticError('nonfinite Qwen3.5 device MoE output')
        return result


__all__ = ['qwen35_device_attention','qwen35_device_moe']
