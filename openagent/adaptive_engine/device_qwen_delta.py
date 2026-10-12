from .device_publisher_projections import device_projection
"""Qwen3.5 device gated-delta mixing with owned pending recurrent state."""
import math


def qwen35_device_delta(torch,hidden,weights,config,*,prefix,previous,mask,
                        max_scratch_bytes,cancel=None,output_gate='silu'):
    """Execute the exact admitted key-by-value recurrent contract on device.

    Padding masks the mixer input and occupies physical convolution slots.
    Previous banks stay unchanged; the caller publishes returned banks only
    after the complete decoder request succeeds.
    """
    c = config
    if output_gate not in ('silu','sigmoid'):
        raise ValueError('explicit supported device delta output gate required')
    if hidden.ndim!=3 or hidden.dtype!=torch.float32 or hidden.shape[-1]!=c.hidden_size:
        raise ValueError('FP32 device Qwen3.5 hidden sequence required')
    batch,tokens,_ = hidden.shape
    if not batch or not tokens or mask.shape!=(batch,tokens) or mask.device!=hidden.device or mask.dtype!=torch.bool:
        raise ValueError('matching nonempty device mixer occupancy required')
    if type(max_scratch_bytes) is not int or max_scratch_bytes<=0:
        raise ValueError('explicit gated-delta device scratch admission required')
    if cancel is not None and not callable(getattr(cancel,'is_set',None)):
        raise ValueError('gated-delta cancellation must provide is_set')
    key_width = c.linear_num_key_heads*c.linear_key_head_dim
    value_width = c.linear_num_value_heads*c.linear_value_head_dim
    width = 2*key_width+value_width
    history_shape = (batch,width,c.linear_conv_kernel_dim)
    state_shape = (batch,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim)
    state_bytes = 4*batch*(width*c.linear_conv_kernel_dim+c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim)
    work = 2*state_bytes+batch*tokens*(12*width+32*value_width+12*c.hidden_size+24*c.linear_num_value_heads)
    work += batch*c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim*16
    if work>max_scratch_bytes:
        raise MemoryError('gated-delta device state/work admission exceeded')
    a = prefix+'linear_attn.'
    expected = {'in_proj_qkv.weight':(width,c.hidden_size),
        'in_proj_z.weight':(value_width,c.hidden_size),
        'in_proj_b.weight':(c.linear_num_value_heads,c.hidden_size),
        'in_proj_a.weight':(c.linear_num_value_heads,c.hidden_size),
        'conv1d.weight':(width,1,c.linear_conv_kernel_dim),
        'norm.weight':(c.linear_value_head_dim,),
        'dt_bias':(c.linear_num_value_heads,), 'A_log':(c.linear_num_value_heads,),
        'out_proj.weight':(c.hidden_size,value_width)}
    for name,shape in expected.items():
        value = weights.get(a+name)
        if value is None or tuple(value.shape)!=shape or value.dtype!=torch.float32 or value.device!=hidden.device:
            raise ValueError('exact Qwen3.5 device mixer weight required: '+name)
    if previous is not None:
        if type(previous) is not tuple or len(previous)!=2:
            raise ValueError('complete device convolution/recurrent banks required')
        for value,shape in zip(previous,(history_shape,state_shape)):
            if tuple(value.shape)!=shape or value.dtype!=torch.float32 or value.device!=hidden.device:
                raise ValueError('device recurrent continuation geometry mismatch')
    def check():
        if cancel is not None and cancel.is_set():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('Qwen3.5 device recurrent execution cancelled')
    check()
    with torch.inference_mode():
        x = hidden*mask[...,None]
        projected = device_projection(torch,x,weights[a+'in_proj_qkv.weight'],cancel=cancel)
        history = torch.zeros(history_shape,device=x.device,dtype=torch.float32) if previous is None else previous[0].clone()
        state = torch.zeros(state_shape,device=x.device,dtype=torch.float32) if previous is None else previous[1].clone()
        convolved = torch.empty_like(projected)
        kernel = weights[a+'conv1d.weight'][:,0]
        for token in range(tokens):
            check()
            history = torch.cat((history[...,1:],projected[:,token,:,None]),dim=-1)
            convolved[:,token] = torch.nn.functional.silu((history*kernel).sum(dim=-1))
        q,k,v = convolved.split((key_width,key_width,value_width),dim=-1)
        q,k = (value.reshape(batch,tokens,c.linear_num_key_heads,c.linear_key_head_dim) for value in (q,k))
        q = q*torch.rsqrt(q.square().sum(dim=-1,keepdim=True)+1e-6)
        k = k*torch.rsqrt(k.square().sum(dim=-1,keepdim=True)+1e-6)
        repeat = c.linear_num_value_heads//c.linear_num_key_heads
        q = q.repeat_interleave(repeat,dim=2)/math.sqrt(c.linear_key_head_dim)
        k = k.repeat_interleave(repeat,dim=2)
        v = v.reshape(batch,tokens,c.linear_num_value_heads,c.linear_value_head_dim)
        beta = (device_projection(torch,x,weights[a+'in_proj_b.weight'],cancel=cancel)).sigmoid()
        decay = (-weights[a+'A_log'].exp()*torch.nn.functional.softplus(
            device_projection(torch,x,weights[a+'in_proj_a.weight'],cancel=cancel)+weights[a+'dt_bias'])).exp()
        output = torch.empty_like(v)
        for token in range(tokens):
            check()
            state = state*decay[:,token,:,None,None]
            prediction = torch.einsum('bhk,bhkv->bhv',k[:,token],state)
            error = (v[:,token]-prediction)*beta[:,token,:,None]
            state = state+k[:,token,:,:,None]*error[:,:,None,:]
            output[:,token] = torch.einsum('bhk,bhkv->bhv',q[:,token],state)
        gate = (device_projection(torch,x,weights[a+'in_proj_z.weight'],cancel=cancel)).reshape(output.shape)
        normalized = output*torch.rsqrt(output.square().mean(dim=-1,keepdim=True)+c.rms_norm_eps)
        normalized = normalized*weights[a+'norm.weight']
        gating = gate.sigmoid() if output_gate=='sigmoid' else torch.nn.functional.silu(gate)
        result = device_projection(torch,(normalized*gating).reshape(batch,tokens,value_width),weights[a+'out_proj.weight'],cancel=cancel)
        check()
        if any(not bool(torch.isfinite(value).all().item()) for value in (result,history,state)):
            raise ArithmeticError('nonfinite Qwen3.5 device recurrent result/state')
        return result,(history,state)


__all__ = ['qwen35_device_delta']
