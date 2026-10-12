"""Whole Qwen4 device forward composition with caller-owned publication."""
from .device_qwen35_layers import _cancel,qwen35_device_moe
from .device_qwen_delta import qwen35_device_delta
from .device_qwen4_layers import qwen4_device_mix,qwen4_device_inject,qwen4_device_ple
from .device_qwen4_attention import qwen4_device_qsa
from .device_logits import device_logits


def qwen4_device_forward(torch,ids,weights,c,*,previous,previous_ple,mask,
                         positions,full_positions,start,rotary_dim,theta,rope_scale,
                         max_layer_bytes,max_head_bytes,max_host_logits_bytes,
                         expert_provider=None,cancel=None,inputs_embeds=None,mrope_section=None,
                         routing_observer=None):
    """Return logits and pending banks; never mutate a committed continuation.

    The owning decoder admits aggregate weights, old/pending banks and output
    before calling. Layer operations separately admit their working sets.
    """
    if ids.ndim!=2 or ids.dtype!=torch.int64 or not all(ids.shape):
        raise ValueError('nonempty device Qwen4 integer token sequence required')
    batch,tokens = ids.shape
    if type(start) is not int or start<0 or start+tokens>c.max_position_embeddings:
        raise ValueError('bounded Qwen4 continuation offset required')
    if mask.shape!=(batch,start+tokens) or mask.dtype!=torch.bool or mask.device!=ids.device:
        raise ValueError('complete device Qwen4 occupancy required')
    if bool((ids<0).any()) or bool((ids>=c.vocab_size).any()):
        raise ValueError('Qwen4 token ID outside vocabulary')
    for banks in (previous,previous_ple):
        if type(banks) is not tuple or len(banks)!=c.num_hidden_layers:
            raise ValueError('complete indexed Qwen4 continuation inventory required')
    if start==0 and any(bank is not None for bank in previous+previous_ple):
        raise ValueError('empty Qwen4 offset requires empty continuation')
    if start and any(bank is None for bank in previous):
        raise ValueError('missing Qwen4 layer continuation')
    for index,bank in enumerate(previous_ple):
        enabled = index+1 in c.ple_layer_ids
        if (not enabled and bank is not None) or (start and enabled and bank is None):
            raise ValueError('Qwen4 lexical continuation inventory mismatch')
    embedding = weights.get('model.embed_tokens.weight')
    head = weights.get('lm_head.weight')
    if any(value is None or tuple(value.shape)!=(c.vocab_size,c.hidden_size) or value.dtype!=torch.float32 or value.device!=ids.device for value in (embedding,head)):
        raise ValueError('exact device Qwen4 embedding/head required')
    _cancel(cancel)
    with torch.inference_mode():
        if inputs_embeds is not None and (inputs_embeds.shape!=(batch,tokens,c.hidden_size)
                or inputs_embeds.dtype!=torch.float32 or inputs_embeds.device!=ids.device):
            raise ValueError('matching Qwen4 device media embeddings required')
        hidden = (embedding[ids] if inputs_embeds is None else inputs_embeds).repeat(1,1,c.hc_count)
        pending,pending_ple = [],[]
        for index,kind in enumerate(c.layer_types):
            _cancel(cancel)
            prefix = f'model.layers.{index}.'
            lexical = None
            if index+1 in c.ple_layer_ids:
                value,lexical = qwen4_device_ple(torch,hidden,ids,weights,c,prefix=prefix,
                    previous=previous_ple[index],mask=mask[:,start:],max_scratch_bytes=max_layer_bytes,cancel=cancel)
                hidden = hidden+value
            mixed,gates = qwen4_device_mix(torch,hidden,weights,c,prefix=prefix+'attn_hyper_connection.',
                inject=True,max_scratch_bytes=max_layer_bytes,cancel=cancel)
            if kind=='linear_attention':
                value,bank = qwen35_device_delta(torch,mixed,weights,c,prefix=prefix,
                    previous=previous[index],mask=mask[:,start:],max_scratch_bytes=max_layer_bytes,
                    cancel=cancel,output_gate='sigmoid' if getattr(c,'output_gate_type',None)=='sigmoid' else 'silu')
            elif kind=='qwen_sparse_attention':
                value,bank = qwen4_device_qsa(torch,mixed,weights,c,prefix=prefix,previous=previous[index],
                    positions=positions,full_positions=full_positions,mask=mask,start=start,rotary_dim=rotary_dim,
                    theta=theta,rope_scale=rope_scale,max_scratch_bytes=max_layer_bytes,cancel=cancel,mrope_section=mrope_section)
            else:
                raise ValueError('unsupported Qwen4 device layer kind')
            hidden = qwen4_device_inject(hidden,value,gates,c)
            mixed,gates = qwen4_device_mix(torch,hidden,weights,c,prefix=prefix+'mlp_hyper_connection.',
                inject=True,max_scratch_bytes=max_layer_bytes,cancel=cancel)
            value = qwen35_device_moe(torch,mixed,weights,c,prefix=prefix,max_scratch_bytes=max_layer_bytes,
                cancel=cancel,expert_provider=expert_provider,layer_index=index,
                normalize_routes=getattr(c,'norm_topk_prob',True),
                routing_observer=None if routing_observer is None else
                    lambda raw, selected, scales, layer=index: routing_observer(layer, raw, selected, scales))
            hidden = qwen4_device_inject(hidden,value,gates,c)
            pending.append(bank);pending_ple.append(lexical)
        mixed,_ = qwen4_device_mix(torch,hidden,weights,c,prefix='model.hyper_connection_mixer.',
            inject=False,max_scratch_bytes=max_layer_bytes,cancel=cancel)
        logits = device_logits(torch,mixed,head,max_device_bytes=max_head_bytes,
            max_host_bytes=max_host_logits_bytes,cancel=cancel)
        _cancel(cancel)
        for bank in pending+pending_ple:
            if bank is not None and any(not bool(torch.isfinite(value).all()) for value in bank):
                raise ArithmeticError('nonfinite Qwen4 pending continuation')
        return logits,tuple(pending),tuple(pending_ple)
