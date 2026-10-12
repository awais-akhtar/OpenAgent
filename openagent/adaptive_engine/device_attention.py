"""Bounded online-softmax grouped-query attention on an explicit device."""
import math


def device_attention(torch,query,keys,values,*,key_mask,query_start,key_start,
                     window,max_scratch_bytes,cancel=None):
    """Return [batch,query_heads,tokens,dim] without a full score allocation.

    Inputs remain caller-owned. Device products, softmax state and weighted
    values use FP32. Block reassociation differs from a monolithic softmax.
    Cancellation discards this operation's local output before publication.
    """
    if (query.ndim!=4 or keys.ndim!=4 or values.shape!=keys.shape or
            query.dtype!=torch.float32 or keys.dtype!=torch.float32 or values.dtype!=torch.float32 or
            query.device!=keys.device or query.device!=values.device or
            query.shape[0]!=keys.shape[0] or query.shape[-1]!=keys.shape[-1]):
        raise ValueError('matching FP32 device attention tensors required')
    batch,heads,tokens,dim = query.shape
    kvheads,key_count = keys.shape[1:3]
    if not all((batch,heads,tokens,dim,kvheads,key_count)) or heads%kvheads:
        raise ValueError('nonempty grouped-query device geometry required')
    if key_mask.shape!=(batch,key_count) or key_mask.device!=query.device or key_mask.dtype!=torch.bool:
        raise ValueError('matching device key occupancy required')
    if any(type(value) is not int or value<0 for value in (query_start,key_start)):
        raise ValueError('nonnegative absolute device attention offsets required')
    if window is not None and (type(window) is not int or window<=0):
        raise ValueError('positive explicit device sliding window required')
    if type(max_scratch_bytes) is not int or max_scratch_bytes<=0:
        raise ValueError('explicit attention scratch admission required')
    if cancel is not None and not callable(getattr(cancel,'is_set',None)):
        raise ValueError('device attention cancellation must provide is_set')
    output_bytes = batch*heads*tokens*dim*4
    query_block,key_block = min(tokens,128),min(key_count,256)
    def charge():
        return (output_bytes+batch*heads*query_block*(key_block*24+dim*24+64)+
                batch*heads*key_block*dim*8+(query_block+key_block)*16)
    while charge()>max_scratch_bytes and (query_block>1 or key_block>1):
        if key_block>=query_block and key_block>1:
            key_block = max(1,key_block//2)
        else:
            query_block = max(1,query_block//2)
    if charge()>max_scratch_bytes:
        raise MemoryError('one attention tile/output exceeds device scratch admission')
    def check_cancel():
        if cancel is not None and cancel.is_set():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('blockwise device attention cancelled')
    check_cancel()
    output = torch.empty_like(query)
    ratio = heads//kvheads
    with torch.inference_mode():
        for qbegin in range(0,tokens,query_block):
            check_cancel()
            qend = min(tokens,qbegin+query_block)
            q = query[:,:,qbegin:qend]
            shape = (batch,heads,qend-qbegin,1)
            maximum = torch.full(shape,-float('inf'),device=query.device,dtype=torch.float32)
            denominator = torch.zeros(shape,device=query.device,dtype=torch.float32)
            accumulator = torch.zeros((*shape[:-1],dim),device=query.device,dtype=torch.float32)
            qpositions = torch.arange(query_start+qbegin,query_start+qend,device=query.device)
            for kbegin in range(0,key_count,key_block):
                check_cancel()
                kend = min(key_count,kbegin+key_block)
                k = keys[:,:,kbegin:kend].repeat_interleave(ratio,dim=1)
                v = values[:,:,kbegin:kend].repeat_interleave(ratio,dim=1)
                positions = torch.arange(key_start+kbegin,key_start+kend,device=query.device)
                allowed = positions[None,:]<=qpositions[:,None]
                if window is not None:
                    allowed &= positions[None,:]>qpositions[:,None]-window
                allowed = allowed[None,None]&key_mask[:,None,None,kbegin:kend]
                scores = (q@k.transpose(-1,-2))/math.sqrt(dim)
                scores = scores.masked_fill(~allowed,-1e30)
                next_maximum = torch.maximum(maximum,scores.amax(dim=-1,keepdim=True))
                previous_scale = (maximum-next_maximum).exp()
                probabilities = (scores-next_maximum).exp()*allowed
                accumulator = accumulator*previous_scale+probabilities@v
                denominator = denominator*previous_scale+probabilities.sum(dim=-1,keepdim=True)
                maximum = next_maximum
            output[:,:,qbegin:qend] = accumulator/denominator.clamp_min(1e-30)
        check_cancel()
        if not bool(torch.isfinite(output).all().item()):
            raise ArithmeticError('nonfinite blockwise device attention output')
    return output


__all__ = ['device_attention']
