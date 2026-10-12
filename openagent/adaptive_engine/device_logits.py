"""Bounded device vocabulary projection with an owned FP32 host boundary."""
import numpy as np
from .device_publisher_projections import device_projection
from .device_execution_policy import admit_fp32


def device_logits(torch,hidden,head,*,max_device_bytes,max_host_bytes,cancel=None):
    admit_fp32(torch,hidden.device)
    if (hidden.ndim not in (2,3) or head.ndim!=2 or hidden.shape[-1]!=head.shape[1] or
            hidden.dtype!=torch.float32 or head.dtype!=torch.float32 or hidden.device!=head.device):
        raise ValueError('matching FP32 device hidden/head projection required')
    if any(type(value) is not int or value<=0 for value in (max_device_bytes,max_host_bytes)):
        raise ValueError('explicit vocabulary device/host projection admission required')
    width,vocab = head.shape[1],head.shape[0]
    rows = hidden.numel()//width
    if not rows or not vocab or rows*vocab*4>max_host_bytes:
        raise MemoryError('vocabulary host result exceeds admission')
    row_charge = 8*vocab+8*width
    block = min(rows,max_device_bytes//row_charge)
    if block<1:
        raise MemoryError('one vocabulary projection row exceeds scratch admission')
    if cancel is not None and not callable(getattr(cancel,'is_set',None)):
        raise ValueError('vocabulary cancellation must provide is_set')
    output = np.empty((rows,vocab),np.float32)
    flattened = hidden.reshape(rows,width)
    with torch.inference_mode():
        for begin in range(0,rows,block):
            if cancel is not None and cancel.is_set():
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('device vocabulary projection cancelled')
            end = min(rows,begin+block)
            projected = device_projection(torch,flattened[begin:end],head,cancel=cancel)
            value = projected.cpu().numpy()
            if not np.isfinite(value).all():
                raise ArithmeticError('nonfinite device vocabulary projection')
            output[begin:end] = value
            del projected,value
    if cancel is not None and cancel.is_set():
        from .execution_service import ExecutionCancelled
        raise ExecutionCancelled('device vocabulary projection cancelled')
    return output.reshape((*hidden.shape[:-1],vocab))


__all__ = ['device_logits']
