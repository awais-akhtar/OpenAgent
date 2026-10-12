"""Device activation rotation for the pinned persisted IQ3 expert contract."""
import math

from .iq3_lattice import ROTATION, rotation_blocks, _signs


def rotate_device_iq3(torch, values, *, max_bytes, cancel=None):
    """Return forward-rotated FP32 rows without staging activations on the host.

    Only deterministic sign metadata is generated on the host. Butterfly
    operands and the returned activation remain on the caller's device.
    """
    from .device_qwen35_layers import _cancel
    if (values.ndim != 2 or not all(values.shape) or values.dtype != torch.float32 or
            values.device.type not in ('cuda','mps') or type(max_bytes) is not int or max_bytes<=0):
        raise ValueError('bounded FP32 accelerator IQ3 activation rows required')
    rows,columns=values.shape
    if values.numel()*24+min(columns,32768)*8+4096>max_bytes:
        raise MemoryError('IQ3 device rotation buffers exceed admission')
    _cancel(cancel)
    if not bool(torch.isfinite(values).all()):
        raise ValueError('finite IQ3 device activations required')
    output=values.clone()
    start=0
    for width in rotation_blocks(columns):
        _cancel(cancel)
        block=output[:,start:start+width]
        signs=torch.tensor(_signs(width),device=values.device,dtype=torch.float32)
        block.mul_(signs)
        stride=1
        while stride<width:
            _cancel(cancel)
            pairs=block.reshape(rows,width//(2*stride),2*stride)
            left=pairs[...,:stride].clone()
            right=pairs[...,stride:].clone()
            pairs[...,:stride]=left+right
            pairs[...,stride:]=left-right
            del left,right,pairs
            stride*=2
        block.div_(math.sqrt(width))
        del signs,block
        start+=width
    _cancel(cancel)
    if not bool(torch.isfinite(output).all()):
        raise ArithmeticError('nonfinite IQ3 device rotation')
    return output


__all__=['ROTATION','rotate_device_iq3']
