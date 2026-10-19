"""Bounded same-device activation transform for persisted E8 expert weights."""
import numpy as np


def rotate_device_e8(torch, values, *, transform, max_bytes, cancel=None):
    """Apply the contract's forward orthogonal transform to FP32 8-lane rows.

    E8 packing stores ``(book_vector @ transform)``. Its consumer therefore
    computes a product against ``transform @ activation``. The activation
    remains on its current device; only the fixed 8x8 transform metadata is
    staged. Processing one 8-lane group at a time bounds temporary storage.
    """
    from .device_qwen35_layers import _cancel

    matrix = np.asarray(transform)
    if (not hasattr(values, 'shape') or values.ndim != 2 or not all(values.shape) or
            values.dtype != torch.float32 or values.device.type not in ('cpu', 'cuda', 'mps') or
            type(max_bytes) is not int or max_bytes <= 0 or
            type(matrix) is not np.ndarray or matrix.shape != (8, 8) or
            matrix.dtype.kind != 'f' or not np.isfinite(matrix).all() or
            not np.allclose(matrix.astype(np.float64) @ matrix.astype(np.float64).T,
                            np.eye(8), rtol=0, atol=1e-6)):
        raise ValueError('bounded FP32 same-device E8 activation rows and orthogonal transform required')
    rows, columns = values.shape
    if columns % 8:
        raise ValueError('E8 device activation width must be divisible by eight')
    # Count the retained input, output, one group result and the device-side
    # transform. The owning expert provider separately admits its matmul and
    # gate/up/down intermediates against the same request work budget.
    required = values.numel() * 4 + rows * 8 * 4 + 8 * 8 * 4 + 4096
    if required > max_bytes:
        raise MemoryError('E8 device rotation buffers exceed admission')
    _cancel(cancel)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError('finite E8 device activations required')
    rotation = torch.tensor(np.array(matrix, dtype=np.float32, copy=True),
                           device=values.device, dtype=torch.float32)
    output = torch.empty_like(values)
    for start in range(0, columns, 8):
        _cancel(cancel)
        transformed = torch.matmul(values[:, start:start + 8], rotation.T)
        output[:, start:start + 8] = transformed
        del transformed
    _cancel(cancel)
    if not bool(torch.isfinite(output).all().item()):
        raise ArithmeticError('nonfinite E8 device rotation')
    return output


__all__ = ['rotate_device_e8']
