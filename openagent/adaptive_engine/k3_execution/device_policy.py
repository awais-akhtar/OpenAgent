"""Explicit K3 accelerator admission; never select a fallback device."""
def admit_device(torch, device, *, allow_cpu=False):
    target = torch.device(device)
    if target.type == 'cpu' and allow_cpu:
        return target
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('requested CUDA/HIP device is unavailable')
        target = torch.device('cuda', torch.cuda.current_device() if target.index is None else target.index)
        if not 0 <= target.index < torch.cuda.device_count():
            raise ValueError('requested CUDA/HIP device index is unavailable')
        with torch.cuda.device(target):
            if not torch.cuda.is_bf16_supported():
                raise ValueError('requested accelerator does not support BF16')
        return target
    if target.type == 'mps':
        if target.index not in (None, 0) or not torch.backends.mps.is_available():
            raise ValueError('requested Metal device is unavailable')
        if not torch.backends.mps.is_macos_or_newer(14, 0):
            raise ValueError('K3 Metal BF16 requires macOS 14 or newer')
        import os
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
            raise ValueError('K3 Metal requires CPU operator fallback to be disabled')
        if os.environ.get('PYTORCH_MPS_FAST_MATH','0') != '0':
            raise ValueError('explicit Metal arithmetic requires fast math to be disabled')
        return torch.device('mps:0')
    raise ValueError('explicit supported K3 device required')
