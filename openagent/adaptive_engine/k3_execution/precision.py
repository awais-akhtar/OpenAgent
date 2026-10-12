"""Admission checks for explicit K3 Torch precision policies."""


def check_precision(torch, device):
    """Reject ambient arithmetic changes without modifying process settings.

    These checks make a requested policy reproducible enough to compare; they
    do not certify model or kernel equivalence.
    """
    if torch.is_autocast_enabled(device.type):
        raise ValueError('K3 explicit arithmetic cannot run inside autocast')
    if device.type != 'cuda':
        return
    matmul = torch.backends.cuda.matmul
    if matmul.allow_tf32 or torch.get_float32_matmul_precision() != 'highest':
        raise ValueError('K3 FP32 routing requires TF32 disabled and highest matmul precision')
    if matmul.allow_bf16_reduced_precision_reduction:
        raise ValueError('K3 BF16 comparison requires reduced-precision reductions disabled')
