"""Source policy admission for explicit accelerator arithmetic owners."""
import os


def admit_no_fallback(device):
    """Do not let a requested Metal owner silently execute CPU operators.

    This reads declared process policy only. It does not probe hardware or
    establish operation placement; later platform evidence must observe that.
    """
    if str(device).split(':',1)[0]=='mps':
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':
            raise ValueError('explicit Metal execution requires CPU operator fallback to be disabled')
        if os.environ.get('PYTORCH_MPS_FAST_MATH','0')!='0':
            raise ValueError('explicit FP32 Metal execution requires fast math to be disabled')


def admit_fp32(torch,device):
    """Reject ambient autocast/TF32 rather than changing process precision."""
    admit_no_fallback(device);target=torch.device(device)
    if torch.is_autocast_enabled(target.type):
        raise ValueError('explicit FP32 decoder arithmetic cannot run inside autocast')
    if target.type=='cuda' and (torch.backends.cuda.matmul.allow_tf32 or
            torch.get_float32_matmul_precision()!='highest'):
        raise ValueError('explicit FP32 decoder arithmetic requires TF32 disabled and highest matmul precision')


__all__=['admit_no_fallback','admit_fp32']
