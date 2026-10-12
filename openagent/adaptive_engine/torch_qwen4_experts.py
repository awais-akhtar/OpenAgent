"""Borrowed quantized Qwen4 expert storage and device expert products."""
from .torch_mixtral_experts import TorchMixtralExperts


class TorchQwen4Experts(TorchMixtralExperts):
    architecture = 'qwen4_exp_text'
