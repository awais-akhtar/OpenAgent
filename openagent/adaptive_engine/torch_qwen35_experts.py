"""Selected quantized Qwen3.5 experts with device-resident activation math."""
from .torch_mixtral_experts import TorchMixtralExperts


class TorchQwen35Experts(TorchMixtralExperts):
    architecture = 'qwen3_5_moe_text'


__all__ = ['TorchQwen35Experts']
