"""Explicit FlashAttention 2 execution for occupied K3 MLA prefixes."""
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path


class K3FlashAttention:
    """Use installed FA2, with pinned interface identity and no eager fallback.

    Source and package pins identify a runtime to compare. They do not identify
    the compiled extension or certify equivalence with another installation.
    """
    def __init__(self, *, expected_version, interface_sha256):
        from packaging.version import Version
        observed = version('flash-attn')
        if observed != expected_version or not Version('2.1.0') <= Version(observed) < Version('3'):
            raise ValueError('pinned FlashAttention 2.1+ required for bottom-right causal alignment')
        if (type(interface_sha256) is not str or len(interface_sha256) != 64
                or any(c not in '0123456789abcdef' for c in interface_sha256)):
            raise ValueError('reviewed FlashAttention interface SHA-256 required')
        from flash_attn.flash_attn_interface import flash_attn_func
        source = inspect.getsourcefile(flash_attn_func)
        if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != interface_sha256:
            raise ValueError('FlashAttention interface differs from source pin')
        self._call = flash_attn_func
        self.identity = hashlib.sha256(json.dumps([observed, interface_sha256,
            'k3-fa2-bf16-causal-chunk-occupied-prefix-v2']).encode()).hexdigest()

    def attention(self, query, keys, values):
        return self.attention_sequence(query[None], keys, values)[0]

    def attention_sequence(self, query, keys, values):
        import torch
        if (query.device.type != 'cuda' or query.dtype != torch.bfloat16
                or keys.device != query.device or values.device != query.device
                or keys.dtype != query.dtype or values.dtype != query.dtype
                or query.ndim != 3 or keys.ndim != 3 or values.ndim != 3
                or keys.shape[1:] != query.shape[1:] or keys.shape[:2] != values.shape[:2]
                or not query.shape[0] or keys.shape[0] < query.shape[0] or not values.shape[-1]
                or query.shape[-1] > 256 or query.shape[-1] % 8
                or values.shape[-1] > query.shape[-1]):
            raise ValueError('FA2 requires supported CUDA BF16 K3 query/key/value geometry')
        if not all(bool(torch.isfinite(x).all()) for x in (query, keys, values)):
            raise ValueError('finite FlashAttention inputs required')
        width = values.shape[-1]
        padded = torch.nn.functional.pad(values, (0, query.shape[-1]-width))
        with torch.inference_mode():
            output = self._call(query[None].contiguous(), keys[None].contiguous(),
                padded[None].contiguous(), dropout_p=0.0,
                softmax_scale=query.shape[-1]**(-0.5), causal=True)
        if (tuple(output.shape) != (1, *query.shape)
                or output.dtype != query.dtype or output.device != query.device
                or not bool(torch.isfinite(output).all())):
            raise ValueError('unexpected FlashAttention output')
        return output[0, :, :, :width].contiguous()
