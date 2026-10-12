"""Opt-in version- and source-pinned MXFP4 library decoding."""
import hashlib
import importlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path


class K3CompressedTensorsDecoder:
    """Decode bounded row tiles through installed compressed-tensors APIs.

    Callers supply reviewed source hashes. This preserves the selected library's
    decode behavior, not the numerical behavior of a separate fused GPU kernel.
    """
    MODULES = (
        'compressed_tensors.compressors.mxfp4.base',
        'compressed_tensors.compressors.nvfp4.base',
        'compressed_tensors.compressors.nvfp4.helpers',
        'compressed_tensors.compressors.mx_utils',
        'compressed_tensors.quantization.lifecycle.forward',
    )

    def __init__(self, *, expected_version, source_hashes):
        if not isinstance(expected_version, str) or not expected_version:
            raise ValueError('explicit compressed-tensors version required')
        if version('compressed-tensors') != expected_version:
            raise ValueError('installed compressed-tensors version differs from pin')
        if not isinstance(source_hashes, dict) or set(source_hashes) != set(self.MODULES):
            raise ValueError('reviewed MXFP4 decoding module hashes required')
        self._sources = {}
        for name in self.MODULES:
            pin = source_hashes[name]
            if (type(pin) is not str or len(pin) != 64
                    or any(c not in '0123456789abcdef' for c in pin)):
                raise ValueError('lowercase SHA-256 source pin required: '+name)
            module = importlib.import_module(name)
            source = inspect.getsourcefile(module)
            if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != pin:
                raise ValueError('unreviewed compressed-tensors source: '+name)
            self._sources[name] = module
        from compressed_tensors.quantization import QuantizationScheme
        self._scheme = QuantizationScheme.model_validate({
            'targets': ['Linear'], 'format': 'mxfp4-pack-quantized',
            'input_activations': None, 'output_activations': None,
            'weights': {'group_size': 32, 'num_bits': 4, 'type': 'float',
                'strategy': 'group', 'symmetric': True, 'dynamic': False,
                'scale_dtype': 'torch.uint8'},
        })
        self._compressor = self._sources[self.MODULES[0]].MXFP4PackedCompressor
        self.identity = hashlib.sha256(json.dumps({'version': expected_version,
            'sources': source_hashes, 'contract': 'k3-library-mxfp4-decode-v1'},
            sort_keys=True).encode()).hexdigest()

    def decode(self, packed, scales, *, rows, cols):
        import torch
        if (packed.dtype != torch.uint8 or scales.dtype != torch.uint8
                or tuple(packed.shape) != (rows, cols//2)
                or tuple(scales.shape) != (rows, cols//32)
                or rows <= 0 or cols <= 0 or cols % 32
                or packed.device != scales.device):
            raise ValueError('admitted same-device MXFP4 row tile required')
        if bool((scales == 255).any()):
            raise ValueError('nonfinite E8M0 scale code')
        with torch.inference_mode():
            result = self._compressor.decompress(
                {'weight_packed': packed, 'weight_scale': scales}, self._scheme)
            weight = result['weight']
            if (tuple(weight.shape) != (rows, cols) or weight.device != packed.device
                    or weight.dtype != torch.bfloat16 or not bool(torch.isfinite(weight).all())):
                raise ValueError('library returned an unexpected MXFP4 decoded tensor')
            return weight
