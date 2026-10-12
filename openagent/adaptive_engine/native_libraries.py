"""Explicit discovery of libraries produced by the engine CMake build."""
from pathlib import Path
from types import MappingProxyType
import sys


DECODER_LIBRARIES = MappingProxyType({
    'mixtral': 'mixtral_decoder',
    'glm_moe_dsa': 'glm_dsa',
    'glm5_next_text': 'glm5_next',
    'inkling_text': 'inkling',
    'kimi_linear': 'kimi',
    'qwen3_5_moe_text': 'qwen35',
    'qwen4_exp_text': 'qwen4',
    'qwen3_5_moe': 'qwen35_vision',
    'qwen4_exp': 'media_ops',
    'inkling_mm_model': 'media_ops',
    'deepseek_v4': 'deepseek_v4',
})
KERNEL_LIBRARIES = frozenset((
    'tensor_ops', 'media_ops', 'mxfp4', 'mixtral_delta', 'qwen4_hyper',
    'kimi_k3_decoder', 'kimi_k3_kda', 'kimi_k3_moe', 'kimi_k3_dense',
    'kimi_k3_blocks', 'kimi_k3_attnres', 'k3_attention',
    'deepseek_v41_ced', 'deepseek_v41_csa2', 'deepseek_v41_engram',
    'deepseek_v41_index', 'deepseek_v41_mhc', 'deepseek_v41_quant',
    'v41_kernels', 'v41_fp8_projection', 'segment_runtime', 'uring_reader',
))


class NativeLibraries:
    """Resolve an explicit build or install directory without loading code.

    File presence is not ABI, numerical, or platform certification. Callers
    still validate each library through its own version and geometry checks.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        if not self.directory.is_dir():
            raise ValueError('native library directory required')
        if sys.platform == 'win32':
            self.prefix, self.suffix = '', '.dll'
        elif sys.platform == 'darwin':
            self.prefix, self.suffix = 'lib', '.dylib'
        else:
            self.prefix, self.suffix = 'lib', '.so'

    def path(self, component):
        if component not in KERNEL_LIBRARIES and component not in DECODER_LIBRARIES.values():
            raise ValueError('unknown engine library component: '+str(component))
        filename = self.prefix+'openagent_'+component+self.suffix
        candidate = self.directory / filename
        resolved = candidate.resolve(strict=True)
        if resolved.parent != self.directory or not resolved.is_file():
            raise ValueError('native library escapes selected directory or is not a file')
        if resolved.stat().st_size == 0:
            raise ValueError('empty native library: '+filename)
        return str(resolved)

    def decoder(self, model_type):
        if model_type not in DECODER_LIBRARIES:
            raise ValueError('no registered resident decoder for '+str(model_type))
        return self.path(DECODER_LIBRARIES[model_type])

    def k3(self, *, packed=False):
        from .execution_factories import K3Libraries
        return K3Libraries(recurrence=self.path('kimi'),
            operators=self.path('tensor_ops'), attention=self.path('k3_attention'),
            mxfp4=self.path('mxfp4') if packed else None)

    def v41(self, *, packed=False):
        result = {'execution': self.path('v41_kernels'),
                  'quant': self.path('deepseek_v41_quant'),
                  'engram': self.path('deepseek_v41_engram')}
        if packed:
            result['mxfp4'] = self.path('mxfp4')
        return result

    def inventory(self):
        components = sorted(KERNEL_LIBRARIES | frozenset(DECODER_LIBRARIES.values()))
        rows = []
        for component in components:
            try:
                path = self.path(component)
            except (OSError, ValueError) as error:
                rows.append({'component': component, 'present': False, 'error': str(error)})
            else:
                rows.append({'component': component, 'present': True, 'path': path})
        return {'directory': str(self.directory), 'libraries': rows, 'verified': False}
