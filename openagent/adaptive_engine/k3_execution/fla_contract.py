"""Reviewed FLA source contracts shared by execution and cache comparison."""
import hashlib
import importlib
import inspect
from pathlib import Path


FLA_REVISION = '3e52d5abb6ad85312846f39c3e946c2e9881a823'
FLA_SOURCES = {
    'fla.modules.conv.short_conv': 'ef03ec92adc42ec35e476e20aed8bd5bc3a84250467e5ca761f51c9b702e0468',
    'fla.modules.conv.causal_conv1d': 'f8531a8cb4268c37bd0d9a8387a310b7cc5a25d594647907d66e7f7de42ccb46',
    'fla.modules.conv.triton.ops': 'de0dc4c6d656857acf0ec30c0cf6f6c76574e329ca07aef49d7afa449ae19b38',
    'fla.modules.conv.triton.kernels': 'a4f171be4088473f6553dc4a173f5cba04b05ab8a4aae758bd493989a6f99ee2',
    'fla.ops.kda.chunk': '35389854d8cf541c6befd46c456d92daf583c9208b40d330618d8ec7de888fdf',
    'fla.ops.kda.fused_recurrent': '189b66270c089d161691cbd37cc4cd474ba079eae919d694da1063d7231eafda',
}


def admit_fla_sources(short_convolution, chunk, recurrent):
    """Reject changed reviewed sources or dispatch aliases before allocation.

    This checks source identity, not numerical correctness or every transitive
    dependency. Runtime versions and hardware admission remain separate checks.
    """
    def check(value, module_name):
        source = inspect.getsourcefile(value)
        if source is None:
            raise ValueError(f'FLA source unavailable: {module_name}')
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if digest != FLA_SOURCES[module_name]:
            raise ValueError(f'unreviewed FLA source: {module_name}')

    for module_name in FLA_SOURCES:
        check(importlib.import_module(module_name), module_name)
    check(short_convolution, 'fla.modules.conv.short_conv')
    check(chunk, 'fla.ops.kda.chunk')
    check(recurrent, 'fla.ops.kda.fused_recurrent')
    return FLA_REVISION
