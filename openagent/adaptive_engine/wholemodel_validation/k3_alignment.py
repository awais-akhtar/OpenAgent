"""Opt-in pinned FLA continuation coordinates, with full physical evidence.

This is not a lossless inverse of publisher state or a numerical certificate.
The oldest full-buffer convolution tap has no native snapshot counterpart.
"""
import inspect
import sys

import numpy as np

from .cache_readers import UnsupportedCacheContract
from .k3 import checked_leaf, config_from_mapping
from .k3_vendor import K3VendorCacheReader, REVISION, vendor_cache_schema
from .runner import digest_file
from ..k3_execution.fla_contract import FLA_REVISION, FLA_SOURCES


LAYOUT = 'kimi-k3-expanded-kv-kfirst-continuation-history-v1'


def empty_comparison_shapes(config):
    return {f'layer.{i}.{name}': [1, config.kda_heads, config.kda_dim, 0]
            for i, kind in enumerate(config.layer_types) if kind == 'kda' and config.kernel == 1
            for name in ('conv_q', 'conv_k', 'conv_v')}


class K3AlignedVendorReader:
    requires_raw_evidence = True

    def __init__(self, config, schema):
        self.config = config_from_mapping(config)
        if self.config.revision != REVISION or self.config.rotary_mode != 'publisher-none':
            raise ValueError('aligned view requires pinned publisher-none semantics')
        self.raw_reader = K3VendorCacheReader(schema, layer_types=self.config.layer_types)
        self.evidence_schema = vendor_cache_schema(self.config.layer_types)
        self.empty_evidence_shapes = {}

    def _observe(self, context):
        c = self.config
        if 'kda' in c.layer_types:
            publisher = inspect.getmodule(type(context.cache))
            for name, module_name in (('chunk_kda', 'fla.ops.kda.chunk'),
                                      ('fused_recurrent_kda', 'fla.ops.kda.fused_recurrent')):
                function = getattr(publisher, name, None)
                source = inspect.getsourcefile(function) if inspect.isfunction(function) else None
                if source is None or digest_file(source) != FLA_SOURCES[module_name]:
                    raise UnsupportedCacheContract('unknown publisher KDA dispatch', fields=[name])
            for module_name, expected in FLA_SOURCES.items():
                module = sys.modules.get(module_name)
                source = inspect.getsourcefile(module) if module is not None else None
                if source is None or digest_file(source) != expected:
                    raise UnsupportedCacheContract('unreviewed FLA state dependency', fields=[module_name])
            # Inspect actual instantiated modules, not a declared backend string.
            modules = {name: module for name, module in context.model.named_modules()}
            for i, kind in enumerate(c.layer_types):
                if kind != 'kda':
                    continue
                for name in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
                    path = f'language_model.model.layers.{i}.self_attn.{name}'
                    module = modules.get(path)
                    source = inspect.getsourcefile(type(module)) if module is not None else None
                    if (source is None or digest_file(source) != FLA_SOURCES['fla.modules.conv.short_conv']
                            or type(module).__name__ != 'ShortConvolution'
                            or module.backend != 'triton' or module.kernel_size != (c.kernel,)):
                        raise UnsupportedCacheContract('unknown convolution implementation/backend', fields=[path])
        return self.raw_reader(context)

    def __call__(self, context):
        raw = self._observe(context)
        c = self.config
        result = dict(raw)
        for i, kind in enumerate(c.layer_types):
            prefix = f'layer.{i}.'
            if kind == 'mla':
                for name, dim in (('key', c.nope + c.rope), ('value', c.value)):
                    result[prefix + name] = checked_leaf(raw[prefix + name],
                        (1, c.heads, context.position, dim), prefix + name, dtype=None)
                continue
            state = checked_leaf(raw[prefix + 'recurrent'],
                (1, c.kda_heads, c.kda_dim, c.kda_dim), prefix + 'recurrent', dtype=None)
            result[prefix + 'recurrent'] = state.swapaxes(-1, -2).copy()
            for name in ('conv_q', 'conv_k', 'conv_v'):
                buffer = checked_leaf(raw[prefix + name],
                    (1, c.kda_heads * c.kda_dim, c.kernel), prefix + name, dtype=None)
                # FLA shifts [1:] before appending. Keep [0] in separate raw evidence.
                result[prefix + name] = buffer.reshape(1, c.kda_heads, c.kda_dim, c.kernel)[..., 1:].copy()
        return result

    def evidence(self, context):
        return self._observe(context)


class K3RawFLAVendorReader(K3AlignedVendorReader):
    """Review actual FLA dispatch while preserving every physical cache tap."""
    requires_raw_evidence = False

    def __call__(self, context):
        return self._observe(context)
