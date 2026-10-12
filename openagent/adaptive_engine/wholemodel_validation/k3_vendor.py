"""Observed expanded cache ABI of the unchanged pinned K3 text implementation.

This is intentionally not the native compressed-latent cache ABI. No projection,
inverse projection, recurrent transpose, history truncation or source patching.
"""
import copy
import inspect

import numpy as np

from .adapters import TorchCausalAdapter
from .cache_readers import UnsupportedCacheContract, detached
from .local_transformers import LocalTransformersFactory
from .runner import digest_file


REVISION = 'f831ab66814297da540d832a5235f8e904f29d06'
SOURCE_SHA256 = '9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a'
LAYOUT = 'kimi-k3-publisher-expanded-kv-raw-fla-v1'


def vendor_cache_schema(layer_types):
    schema = {'position': 'global'}
    for layer, kind in enumerate(layer_types):
        if kind not in ('kda', 'mla'):
            raise ValueError('explicit KDA/MLA schedule required')
        leaves = ('recurrent', 'conv_q', 'conv_k', 'conv_v') if kind == 'kda' else ('key', 'value')
        for leaf in leaves:
            schema[f'layer.{layer}.{leaf}'] = layer
    return schema


class K3VendorCacheReader:
    """Read all four publisher cache inventories, including inactive slots.

    MLA lengths are observed from actual occupied KV tensors. KDA recurrent
    layout and full convolution buffers stay in the pinned FLA physical order;
    matching dimensions alone does not establish native semantic equivalence.
    """

    def __init__(self, schema, *, layer_types):
        self.layer_types = tuple(layer_types)
        if not self.layer_types or 'mla' not in self.layer_types:
            raise UnsupportedCacheContract('publisher cache has no observable KV prefix length',
                                           fields=['position'])
        if dict(schema) != vendor_cache_schema(self.layer_types):
            raise ValueError('complete expanded publisher cache schema required')

    def __call__(self, context):
        cache = context.cache
        source = inspect.getsourcefile(type(cache))
        if (type(cache).__name__ != 'KimiDynamicCache' or source is None
                or digest_file(source) != SOURCE_SHA256):
            raise UnsupportedCacheContract('cache class is not the unchanged pinned publisher class',
                                           fields=['cache_class'])
        expected = ['linear_attention' if kind == 'kda' else 'full_attention'
                    for kind in self.layer_types]
        if cache.layer_types != expected:
            raise ValueError('publisher cache schedule differs from declared schedule')
        if cache.config.num_hidden_layers != len(expected):
            raise ValueError('publisher config layer count differs from observed cache')
        full = [i for i, kind in enumerate(self.layer_types) if kind == 'mla']
        linear = [i for i, kind in enumerate(self.layer_types) if kind == 'kda']
        if cache.transformer_layers != full or cache.last_linear_layer != (linear[-1] if linear else -1):
            raise ValueError('publisher cache ownership inventory mismatch')
        fields = ('key_cache', 'value_cache', 'conv_states', 'recurrent_states')
        inventories = {name: getattr(cache, name) for name in fields}
        if any(type(items) is not list or len(items) != len(expected)
               for items in inventories.values()):
            raise ValueError('missing or extra publisher cache slots')
        # Additional tensor inventories require a new reviewed ABI, not filtering.
        allowed = {'config', 'layer_types', 'transformer_layers', 'last_linear_layer', *fields}
        if set(vars(cache)) != allowed:
            raise UnsupportedCacheContract('unknown publisher cache members',
                                           fields=sorted(set(vars(cache)) - allowed))
        result = {}
        observed_position = None
        for layer, kind in enumerate(self.layer_types):
            prefix = f'layer.{layer}.'
            key, value, conv, recurrent = (inventories[name][layer] for name in fields)
            if kind == 'mla':
                if conv is not None or recurrent is not None:
                    raise ValueError('unexpected recurrent state on MLA layer')
                for name, tensor in (('key', key), ('value', value)):
                    array = detached(tensor)
                    if (array.ndim != 4 or array.shape[0] != 1 or array.dtype.kind != 'f'
                            or not array.size or not np.isfinite(array).all()):
                        raise ValueError('invalid occupied publisher KV: ' + prefix + name)
                    config = cache.config
                    dimension = (config.qk_nope_head_dim + config.qk_rope_head_dim
                                 if name == 'key' else config.v_head_dim)
                    if array.shape[1] != config.num_attention_heads or array.shape[3] != dimension:
                        raise ValueError('publisher MLA geometry differs from actual config')
                    if observed_position is None:
                        observed_position = array.shape[2]
                    if array.shape[2] != observed_position or observed_position != context.position:
                        raise ValueError('observed publisher KV lengths disagree with append position')
                    result[prefix + name] = array
                if result[prefix + 'key'].shape[:3] != result[prefix + 'value'].shape[:3]:
                    raise ValueError('publisher key/value head geometry differs')
            else:
                if key is not None or value is not None:
                    raise ValueError('unexpected expanded KV on KDA layer')
                if type(conv) is not tuple or len(conv) != 3:
                    raise UnsupportedCacheContract('all Q/K/V convolution states required',
                                                   fields=[prefix + 'conv_q', prefix + 'conv_k', prefix + 'conv_v'])
                for name, tensor, ndim in [('recurrent', recurrent, 4),
                        *[(name, tensor, 3) for name, tensor in
                          zip(('conv_q', 'conv_k', 'conv_v'), conv)]]:
                    array = detached(tensor)
                    if (array.ndim != ndim or array.shape[0] != 1 or array.dtype.kind != 'f'
                            or not array.size or not np.isfinite(array).all()):
                        raise UnsupportedCacheContract('missing/unsupported raw FLA state',
                                                       fields=[prefix + name])
                    config = cache.config.linear_attn_config
                    heads, dimension = config['num_heads'], config['head_dim']
                    if name == 'recurrent':
                        if array.shape != (1, heads, dimension, dimension):
                            raise ValueError('publisher recurrent geometry differs from actual config')
                    elif array.shape[1] != heads * dimension:
                        raise ValueError('publisher convolution channel geometry differs from actual config')
                    result[prefix + name] = array
        result['position'] = np.asarray(observed_position, dtype=np.int64)
        return result


def create_k3_vendor_cache_reader(*, schema, options):
    return K3VendorCacheReader(schema, layer_types=options['layer_types'])


def create_k3_expanded_authoritative_adapter(*, offline_root, options):
    """Existing offline loader, unchanged publisher execution, concrete reader.

    Wrapper/auto-model availability and accelerator/FLA dependencies are genuine
    prerequisites. This factory never replaces them or aliases native execution.
    """
    metadata = copy.deepcopy(options['metadata'])
    hybrid = options.get('hybrid_cache_alignment', False)
    pinned_raw = options.get('require_pinned_fla', False)
    if type(pinned_raw) is not bool or (hybrid and pinned_raw):
        raise ValueError('choose raw pinned FLA or transformed hybrid alignment')
    if type(hybrid) is not bool:
        raise ValueError('hybrid_cache_alignment must be boolean')
    if hybrid:
        from .k3_alignment import (FLA_REVISION, FLA_SOURCES, K3AlignedVendorReader,
                                   LAYOUT as ALIGNED_LAYOUT, empty_comparison_shapes)
        from .k3 import config_from_mapping
        config = config_from_mapping(options['execution_config'])
        layout, empties = ALIGNED_LAYOUT, empty_comparison_shapes(config)
    else:
        layout, empties = LAYOUT, {}
    if (metadata.get('revision') != REVISION or metadata.get('authoritative') is not True
            or metadata.get('unchanged') is not True):
        raise ValueError('exact unchanged publisher provenance required')
    if metadata.get('cache_layout') != layout or metadata.get('empty_cache_shapes', {}) != empties:
        raise ValueError('explicit expanded/raw-FLA layout with no structural empties required')
    schedule = options['layer_types']
    if len(schedule) != metadata['num_layers']:
        raise ValueError('publisher schedule must cover every model layer')
    if any(name in options for name in ('cache_reader', 'cache_reader_factory', 'forward_call')):
        raise ValueError('concrete unchanged publisher adapter does not accept overrides')
    factory = LocalTransformersFactory(offline_root, options)
    if SOURCE_SHA256 not in {row['sha256'] for row in options['_artifact_pins']
                            if row['kind'] == 'source'}:
        raise ValueError('pinned publisher text implementation must be a source artifact')
    reader = K3VendorCacheReader(metadata['cache_schema'], layer_types=schedule)
    if pinned_raw:
        from .k3_alignment import K3RawFLAVendorReader, FLA_SOURCES, FLA_REVISION
        from .k3 import config_from_mapping
        config = config_from_mapping(options['execution_config'])
        if config.layer_types != tuple(schedule):
            raise ValueError('raw FLA publisher schedule differs from execution config')
        hashes = {row['sha256'] for row in options['_artifact_pins'] if row['kind'] == 'source'}
        if 'kda' in schedule and not set(FLA_SOURCES.values()) <= hashes:
            raise ValueError('raw FLA comparison requires every reviewed dependency source')
        reader = K3RawFLAVendorReader(config, metadata['cache_schema'])
        metadata['fla_source_contract'] = {'revision': FLA_REVISION, 'sources': dict(FLA_SOURCES),
                                         'state_transformations': False, 'certified': False}
    if hybrid:
        if tuple(schedule) != config.layer_types:
            raise ValueError('aligned publisher schedule differs from execution config')
        hashes = {row['sha256'] for row in options['_artifact_pins'] if row['kind'] == 'source'}
        if 'kda' in schedule and not set(FLA_SOURCES.values()) <= hashes:
            raise ValueError('aligned hybrid cache requires every reviewed FLA state source pin')
        reader = K3AlignedVendorReader(config, metadata['cache_schema'])
        metadata['evidence_schema'] = copy.deepcopy(reader.evidence_schema)
        metadata['empty_evidence_shapes'] = {}
        metadata['cache_alignment'] = dict(
            certified=False, fla_revision=FLA_REVISION, fla_sources=dict(FLA_SOURCES),
            recurrent='observed V-first transposed to K-first',
            history='head-major oldest-first suffix [1:] of observed full kernel buffer',
            limitation='oldest physical tap retained as evidence, not cross-role comparable')
    return TorchCausalAdapter(factory, metadata, cache_reader=reader,
                              input_device=options.get('input_device'),
                              position_ids=True, cache_position=True)
