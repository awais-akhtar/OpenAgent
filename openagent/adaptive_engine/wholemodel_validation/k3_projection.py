"""Explicit forward MLA comparison view; raw native state remains evidence.

Derived from pinned publisher projection ordering, not an inverse of expanded
KV. NumPy FP32 matmul does not certify publisher Torch/accelerator arithmetic.
"""
import copy
from dataclasses import asdict

import numpy as np

from .adapters import NativeExecutorAdapter
from .cache_readers import UnsupportedCacheContract
from .comparison import Tolerance
from .k3 import (K3CandidateFactory, K3ExecutionCacheReader, checked_leaf,
                 config_from_mapping, k3_cache_schema, k3_empty_cache_shapes,
                 k3_metadata, k3_text_forward)
from .k3_vendor import LAYOUT, REVISION, SOURCE_SHA256, vendor_cache_schema


def project_mla_cache(latent, rotary_key, weight, config):
    """[1,T,R] @ W[H*(N+V),R].T -> expanded [1,H,T,N+rope/V].

    Input latent has already been normalized by execution. Applying RMSNorm
    again here would change the observed cache and is deliberately prohibited.
    """
    c = config_from_mapping(config)
    if c.rotary_mode != 'publisher-none' or c.revision != REVISION:
        raise UnsupportedCacheContract('projection requires pinned publisher-none rotary semantics',
                                       fields=['rotary_key'])
    latent = np.asarray(latent)
    if latent.ndim != 3 or latent.shape[0] != 1 or latent.shape[1] <= 0:
        raise ValueError('nonempty batch-one occupied latent cache required')
    tokens = latent.shape[1]
    latent = checked_leaf(latent, (1, tokens, c.kvrank), 'latent')
    rotary_key = checked_leaf(rotary_key, (1, tokens, c.rope), 'rotary_key')
    weight = checked_leaf(weight, (c.heads * (c.nope + c.value), c.kvrank), 'kv_b_proj.weight')
    # Publisher output rows are head-major, then nonrotary-key followed by value.
    with np.errstate(over='raise', invalid='raise'):
        expanded = np.matmul(latent, weight.T)
    expanded = expanded.reshape(1, tokens, c.heads, c.nope + c.value).transpose(0, 2, 1, 3)
    rotary = np.broadcast_to(rotary_key[:, None], (1, c.heads, tokens, c.rope))
    key = np.concatenate((expanded[..., :c.nope], rotary), axis=-1)
    value = np.array(expanded[..., c.nope:], copy=True, order='C')
    if not np.isfinite(key).all() or not np.isfinite(value).all():
        raise ValueError('nonfinite forward MLA comparison projection')
    return key, value


class K3ProjectedMLAReader:
    """Expanded MLA view with complete raw snapshot evidence.

    Hybrid execution is rejected: the publisher FLA state transpose and full
    convolution buffer ABI are not aligned by an MLA projection. Rejecting it
    here prevents a partial MLA comparison masquerading as whole-cache parity.
    Explicit hybrid_alignment instead selects pinned K-first continuation
    coordinates; the matching vendor reader must retain its full raw buffers.
    """
    requires_raw_evidence = True

    def __init__(self, config, schema, *, hybrid_alignment=False):
        self.config = config_from_mapping(config)
        self.hybrid_alignment = hybrid_alignment
        if any(kind != 'mla' for kind in self.config.layer_types) and not hybrid_alignment:
            raise UnsupportedCacheContract('hybrid KDA continuation layout is not aligned by MLA projection',
                fields=[f'layer.{i}.{name}' for i, kind in enumerate(self.config.layer_types)
                        if kind == 'kda' for name in ('recurrent', 'conv_q', 'conv_k', 'conv_v')])
        if self.config.rotary_mode != 'publisher-none':
            raise ValueError('split-half native rotary cache cannot use publisher-none comparison layout')
        if self.config.projection_epsilon != 1e-6:
            raise ValueError('pinned publisher MLA RMSNorm uses epsilon=1e-6')
        if dict(schema) != vendor_cache_schema(self.config.layer_types):
            raise ValueError('complete expanded publisher MLA schema required')
        self.raw_reader = K3ExecutionCacheReader(self.config, k3_cache_schema(self.config))
        self.evidence_schema = k3_cache_schema(self.config)
        self.empty_evidence_shapes = k3_empty_cache_shapes(self.config)
        for layer in range(len(self.config.layer_types)):
            if self.config.layer_types[layer] == 'mla':
                self.evidence_schema[f'layer.{layer}.kv_b_proj.weight'] = layer

    def _weight(self, context, layer):
        name = f'language_model.model.layers.{layer}.self_attn.kv_b_proj.weight'
        weight = context.model._weights[name]
        if weight.flags.writeable:
            raise ValueError('projection requires actual immutable executor kv_b_proj weight')
        return weight

    def __call__(self, context):
        raw = self.raw_reader(context)
        result = {'position': raw['position']}
        for layer in range(len(self.config.layer_types)):
            prefix = f'layer.{layer}.'
            if self.config.layer_types[layer] == 'kda':
                result[prefix + 'recurrent'] = raw[prefix + 'state'].copy()
                for name in ('conv_q', 'conv_k', 'conv_v'):
                    result[prefix + name] = raw[prefix + name].copy()
                continue
            key, value = project_mla_cache(raw[prefix + 'latent'], raw[prefix + 'rotary_key'],
                                          self._weight(context, layer), self.config)
            result[prefix + 'key'], result[prefix + 'value'] = key, value
        return result

    def evidence(self, context):
        raw = self.raw_reader(context)
        for layer in range(len(self.config.layer_types)):
            if self.config.layer_types[layer] == 'mla':
                raw[f'layer.{layer}.kv_b_proj.weight'] = np.array(self._weight(context, layer), copy=True)
        return raw


def create_k3_projected_mla_reader(*, schema, options):
    return K3ProjectedMLAReader(options['execution_config'], schema)


def create_k3_projected_candidate_adapter(*, offline_root, options):
    """Existing pinned native checkpoint loader, opt-in projected cache view.

    projection_tolerance must be explicitly provided and equal manifest cache
    tolerance. No automatic error budget, weight substitution or oracle change.
    """
    supplied = copy.deepcopy(options['metadata'])
    if any(name in options for name in ('cache_reader', 'cache_reader_factory', 'forward_call', 'cache_getter')):
        raise ValueError('concrete native projection adapter does not accept cache/forward overrides')
    factory = K3CandidateFactory(offline_root, options)
    c = factory.config
    hybrid = options.get('hybrid_cache_alignment', False)
    if type(hybrid) is not bool:
        raise ValueError('hybrid_cache_alignment must be boolean')
    from .k3_alignment import LAYOUT as ALIGNED_LAYOUT, empty_comparison_shapes
    layout = ALIGNED_LAYOUT if hybrid else LAYOUT
    empties = empty_comparison_shapes(c) if hybrid else {}
    if supplied.get('authoritative') is True:
        raise ValueError('native forward projection is not an authoritative implementation')
    if supplied.get('cache_layout') != layout or supplied.get('empty_cache_shapes', {}) != empties:
        raise ValueError('explicit expanded publisher comparison layout with no empty cache leaves required')
    reader = K3ProjectedMLAReader(c, supplied['cache_schema'], hybrid_alignment=hybrid)
    tolerance = Tolerance(**options['projection_tolerance'])
    # Reuse native identity/config checks without replacing the caller's view.
    compressed = dict(supplied, cache_schema=k3_cache_schema(c),
                      empty_cache_shapes=k3_empty_cache_shapes(c))
    metadata = k3_metadata(compressed, c)
    metadata['cache_schema'] = copy.deepcopy(supplied['cache_schema'])
    metadata['cache_layout'] = layout
    metadata['empty_cache_shapes'] = empties
    metadata['evidence_schema'] = copy.deepcopy(reader.evidence_schema)
    metadata['empty_evidence_shapes'] = copy.deepcopy(reader.empty_evidence_shapes)
    metadata['cache_projection'] = dict(
        kind='observed-normalized-latent-forward-kv-b-proj', certified=False,
        arithmetic='numpy-float32-matmul', tolerance=asdict(tolerance),
        leaves=[name for name, layer in metadata['cache_schema'].items()
                if layer != 'global' and c.layer_types[layer] == 'mla'],
        publisher_revision=REVISION, publisher_source_sha256=SOURCE_SHA256,
        ordering='head-major:[nonrotary-key,value];key=[nonrotary-key,shared-rotary]',
        limitation='forward view cannot certify latent nullspace or reassociated attention parity')
    return NativeExecutorAdapter(factory, metadata, cache_reader=reader,
                                 forward_call=k3_text_forward, close_call=options.get('close_call'))
