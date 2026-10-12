"""Raw FLA candidate caches with an explicit forward MLA comparison view."""
import copy
from dataclasses import asdict
import numpy as np

from .adapters import NativeExecutorAdapter
from .k3 import K3CandidateFactory, config_from_mapping, indexed, checked_leaf, k3_text_forward
from .k3_vendor import LAYOUT, vendor_cache_schema
from .comparison import Tolerance


class K3FLACacheReader:
    """Keep full convolution buffers and transposed recurrent storage intact.

    MLA is a forward projection through the candidate's selected arithmetic,
    not an inverse reconstruction of publisher state or a parity certificate.
    """
    requires_raw_evidence = True

    def __init__(self, config, schema):
        self.config = config_from_mapping(config)
        if dict(schema) != vendor_cache_schema(self.config.layer_types):
            raise ValueError('complete raw-FLA/expanded-MLA cache schema required')
        self.evidence_schema = dict(schema)
        for i, kind in enumerate(self.config.layer_types):
            if kind == 'mla':
                self.evidence_schema.update({f'layer.{i}.{name}': i
                    for name in ('latent', 'rotary_key', 'projection_weight')})
        self.empty_evidence_shapes = {}
        if self.config.rope == 0:
            self.empty_evidence_shapes = {f'layer.{i}.rotary_key': [1, None, 0]
                for i, kind in enumerate(self.config.layer_types) if kind == 'mla'}

    def _read(self, context):
        from ..k3_execution.executor import K3Execution, K3Snapshot
        from ..k3_execution.fla_mixer import FLAMixerSnapshot
        c, model, snapshot = self.config, context.model, context.cache
        if type(model) is not K3Execution or type(snapshot) is not K3Snapshot:
            raise ValueError('actual K3 executor snapshot required')
        if (model.config != c or snapshot.identity != model._identity
                or snapshot.position != context.position or model.position != context.position
                or context.position <= 0 or model._arithmetic is None):
            raise ValueError('occupied BF16 K3 snapshot owner/position mismatch')
        n = context.position
        kda = indexed(snapshot.kda, {i for i, t in enumerate(c.layer_types) if t == 'kda'}, 'KDA')
        mla = indexed(snapshot.mla, {i for i, t in enumerate(c.layer_types) if t == 'mla'}, 'MLA')
        raw = {'position': np.asarray(n, dtype=np.int64)}
        evidence = dict(raw)
        for i, state in kda.items():
            if (type(state) is not FLAMixerSnapshot or state.position != n
                    or state.identity != model._kda[i].identity
                    or model._kda[i].position != n or len(state.convolution) != 3):
                raise ValueError('raw FLA snapshot identity/geometry mismatch')
            prefix = f'layer.{i}.'
            raw[prefix+'recurrent'] = checked_leaf(state.recurrent,
                (1, c.kda_heads, c.kda_dim, c.kda_dim), prefix+'recurrent')
            for name, buffer in zip(('conv_q', 'conv_k', 'conv_v'), state.convolution):
                raw[prefix+name] = checked_leaf(buffer,
                    (1, c.kda_heads*c.kda_dim, c.kernel), prefix+name)
            evidence.update({name: value for name, value in raw.items() if name.startswith(prefix)})
        for i, pages in mla.items():
            cache = model._mla[i]
            if cache.length != n or not pages:
                raise ValueError('occupied MLA pages required')
            required = n*(c.heads*(c.nope+c.rope+c.value)*16+(c.kvrank+c.rope)*8)
            if required > model._arithmetic.max_bytes:
                raise MemoryError('comparison projection exceeds explicit arithmetic budget')
            rows = []
            for j, page in enumerate(pages):
                if (page.ndim != 2 or not 1 <= len(page) <= cache.page_tokens
                        or j < len(pages)-1 and len(page) != cache.page_tokens):
                    raise ValueError('invalid occupied MLA page extent')
                rows.append(checked_leaf(page, (len(page), c.kvrank+c.rope), 'MLA page'))
            if sum(len(row) for row in rows) != n:
                raise ValueError('MLA page count differs from prefix')
            joined = np.concatenate(rows)
            prefix = f'layer.{i}.'
            weight = model._weights[f'language_model.model.layers.{i}.self_attn.kv_b_proj.weight']
            expanded = np.stack([model._matrix(weight, row[:c.kvrank]) for row in joined])
            expanded = expanded.reshape(n, c.heads, c.nope+c.value)
            key = np.empty((n, c.heads, c.nope+c.rope), np.float32)
            key[:, :, :c.nope] = expanded[:, :, :c.nope]
            key[:, :, c.nope:] = joined[:, None, c.kvrank:]
            raw[prefix+'key'] = key.transpose(1, 0, 2)[None]
            raw[prefix+'value'] = expanded[:, :, c.nope:].transpose(1, 0, 2)[None]
            evidence.update({prefix+'latent': joined[None, :, :c.kvrank],
                prefix+'rotary_key': joined[None, :, c.kvrank:],
                prefix+'projection_weight': weight.copy(),
                prefix+'key': raw[prefix+'key'], prefix+'value': raw[prefix+'value']})
        return raw, evidence

    def __call__(self, context):
        return self._read(context)[0]

    def evidence(self, context):
        return self._read(context)[1]


def create_k3_fla_candidate_adapter(*, offline_root, options):
    if not {'fla_policy', 'arithmetic_policy', 'routing_policy'} <= options.keys():
        raise ValueError('explicit FLA, BF16 and routing policies required')
    metadata = copy.deepcopy(options['metadata'])
    factory = K3CandidateFactory(offline_root, options)
    c = factory.config
    if (metadata.get('authoritative') is True or metadata.get('cache_layout') != LAYOUT
            or metadata.get('model_type') != 'kimi_k3' or metadata.get('revision') != c.revision
            or metadata.get('num_layers') != len(c.layer_types)
            or metadata.get('vocabulary_size') != c.vocab):
        raise ValueError('explicit nonauthoritative raw-FLA comparison metadata required')
    reader = K3FLACacheReader(c, metadata['cache_schema'])
    tolerance = Tolerance(**options['projection_tolerance'])
    if any(type(v) not in (float, int) for v in (tolerance.atol, tolerance.rtol)):
        raise ValueError('numeric nonboolean projection tolerance required')
    metadata['evidence_schema'] = reader.evidence_schema
    metadata['empty_evidence_shapes'] = reader.empty_evidence_shapes
    metadata['execution_scope'] = 'text-tokens-only'
    metadata['cache_projection'] = {'certified': False,
        'kind': 'candidate-bf16-forward-mla-projection',
        'tolerance': asdict(tolerance),
        'leaves': [name for name, layer in metadata['cache_schema'].items()
                   if layer != 'global' and c.layer_types[layer] == 'mla'],
        'limitation': 'raw FLA and projected MLA require authoritative numerical comparison'}
    import torch
    return NativeExecutorAdapter(factory, metadata, cache_reader=reader,
        forward_call=k3_text_forward, close_call=options.get('close_call'), torch=torch)
