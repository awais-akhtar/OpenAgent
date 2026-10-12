"""Explicit complete compressed DSA observations for future pinned comparisons."""
import hashlib
import json
from pathlib import Path

import numpy as np

from ..independent_glm_dsa import IndependentGlmDsaDecoder
from .adapters import NativeExecutorAdapter, resolve_callable
from .publisher_family_candidate import FamilyExecutor, physical_cache


def contract(config):
    return IndependentGlmDsaDecoder(config, {}, _inventory_only=True).config


def schema(config):
    c = contract(config)
    result = {'position':'global', 'mask':'global'}
    for layer, kind in enumerate(c.indexer_types):
        for name in ('latent','rotary') + (('indexer',) if kind=='full' else ()):
            result[f'layer.{layer}.{name}'] = layer
    return result


def _identity(config):
    return hashlib.sha256(json.dumps(vars(config), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).digest()


class GlmDsaCacheReader:
    """Read stored state; no latent expansion, rounding or token replay.

    A pinned oracle may supply exporter(context, max_bytes=...) for its exact
    physical cache layout. The exporter must return all original compressed
    leaves in the declared layout, including the actual committed prefix mask.
    """
    def __init__(self, config, *, max_export_bytes, exporter=None):
        self.config = contract(config)
        self.schema = schema(config)
        self.identity = _identity(self.config)
        if type(max_export_bytes) is not int or max_export_bytes <= 0:
            raise ValueError('positive complete DSA observation budget required')
        if exporter is not None and not callable(exporter):
            raise TypeError('explicit DSA physical oracle exporter required')
        self.max_bytes, self.exporter = max_export_bytes, exporter

    def __call__(self, context):
        owner = getattr(context.model, 'decoder', context.model)
        position = context.position
        if type(position) is not int or not 0 < position <= self.config.max_position_embeddings:
            raise ValueError('occupied DSA continuation required')
        if self.exporter is None:
            raw = getattr(owner, 'contract', getattr(owner, 'config', None))
            raw = dict(raw) if isinstance(raw, dict) else vars(raw)
            observed = contract(raw)
            if _identity(observed) != self.identity or owner.position != position:
                raise ValueError('DSA owner configuration or continuation differs')
            value = owner.export_cache(max_bytes=self.max_bytes)
        else:
            value = self.exporter(context, max_bytes=self.max_bytes)
        if type(value) is not dict or set(value) != set(self.schema):
            raise ValueError('complete exact compressed DSA cache leaves required')
        sizes = 0
        for name, array in value.items():
            if type(array) is not np.ndarray:
                raise TypeError('DSA exporter must detach exact NumPy cache leaves')
            sizes += array.nbytes
            if name=='position':
                if array.dtype != np.int64 or array.shape != (1,) or array[0] != position:
                    raise ValueError('DSA physical continuation differs')
            elif name=='mask':
                if array.dtype != np.uint8 or array.shape != (1,position) or np.any(array>1):
                    raise ValueError('exact batch-one DSA prefix mask required')
            else:
                width = {'latent':self.config.kv_lora_rank,
                    'rotary':self.config.qk_rope_head_dim,
                    'indexer':self.config.index_head_dim}[name.rsplit('.',1)[1]]
                if array.dtype != np.float32 or array.shape != (1,position,width):
                    raise ValueError('DSA compressed physical cache geometry differs: '+name)
            flat = array.reshape(-1)
            for start in range(0,flat.size,262144):
                if not np.isfinite(flat[start:start+262144]).all():
                    raise ValueError('nonfinite DSA physical cache evidence')
        if 3*sizes > self.max_bytes:
            raise MemoryError('detached DSA observation copies exceed admission')
        return value


def create_reader(*, schema, options):
    reader = GlmDsaCacheReader(options['config'], max_export_bytes=options['max_export_bytes'],
        exporter=resolve_callable(options['exporter']) if 'exporter' in options else None)
    if dict(schema) != reader.schema:
        raise ValueError('complete original compressed DSA schema required')
    return reader


def create_candidate(*, offline_root, options):
    """Construct adapter metadata; decoder loading remains a future runner action."""
    root = Path(offline_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('pinned local offline DSA root required')
    c = contract(options['config'])
    metadata = options['metadata']
    if (metadata['model_type']!='glm_moe_dsa' or metadata['num_layers']!=c.num_hidden_layers
            or metadata['vocabulary_size']!=c.vocab_size or metadata['cache_schema']!=schema(options['config'])
            or metadata.get('empty_cache_shapes') or metadata.get('evidence_schema')):
        raise ValueError('exact compressed DSA candidate metadata required')
    loader = resolve_callable(options['decoder_factory'])
    def loaded(*, seed):
        decoder = loader(seed=seed, offline_root=str(root), options=dict(options.get('decoder_options',{})))
        try:
            return FamilyExecutor(decoder, vocabulary_size=c.vocab_size,
                max_observation_bytes=options['max_observation_bytes'], cache_keyword=False)
        except BaseException:
            decoder.close()
            raise
    reader = GlmDsaCacheReader(options['config'], max_export_bytes=options['max_cache_export_bytes'])
    return NativeExecutorAdapter(loaded, metadata, cache_reader=reader, cache_getter=physical_cache)


__all__ = ['schema','GlmDsaCacheReader','create_reader','create_candidate']
