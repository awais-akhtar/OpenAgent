"""Explicit complete-decoder observation adapters for offline family comparisons.

Factories remain caller-selected and artifact-pinned by the comparison runner.
Constructing a configuration does not load a decoder or execute a comparison.
"""
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .adapters import NativeExecutorAdapter, resolve_callable
from .cache_readers import detached
from .publisher_family_caches import PublisherFamilyCacheReader, schema, empty_shapes


class FamilyExecutor:
    """Own one decoder and expose its complete physical state to the reader."""

    def __init__(self, decoder, *, vocabulary_size, max_observation_bytes,routing_geometry=None,cache_keyword=True):
        if decoder is None or any(not callable(getattr(decoder, name, None))
                                  for name in ('forward', 'snapshot', 'restore', 'close')):
            raise ValueError('complete publisher decoder lifecycle is required')
        if type(vocabulary_size) is not int or vocabulary_size <= 0:
            raise ValueError('positive publisher vocabulary required')
        if type(max_observation_bytes) is not int or max_observation_bytes <= 0:
            raise ValueError('bounded whole-decoder observation admission required')
        if type(cache_keyword) is not bool:
            raise ValueError('explicit publisher cache invocation policy required')
        self.decoder = decoder
        self.vocabulary_size = vocabulary_size
        self.max_observation_bytes = max_observation_bytes
        self.routing_geometry=routing_geometry
        self.cache_keyword=cache_keyword
        self.last_output = None
        self.closed = False
        self.identity = getattr(decoder, 'identity', getattr(decoder, '_identity', None))
        if type(self.identity) is not str or not self.identity:
            raise ValueError('stable publisher decoder identity required')

    def forward(self, tokens):
        if self.closed:
            raise RuntimeError('closed publisher observation executor')
        ids = np.asarray(tokens)
        if ids.ndim != 1 or ids.dtype != np.dtype('int64') or not ids.size:
            raise ValueError('nonempty one-dimensional int64 observation tokens required')
        if np.any(ids < 0) or np.any(ids >= self.vocabulary_size):
            raise ValueError('publisher observation token outside vocabulary')
        # Include the decoder output and both detached/canonical host copies.
        routing_bytes=0 if self.routing_geometry is None else ids.size*int(np.prod(self.routing_geometry,dtype=object))*32
        if ids.size * (self.vocabulary_size * 12 + 8)+routing_bytes > self.max_observation_bytes:
            raise MemoryError('complete token logits exceed observation admission')
        kwargs={} if self.routing_geometry is None or not getattr(self.decoder,'routing_observation_supported',True) else {'capture_routing':True}
        if self.cache_keyword:kwargs['use_cache']=True
        self.last_output=None
        output = self.decoder.forward(ids[None, :],**kwargs)
        value = (output if isinstance(output,np.ndarray) else output.get('logits')
                 if isinstance(output, Mapping) else getattr(output, 'logits', None))
        if value is None:
            raise ValueError('complete publisher decoder must expose token logits')
        if tuple(value.shape) != (1, ids.size, self.vocabulary_size):
            raise ValueError('publisher token-to-logit geometry differs')
        self.last_output = output
        return detached(value[0])

    def snapshot(self):
        if self.closed:
            raise RuntimeError('closed publisher observation executor')
        return self.decoder.snapshot()

    def restore(self, receipt):
        if self.closed:
            raise RuntimeError('closed publisher observation executor')
        self.decoder.restore(receipt)
        self.last_output = None

    def release_snapshot(self, receipt):
        if self.closed:
            raise RuntimeError('closed publisher observation executor')
        release = getattr(self.decoder, 'release_snapshot', None)
        if callable(release):
            release(receipt)

    def close(self):
        if not self.closed:
            # Keep ownership if closure fails because a borrowed resource is live.
            self.decoder.close()
            self.closed = True
            self.last_output = None
            self.decoder = None


def physical_cache(executor):
    """Reader observes the owner directly; no retained snapshot receipt needed."""
    if executor.closed:
        raise RuntimeError('closed publisher cache observation')
    return None


def create_candidate(*, offline_root, options):
    """Runner factory with exact family/config/schema and explicit loader identity.

    loader(seed=..., offline_root=..., options=...) must return a fresh decoder.
    All loader source, dependency, configuration and weights must be included in
    the runner's candidate artifact pins; this function never invents pins.
    """
    root = Path(offline_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('offline publisher artifact directory required')
    family = options['family']
    config = options['config']
    metadata = options['metadata']
    if dict(metadata['cache_schema']) != schema(family, config):
        raise ValueError('complete publisher family cache schema required')
    if metadata.get('empty_cache_shapes',{}) != empty_shapes(family,config):
        raise ValueError('explicit family structural empty cache declarations required')
    reader_type=PublisherFamilyCacheReader
    if family=='qwen3_5_moe_text':
        from .qwen35_publisher_caches import Qwen35PublisherCacheReader
        reader_type=Qwen35PublisherCacheReader
    if family=='qwen4_exp_text':
        from .qwen4_publisher_caches import Qwen4PublisherCacheReader
        reader_type=Qwen4PublisherCacheReader
    reader = reader_type(family=family, config=config, role='candidate',
        max_export_bytes=options['max_cache_export_bytes'])
    routing_geometry=None
    if metadata.get('evidence_schema'):
        declared=reader.enable_routing_observations(max_export_bytes=options['max_routing_export_bytes'])
        if metadata['evidence_schema']!=declared or metadata.get('empty_evidence_shapes'):
            raise ValueError('complete observed GLM routing evidence schema required')
        routing_geometry=(reader.config.num_hidden_layers,
                          reader.config.num_experts if family in ('qwen4_exp_text','qwen3_5_moe_text') else reader.config.n_routed_experts)
    layers = len(reader.geometry or reader.config.layer_types)
    vocabulary = ((reader.config.get('unpadded_vocab_size') or reader.config['vocab_size']) if family == 'inkling_text'
                  else reader.config.vocab_size)
    if metadata['num_layers'] != layers or metadata['vocabulary_size'] != vocabulary:
        raise ValueError('publisher observation metadata differs from complete decoder contract')
    loader = resolve_callable(options['decoder_factory'])
    loader_options = dict(options.get('decoder_options', {}))

    def loaded(*, seed):
        decoder = loader(seed=seed, offline_root=str(root), options=dict(loader_options))
        try:
            return FamilyExecutor(decoder, vocabulary_size=vocabulary,
                max_observation_bytes=options['max_observation_bytes'],routing_geometry=routing_geometry,
                cache_keyword=family not in ('qwen4_exp_text','qwen3_5_moe_text'))
        except BaseException as primary:
            try:
                decoder.close()
            except BaseException as cleanup:
                # Preserve the failed owner for the caller's explicit retry;
                # comparison construction must not discard a live decoder.
                primary.cleanup = decoder.close
                primary.add_note('candidate decoder cleanup remains pending: '+str(cleanup))
            raise

    return NativeExecutorAdapter(loaded, metadata, cache_reader=reader,
        cache_getter=physical_cache)


__all__ = ['FamilyExecutor', 'physical_cache', 'create_candidate']
