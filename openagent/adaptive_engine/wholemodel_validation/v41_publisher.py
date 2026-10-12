"""Concrete publisher-shaped candidate; no private cache or oracle equations."""
import copy
from contextlib import ExitStack
from dataclasses import asdict
import inspect
import json
from pathlib import Path

import numpy as np

from .adapters import NativeExecutorAdapter
from .runner import digest_file, pinned_artifacts


REVISION = '2cba9e42aa026125f3ed06c6d98c1db82f7ca027'
LAYOUT = 'v41-publisher-public-packed-window-compressed-index-pending-v1'


def publisher_v41_cache_schema(profile):
    schema = {'position': 'global', 'token_history': 'global', 'token_types': 'global'}
    for index, layer in enumerate(profile.layers):
        if type(layer.layer) is not int or layer.layer != index:
            raise ValueError('ordered complete publisher layer profile required')
        for kind in ('window', 'compressed', 'index'):
            for component in ('values', 'scales'):
                schema[f'layer.{index}.{kind}.{component}'] = index
        for component in ('value', 'gate'):
            schema[f'layer.{index}.pending.{component}'] = index
    return schema


def _packed_widths(width, encoding):
    from ..v41_execution.publisher_cache import packed_nbytes
    total = packed_nbytes(width, encoding)
    values = width if encoding == 'fp8-e4m3-g32-ue8m0' else width // 2
    return values, total - values


def publisher_v41_empty_cache_shapes(profile):
    result = {'token_history': [0], 'token_types': [0]}
    for layer in profile.layers:
        for kind, width, encoding in (
                ('window', profile.head_dim, layer.window_encoding),
                ('compressed', profile.head_dim, layer.compressed_encoding),
                ('index', profile.index_head_dim, layer.index_encoding)):
            for component, extent in zip(('values', 'scales'), _packed_widths(width, encoding)):
                result[f'layer.{layer.layer}.{kind}.{component}'] = [0, extent]
        for component in ('value', 'gate'):
            result[f'layer.{layer.layer}.pending.{component}'] = [0, profile.head_dim]
    return result


def _snapshot_token_types(snapshot):
    types = snapshot.token_types
    if type(types) is not tuple:
        raise ValueError('publisher token_types must be a tuple')
    # The public legacy empty tuple explicitly denotes all-text history.
    if not types:
        return (-1,) * len(snapshot.tokens)
    if (len(types) != len(snapshot.tokens)
            or any(type(kind) is not int or not -1 <= kind <= 3 for kind in types)):
        raise ValueError('publisher token_types length/value mismatch')
    return types


class PublisherV41CacheReader:
    """Exact public snapshot inventory, retaining packed codes and scale bytes.

    Shared caches are not duplicated into non-owner slots. Their empty physical
    inventories and profile ownership remain explicit. Engram uses global token
    history; this snapshot exposes no separate persistent Engram tensor state.
    """
    full_layer_cache_evidence = True
    authoritative_layout_certified = False

    def __init__(self, profile, schema):
        self.profile = profile
        self.schema = publisher_v41_cache_schema(profile)
        if dict(schema) != self.schema:
            raise ValueError('complete public packed publisher cache schema required')

    def __call__(self, context):
        from ..v41_execution.publisher import PublisherV41Execution, PublisherV41Snapshot
        from ..v41_execution.publisher_cache import PackedPublisherRow
        model, saved, p = context.model, context.cache, self.profile
        if type(model) is not PublisherV41Execution or type(saved) is not PublisherV41Snapshot:
            raise ValueError('exact PublisherV41Execution and public snapshot required')
        position = context.position
        if (model.profile != p or saved.identity != model.identity
                or type(position) is not int or not 0 <= position <= p.context
                or model.length != position or type(saved.tokens) is not tuple
                or len(saved.tokens) != position
                or any(type(t) is not int or not 0 <= t < p.vocabulary for t in saved.tokens)):
            raise ValueError('publisher snapshot identity/profile/prefix mismatch')
        if type(saved.state) is not tuple or len(saved.state) != 4:
            raise ValueError('publisher snapshot needs all four cache inventories')
        if any(type(kind) is not tuple or len(kind) != len(p.layers) for kind in saved.state):
            raise ValueError('publisher snapshot has missing/extra layer slots')
        token_types = _snapshot_token_types(saved)
        # Public observation detects stale same-length snapshots without accessing
        # mutable private executor fields or changing state with restore().
        live = model.snapshot()
        if (live.identity != saved.identity or live.tokens != saved.tokens
                or _snapshot_token_types(live) != token_types):
            raise ValueError('stale publisher token/type history')
        result = {'position': np.asarray(position, dtype=np.int64),
                  'token_history': np.asarray(saved.tokens, dtype=np.int64),
                  'token_types': np.asarray(token_types, dtype=np.int8)}
        for index, layer in enumerate(p.layers):
            ratio = layer.compression_ratio
            if type(ratio) is not int or ratio < 0:
                raise ValueError('invalid publisher compression ratio')
            owned = bool(ratio and layer.kv_owner == index)
            for kind_index, kind in enumerate(('window', 'compressed', 'index', 'pending')):
                rows, live_rows = saved.state[kind_index][index], live.state[kind_index][index]
                count = (min(position, p.window) if kind_index == 0 else
                         position % ratio if kind_index == 3 and owned else
                         position // ratio if owned else 0)
                if type(rows) is not tuple or len(rows) != count or len(live_rows) != count:
                    raise ValueError('publisher cache owner/occupied extent mismatch')
                prefix = f'layer.{index}.{kind}.'
                if kind_index == 3:
                    values, gates = [], []
                    for row, live_row in zip(rows, live_rows):
                        if type(row) is not tuple or len(row) != 2:
                            raise ValueError('complete pending value/gate pair required')
                        for component, actual, current in zip((values, gates), row, live_row):
                            if (type(actual) is not np.ndarray or actual.dtype != np.float32
                                    or actual.shape != (p.head_dim,) or not np.isfinite(actual).all()
                                    or actual.tobytes() != current.tobytes()):
                                raise ValueError('invalid/stale observed pending FP32 row')
                            component.append(actual)
                    for name, arrays in (('value', values), ('gate', gates)):
                        result[prefix + name] = (np.stack(arrays).copy() if arrays else
                                                np.empty((0, p.head_dim), dtype=np.float32))
                    continue
                width = p.index_head_dim if kind_index == 2 else p.head_dim
                encoding = (layer.window_encoding if kind_index == 0 else
                            layer.compressed_encoding if kind_index == 1 else layer.index_encoding)
                for row, current in zip(rows, live_rows):
                    if (type(row) is not PackedPublisherRow or row.size != width
                            or row.encoding != encoding or row != current):
                        raise ValueError('invalid/stale publisher packed row')
                    # Revalidate public buffers, including forged dataclass values;
                    # this validates codes but never decodes or repacks cache rows.
                    PackedPublisherRow(row.encoding, row.size, row.values, row.scales)
                for name, extent in zip(('values', 'scales'), _packed_widths(width, encoding)):
                    data = b''.join(getattr(row, name) for row in rows)
                    result[prefix + name] = np.frombuffer(data, dtype=np.uint8).reshape(count, extent).copy()
        if set(result) != set(self.schema):
            raise ValueError('incomplete public publisher cache export')
        return result


def publisher_v41_text_forward(executor, token_ids):
    # NativeSession uses np.int64; publisher forward explicitly requires Python ints.
    return executor.forward([int(token) for token in token_ids])


class PublisherV41CandidateFactory:
    """Pinned local checkpoint -> existing payload/arithmetic/public executor."""
    def __init__(self, offline_root, options):
        from ..publisher_checkpoint.v41_profile import v41_publisher_profile
        self.root, self.options = Path(offline_root).resolve(), dict(options)
        self.resources = {}
        self.bf16_projection_bytes=options.get('bf16_projection_max_bytes',0)
        if type(self.bf16_projection_bytes) is not int or self.bf16_projection_bytes<0:
            raise ValueError('nonnegative explicit streamed BF16 candidate budget required')
        loader_options = options.get('loader_options', {})
        allowed_loader_options = {'max_chunk_bytes', 'max_expert_bytes', 'max_engram_bytes',
                                  'max_engram_rows', 'max_projection_bytes'}
        if not isinstance(loader_options, dict) or set(loader_options) - allowed_loader_options:
            raise ValueError('only publisher provider admission budgets may override loader options')
        self.loader_options = copy.deepcopy(loader_options)
        rows = self._pins()
        self.checkpoint = (self.root / options['checkpoint_path']).resolve()
        if not self.checkpoint.is_relative_to(self.root) or not self.checkpoint.is_dir():
            raise ValueError('local checkpoint directory beneath offline_root required')
        self._checkpoint_pins(rows)
        config_path = self.checkpoint / 'config.json'
        with config_path.open(encoding='utf-8') as stream:
            self.profile = v41_publisher_profile(json.load(stream), capacity=options['capacity'])
        if options['metadata']['config_sha256'] != digest_file(config_path):
            raise ValueError('publisher config metadata differs from actual checkpoint')
        self.libraries = {}
        libraries = options['libraries']
        if not {'execution', 'mxfp4'} <= set(libraries) or set(libraries) - {'execution', 'mxfp4', 'fp8'}:
            raise ValueError('explicit prebuilt execution/mxfp4 and optional fp8 libraries required')
        for name, path in libraries.items():
            resolved = (self.root / path).resolve()
            if resolved not in rows or rows[resolved]['kind'] not in ('source', 'dependency'):
                raise ValueError('prebuilt library requires a source/dependency artifact pin')
            self.libraries[name] = resolved
        self.tokenizer = (self.root / options['tokenizer_path']).resolve()
        if self.tokenizer not in rows or rows[self.tokenizer]['kind'] != 'tokenizer':
            raise ValueError('actual tokenizer JSON requires a tokenizer artifact pin')

    def _pins(self):
        return {Path(row['resolved_path']): row for row in
                pinned_artifacts({'artifacts': self.options['_artifact_pins']}, self.root)}

    def _checkpoint_pins(self, rows):
        for path in self.checkpoint.rglob('*'):
            if path.is_file() and path.resolve() not in rows:
                raise ValueError('unpinned publisher checkpoint file: ' + str(path))
            if path.is_file() and (path.suffix == '.safetensors' or path.name.endswith('.safetensors.index.json')):
                if rows[path.resolve()]['kind'] != 'weights':
                    raise ValueError('publisher weight shards and indexes must be weights artifacts')
        config = self.checkpoint / 'config.json'
        if config not in rows or rows[config]['kind'] != 'config':
            raise ValueError('publisher config.json must be pinned as config')

    def __call__(self, *, seed):
        from tokenizers import Tokenizer
        from ..publisher_checkpoint.service import PublisherCheckpoint
        from ..publisher_checkpoint.v41_adapter import load_v41
        from ..v41_execution.publisher import PublisherV41Execution
        from ..v41_execution.publisher_arithmetic import NativePublisherV41Arithmetic
        from ..owned_resources import OwnedResources
        from ..v41_execution.bf16_projections import PublisherBF16Projections
        from ..publisher_tensor_reader import PublisherTensorReader
        from ..publisher_checkpoint.v41_provider import V41TokenEmbeddingProvider
        from ..native_library_identity import NativeLibraryIdentity
        rows = self._pins()
        self._checkpoint_pins(rows)
        hashes = {r['sha256'] for r in rows.values() if r['kind'] == 'source'}
        implementations=(PublisherV41Execution, NativePublisherV41Arithmetic, load_v41,
            V41TokenEmbeddingProvider,PublisherTensorReader,OwnedResources,NativeLibraryIdentity)
        if self.bf16_projection_bytes:implementations+=(PublisherBF16Projections,)
        for implementation in implementations:
            source = inspect.getsourcefile(implementation)
            if source is None or digest_file(source) not in hashes:
                raise ValueError('actual publisher execution/transport/loader source must be pinned')
        with OwnedResources() as resources:
            checkpoint = PublisherCheckpoint.open(self.checkpoint, family='deepseek_v41')
            if callable(getattr(checkpoint,'close',None)):resources.callback(checkpoint.close)
            bf16=None
            if self.bf16_projection_bytes:
                bf16=PublisherBF16Projections(checkpoint,max_bytes=self.bf16_projection_bytes)
                resources.callback(bf16.close)
            payload = load_v41(checkpoint, capacity=self.options['capacity'],
                max_weight_bytes=self.options['max_weight_bytes'],
                use_providers=True,sharded_projections=() if bf16 is None else tuple(sorted(bf16.sources)),
                **self.loader_options)
            for provider in (payload.engram_provider, payload.expert_provider, payload.fp8_provider,
                             payload.embedding_provider):
                if provider is not None:
                    resources.callback(provider.close)
            if (payload.profile != self.profile or payload.unsupported
                    or payload.config_sha256 != self.options['metadata']['config_sha256']):
                raise ValueError('publisher payload differs from admitted profile/config')
            arithmetic = NativePublisherV41Arithmetic(payload,
                execution_library=self.libraries['execution'], mxfp4_library=self.libraries['mxfp4'],
                fp8_library=self.libraries.get('fp8'), tokenizer=Tokenizer.from_file(str(self.tokenizer)),
                max_bytes=self.options['max_arithmetic_bytes'],projection_backend=bf16)
            resources.callback(arithmetic.close)
            if bf16 is not None:bf16.bind_arithmetic(arithmetic)
            executor = PublisherV41Execution.from_payload(payload, arithmetic,
                library=self.libraries['execution'], max_bytes=self.options['max_bytes'])
            resources.callback(executor.close)
            if executor.length != 0:
                raise ValueError('fresh empty publisher session required')
            self._checkpoint_pins(self._pins())
            self.resources[executor] = resources.pop_all()
            return executor

    def close(self, executor):
        resources = self.resources[executor]
        resources.close()
        del self.resources[executor]


def create_v41_publisher_candidate_adapter(*, offline_root, options):
    metadata = copy.deepcopy(options['metadata'])
    if any(key in options for key in ('executor_factory', 'cache_reader', 'cache_reader_factory',
                                     'cache_getter', 'forward_call', 'close_call')):
        raise ValueError('concrete publisher candidate does not accept executor/cache overrides')
    if metadata.get('authoritative') is True:
        raise ValueError('publisher-shaped native candidate is not an authoritative oracle')
    if metadata.get('model_type') != 'deepseek_v41' or metadata.get('revision') != REVISION:
        raise ValueError('exact pinned DeepSeek V4.1 identity required')
    factory = PublisherV41CandidateFactory(offline_root, options)
    p = factory.profile
    if (metadata.get('num_layers') != len(p.layers) or metadata.get('vocabulary_size') != p.vocabulary
            or metadata.get('cache_layout') != LAYOUT
            or metadata.get('empty_cache_shapes', {}) != publisher_v41_empty_cache_shapes(p)):
        raise ValueError('explicit matching publisher geometry/layout/empty shapes required')
    reader = PublisherV41CacheReader(p, metadata['cache_schema'])
    metadata['execution_scope'] = 'publisher-text-tokens-only'
    metadata['publisher_profile'] = asdict(p)
    metadata['authoritative_layout_certified'] = False
    metadata['implementation'] = 'native-c-publisher-v41-text-contract-v1'
    metadata['inference_supported'] = False
    metadata['published_checkpoint_supported'] = False
    return NativeExecutorAdapter(factory, metadata, cache_reader=reader,
                                 forward_call=publisher_v41_text_forward, close_call=factory.close)
