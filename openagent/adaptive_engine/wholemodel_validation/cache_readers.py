"""Lossless, explicit cache readers. Unknown family caches fail closed."""
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .comparison import canonical


def detached(value):
    """Export real Torch tensors without rounding float32/64 or aliasing cache.

    NumPy lacks bfloat16: widening to float32 is exact, not quantization.
    """
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        if str(value.dtype) == 'torch.bfloat16':
            value = value.float()
        value = value.numpy()
    return canonical(value)


@dataclass(frozen=True)
class CacheContext:
    cache: Any
    model: Any
    output: Any
    position: int
    appended_tokens: int


class UnsupportedCacheContract(ValueError):
    """Typed missing/unsupported observed state, never a skipped success."""
    def __init__(self, message, *, fields):
        super().__init__(message)
        self.unsupported_fields = list(fields)


def declared_empty(value, pattern):
    return (not value.size and isinstance(pattern, (tuple, list))
            and len(pattern) == value.ndim and 0 in pattern
            and all(p is None or (type(p) is int and p >= 0 and p == n)
                    for p, n in zip(pattern, value.shape)))


def export_cache(reader, context, schema, *, empty_shapes=None):
    raw = reader(context)
    if not isinstance(raw, Mapping) or set(raw) != set(schema):
        raise ValueError('cache reader must return exactly every declared cache leaf')
    result = {name: detached(value) for name, value in raw.items()}
    empty_shapes = empty_shapes or {}
    if any((not value.size and not declared_empty(value, empty_shapes.get(name)))
           or not np.isfinite(value).all() for name, value in result.items()):
        raise ValueError('cache reader returned undeclared empty/nonfinite logical state')
    return result


class StandardKVReader:
    """Only explicitly authorized non-sliding KV-only legacy/DynamicCache.

    Never calls to_legacy_cache: converting an unknown hybrid cache can drop
    recurrent or MLA state. Fixed/sliding/quantized/offloaded caches require a
    caller-supplied reader. Canonical layout is [batch, heads, tokens, dim].
    """

    def __init__(self, num_layers, *, kv_only, schema):
        if kv_only is not True:
            raise ValueError('standard KV reader requires explicit kv_only=True assertion')
        self.num_layers = num_layers
        self.schema = {'position': 'global'}
        for layer in range(num_layers):
            self.schema[f'layer.{layer}.key'] = layer
            self.schema[f'layer.{layer}.value'] = layer
        if dict(schema) != self.schema:
            raise ValueError('standard KV schema must contain position and every layer key/value')

    def __call__(self, context):
        cache = context.cache
        if type(cache) in (tuple, list):
            pairs = cache
        elif (type(cache).__name__ == 'DynamicCache'
              and type(cache).__module__.startswith('transformers.')):
            # Only exact standard layer classes with full occupied tensors.
            if getattr(cache, 'offloading', False):
                raise ValueError('offloaded cache requires an explicit reader')
            if hasattr(cache, 'layers'):
                layers = cache.layers
                if any(type(layer).__name__ != 'DynamicLayer'
                       or not type(layer).__module__.startswith('transformers.') for layer in layers):
                    raise ValueError('nonstandard cache layer requires a family reader')
                pairs = [(layer.keys, layer.values) for layer in layers]
            elif hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
                if len(cache.key_cache) != len(cache.value_cache):
                    raise ValueError('inconsistent DynamicCache key/value layers')
                pairs = list(zip(cache.key_cache, cache.value_cache))
            else:
                raise ValueError('unrecognized DynamicCache layout; supply explicit reader')
        else:
            raise ValueError('unsupported cache type; explicit family cache reader required')
        if len(pairs) != self.num_layers:
            raise ValueError('cache must expose every layer')
        result = {'position': np.array(context.position, dtype=np.int64)}
        for layer, pair in enumerate(pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError('extra layer state requires an explicit reader')
            for name, tensor in zip(('key', 'value'), pair):
                value = detached(tensor)
                if value.ndim != 4 or value.shape[0] != 1 or value.shape[2] != context.position:
                    raise ValueError('KV reader needs fully occupied non-sliding [1, heads, tokens, dim]')
                result[f'layer.{layer}.{name}'] = value
        return result


class MappedCacheReader:
    """Explicit accessor for EVERY leaf, including recurrent/MLA extras.

    Accessors receive CacheContext and return observed state, not recomputed
    equations. No default selections or filtering. Caller owns occupied slicing
    and lossless layout mapping for that exact family/version.
    """

    def __init__(self, schema, readers):
        if not schema or set(readers) != set(schema) or any(not callable(r) for r in readers.values()):
            raise ValueError('one callable reader required for every declared cache leaf')
        self.readers = dict(readers)

    def __call__(self, context):
        return {name: reader(context) for name, reader in self.readers.items()}


class SnapshotLayerReader:
    """Explicit complete mapping of native (observed_position, layer_tuples).

    Each layer spec lists EVERY snapshot tuple entry in its actual order.
    A leaf declares name, optional occupied token_axis, optional transpose axes,
    and optional batch_axis insertion. Recurrent matrices/histories have no
    token_axis. Unknown/extra members fail instead of being filtered out.
    """

    def __init__(self, schema, *, layers, position_name):
        self.schema = dict(schema)
        self.layers = layers
        self.position_name = position_name
        declared = {position_name: 'global'}
        for index, leaves in enumerate(layers):
            if not leaves:
                raise ValueError('every snapshot layer needs explicit leaf mappings')
            for spec in leaves:
                name = spec['name']
                if name in declared:
                    raise ValueError('duplicate snapshot leaf mapping')
                declared[name] = index
                if set(spec) - {'name', 'token_axis', 'axes', 'batch_axis'}:
                    raise ValueError('unsupported snapshot mapping operation')
        if declared != self.schema:
            raise ValueError('snapshot mappings must exactly cover declared cache_schema')

    def __call__(self, context):
        snapshot = context.cache
        if not isinstance(snapshot, (tuple, list)) or len(snapshot) != 2:
            raise ValueError('reader requires native (position, layer_tuples) snapshot')
        position, layers = snapshot
        if not isinstance(position, (int, np.integer)) or position != context.position:
            raise ValueError('observed native position disagrees with appended token count')
        if len(layers) != len(self.layers):
            raise ValueError('snapshot has missing/extra layers')
        result = {self.position_name: np.asarray(position, dtype=np.int64)}
        for observed, specs in zip(layers, self.layers):
            if not isinstance(observed, (tuple, list)) or len(observed) != len(specs):
                raise ValueError('every observed native layer member needs a mapping')
            for value, spec in zip(observed, specs):
                value = detached(value)
                if 'token_axis' in spec:
                    axis = spec['token_axis']
                    if type(axis) is not int or not 0 <= axis < value.ndim or value.shape[axis] < position:
                        raise ValueError('invalid occupied snapshot token axis')
                    slices = [slice(None)] * value.ndim
                    slices[axis] = slice(0, position)
                    value = value[tuple(slices)]
                if 'axes' in spec:
                    value = np.transpose(value, spec['axes'])
                if 'batch_axis' in spec:
                    value = np.expand_dims(value, axis=spec['batch_axis'])
                result[spec['name']] = value
        return result


def create_snapshot_reader(*, schema, options):
    return SnapshotLayerReader(schema, layers=options['layers'],
                               position_name=options['position_name'])


def observed_path(context, selector):
    """Read an explicitly named observed object; no eval or method invocation."""
    if set(selector) != {'source', 'path'}:
        raise ValueError('selector requires exactly source and path')
    source = selector['source']
    if source not in {'cache', 'model', 'output', 'context'}:
        raise ValueError('unknown observed-state source')
    value = context if source == 'context' else getattr(context, source)
    path = selector['path']
    if not isinstance(path, list):
        raise ValueError('observed path must be an explicit list of keys/indices/attributes')
    for step in path:
        if type(step) is int:
            value = value[step]
        elif isinstance(step, str) and step:
            value = value[step] if isinstance(value, Mapping) else getattr(value, step)
        else:
            raise ValueError('invalid observed path component')
    return value


class ObservedPathReader:
    """Declarative occupied recurrent/MLA/KV state export with exact coverage.

    No family aliases or assumed field names. Every leaf declares its observed
    source/path; optional slicing/transpose maps physical to canonical layout.
    This does not prove the caller's schema includes all internal family state.
    """

    def __init__(self, schema, *, leaves):
        if not schema or set(leaves) != set(schema):
            raise ValueError('observed paths must explicitly cover every cache leaf')
        self.leaves = leaves
        for spec in leaves.values():
            if not {'source', 'path'} <= set(spec) or set(spec) - {'source', 'path', 'token_axis', 'length', 'axes', 'batch_axis'}:
                raise ValueError('invalid observed-state mapping')
            if 'length' in spec and 'token_axis' not in spec:
                raise ValueError('occupied length requires explicit token_axis')

    def __call__(self, context):
        result = {}
        for name, spec in self.leaves.items():
            value = detached(observed_path(context, {key: spec[key] for key in ('source', 'path')}))
            if 'token_axis' in spec:
                axis = spec['token_axis']
                length = (observed_path(context, spec['length']) if 'length' in spec else context.position)
                length = detached(length)
                if length.size != 1 or length.dtype.kind not in 'iu':
                    raise ValueError('occupied length must be an observed scalar integer')
                length = int(length.reshape(-1)[0])
                if type(axis) is not int or not 0 <= axis < value.ndim or not 0 < length <= value.shape[axis]:
                    raise ValueError('invalid occupied observed-state axis or length')
                slices = [slice(None)] * value.ndim
                slices[axis] = slice(0, length)
                value = value[tuple(slices)]
            if 'axes' in spec:
                value = np.transpose(value, spec['axes'])
            if 'batch_axis' in spec:
                value = np.expand_dims(value, spec['batch_axis'])
            result[name] = value
        return result


def create_observed_path_reader(*, schema, options):
    return ObservedPathReader(schema, leaves=options['leaves'])
