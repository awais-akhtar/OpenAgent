"""Export observed V4.1 incremental state without inventing oracle layouts."""
import numpy as np


V41_REPLAY_SCHEMA = {'position': 'global', 'token_history': 'global'}


class V41ReplaySnapshotReader:
    """Export only persistent state actually exposed by current V41Snapshot.

    This legacy reader exports global history only. Use the incremental reader
    for local/global/index/Engram state. Global history alone cannot satisfy a
    whole-layer cache comparison gate.
    """
    full_layer_cache_evidence = False

    def __init__(self, schema):
        if dict(schema) != V41_REPLAY_SCHEMA:
            raise ValueError('V41 snapshot exposes only explicit global replay position/token history')

    def __call__(self, context):
        from ..v41_execution.execution import V41Execution, V41Snapshot
        executor, snapshot = context.model, context.cache
        if type(executor) is not V41Execution or type(snapshot) is not V41Snapshot:
            raise ValueError('V41 reader requires current executor and its actual snapshot')
        if snapshot.identity != executor._identity or snapshot.tokens != executor._tokens:
            raise ValueError('foreign/stale V41 replay snapshot')
        if len(snapshot.tokens) != context.position or executor.length != context.position:
            raise ValueError('V41 replay prefix position mismatch')
        if any(type(token) is not int or not 0 <= token < executor.config.vocabulary for token in snapshot.tokens):
            raise ValueError('invalid observed V41 replay tokens')
        return {'position': np.asarray(len(snapshot.tokens), dtype=np.int64),
                'token_history': np.asarray(snapshot.tokens, dtype=np.int64)}


def create_v41_replay_reader(*, schema, options):
    return V41ReplaySnapshotReader(schema)


def create_v41_full_cache_reader(*, schema, options):
    return V41IncrementalCacheReader(schema)


def create_v41_publisher_candidate_adapter(*, offline_root, options):
    """Concrete public packed publisher candidate, distinct from private V41Execution."""
    from .v41_publisher import create_v41_publisher_candidate_adapter as create
    return create(offline_root=offline_root, options=options)


def v41_cache_schema(config):
    schema = dict(V41_REPLAY_SCHEMA)
    for layer in range(config.layers):
        for name in ('local_key', 'local_value', 'pending', 'gates', 'compressed',
                     'index', 'engram_position', 'engram_tokens'):
            schema[f'layer.{layer}.{name}'] = layer
    return schema


def v41_empty_cache_shapes(config):
    result = {'token_history': [0]}
    packed = config.hidden // 16 * 9
    for layer in range(config.layers):
        for name in ('local_key', 'local_value', 'compressed'):
            result[f'layer.{layer}.{name}'] = [0, packed]
        for name in ('pending', 'gates', 'index'):
            result[f'layer.{layer}.{name}'] = [0, config.hidden]
        result[f'layer.{layer}.engram_tokens'] = [0]
    return result


class V41IncrementalCacheReader:
    """All native persistent leaves; authoritative alignment is separate.

    Packed rows are compared as bytes, not silently requantized. A publisher
    adapter must expose a matching logical contract before it can be compared
    with this equation schema.
    """
    full_layer_cache_evidence = True
    authoritative_layout_certified = False

    def __init__(self, schema):
        self.schema = dict(schema)

    def __call__(self, context):
        from ..v41_execution import V41Execution, V41Snapshot
        model, snapshot = context.model, context.cache
        if type(model) is not V41Execution or type(snapshot) is not V41Snapshot:
            raise ValueError('current V4.1 executor and snapshot required')
        if self.schema != v41_cache_schema(model.config):
            raise ValueError('complete V4.1 incremental cache schema required')
        if (snapshot.identity != model._identity or len(snapshot.tokens) != context.position
                or snapshot.tokens != model._tokens or model.length != context.position):
            raise ValueError('snapshot identity or prefix mismatch')
        model._validate_cache(snapshot.cache, snapshot.tokens)
        c, state = model.config, snapshot.cache
        output = {'position': np.asarray(context.position, dtype=np.int64),
                  'token_history': np.asarray(snapshot.tokens, dtype=np.int64)}
        packed = c.hidden // 16 * 9
        for layer in range(c.layers):
            prefix = f'layer.{layer}.'
            rows = state.local[layer]
            for index, name in enumerate(('local_key', 'local_value')):
                data = b''.join(row[index] for row in rows)
                output[prefix + name] = np.frombuffer(data, dtype=np.uint8).reshape(len(rows), packed)
            output[prefix + 'compressed'] = np.frombuffer(b''.join(state.compressed[layer]),
                dtype=np.uint8).reshape(len(state.compressed[layer]), packed)
            for name, field in (('pending', state.pending), ('gates', state.gates), ('index', state.index)):
                output[prefix + name] = np.frombuffer(b''.join(field[layer]),
                    dtype=np.float32).reshape(len(field[layer]), c.hidden)
            _, position, tokens = state.engram[layer]
            output[prefix + 'engram_position'] = np.asarray(position, dtype=np.int64)
            output[prefix + 'engram_tokens'] = np.asarray(tokens, dtype=np.int64)
        return output
