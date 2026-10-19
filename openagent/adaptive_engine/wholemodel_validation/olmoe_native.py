"""Pinned whole-model adapter for the native leased OLMoE decoder."""
from pathlib import Path

import numpy as np

from .adapters import NativeExecutorAdapter


def cache_schema(num_layers):
    if type(num_layers) is not int or not 1 <= num_layers <= 256:
        raise ValueError('bounded OLMoE layer inventory required')
    result = {'position': 'global'}
    for layer in range(num_layers):
        result[f'layer.{layer}.key'] = layer
        result[f'layer.{layer}.value'] = layer
    return result


def read_complete_cache(context):
    """Map every observed native OLMoE KV leaf without reconstructing state."""
    state = context.cache
    if (type(state) is not dict or set(state) !=
            {'format', 'edge_execution_identity', 'range_state'} or
            state.get('format') != 'openagent.native-olmoe-decoder.v1' or
            type(state.get('edge_execution_identity')) is not str or
            not state['edge_execution_identity']):
        raise ValueError('complete owned native OLMoE decoder state required')
    physical = state['range_state']
    if (type(physical) is not dict or type(physical.get('position')) is not int or
            physical['position'] != context.position or
            type(physical.get('keys')) is not list or
            type(physical.get('values')) is not list or
            len(physical['keys']) != len(physical['values']) or
            not physical['keys']):
        raise ValueError('native OLMoE KV state position or layer inventory differs')
    result = {'position': np.asarray(physical['position'], dtype=np.int64)}
    for index, (key, value) in enumerate(zip(physical['keys'], physical['values'])):
        for name, tensor in (('key', key), ('value', value)):
            if (not isinstance(tensor, np.ndarray) or tensor.dtype != np.float32 or
                    tensor.ndim != 3 or tensor.shape[1] != context.position or
                    not np.isfinite(tensor).all()):
                raise ValueError(f'invalid native OLMoE {name} cache at layer {index}')
            # Native physical layout is [heads, occupied_tokens, dim]. The
            # comparison contract uses the same logical axes with batch=1.
            result[f'layer.{index}.{name}'] = tensor[None, ...]
    return result


class _OwnedOlmoeExecutor:
    def __init__(self, session, edge, engine):
        self.session, self.edge, self.engine = session, edge, engine
        self.closed = False

    def forward(self, token_ids):
        if self.closed:
            raise RuntimeError('native OLMoE comparison executor is closed')
        return self.session.forward(token_ids)

    def snapshot_state(self):
        if self.closed:
            raise RuntimeError('native OLMoE comparison executor is closed')
        return self.session.snapshot_state()

    def restore_state(self, state):
        if self.closed:
            raise RuntimeError('native OLMoE comparison executor is closed')
        self.session.restore_state(state)

    def close(self):
        if self.closed:
            return
        errors = []
        for owner in (self.session, self.edge, self.engine):
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            for error in errors[1:]:
                errors[0].add_note('additional native OLMoE owner cleanup failed: '+repr(error))
            raise errors[0]
        self.closed = True


def _artifact_path(root, relative, pins, *, kinds):
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError('relative pinned OLMoE artifact path required')
    from ..session_recovery import _path
    path = _path(root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('OLMoE adapter artifact must be a regular file below offline root')
    rows = [row for row in pins if row.get('path') == relative and row.get('kind') in kinds]
    if len(rows) != 1:
        raise ValueError('OLMoE adapter artifact must have one exact role pin: '+relative)
    from .runner import digest_file
    if digest_file(path) != rows[0].get('sha256'):
        raise ValueError('OLMoE adapter artifact differs from runner pin: '+relative)
    return path


def create_olmoe_native_adapter(*, offline_root, options):
    """Create a candidate adapter from already-pinned local files only.

    Required options: metadata, manifest, library, context_tokens and explicit
    positive resident/cache/transfer/staging/snapshot byte budgets. The
    manifest is a weights pin; the existing native library is a dependency
    pin. No build, download, or fallback is attempted.
    """
    root = Path(offline_root).resolve(strict=True)
    if not root.is_dir() or type(options) is not dict:
        raise ValueError('offline OLMoE artifact root and explicit options required')
    pins = options.get('_artifact_pins')
    if type(pins) is not list:
        raise ValueError('runner-supplied complete OLMoE artifact pins required')
    metadata = dict(options.get('metadata', {}))
    required = {'model_type', 'revision', 'config_sha256', 'weights_sha256',
                'tokenizer_sha256', 'vocabulary_size', 'num_layers', 'dtype',
                'implementation', 'implementation_revision', 'device',
                'dependencies', 'cache_schema'}
    if (not required.issubset(metadata) or metadata['model_type'] != 'olmoe' or
            metadata['cache_schema'] != cache_schema(metadata['num_layers'])):
        raise ValueError('complete OLMoE comparison metadata and physical cache schema required')
    manifest_path = _artifact_path(root, options.get('manifest'), pins, kinds={'weights', 'dependency'})
    library_path = _artifact_path(root, options.get('library'), pins, kinds={'dependency'})
    budgets = ('context_tokens', 'resident_bytes', 'expert_cache_bytes', 'transfer_bytes',
               'staging_bytes', 'snapshot_bytes')
    if any(type(options.get(name)) is not int or options[name] <= 0 for name in budgets):
        raise ValueError('positive explicit native OLMoE execution budgets required')

    def factory(*, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError('nonnegative OLMoE comparison session seed required')
        from ..expert_runtime import SparseExpertManifest
        from ..native_leased_olmoe_segments import NativeLeasedOlmoeSegmentEngine
        from ..native_olmoe_segments import NativeOlmoeEdge
        manifest = SparseExpertManifest.read(manifest_path)
        engine = edge = session = None
        try:
            engine = NativeLeasedOlmoeSegmentEngine(
                root, manifest, 0, metadata['num_layers'], library_path,
                options['resident_bytes'], options['expert_cache_bytes'],
                options['context_tokens'], max_bytes=options['transfer_bytes'],
                max_staging_bytes=options['staging_bytes'],
                max_snapshot_bytes=options['snapshot_bytes'])
            edge = NativeOlmoeEdge(
                root, library_path, options['resident_bytes'],
                max_bytes=options['transfer_bytes'])
            session = engine.create_decoder_session(edge,
                context_tokens=options['context_tokens'])
            return _OwnedOlmoeExecutor(session, edge, engine)
        except BaseException as primary:
            for owner in (session, edge, engine):
                if owner is None:
                    continue
                try:
                    owner.close()
                except BaseException as cleanup:
                    primary.add_note('partial native OLMoE adapter construction cleanup failed: '+repr(cleanup))
            raise

    return NativeExecutorAdapter(factory, metadata, cache_reader=read_complete_cache)


__all__ = ['cache_schema', 'read_complete_cache', 'create_olmoe_native_adapter']
