"""Concrete current K3Execution compressed-cache mapping and factory wiring.

The standard adapter has text-token scope. The separate k3_media adapter uses
the same publisher handoff for complete expanded media prefixes. This module
exports raw compressed state. Explicit
forward comparison projection lives in k3_projection, not this raw reader.
No vendor source patching, FLA substitutes, builds or parity assertions.
"""
import copy
from dataclasses import asdict
import inspect
from pathlib import Path

import numpy as np

from .adapters import NativeExecutorAdapter, TorchCausalAdapter, reader_from_options
from .cache_readers import detached
from .local_transformers import LocalTransformersFactory
from .runner import digest_file, pinned_artifacts


def k3_cache_schema(config):
    schema = {'position': 'global'}
    for index, kind in enumerate(config.layer_types):
        prefix = f'layer.{index}.'
        schema[prefix + 'position'] = index
        leaves = ('state', 'conv_q', 'conv_k', 'conv_v') if kind == 'kda' else ('latent', 'rotary_key')
        for leaf in leaves:
            schema[prefix + leaf] = index
    return schema


def k3_empty_cache_shapes(config,*,fla=False):
    """Explicit structural zero extents, not placeholders for missing state."""
    result = {}
    for index, kind in enumerate(config.layer_types):
        if kind == 'kda' and config.kernel == 1 and not fla:
            for name in ('conv_q', 'conv_k', 'conv_v'):
                result[f'layer.{index}.{name}'] = [1, config.kda_heads, config.kda_dim, 0]
        if kind == 'mla' and config.rope == 0:
            result[f'layer.{index}.rotary_key'] = [1, None, 0]
    return result


def config_from_mapping(value):
    from ..k3_execution import K3Config
    if isinstance(value, K3Config):
        config = value
    else:
        data = copy.deepcopy(dict(value))
        data['layer_types'] = tuple(data['layer_types'])
        if data.get('vision') is not None:
            from ..k3_execution.vision import K3VisionConfig
            vision=dict(data['vision']);vision['merge']=tuple(vision['merge'])
            data['vision']=K3VisionConfig(**vision)
        config = K3Config(**data)
    return config


def checked_leaf(value, shape, name, *, dtype=np.float32):
    array = detached(value)
    if array.shape != shape or (dtype is not None and array.dtype != dtype) or array.dtype.kind != 'f' or not np.isfinite(array).all():
        raise ValueError('invalid observed K3 cache geometry/dtype: ' + name)
    return array


def validate_k3_leaves(raw,config,position,*,require_fp32=True,fla=False):
    """Both readers must export exactly the SAME explicitly observed layout."""
    if type(position) is not int or not 0<position<=config.capacity or type(fla) is not bool:
        raise ValueError('occupied bounded K3 physical observation position and layout required')
    schema = k3_cache_schema(config)
    if set(raw) != set(schema):
        raise ValueError('K3 cache reader must expose every declared layer and state leaf')
    result = {}
    for name in ['position', *(f'layer.{i}.position' for i in range(len(config.layer_types)))]:
        value = detached(raw[name])
        if value.shape != () or value.dtype.kind not in 'iu' or int(value) != position:
            raise ValueError('observed K3 prefix positions do not agree: ' + name)
        result[name] = value
    for index, kind in enumerate(config.layer_types):
        prefix = f'layer.{index}.'
        if kind == 'kda':
            shapes = {'state': (1, config.kda_heads, config.kda_dim, config.kda_dim)}
            shapes.update({name: (1,config.kda_heads*config.kda_dim,config.kernel) if fla else
                           (1,config.kda_heads,config.kda_dim,config.kernel-1)
                           for name in ('conv_q', 'conv_k', 'conv_v')})
        else:
            shapes = {'latent': (1, position, config.kvrank), 'rotary_key': (1, position, config.rope)}
        for name, shape in shapes.items():
            result[prefix + name] = checked_leaf(raw[prefix + name], shape, prefix + name,
                                                dtype=np.float32 if require_fp32 else None)
    return result


def indexed(items, expected, name):
    if type(items) is not tuple:
        raise ValueError(name + ' snapshot inventory must be tuple')
    result = {}
    for item in items:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError('invalid ' + name + ' snapshot entry')
        index, value = item
        if type(index) is not int or index not in expected or index in result:
            raise ValueError('duplicate/foreign ' + name + ' layer')
        result[index] = value
    if set(result) != expected:
        raise ValueError('missing ' + name + ' layers')
    return result


class K3ExecutionCacheReader:
    """Observe all current K3Snapshot KDA members and compressed MLA pages.

    KDA state [1, heads, key, value]; histories [1, heads, dim, oldest-first taps].
    MLA normalized latent [1, occupied tokens, rank], rotary key [1, tokens, rope].
    Page boundaries are physical storage only; concatenate occupied rows, without
    expanding latent keys/values or projecting/normalizing any cache values.
    """

    def __init__(self,config,schema,*,fla=False,evidence_schema=None):
        self.config = config_from_mapping(config)
        if type(fla) is not bool:raise ValueError('explicit physical FLA cache policy required')
        self.fla=fla
        self.evidence_schema={} if evidence_schema is None else dict(evidence_schema)
        if dict(schema) != k3_cache_schema(self.config):
            raise ValueError('K3 cache_schema must be the complete compressed-state schema')

    def __call__(self, context):
        from ..k3_execution.executor import K3Execution, K3Snapshot
        from ..k3_execution.cache import MLALatentPages
        executor, snapshot, config = context.model, context.cache, self.config
        if type(executor) is not K3Execution or type(snapshot) is not K3Snapshot:
            raise ValueError('reader requires current K3Execution and its actual K3Snapshot')
        if executor.config != config or snapshot.identity != executor._identity:
            raise ValueError('K3 snapshot owner/config mismatch')
        if (executor._fla_policy is not None)!=self.fla:
            raise ValueError('observed K3 recurrence owner differs from physical cache policy')
        position = snapshot.position
        if type(position) is not int or position != context.position or executor.position != position:
            raise ValueError('K3 occupied prefix disagrees with appended-token position')
        expected_kda = {i for i, kind in enumerate(config.layer_types) if kind == 'kda'}
        expected_mla = {i for i, kind in enumerate(config.layer_types) if kind == 'mla'}
        kda = indexed(snapshot.kda, expected_kda, 'KDA')
        mla = indexed(snapshot.mla, expected_mla, 'MLA')
        if set(executor._kda) != expected_kda or set(executor._mla) != expected_mla:
            raise ValueError('live K3 cache inventory differs from declared schedule')
        raw = {'position': np.asarray(position, dtype=np.int64)}
        for index, item in kda.items():
            if self.fla:
                from ..k3_execution.fla_mixer import FLAMixerSnapshot
                mixer=executor._kda[index];torch=mixer._torch
                if (type(item) is not FLAMixerSnapshot or item.identity!=mixer.identity or
                        item.position!=position or mixer.position!=position or
                        type(item.convolution) is not tuple or len(item.convolution)!=3):
                    raise ValueError('complete original FLA physical snapshot required')
                prefix=f'layer.{index}.'
                raw[prefix+'position']=np.asarray(position,dtype=np.int64)
                if (not isinstance(item.recurrent,torch.Tensor) or item.recurrent.dtype!=torch.float32 or
                        item.recurrent.device!=mixer.device):
                    raise ValueError('original FP32 FLA recurrence required')
                raw[prefix+'state']=checked_leaf(item.recurrent,
                    (1,config.kda_heads,config.kda_dim,config.kda_dim),prefix+'state')
                for name,history in zip(('conv_q','conv_k','conv_v'),item.convolution):
                    if (not isinstance(history,torch.Tensor) or history.dtype!=torch.bfloat16 or history.device!=mixer.device):
                        raise ValueError('original BF16 FLA convolution buffer required')
                    raw[prefix+name]=checked_leaf(history,
                        (1,config.kda_heads*config.kda_dim,config.kernel),prefix+name)
                continue
            if type(item) is not tuple or len(item) != 3:
                raise ValueError('KDA snapshot must contain position/state/three histories')
            local_position, state, histories = item
            if type(local_position) is not int or local_position != position or executor._kda[index].position != position:
                raise ValueError('KDA layer prefix position mismatch')
            if type(histories) is not tuple or len(histories) != 3:
                raise ValueError('all Q/K/V KDA convolution histories required')
            prefix = f'layer.{index}.'
            raw[prefix + 'position'] = np.asarray(local_position, dtype=np.int64)
            state = checked_leaf(state, (config.kda_heads, config.kda_dim, config.kda_dim), prefix + 'state')
            raw[prefix + 'state'] = state[None]
            for name, history in zip(('conv_q', 'conv_k', 'conv_v'), histories):
                history = checked_leaf(history, (config.kernel - 1, config.kda_heads, config.kda_dim), prefix + name)
                raw[prefix + name] = history.transpose(1, 2, 0)[None]
        for index, pages in mla.items():
            cache = executor._mla[index]
            if type(cache) is not MLALatentPages or cache.length != position or cache.rank != config.kvrank or cache.rope != config.rope:
                raise ValueError('compressed MLA live geometry/position mismatch')
            if type(pages) is not tuple or not pages:
                raise ValueError('nonempty occupied compressed MLA pages required')
            copied, count = [], 0
            for page_index, page in enumerate(pages):
                value = detached(page)
                if value.ndim != 2 or value.shape[1] != config.kvrank + config.rope:
                    raise ValueError('compressed MLA snapshot contains expanded or foreign layout')
                extent = value.shape[0]
                if not 1 <= extent <= cache.page_tokens or (page_index < len(pages) - 1 and extent != cache.page_tokens):
                    raise ValueError('invalid occupied compressed MLA page extent')
                checked_leaf(value, (extent, config.kvrank + config.rope), f'layer.{index}.page.{page_index}')
                copied.append(value)
                count += extent
            if count != position:
                raise ValueError('compressed MLA page row count disagrees with exact prefix')
            joined = np.concatenate(copied, axis=0)
            prefix = f'layer.{index}.'
            raw[prefix + 'position'] = np.asarray(count, dtype=np.int64)
            raw[prefix + 'latent'] = joined[None, :, :config.kvrank]
            raw[prefix + 'rotary_key'] = joined[None, :, config.kvrank:]
        return validate_k3_leaves(raw,config,position,fla=self.fla)

    def evidence(self,context):
        value=getattr(context.model,'_comparison_observations',None)
        if not isinstance(value,dict) or set(value)!=set(self.evidence_schema):
            raise ValueError('complete actual executed K3 routing/arithmetic observations required')
        return {name:detached(array) for name,array in value.items()}


class K3AuthoritativeCacheReader:
    """Guard a caller-injected observed vendor reader; NEVER infer alignment.

    Expanded vendor KV cannot be inverted into this latent schema here. Vendor
    recurrent matrices/history order and normalized latent/rotary coordinates
    must be directly observable and explicitly mapped by the caller's reader.
    """

    def __init__(self,config,schema,reader,*,fla=False):
        self.config = config_from_mapping(config)
        if type(fla) is not bool:raise ValueError('explicit authoritative physical recurrence policy required')
        self.fla=fla
        if dict(schema) != k3_cache_schema(self.config) or not callable(reader):
            raise ValueError('explicit authoritative all-layer compressed cache reader required')
        self.reader = reader

    def __call__(self, context):
        return validate_k3_leaves(self.reader(context),self.config,context.position,require_fp32=False,fla=self.fla)


def k3_text_forward(executor, token_ids):
    """Native cache append offset and rotary prefix positions share one origin."""
    start = executor.position
    return executor.forward(token_ids, position_ids=np.arange(start, start + len(token_ids), dtype=np.int64))


def create_k3_cache_reader(*, schema, options):
    return K3ExecutionCacheReader(options['execution_config'],schema,fla=options.get('physical_fla_cache',False),
        evidence_schema=options.get('evidence_schema'))


def observation_options(options,config,metadata):
    policy=options.get('observations')
    if policy is None:
        if metadata.get('evidence_schema'):raise ValueError('executed K3 evidence requires an explicit observation policy')
        return None
    if type(policy) is not dict or set(policy)!={'max_bytes','routing','projection_names'}:
        raise ValueError('exact bounded executed K3 observation policy required')
    from ..k3_execution.observations import K3ObservationRecorder,observation_schema
    policy=copy.deepcopy(policy)
    if type(policy['projection_names']) is not list:raise ValueError('explicit named projection observation list required')
    policy['projection_names']=tuple(policy['projection_names'])
    K3ObservationRecorder(config,**policy)
    expected=observation_schema(config,routing=policy['routing'],projection_names=policy['projection_names'])
    if not expected or metadata.get('evidence_schema')!=expected or metadata.get('empty_evidence_shapes'):
        raise ValueError('complete executed K3 observation schema without empty placeholders required')
    return policy


def observed_text_forward(policy):
    def forward(executor,tokens):
        executor._comparison_observations=None
        try:
            with executor.observation_scope(**policy) as recorder:
                logits=k3_text_forward(executor,tokens)
                executor._comparison_observations=dict(recorder.export(rows=len(tokens)))
                return logits
        except BaseException:
            executor._comparison_observations=None
            raise
    return forward


def verify_checkpoint(root, path, pins):
    root = Path(root).resolve()
    directory = (root / path).resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise ValueError('checkpoint must be a local directory beneath offline_root')
    rows = pinned_artifacts({'artifacts': pins}, root)
    indexed_paths = {Path(row['resolved_path']): row for row in rows}
    for file in directory.rglob('*'):
        if file.is_file() and (not file.resolve().is_relative_to(directory) or file.resolve() not in indexed_paths):
            raise ValueError('unpinned/escaping local K3 checkpoint file: ' + str(file))
    if directory / 'config.json' not in indexed_paths or indexed_paths[directory / 'config.json']['kind'] != 'config':
        raise ValueError('K3 checkpoint config.json must be pinned as config')
    return directory, indexed_paths


class K3CandidateFactory:
    """Concrete local publisher-checkpoint handoff to existing K3Execution.

    Config/library choices are caller-injected. Existing load_k3 conversion is
    used unchanged and must produce exactly that config. No admission flags are
    promoted; exact publisher semantics remain the executor's explicit gap.
    """

    def __init__(self, root, options):
        self.root, self.options = Path(root).resolve(), dict(options)
        self.config = config_from_mapping(options['execution_config'])
        self.pins = options['_artifact_pins']
        self.checkpoint, rows = verify_checkpoint(self.root, options['checkpoint_path'], self.pins)
        libraries = options['libraries']
        if not {'kimi', 'attention', 'operators'} <= set(libraries) or set(libraries) - {'kimi', 'attention', 'operators', 'mxfp4'}:
            raise ValueError('explicit kimi, compressed attention and operators libraries required')
        self.libraries = {}
        for key, path in libraries.items():
            resolved = (self.root / path).resolve()
            if resolved not in rows or rows[resolved]['kind'] not in ('source', 'dependency'):
                raise ValueError('each prebuilt native library must be a pinned source/dependency artifact')
            self.libraries[key] = resolved
        self.execution_policies = {}
        for name in ('routing_policy','arithmetic_policy','fla_policy','metal_policy','packed_decoder_policy','attention_policy'):
            if name in options:
                if not isinstance(options[name], dict):
                    raise ValueError(name + ' must contain explicit constructor options')
                self.execution_policies[name] = copy.deepcopy(options[name])
        if 'fla_policy' in self.execution_policies and 'metal_policy' in self.execution_policies:
            raise ValueError('choose exactly one physical device recurrence implementation')
        if ({'fla_policy','metal_policy'} & self.execution_policies.keys() and
                not {'arithmetic_policy','routing_policy'}<=self.execution_policies.keys()):
            raise ValueError('physical device recurrence requires explicit arithmetic and routing policies')
        if 'packed_decoder_policy' in self.execution_policies and 'arithmetic_policy' not in self.execution_policies:
            raise ValueError('library packed decoding requires explicit arithmetic policy')
        if 'attention_policy' in self.execution_policies and 'arithmetic_policy' not in self.execution_policies:
            raise ValueError('FlashAttention requires explicit arithmetic policy')
        if self.config.expert_format != 'fp32' and 'mxfp4' not in self.libraries and 'arithmetic_policy' not in self.execution_policies:
            raise ValueError('packed experts require explicit prebuilt MXFP4 library')
        if options['metadata']['config_sha256'] != digest_file(self.checkpoint / 'config.json'):
            raise ValueError('candidate metadata config identity differs from actual checkpoint')

    def __call__(self, *, seed):
        from ..publisher_checkpoint.service import PublisherCheckpoint
        from ..publisher_checkpoint.k3_adapter import load_k3, k3_route_identity
        from ..k3_execution.executor import K3Execution
        verify_checkpoint(self.root, self.options['checkpoint_path'], self.pins)
        source = inspect.getsourcefile(K3Execution)
        if source is None or digest_file(source) not in {r['sha256'] for r in self.pins if r['kind'] == 'source'}:
            raise ValueError('actual K3Execution source is not covered by source pins')
        checkpoint = PublisherCheckpoint.open(self.checkpoint, family='kimi_k3')
        c = self.config
        # Candidate/evidence tooling constructs K3Execution directly instead of
        # going through create_k3_execution_service. Bind the exact P02 tuple
        # here, before load_k3 can read or convert any model payload.
        if 'metal_policy' in self.execution_policies:
            route_backend, route_arithmetic = 'metal', 'bf16'
        elif 'fla_policy' in self.execution_policies:
            route_backend, route_arithmetic = 'fla', 'bf16'
        elif 'arithmetic_policy' in self.execution_policies:
            route_backend, route_arithmetic = 'device', 'bf16'
        else:
            route_backend, route_arithmetic = 'native', 'fp32-approximate'
        k3_route_identity(checkpoint, backend=route_backend,
                          arithmetic=route_arithmetic,
                          expert_format=c.expert_format)
        payload = load_k3(checkpoint, capacity=c.capacity,
            max_weight_bytes=self.options['max_weight_bytes'], expert_format=c.expert_format,
            projection_epsilon=c.projection_epsilon, l2_epsilon=c.l2_epsilon,
            rotary_mode=c.rotary_mode, rotary_theta=c.rotary_theta,
            include_vision=c.vision is not None,
            vision_tower_epsilon=2**-23 if c.vision is None else c.vision.tower_epsilon,
            source_backend=route_backend, source_arithmetic=route_arithmetic)
        if payload.config != c:
            raise ValueError('injected K3Config differs from unchanged local checkpoint handoff')
        if payload.config_sha256 != self.options['metadata']['config_sha256']:
            raise ValueError('loaded K3 payload config hash differs from declared identity')
        policies = {}
        if 'routing_policy' in self.execution_policies:
            from ..k3_execution.routing import TorchK3Router
            policies['routing'] = TorchK3Router(**self.execution_policies['routing_policy'])
        if 'arithmetic_policy' in self.execution_policies:
            from ..k3_execution.bf16_arithmetic import K3BF16Arithmetic
            arithmetic_options = copy.deepcopy(self.execution_policies['arithmetic_policy'])
            if 'packed_decoder' in arithmetic_options:
                raise ValueError('use packed_decoder_policy for the pinned decoder')
            if 'attention_backend' in arithmetic_options:
                raise ValueError('use attention_policy for the pinned attention backend')
            if 'attention_policy' in self.execution_policies:
                from ..k3_execution.flash_attention import K3FlashAttention
                arithmetic_options['attention_backend'] = K3FlashAttention(
                    **self.execution_policies['attention_policy'])
            if 'packed_decoder_policy' in self.execution_policies:
                from ..k3_execution.compressed_tensors import K3CompressedTensorsDecoder
                arithmetic_options['packed_decoder'] = K3CompressedTensorsDecoder(
                    **self.execution_policies['packed_decoder_policy'])
            policies['arithmetic'] = K3BF16Arithmetic(**arithmetic_options)
        if 'fla_policy' in self.execution_policies:
            from ..k3_execution.fla_mixer import FLAK3Policy
            policies['fla_policy'] = FLAK3Policy(**self.execution_policies['fla_policy'])
        if 'metal_policy' in self.execution_policies:
            from ..k3_execution.metal_mixer import MetalK3Policy
            policies['fla_policy']=MetalK3Policy(**self.execution_policies['metal_policy'])
        executor = K3Execution(c, payload.tensors, library=self.libraries['kimi'],
            attention_library=self.libraries['attention'], operators_library=self.libraries['operators'],
            mxfp4_library=self.libraries.get('mxfp4'), max_bytes=self.options['max_bytes'], **policies)
        try:
            verify_checkpoint(self.root, self.options['checkpoint_path'], self.pins)
        except BaseException:
            executor.close()
            raise
        return executor


def k3_metadata(value,config,*,fla=False):
    if type(fla) is not bool:raise ValueError('explicit physical recurrence metadata policy required')
    metadata = copy.deepcopy(value)
    if metadata.get('model_type') != 'kimi_k3' or metadata.get('revision') != config.revision:
        raise ValueError('K3 metadata requires exact wrapper identity and pinned executor revision')
    if metadata.get('num_layers') != len(config.layer_types) or metadata.get('vocabulary_size') != config.vocab:
        raise ValueError('K3 metadata geometry differs from explicit execution config')
    if metadata.get('cache_schema') != k3_cache_schema(config):
        raise ValueError('explicit complete compressed K3 cache_schema required')
    empties=k3_empty_cache_shapes(config,fla=fla)
    if metadata.get('empty_cache_shapes', {}) != empties:
        raise ValueError('K3 structural empty shapes must be explicitly declared exactly')
    metadata['cache_layout']=('k3-observed-normalized-latent-rotary-fla-raw-full-convolution-v1' if fla else
        'k3-observed-normalized-latent-rotary-kda-oldest-first-v1')
    metadata['execution_scope'] = 'text-tokens-only'
    metadata['execution_config'] = asdict(config)
    return metadata


def create_k3_candidate_adapter(*, offline_root, options):
    factory = K3CandidateFactory(offline_root, options)
    fla=bool({'fla_policy','metal_policy'} & options.keys())
    metadata=k3_metadata(options['metadata'],factory.config,fla=fla)
    if metadata.get('authoritative') is True:
        raise ValueError('original K3Execution candidate is not an authoritative vendor oracle')
    observations=observation_options(options,factory.config,metadata)
    kwargs={}
    if fla:
        import torch
        kwargs['torch']=torch
    return NativeExecutorAdapter(factory,metadata,
        cache_reader=K3ExecutionCacheReader(factory.config,metadata['cache_schema'],fla=fla,evidence_schema=metadata.get('evidence_schema')),
        forward_call=k3_text_forward if observations is None else observed_text_forward(observations),
        close_call=options.get('close_call'),**kwargs)


def create_k3_authoritative_adapter(*, offline_root, options):
    """Local unchanged vendor loader plus REQUIRED explicit observed-state reader.

    Uses existing authorized/pinned LocalTransformersFactory. If its auto-model
    entry point or actual compressed vendor cache is unavailable, fail, not alias
    to K3Execution or fabricate state from expanded KV.
    """
    config = config_from_mapping(options['execution_config'])
    fla=options.get('physical_fla_cache',False)
    metadata=k3_metadata(options['metadata'],config,fla=fla)
    if metadata.get('authoritative') is not True or metadata.get('unchanged') is not True:
        raise ValueError('authoritative K3 factory requires reviewed unchanged vendor provenance')
    reader = reader_from_options(options)
    if reader is None:
        raise ValueError('K3 authoritative factory requires explicit observed compressed/KDA cache reader')
    factory = LocalTransformersFactory(offline_root, options)
    return TorchCausalAdapter(factory, metadata,
        cache_reader=K3AuthoritativeCacheReader(config,metadata['cache_schema'],reader,fla=fla),
        input_device=options.get('input_device'), position_ids=options.get('position_ids', True),
        cache_position=options.get('cache_position', False), forward_call=options.get('forward_call'))
