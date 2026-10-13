"""Explicit construction of native sessions from admitted local payloads."""
from dataclasses import dataclass, replace
from contextlib import contextmanager
from pathlib import Path

from .execution_service import ExecutionLimits, NativeExecutionService


def compose_public_publisher_service(backing_service, *, runtime_library, family, config,
        model_directory, checkpoint_identity, executor_bytes, max_resource_bytes,
        max_work_bytes, max_snapshot_bytes, max_batch_rows=32, max_sessions=1,
        max_snapshots_per_session=4):
    """Own concrete K3/V4.1 factories through the public native dispatcher.

    Ownership of backing_service transfers to the result. Each live public
    conversation constructs an independent embedding/head owner and a complete
    state owner, so a budgeted backing factory needs two slots per conversation.
    """
    from .publisher_public_adapters import NativePublisherPublicAdapters
    from .segment_edge_runtime import SegmentEdgeExecution, SegmentRange
    if type(backing_service) is not NativeExecutionService:
        raise TypeError('concrete publisher execution service required')
    if (type(max_sessions) is not int or max_sessions<=0 or
            type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62):
        raise ValueError('bounded explicit public publisher session and snapshot counts required')
    if backing_service._sessions or backing_service._opening or backing_service._closed:
        raise ValueError('fresh empty backing publisher service required for ownership transfer')
    slots=getattr(backing_service.factory,'count_limit',None)
    if slots is not None and slots<2*max_sessions:
        raise MemoryError('backing publisher factory must reserve both Edge and session slots')
    if any(type(value) is not int or value<=0 for value in
           (executor_bytes,max_resource_bytes,max_work_bytes,max_snapshot_bytes)):
        raise ValueError('positive complete public publisher resource budgets required')
    if 3*executor_bytes*max_sessions>max_resource_bytes:
        raise MemoryError('all public engine, Edge and session reservations must fit aggregate admission')
    owner=None
    pending={}
    def cleanup():
        for key,execution in tuple(pending.items()):
            execution.close()
            del pending[key]
        if owner is not None:
            owner.retry_cleanup()
            if owner.objects:
                raise RuntimeError('public publisher resources still borrow the backing service')
            owner.resource_ledger.close()
        backing_service.close()
    try:
        owner=NativePublisherPublicAdapters(runtime_library,backing_service,family=family,config=config,
            model_directory=model_directory,checkpoint_identity=checkpoint_identity,
            executor_bytes=executor_bytes,max_resource_bytes=max_resource_bytes,
            max_snapshot_bytes=max_snapshot_bytes,max_work_bytes=max_work_bytes,max_batch_rows=max_batch_rows)
        if (backing_service.limits.vocabulary!=owner.vocabulary or
                backing_service.limits.context!=owner.context):
            raise ValueError('backing service limits differ from original publisher geometry')
        owner=owner.register()
        limits=replace(backing_service.limits,max_sessions=max_sessions,
            max_snapshots_per_session=max_snapshots_per_session,
            prefill_chunk_tokens=min(backing_service.limits.prefill_chunk_tokens,max_batch_rows))
        def construct():
            if pending or owner.pending:
                raise RuntimeError('public publisher construction cleanup is pending')
            execution=None
            try:
                execution=SegmentEdgeExecution(runtime_library,
                    edge_adapter=owner.edge_id.decode(),model_directory=str(owner.root),
                    checkpoint_identity=owner.identity,
                    ranges=(SegmentRange(owner.segment_id.decode(),str(owner.root),0,owner.layers),),
                    context_tokens=owner.context,native_memory_bytes=executor_bytes,
                    max_transfer_bytes=max_work_bytes,max_snapshot_bytes=max_snapshot_bytes,
                    max_snapshots=max_snapshots_per_session+2)
                return execution
            except BaseException as primary:
                if execution is not None:
                    try:
                        execution.close()
                    except BaseException as failure:
                        pending[id(execution)]=execution
                        primary.add_note('public publisher executor cleanup retained: '+repr(failure))
                raise
        result=NativeExecutionService(construct,limits,factory_cleanup=cleanup)
        result.public_adapters=owner
        result.resource_ledger=owner.resource_ledger
        return result
    except BaseException as primary:
        try:
            cleanup()
        except BaseException as failure:
            primary.cleanup=cleanup
            primary.add_note('public publisher ownership transfer cleanup retained: '+repr(failure))
        raise


def create_public_k3_execution_service(payload, libraries, *, runtime_library,
        model_directory, checkpoint_identity, runtime_max_bytes, max_resource_bytes,
        max_work_bytes, max_snapshot_bytes, max_batch_rows=32, max_sessions=1,
        max_snapshots_per_session=4, **options):
    """Construct original native K3, then serve its complete public range."""
    if any(options.get(name) is not None for name in ('arithmetic','fla_policy')):
        raise ValueError('the public native K3 factory requires its counted CPU arithmetic')
    backing=create_k3_execution_service(payload,libraries,runtime_max_bytes=runtime_max_bytes,
        max_sessions=2*max_sessions,**options)
    return compose_public_publisher_service(backing,runtime_library=runtime_library,family='kimi-k3',
        config=payload.config,model_directory=model_directory,checkpoint_identity=checkpoint_identity,
        executor_bytes=runtime_max_bytes,max_resource_bytes=max_resource_bytes,
        max_work_bytes=max_work_bytes,max_snapshot_bytes=max_snapshot_bytes,max_batch_rows=max_batch_rows,
        max_sessions=max_sessions,max_snapshots_per_session=max_snapshots_per_session)


def create_public_publisher_v41_execution_service(payload, arithmetic_factory, *, library,
        runtime_library, model_directory, checkpoint_identity, runtime_max_bytes,
        executor_bytes, max_resource_bytes, max_work_bytes, max_snapshot_bytes,
        max_batch_rows=32, max_sessions=1, max_snapshots_per_session=4, **options):
    """Compose the concrete publisher V4.1 arithmetic factory through public ABI."""
    backing=create_publisher_v41_execution_service(payload,arithmetic_factory,
        library=library,runtime_max_bytes=runtime_max_bytes,max_sessions=2*max_sessions,**options)
    return compose_public_publisher_service(backing,runtime_library=runtime_library,family='deepseek-v41',
        config=payload.profile,model_directory=model_directory,checkpoint_identity=checkpoint_identity,
        executor_bytes=executor_bytes,max_resource_bytes=max_resource_bytes,
        max_work_bytes=max_work_bytes,max_snapshot_bytes=max_snapshot_bytes,max_batch_rows=max_batch_rows,
        max_sessions=max_sessions,max_snapshots_per_session=max_snapshots_per_session)


def open_public_publisher_v41_service(directory, *, runtime_library, executor_bytes,
        max_resource_bytes, max_work_bytes, max_snapshot_bytes, max_batch_rows=32,
        max_snapshots_per_session=4, **options):
    """Load the original complete native V4.1 source and own its public ABI."""
    if options.get('device') is not None:
        raise ValueError('native public V4.1 composition requires its original CPU arithmetic owner')
    backing=open_publisher_v41_service(directory,**options)
    payload=backing.publisher_payload
    return compose_public_publisher_service(backing,runtime_library=runtime_library,family='deepseek-v41',
        config=payload.profile,model_directory=directory,checkpoint_identity=payload.config_sha256,
        executor_bytes=executor_bytes,max_resource_bytes=max_resource_bytes,
        max_work_bytes=max_work_bytes,max_snapshot_bytes=max_snapshot_bytes,max_batch_rows=max_batch_rows,
        max_snapshots_per_session=max_snapshots_per_session)


@dataclass(frozen=True)
class K3Libraries:
    recurrence: str | None = None
    operators: str | None = None
    attention: str | None = None
    mxfp4: str | None = None

    def resolved(self,*,require_native=True):
        result = {}
        for name in ('recurrence', 'operators', 'attention', 'mxfp4'):
            value = getattr(self, name)
            if value is None and (name == 'mxfp4' or not require_native):
                result[name] = None
                continue
            if type(value) is not str or not value:
                raise ValueError('explicit native library path required: ' + name)
            path = Path(value).resolve(strict=True)
            if not path.is_file():
                raise ValueError('native library is not a file: ' + name)
            result[name] = str(path)
        return result


def create_k3_batch_execution(payload, libraries, *, runtime_max_bytes,
                              max_sessions, max_total_model_bytes,
                              max_batch_bytes, max_snapshot_bytes,
                              expert_provider=None, routing=None,
                              fla_policy=None, arithmetic=None, sharded_projections=None,
                              require_publisher_exact=False,exact_evidence=None,
                              native_projections=None,native_projection_work_bytes=64*1024**2):
    """Reuse checkpoint/library admission with independent mutable batch slots.

    Provider and arithmetic ownership match create_k3_execution_service. The
    aggregate model limit reserves each slot's full declared executor budget;
    batch inputs/results and rollback snapshots have separate explicit budgets.
    """
    from .k3_execution.batch import K3BatchExecution
    service = create_k3_execution_service(payload, libraries,
        runtime_max_bytes=runtime_max_bytes, max_new_tokens=1, max_sessions=max_sessions,
        expert_provider=expert_provider, routing=routing,
        fla_policy=fla_policy, arithmetic=arithmetic, sharded_projections=sharded_projections,
        require_publisher_exact=require_publisher_exact,exact_evidence=exact_evidence,
        native_projections=native_projections,native_projection_work_bytes=native_projection_work_bytes,
        max_total_model_bytes=max_total_model_bytes)
    try:
        return K3BatchExecution(service.factory, max_sessions=max_sessions,
            max_model_bytes=max_total_model_bytes, max_batch_bytes=max_batch_bytes,
            max_snapshot_bytes=max_snapshot_bytes, owner_service=service)
    except BaseException as primary:
        try:
            service.close()
        except BaseException as cleanup:
            primary.cleanup = service.close
            primary.add_note('K3 batch service cleanup retained: '+repr(cleanup))
        raise


def create_segment_edge_execution_service(
    library, *, edge_adapter, model_directory, checkpoint_identity, ranges,
    vocabulary, context_tokens, native_memory_bytes,
    max_transfer_bytes=256*1024**2, max_snapshot_bytes=1024**3,
    max_new_tokens=256, max_sessions=1, prefill_chunk_tokens=32,
    max_snapshots_per_session=4, backend_mask=1 << 8,
):
    """Use explicitly registered external native Segment/Edge model adapters.

    Model vocabulary and artifact identity are supplied by checkpoint admission,
    then checked against the opened native edge. This is not an adapter registry
    implementation or certification of any checkpoint/backend combination.
    Native memory is a per-adapter budget, not an aggregate process-RSS bound.
    """
    from itertools import islice
    from .segment_edge_runtime import (
        SegmentEdgeExecution, SegmentRange, cleanup_failed_segment_edge_executors,
    )

    selected = tuple(islice(iter(ranges), 4097))
    if not selected or len(selected) > 4096 or any(type(item) is not SegmentRange for item in selected):
        raise ValueError('explicit bounded native layer range inventory required')
    if type(max_snapshots_per_session) is not int or not 1 <= max_snapshots_per_session <= 62:
        raise ValueError('public snapshot limit must be between one and 62')
    limits = ExecutionLimits(vocabulary, context_tokens, max_new_tokens, max_sessions,
        prefill_chunk_tokens, max_snapshots_per_session,
        max_logit_buffer_bytes=max_transfer_bytes)
    pending = {}

    def cleanup():
        errors = []
        for key, executor in tuple(pending.items()):
            try:
                executor.close()
            except BaseException as error:
                errors.append(error)
            else:
                del pending[key]
        try:
            cleanup_failed_segment_edge_executors()
        except BaseException as error:
            errors.append(error)
        if errors:
            raise RuntimeError('native Segment/Edge factory cleanup remains incomplete') from errors[0]

    def create_executor():
        if pending:
            raise RuntimeError('failed native executor cleanup is pending; close the service')
        executor = SegmentEdgeExecution(library, edge_adapter=edge_adapter,
            model_directory=model_directory, checkpoint_identity=checkpoint_identity,
            ranges=selected, context_tokens=context_tokens, native_memory_bytes=native_memory_bytes,
            max_transfer_bytes=max_transfer_bytes, max_snapshot_bytes=max_snapshot_bytes,
            max_snapshots=max_snapshots_per_session+2, backend_mask=backend_mask)
        try:
            if executor.vocabulary != vocabulary:
                raise ValueError('native edge vocabulary differs from admitted checkpoint')
            if executor.max_prefill_tokens < limits.admitted_prefill_tokens:
                raise ValueError('prefill chunk exceeds the native adapter or transfer capacity')
        except BaseException as error:
            try:
                executor.close()
            except BaseException as failure:
                pending[id(executor)] = executor
                error.add_note('native executor cleanup pending: '+str(failure))
            raise
        return executor

    return NativeExecutionService(create_executor, limits, factory_cleanup=cleanup)


def create_native_family_public_execution_service(runtime_library,decoder,*,model_directory,
        max_resource_bytes,max_work_bytes,max_batch_rows=32,max_new_tokens=256,max_sessions=1,
        prefill_chunk_tokens=32,max_snapshots_per_session=4,max_transfer_bytes=256*1024**2,
        max_snapshot_bytes=1024**3,own_decoder=False):
    """Use native C model registration and the public Segment/Edge dispatcher."""
    from .native_family_public_adapters import NativeFamilyPublicAdapters
    from .segment_edge_runtime import SegmentRange
    owner=NativeFamilyPublicAdapters(runtime_library,decoder,model_directory=model_directory,
        max_resource_bytes=max_resource_bytes,max_work_bytes=max_work_bytes,
        max_batch_rows=max_batch_rows,own_decoder=own_decoder)
    try:
        service=create_segment_edge_execution_service(runtime_library,edge_adapter=owner.edge_id,
            model_directory=str(owner.root),checkpoint_identity=owner.identity,
            ranges=(SegmentRange(owner.segment_id,str(owner.root),0,owner.layers),),
            vocabulary=owner.vocabulary,context_tokens=owner.context,native_memory_bytes=max_resource_bytes,
            max_transfer_bytes=max_transfer_bytes,max_snapshot_bytes=max_snapshot_bytes,
            max_new_tokens=max_new_tokens,max_sessions=max_sessions,prefill_chunk_tokens=prefill_chunk_tokens,
            max_snapshots_per_session=max_snapshots_per_session)
        cleanup=service._factory_cleanup
        def close_factory():
            if cleanup is not None:cleanup()
            owner.close()
        service._factory_cleanup=close_factory
        service.native_model_owner=owner
        return service
    except BaseException as error:
        try:owner.close()
        except BaseException as failure:
            error.cleanup=owner.close;error.add_note('native public family ownership cleanup retained: '+str(failure))
        raise


def create_k3_execution_service(payload, libraries, *, runtime_max_bytes,
                                max_new_tokens=256, max_sessions=1,
                                prefill_chunk_tokens=512, expert_provider=None, routing=None,
                                fla_policy=None, arithmetic=None, sharded_projections=None,
                                require_publisher_exact=False,exact_evidence=None,
                                max_total_model_bytes=None,native_projections=None,
                                native_projection_work_bytes=64*1024**2):
    """Connect a prepared K3 payload without claiming publisher equivalence.

    The caller owns any expert provider and keeps it open until all sessions
    close. Each factory call constructs independent recurrent/attention state.
    No tokenizer, library build, network call, or automatic provider routing
    is performed here.
    """
    from .k3_execution import K3Config, K3Execution
    from .publisher_checkpoint import K3ExecutionPayload

    if type(payload) is not K3ExecutionPayload or type(payload.config) is not K3Config:
        raise ValueError('prepared K3 execution payload required')
    if type(libraries) is not K3Libraries:
        raise ValueError('explicit K3 library inventory required')
    if type(runtime_max_bytes) is not int or runtime_max_bytes <= 0:
        raise ValueError('positive per-executor memory budget required')
    if payload.expert_provider is not None:
        if expert_provider is not None and expert_provider is not payload.expert_provider:
            raise ValueError('explicit provider differs from prepared checkpoint provider')
        expert_provider = payload.expert_provider
    paths = libraries.resolved(require_native=arithmetic is None)
    prepared_shards = payload.sharded_projections
    if sharded_projections is None:
        sharded_projections = prepared_shards
    elif prepared_shards and sharded_projections != prepared_shards:
        raise ValueError('explicit sharded projections differ from prepared payload omission contract')
    if (type(sharded_projections) is not tuple or
            any(type(name) is not str for name in sharded_projections) or
            len(set(sharded_projections))!=len(sharded_projections)):
        raise ValueError('explicit unique sharded K3 projection names required')
    local_tensors = {name:value for name,value in payload.tensors.items()
                     if name not in sharded_projections}
    from .shared_expert_provider import shared_expert_provider
    expert_provider = shared_expert_provider(expert_provider)
    if payload.config.expert_format != 'fp32' and paths['mxfp4'] is None and arithmetic is None:
        raise ValueError('packed K3 execution requires the MXFP4 library')
    limits = ExecutionLimits(
        vocabulary=payload.config.vocab, context=payload.config.capacity,
        max_new_tokens=max_new_tokens, max_sessions=max_sessions,
        prefill_chunk_tokens=prefill_chunk_tokens,
    )
    if limits.prefill_chunk_tokens > 4096:
        raise ValueError('K3 prefill chunk exceeds the executor per-call limit')

    def create_executor():
        return K3Execution(
            payload.config, local_tensors, library=paths['recurrence'],
            operators_library=paths['operators'], attention_library=paths['attention'],
            mxfp4_library=paths['mxfp4'], max_bytes=runtime_max_bytes,
            expert_provider=expert_provider, require_publisher_exact=require_publisher_exact,
            exact_evidence=exact_evidence,
            routing=routing,
            fla_policy=fla_policy,
            arithmetic=arithmetic,
            sharded_projections=sharded_projections,
            native_projections=native_projections,native_projection_work_bytes=native_projection_work_bytes,
        )

    from .executor_budget import BudgetedExecutorFactory
    factory=BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    return NativeExecutionService(factory, limits,factory_cleanup=factory.retry_cleanup)


def create_v41_execution_service(config, tensors, *, libraries, engram, compressed_ids,
                                  runtime_max_bytes, expert_provider=None,
                                  max_new_tokens=256, max_sessions=1, prefill_chunk_tokens=512,
                                  max_total_model_bytes=None):
    """Use the declared V4.1 equation schema in independent native sessions.

    This is not a publisher checkpoint adapter. Table and packed expert
    providers remain caller-owned. Each executor validates its own weights.
    """
    from itertools import islice
    from types import MappingProxyType
    from .v41_execution import V41Config, V41Execution

    if type(config) is not V41Config or type(runtime_max_bytes) is not int or runtime_max_bytes<=0:
        raise ValueError('explicit V4.1 config and positive runtime budget required')
    descriptors=tuple(islice(iter(engram),config.layers+1))
    if len(descriptors)!=config.layers:
        raise ValueError('one Engram descriptor per layer required')
    frozen=[]
    for descriptor in descriptors:
        value=dict(descriptor)
        if 'heads' in value:
            heads=tuple(islice(iter(value['heads']),25))
            if not 1<=len(heads)<=24: raise ValueError('Engram head inventory exceeds admission')
            value['heads']=tuple(MappingProxyType(dict(head)) for head in heads)
        frozen.append(MappingProxyType(value))
    descriptors=tuple(frozen)
    token_map=tuple(islice(iter(compressed_ids),config.vocabulary+1))
    if len(token_map)!=config.vocabulary:
        raise ValueError('complete compressed vocabulary map required')
    paths={name:str(Path(path).resolve(strict=True)) for name,path in dict(libraries).items()}
    weights=MappingProxyType(dict(tensors))
    from .shared_expert_provider import shared_expert_provider
    expert_provider=shared_expert_provider(expert_provider)
    limits=ExecutionLimits(config.vocabulary,config.capacity,max_new_tokens,max_sessions,
                           prefill_chunk_tokens)

    def create_executor():
        return V41Execution(config,weights,libraries=paths,engram=descriptors,
            compressed_ids=token_map,max_bytes=runtime_max_bytes,expert_provider=expert_provider)

    from .executor_budget import BudgetedExecutorFactory
    factory=BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


class _OwnedPublisherExecutor:
    """Keep factory-created arithmetic alive and close both owned resources."""
    def __init__(self, model, arithmetic, projection_resources=None):
        for name, resource in (('model', model), ('arithmetic', arithmetic)):
            if resource is None or not callable(getattr(resource, 'close', None)):
                raise ValueError('owned publisher '+name+' must provide close')
            budget = getattr(resource, 'max_bytes', None)
            if type(budget) is not int or budget < 0:
                raise ValueError('owned publisher '+name+' must declare max_bytes')
        if projection_resources is not None and not callable(getattr(projection_resources, 'close', None)):
            raise ValueError('owned publisher projection resources must provide close')
        self.model, self.arithmetic = model, arithmetic
        self._model_closed = False
        self._arithmetic_closed = False
        self.projection_resources=projection_resources
        self._projection_closed=False

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def declared_executor_bytes(self):
        total=self.model.max_bytes+self.arithmetic.max_bytes
        seen=set()
        if self.projection_resources is not None:
            for method,_,_ in self.projection_resources._callbacks:
                owner=getattr(method,'__self__',None)
                if owner is None or id(owner) in seen:
                    continue
                seen.add(id(owner))
                amount=getattr(owner,'max_bytes',0)
                if type(amount) is not int or amount<0:
                    raise ValueError('invalid declared publisher projection ownership budget')
                total+=amount
        return total

    @contextmanager
    def cancellation_scope(self, cancel):
        from contextlib import ExitStack
        with ExitStack() as scopes:
            for resource in (self.model,self.arithmetic):
                scope=getattr(resource,'cancellation_scope',None)
                if callable(scope): scopes.enter_context(scope(cancel))
            yield

    def close(self):
        errors = []
        for name, resource in (('model', self.model), ('arithmetic', self.arithmetic),
                               ('projection',self.projection_resources)):
            if getattr(self, '_' + name + '_closed'):
                continue
            try:
                close = getattr(resource, 'close', None)
                if callable(close):
                    close()
                setattr(self, '_' + name + '_closed', True)
            except BaseException as error:
                errors.append(error)
                # Arithmetic can still be borrowed by a model whose close
                # failed. Keep dependency ownership and its budget for retry.
                break
        if errors:
            raise errors[0]


class _StructuredNativeSessionExecutor:
    """Adapt batch-one structured native output to the session logit contract."""
    def __init__(self, decoder):
        self.decoder = decoder

    def __getattr__(self, name):
        return getattr(self.decoder, name)

    def forward(self, token_ids):
        result = self.decoder.forward(token_ids)
        logits = getattr(result,'logits',result)
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError('native sessions require batch-one structured logits')
        return logits[0]


def open_native_conditional_directory_service(directory,*,family,library,text_library,
        context_tokens,max_weight_bytes,runtime_max_bytes,max_total_model_bytes,
        max_media_bytes,max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=512,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        read_limits=None,arithmetic='fp32',mtp_max_bytes=0,mtp_work_bytes=0,max_expert_bytes=0,
        text_device=None,max_projection_bytes=0,segment_runtime_library=None,segment_work_bytes=0,
        segment_layer_span=0,native_composition=False,native_store_runtime_library=None,
        native_expert_max_bytes=0,native_ple_max_bytes=0,max_checkpoint_buffer_bytes=16*1024**2):
    """Serve preprocessed media and generated tokens with physical recovery."""
    import json
    from .native_checkpoint import _pairs
    from .publisher_family_contracts import component_contract
    from .multimodal_session import MultimodalSessionExecutor
    from .executor_budget import BudgetedExecutorFactory
    if family not in ('qwen3_5_moe','qwen4_exp','inkling_mm_model','glm5_next'):
        raise ValueError('explicit supported conditional publisher family required')
    if family!='inkling_mm_model' and (mtp_max_bytes or mtp_work_bytes):
        raise ValueError('these MTP stage budgets require the Inkling family')
    if type(max_media_bytes) is not int or not 0<max_media_bytes<runtime_max_bytes:
        raise ValueError('media staging must fit the aggregate executor budget')
    root=Path(directory).resolve(strict=True)
    with (root/'config.json').open('rb') as source: raw=source.read(1024**2+1)
    if len(raw)>1024**2: raise MemoryError('conditional config exceeds metadata admission')
    config=json.loads(raw,object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if type(config) is not dict or config.get('model_type')!=family:
        raise ValueError('conditional service identity differs from checkpoint')
    components=component_contract(config,arithmetic=arithmetic);c=components.text_config
    if family=='qwen4_exp' and context_tokens!=c['max_position_embeddings']:
        raise ValueError('Qwen4 service context must match checkpoint capacity')
    vocabulary=c.get('unpadded_vocab_size') or c['vocab_size']
    limits=ExecutionLimits(vocabulary,context_tokens,max_new_tokens,max_sessions,
        prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes=max_logit_buffer_bytes,
        max_checkpoint_buffer_bytes=max_checkpoint_buffer_bytes)
    pending={}
    def construct():
        if pending: raise RuntimeError('conditional construction cleanup remains pending')
        from .publisher_conditional_loader import load_publisher_conditional_decoder
        decoder=load_publisher_conditional_decoder(root,family=family,library=library,
            text_library=text_library,max_weight_bytes=max_weight_bytes,
            max_bytes=runtime_max_bytes-max_media_bytes,
            max_conversion_bytes=min(max_weight_bytes//8,runtime_max_bytes//16),
            arithmetic=arithmetic,context_tokens=context_tokens,max_snapshots=max_snapshots_per_session+2,
            mtp_max_bytes=mtp_max_bytes,mtp_work_bytes=mtp_work_bytes,max_expert_bytes=max_expert_bytes,
            text_device=text_device,max_projection_bytes=max_projection_bytes,
            segment_runtime_library=segment_runtime_library,segment_work_bytes=segment_work_bytes,
            segment_layer_span=segment_layer_span,native_composition=native_composition,
            native_store_runtime_library=native_store_runtime_library,
            native_expert_max_bytes=native_expert_max_bytes,native_ple_max_bytes=native_ple_max_bytes)
        try:
            if decoder.components.identity!=components.identity:
                raise ValueError('conditional checkpoint contract changed after admission')
            return MultimodalSessionExecutor(decoder,max_media_bytes=max_media_bytes)
        except BaseException as error:
            try: decoder.close()
            except BaseException as failure:
                pending[id(decoder)]=decoder
                error.cleanup=decoder.close
                error.add_note('conditional native owner cleanup retained: '+str(failure))
            raise
    factory=BudgetedExecutorFactory(construct,executor_bytes=runtime_max_bytes,
        max_total_bytes=max_total_model_bytes,max_executors=max_sessions)
    def cleanup():
        for key,decoder in tuple(pending.items()):
            decoder.close();del pending[key]
        factory.retry_cleanup()
    return NativeExecutionService(factory,limits,factory_cleanup=cleanup)


def open_flat_native_directory_service(directory, *, family, library,
                                       max_weight_bytes, runtime_max_bytes,
                                       context_tokens, max_total_model_bytes,
                                       max_new_tokens=256, max_sessions=1,
                                       prefill_chunk_tokens=32,
                                       max_snapshots_per_session=4, read_limits=None):
    """Serve actual GLM5-next or Inkling native state with durable recovery.

    Each session independently owns all weights and continuation banks. The
    aggregate factory reserves the full declared model budget before loading.
    """
    import json
    from .native_checkpoint import load_native_text_decoder, _pairs
    from .flat_native_session import FlatNativeSession
    from .executor_budget import BudgetedExecutorFactory
    if family not in ('glm5_next_text','inkling_text'):
        raise ValueError('explicit flat-state native family required')
    root = Path(directory).resolve(strict=True)
    path = Path(library).resolve(strict=True)
    if not root.is_dir() or not path.is_file():
        raise ValueError('local checkpoint directory and explicit native library required')
    with (root/'config.json').open('rb') as source:
        encoded = source.read(1024**2+1)
    if len(encoded)>1024**2:
        raise MemoryError('native service configuration exceeds admission')
    config = json.loads(encoded,object_pairs_hook=_pairs)
    if type(config) is not dict or config.get('model_type')!=family:
        raise ValueError('native service family differs from local checkpoint')
    vocabulary = (config.get('unpadded_vocab_size') or config['vocab_size']) if family=='inkling_text' else config['vocab_size']
    if family=='inkling_text' and context_tokens!=config['max_position_embeddings']:
        raise ValueError('Inkling service context must match checkpoint capacity')
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded native service snapshot count required')
    limits = ExecutionLimits(vocabulary,context_tokens,max_new_tokens,max_sessions,
                             prefill_chunk_tokens,max_snapshots_per_session)
    pending = {}
    def cleanup():
        errors = []
        for key,decoder in tuple(pending.items()):
            try:
                decoder.close()
            except BaseException as failure:
                errors.append(failure)
            else:
                pending.pop(key,None)
        budget.retry_cleanup()
        if errors:
            raise RuntimeError('flat native checkpoint construction cleanup pending') from errors[0]
    def construct():
        if pending:
            raise RuntimeError('flat native checkpoint cleanup pending')
        decoder = load_native_text_decoder(root,model_type=family,library=path,
            max_weight_bytes=max_weight_bytes,max_bytes=runtime_max_bytes,
            context_tokens=context_tokens if family=='glm5_next_text' else None,read_limits=read_limits)
        try:
            executor = FlatNativeSession(decoder,family=family,max_bytes=runtime_max_bytes,
                                         max_snapshots=max_snapshots_per_session+2)
            if executor.capacity!=context_tokens or executor.vocabulary!=vocabulary:
                raise ValueError('native checkpoint geometry changed after service admission')
            if family=='glm5_next_text' and not decoder._head.p:
                raise ValueError('GLM service requires an executable vocabulary head')
            return executor
        except BaseException as failure:
            try:
                decoder.close()
            except BaseException as cleanup_error:
                pending[id(decoder)] = decoder
                failure.add_note('native construction resource retained: '+str(cleanup_error))
            raise
    budget = BudgetedExecutorFactory(construct,executor_bytes=runtime_max_bytes,
        max_total_bytes=max_total_model_bytes,max_executors=max_sessions)
    return NativeExecutionService(budget,limits,factory_cleanup=cleanup)


def open_publisher_glm5_text_service(directory,*,library,expert_library,
        context_tokens,max_weight_bytes,runtime_max_bytes,max_expert_bytes,
        max_conversion_bytes,max_total_model_bytes,max_new_tokens=256,
        max_sessions=1,prefill_chunk_tokens=32,max_snapshots_per_session=4):
    """Serve original GLM text tensors with borrowed selected expert callbacks."""
    import json
    from .native_checkpoint import _pairs
    from .publisher_family_contracts import component_contract
    from .publisher_glm5_loader import load_publisher_glm5_text_decoder
    from .flat_native_session import FlatNativeSession
    from .executor_budget import BudgetedExecutorFactory
    root=Path(directory).resolve(strict=True)
    with (root/'config.json').open('rb') as stream: raw=stream.read(1024**2+1)
    if len(raw)>1024**2: raise MemoryError('GLM service metadata exceeds admission')
    config=json.loads(raw,object_pairs_hook=_pairs)
    components=component_contract(config,arithmetic='fp32')
    if components.variant.wrapper!='glm5_next': raise ValueError('exact GLM publisher identity required')
    limits=ExecutionLimits(components.text_config['vocab_size'],context_tokens,max_new_tokens,
        max_sessions,prefill_chunk_tokens,max_snapshots_per_session)
    pending={}
    def construct():
        if pending: raise RuntimeError('publisher GLM native cleanup remains pending')
        owner=load_publisher_glm5_text_decoder(root,library=library,expert_library=expert_library,
            context_tokens=context_tokens,max_weight_bytes=max_weight_bytes,max_bytes=runtime_max_bytes,
            max_expert_bytes=max_expert_bytes,max_conversion_bytes=max_conversion_bytes)
        try:
            session=FlatNativeSession(owner.decoder,family='glm5_next_text',max_bytes=runtime_max_bytes,
                max_snapshots=max_snapshots_per_session+2)
            if session.capacity!=context_tokens or session.vocabulary!=limits.vocabulary:
                raise ValueError('publisher GLM geometry changed after service admission')
            return _OwnedPublisherExecutor(session,owner)
        except BaseException as error:
            try: owner.close()
            except BaseException as failure:
                pending[id(owner)]=owner;error.cleanup=owner.close
                error.add_note('publisher GLM owner retained: '+str(failure))
            raise
    budget=BudgetedExecutorFactory(construct,executor_bytes=runtime_max_bytes,
        max_total_bytes=max_total_model_bytes,max_executors=max_sessions)
    def cleanup():
        for key,owner in tuple(pending.items()): owner.close();del pending[key]
        budget.retry_cleanup()
    return NativeExecutionService(budget,limits,factory_cleanup=cleanup)


def _glm_dsa_device_service(config,construct,*,runtime_max_bytes,max_total_model_bytes,
        max_new_tokens,max_sessions,prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes):
    from .independent_glm_dsa import IndependentGlmDsaDecoder
    from .executor_budget import BudgetedExecutorFactory
    inventory=IndependentGlmDsaDecoder(config,{},_inventory_only=True)
    settings=dict(vars(inventory.config))
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded GLM DSA service snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded GLM DSA device prefill required')
    limits=ExecutionLimits(settings['vocab_size'],settings['max_position_embeddings'],
        max_new_tokens,max_sessions,prefill_chunk_tokens,max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create():
        return _StructuredNativeSessionExecutor(construct())
    def validate(executor):
        if executor.config!=settings:
            raise ValueError('GLM DSA configuration changed after service admission')
    factory=BudgetedExecutorFactory(create,executor_bytes=runtime_max_bytes,
        max_total_bytes=max_total_model_bytes,max_executors=max_sessions,validate=validate)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def create_device_glm_dsa_execution_service(config,tensors,*,device,runtime_max_bytes,
        max_total_model_bytes,max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=128,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        max_work_bytes=64*1024**2,max_head_bytes=64*1024**2,expert_provider=None):
    """Own resident GLM DSA accelerator sessions and complete physical recovery."""
    import copy
    from .torch_glm_dsa_decoder import TorchGlmDsaDecoder
    settings=copy.deepcopy(dict(config));weights=dict(tensors)
    def construct():
        return TorchGlmDsaDecoder(settings,weights,device=device,max_bytes=runtime_max_bytes,
            max_snapshots=max_snapshots_per_session+2,max_work_bytes=max_work_bytes,max_head_bytes=max_head_bytes,
            expert_provider=expert_provider)
    return _glm_dsa_device_service(settings,construct,runtime_max_bytes=runtime_max_bytes,
        max_total_model_bytes=max_total_model_bytes,max_new_tokens=max_new_tokens,max_sessions=max_sessions,
        prefill_chunk_tokens=prefill_chunk_tokens,max_snapshots_per_session=max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)


def open_device_glm_dsa_directory_service(directory,*,device,max_weight_bytes,runtime_max_bytes,
        max_total_model_bytes,max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=128,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        max_work_bytes=64*1024**2,max_head_bytes=64*1024**2,read_limits=None,cancel=None,expert_provider=None):
    """Admit a complete local GLM DSA checkpoint before device session creation."""
    import json
    from .native_checkpoint import _pairs
    from .torch_glm_dsa_decoder import TorchGlmDsaDecoder
    root=Path(directory).resolve(strict=True)
    with (root/'config.json').open('rb') as source:raw=source.read(1024**2+1)
    if len(raw)>1024**2:raise MemoryError('GLM DSA service configuration exceeds admission')
    config=json.loads(raw,object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    def construct():
        return TorchGlmDsaDecoder.from_directory(root,device=device,max_weight_bytes=max_weight_bytes,
            max_bytes=runtime_max_bytes,max_snapshots=max_snapshots_per_session+2,
            max_work_bytes=max_work_bytes,max_head_bytes=max_head_bytes,read_limits=read_limits,cancel=cancel,
            expert_provider=expert_provider)
    return _glm_dsa_device_service(config,construct,runtime_max_bytes=runtime_max_bytes,
        max_total_model_bytes=max_total_model_bytes,max_new_tokens=max_new_tokens,max_sessions=max_sessions,
        prefill_chunk_tokens=prefill_chunk_tokens,max_snapshots_per_session=max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)


def create_device_inkling_execution_service(config,tensors,*,device,runtime_max_bytes,
        max_total_model_bytes,max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=128,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        max_work_bytes=64*1024**2,max_head_bytes=64*1024**2,
        expert_provider=None,projection_provider=None):
    """Serve exact in-memory Inkling tensors in separate owned device sessions.

    Input tensors and providers remain borrowed during construction. Each
    decoder admits and copies its resident tensors before it can be published.
    """
    from .independent_inkling import configuration
    from .torch_inkling_decoder import TorchInklingDecoder
    from .executor_budget import BudgetedExecutorFactory
    import copy
    settings=copy.deepcopy(dict(config));weights=dict(tensors)
    c,_=configuration(settings)
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded Inkling service snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded Inkling device prefill required')
    limits=ExecutionLimits(c['unpadded_vocab_size'] or c['vocab_size'],c['max_position_embeddings'],
        max_new_tokens,max_sessions,prefill_chunk_tokens,max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create():
        return _StructuredNativeSessionExecutor(TorchInklingDecoder(settings,weights,device=device,
            max_bytes=runtime_max_bytes,max_snapshots=max_snapshots_per_session+2,
            max_work_bytes=max_work_bytes,max_head_bytes=max_head_bytes,
            expert_provider=expert_provider,projection_provider=projection_provider))
    factory=BudgetedExecutorFactory(create,executor_bytes=runtime_max_bytes,
        max_total_bytes=max_total_model_bytes,max_executors=max_sessions)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def create_device_glm5_execution_service(config,tensors,*,device,runtime_max_bytes,
        max_total_model_bytes,context_tokens=None,max_new_tokens=256,max_sessions=1,
        prefill_chunk_tokens=128,max_snapshots_per_session=4,
        max_logit_buffer_bytes=256*1024**2,max_work_bytes=64*1024**2,max_head_bytes=64*1024**2,
        expert_provider=None,projection_provider=None):
    """Serve exact GLM KDA/DSA device state through transactional generation."""
    from .independent_glm5_next import Glm5NextTextContract
    from .torch_glm5_decoder import TorchGlm5NextDecoder
    from .executor_budget import BudgetedExecutorFactory
    import copy
    settings=copy.deepcopy(dict(config));weights=dict(tensors)
    contract=Glm5NextTextContract.parse(settings)
    capacity=contract.max_position_embeddings if context_tokens is None else context_tokens
    if type(capacity) is not int or not 1<=capacity<=contract.max_position_embeddings:
        raise ValueError('bounded GLM service context required')
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded GLM service snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded GLM device prefill required')
    limits=ExecutionLimits(contract.vocab_size,capacity,max_new_tokens,max_sessions,
        prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create():
        return _StructuredNativeSessionExecutor(TorchGlm5NextDecoder(settings,weights,device=device,
            max_bytes=runtime_max_bytes,max_tokens=capacity,max_snapshots=max_snapshots_per_session+2,
            max_work_bytes=max_work_bytes,max_head_bytes=max_head_bytes,
            expert_provider=expert_provider,projection_provider=projection_provider))
    factory=BudgetedExecutorFactory(create,executor_bytes=runtime_max_bytes,
        max_total_bytes=max_total_model_bytes,max_executors=max_sessions)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def open_device_glm5_directory_service(directory,*,device,max_weight_bytes,runtime_max_bytes,
        max_total_model_bytes,max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=128,
        context_tokens=None,max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        max_work_bytes=64*1024**2,max_head_bytes=64*1024**2,read_limits=None,cancel=None,
        expert_provider=None,projection_provider=None):
    """Own independent GLM device continuation with declared aggregate admission.

    This entry point admits the exact text checkpoint namespace. Publisher
    wrapper conversion remains the responsibility of its explicit loader.
    Providers are borrowed and must outlive every service executor.
    """
    import json
    from .native_checkpoint import _pairs
    from .independent_glm5_next import Glm5NextTextContract
    from .torch_glm5_decoder import TorchGlm5NextDecoder
    from .executor_budget import BudgetedExecutorFactory
    root=Path(directory).resolve(strict=True)
    with (root/'config.json').open('rb') as source:
        raw=source.read(1024**2+1)
    if len(raw)>1024**2:
        raise MemoryError('GLM accelerator configuration exceeds admission')
    config=json.loads(raw,object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    contract=Glm5NextTextContract.parse(config)
    capacity=contract.max_position_embeddings if context_tokens is None else context_tokens
    if type(capacity) is not int or not 1<=capacity<=contract.max_position_embeddings:
        raise ValueError('bounded GLM service context required')
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded GLM service snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded GLM device prefill required')
    limits=ExecutionLimits(contract.vocab_size,capacity,max_new_tokens,max_sessions,
        prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create():
        return _StructuredNativeSessionExecutor(TorchGlm5NextDecoder.from_directory(root,device=device,
            max_weight_bytes=max_weight_bytes,max_bytes=runtime_max_bytes,read_limits=read_limits,cancel=cancel,
            max_tokens=capacity,max_snapshots=max_snapshots_per_session+2,
            max_work_bytes=max_work_bytes,max_head_bytes=max_head_bytes,
            expert_provider=expert_provider,projection_provider=projection_provider))
    def validate(executor):
        if executor.decoder.contract!=contract or executor.decoder.max_tokens!=capacity:
            raise ValueError('GLM configuration changed after service admission')
    factory=BudgetedExecutorFactory(create,executor_bytes=runtime_max_bytes,
        max_total_bytes=max_total_model_bytes,max_executors=max_sessions,validate=validate)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def open_device_inkling_directory_service(directory,*,device,max_weight_bytes,runtime_max_bytes,
        max_total_model_bytes,max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=128,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        max_work_bytes=64*1024**2,max_head_bytes=64*1024**2,read_limits=None,cancel=None,
        expert_provider=None,projection_provider=None):
    """Admit separately owned Inkling device caches and bounded generation."""
    import json
    from .native_checkpoint import _pairs
    from .independent_inkling import configuration
    from .torch_inkling_decoder import TorchInklingDecoder
    from .executor_budget import BudgetedExecutorFactory
    root=Path(directory).resolve(strict=True)
    with (root/'config.json').open('rb') as source:
        raw=source.read(1024**2+1)
    if len(raw)>1024**2:
        raise MemoryError('Inkling accelerator configuration exceeds admission')
    config=json.loads(raw,object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    c,_=configuration(config)
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded Inkling service snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded Inkling device prefill required')
    limits=ExecutionLimits(c['unpadded_vocab_size'] or c['vocab_size'],c['max_position_embeddings'],
        max_new_tokens,max_sessions,prefill_chunk_tokens,max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create():
        return _StructuredNativeSessionExecutor(TorchInklingDecoder.from_directory(root,device=device,
            max_weight_bytes=max_weight_bytes,max_bytes=runtime_max_bytes,read_limits=read_limits,cancel=cancel,
            max_snapshots=max_snapshots_per_session+2,max_work_bytes=max_work_bytes,max_head_bytes=max_head_bytes,
            expert_provider=expert_provider,projection_provider=projection_provider))
    def validate(executor):
        if dict(executor.config)!=dict(c):
            raise ValueError('Inkling configuration changed after service admission')
    factory=BudgetedExecutorFactory(create,executor_bytes=runtime_max_bytes,
        max_total_bytes=max_total_model_bytes,max_executors=max_sessions,validate=validate)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def open_device_qwen4_directory_service(directory,*,device,max_weight_bytes,runtime_max_bytes,
        max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=512,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        read_limits=None,cancel=None,max_total_model_bytes=None,
        max_layer_bytes=64*1024**2,max_head_bytes=64*1024**2,expert_provider=None,
        max_projection_bytes=None,max_projection_host_bytes=None,max_projection_cache_bytes=0):
    """Own Qwen4 device generation sessions loaded from exact local tensors."""
    import json
    from .native_checkpoint import _pairs
    from .independent_qwen4 import IndependentQwen4TextDecoder
    from .torch_qwen4_decoder import TorchQwen4Decoder
    from .executor_budget import BudgetedExecutorFactory
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('local Qwen4 device checkpoint directory required')
    with (root/'config.json').open('rb') as source:
        encoded = source.read(1024**2+1)
    if len(encoded)>1024**2:
        raise MemoryError('Qwen4 configuration exceeds admission')
    raw = json.loads(encoded,object_pairs_hook=_pairs)
    c = IndependentQwen4TextDecoder(raw,{},_inventory_only=True).config
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded Qwen4 snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded Qwen4 device prefill chunk required')
    limits = ExecutionLimits(c.vocab_size,c.max_position_embeddings,max_new_tokens,max_sessions,
        prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create_executor():
        return _StructuredNativeSessionExecutor(TorchQwen4Decoder.from_directory(root,device=device,
            max_weight_bytes=max_weight_bytes,max_bytes=runtime_max_bytes,read_limits=read_limits,cancel=cancel,
            max_snapshots=max_snapshots_per_session+2,max_layer_bytes=max_layer_bytes,
            max_head_bytes=max_head_bytes,expert_provider=expert_provider,
            max_projection_bytes=max_projection_bytes,max_projection_host_bytes=max_projection_host_bytes,
            max_projection_cache_bytes=max_projection_cache_bytes))
    def validate(executor):
        if vars(executor.config)!=vars(c):
            raise ValueError('Qwen4 checkpoint configuration changed after service admission')
    factory = BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions,validate=validate)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def create_device_qwen4_execution_service(config,tensors,*,device,runtime_max_bytes,
        max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=512,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        max_total_model_bytes=None,max_layer_bytes=64*1024**2,max_head_bytes=64*1024**2,expert_provider=None):
    """Construct separately owned Qwen4 hybrid-state device sessions."""
    import json
    from types import MappingProxyType
    from .independent_qwen4 import IndependentQwen4TextDecoder
    from .torch_qwen4_decoder import TorchQwen4Decoder
    from .executor_budget import BudgetedExecutorFactory
    raw = json.loads(json.dumps(dict(config),allow_nan=False))
    c = IndependentQwen4TextDecoder(raw,{},_inventory_only=True).config
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded Qwen4 snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded Qwen4 device prefill chunk required')
    weights = MappingProxyType(dict(tensors))
    limits = ExecutionLimits(c.vocab_size,c.max_position_embeddings,max_new_tokens,max_sessions,
        prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create_executor():
        return _StructuredNativeSessionExecutor(TorchQwen4Decoder(raw,weights,device=device,max_bytes=runtime_max_bytes,
            max_snapshots=max_snapshots_per_session+2,max_layer_bytes=max_layer_bytes,
            max_head_bytes=max_head_bytes,expert_provider=expert_provider))
    factory = BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def create_device_qwen35_execution_service(config,tensors,*,device,runtime_max_bytes,
        max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=512,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        max_total_model_bytes=None,max_layer_bytes=64*1024**2,max_head_bytes=64*1024**2,expert_provider=None):
    """Own independent Qwen3.5 device hybrid-state generation sessions."""
    import json
    from types import MappingProxyType
    from .independent_qwen35 import IndependentQwen35TextDecoder
    from .torch_qwen35_decoder import TorchQwen35Decoder
    from .executor_budget import BudgetedExecutorFactory
    raw = json.loads(json.dumps(dict(config),allow_nan=False))
    c = IndependentQwen35TextDecoder(raw,{},_inventory_only=True).config
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded Qwen3.5 snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded Qwen3.5 prefill chunk required')
    limits = ExecutionLimits(c.vocab_size,c.max_position_embeddings,max_new_tokens,max_sessions,
        prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes=max_logit_buffer_bytes)
    weights = MappingProxyType(dict(tensors))
    def create_executor():
        return _StructuredNativeSessionExecutor(TorchQwen35Decoder(raw,weights,device=device,
            max_bytes=runtime_max_bytes,max_snapshots=max_snapshots_per_session+2,
            max_layer_bytes=max_layer_bytes,max_head_bytes=max_head_bytes,expert_provider=expert_provider))
    factory = BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def open_device_qwen35_directory_service(directory,*,device,max_weight_bytes,runtime_max_bytes,
        max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=512,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,
        read_limits=None,cancel=None,max_total_model_bytes=None,
        max_layer_bytes=64*1024**2,max_head_bytes=64*1024**2,expert_provider=None):
    """Open exact local Qwen3.5 checkpoints with aggregate executor admission."""
    import json
    from .native_checkpoint import _pairs
    from .independent_qwen35 import IndependentQwen35TextDecoder
    from .torch_qwen35_decoder import TorchQwen35Decoder
    from .executor_budget import BudgetedExecutorFactory
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('local Qwen3.5 checkpoint directory required')
    with (root/'config.json').open('rb') as source:
        encoded = source.read(1024**2+1)
    if len(encoded)>1024**2:
        raise MemoryError('Qwen3.5 configuration exceeds admission')
    raw = json.loads(encoded,object_pairs_hook=_pairs)
    c = IndependentQwen35TextDecoder(raw,{},_inventory_only=True).config
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded Qwen3.5 snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded Qwen3.5 prefill chunk required')
    limits = ExecutionLimits(c.vocab_size,c.max_position_embeddings,max_new_tokens,max_sessions,
        prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create_executor():
        return _StructuredNativeSessionExecutor(TorchQwen35Decoder.from_directory(root,device=device,
            max_weight_bytes=max_weight_bytes,max_bytes=runtime_max_bytes,
            max_snapshots=max_snapshots_per_session+2,read_limits=read_limits,cancel=cancel,
            max_layer_bytes=max_layer_bytes,max_head_bytes=max_head_bytes,expert_provider=expert_provider))
    def validate(executor):
        if vars(executor.config)!=vars(c):
            raise ValueError('Qwen3.5 configuration changed after service admission')
    factory = BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions,validate=validate)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def create_device_mixtral_execution_service(config,tensors,*,device,runtime_max_bytes,
        max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=512,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,expert_provider=None,
        max_total_model_bytes=None,max_attention_bytes=64*1024**2,max_head_bytes=64*1024**2):
    """Construct independently owned full device Mixtral session executors."""
    import json
    from types import MappingProxyType
    from .native_mixtral_decoder import configuration
    from .torch_mixtral_decoder import TorchMixtralDecoder
    raw = json.loads(json.dumps(dict(config),allow_nan=False))
    c,context = configuration(raw,None)
    if type(runtime_max_bytes) is not int or runtime_max_bytes<=0:
        raise ValueError('explicit device decoder memory budget required')
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded device snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded device prefill chunk required')
    weights = MappingProxyType(dict(tensors))
    limits = ExecutionLimits(c['vocab_size'],context,max_new_tokens,max_sessions,
        prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create_executor():
        decoder = TorchMixtralDecoder(raw,weights,device=device,max_bytes=runtime_max_bytes,
                                     max_snapshots=max_snapshots_per_session+2,expert_provider=expert_provider,
                                     max_attention_bytes=max_attention_bytes,max_head_bytes=max_head_bytes)
        return _StructuredNativeSessionExecutor(decoder)
    from .executor_budget import BudgetedExecutorFactory
    factory = BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def open_device_mixtral_directory_service(directory,*,device,max_weight_bytes,runtime_max_bytes,
        max_new_tokens=256,max_sessions=1,prefill_chunk_tokens=512,
        max_snapshots_per_session=4,max_logit_buffer_bytes=256*1024**2,read_limits=None,cancel=None,
        expert_provider=None,max_total_model_bytes=None,max_attention_bytes=64*1024**2,max_head_bytes=64*1024**2):
    """Load each admitted device session from an exact local family checkpoint."""
    import json
    from .native_mixtral_decoder import configuration
    from .torch_mixtral_decoder import TorchMixtralDecoder
    from .native_checkpoint import _pairs
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('local Mixtral device checkpoint directory required')
    with (root/'config.json').open('rb') as source:
        raw = source.read(1024*1024+1)
    if len(raw)>1024*1024:
        raise MemoryError('device checkpoint configuration exceeds admission')
    config = json.loads(raw,object_pairs_hook=_pairs)
    c,context = configuration(config,None)
    if type(max_snapshots_per_session) is not int or not 1<=max_snapshots_per_session<=62:
        raise ValueError('bounded device snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1<=prefill_chunk_tokens<=4096:
        raise ValueError('bounded device prefill chunk required')
    limits = ExecutionLimits(c['vocab_size'],context,max_new_tokens,max_sessions,
        prefill_chunk_tokens,max_snapshots_per_session,max_logit_buffer_bytes=max_logit_buffer_bytes)
    def create_executor():
        decoder = TorchMixtralDecoder.from_directory(root,device=device,max_weight_bytes=max_weight_bytes,
            max_bytes=runtime_max_bytes,max_snapshots=max_snapshots_per_session+2,
            read_limits=read_limits,cancel=cancel,expert_provider=expert_provider,
            max_attention_bytes=max_attention_bytes,max_head_bytes=max_head_bytes)
        return _StructuredNativeSessionExecutor(decoder)
    def validate(executor):
        if dict(executor.config)!=dict(c):
            raise ValueError('device checkpoint configuration changed after service admission')
    from .executor_budget import BudgetedExecutorFactory
    factory = BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions,validate=validate)
    return NativeExecutionService(factory,limits,factory_cleanup=factory.retry_cleanup)


def create_mixtral_execution_service(config, tensors, *, library, runtime_max_bytes,
                                    context_tokens=None, max_new_tokens=256,
                                    max_sessions=1, prefill_chunk_tokens=512,
                                    max_snapshots_per_session=4,
                                    max_logit_buffer_bytes=256*1024**2,
                                    expert_provider=None,max_total_model_bytes=None):
    """Create native Mixtral sessions with resident or explicitly streamed experts.

    Weight copies, cache state and snapshots are admitted per decoder. This
    An optional borrowed provider requires dense/router-only tensors and owns
    its own explicit compiled-kernel contract and expert cache admission.
    Tensors remain caller-owned until each decoder has made its private copy.
    """
    from types import MappingProxyType
    from .native_mixtral_decoder import NativeMixtralDecoder, configuration

    if type(runtime_max_bytes) is not int or runtime_max_bytes <= 0:
        raise ValueError('positive per-decoder memory budget required')
    if type(prefill_chunk_tokens) is not int or not 1 <= prefill_chunk_tokens <= 4096:
        raise ValueError('Mixtral prefill chunks must contain one to 4096 tokens')
    if type(max_snapshots_per_session) is not int or not 1 <= max_snapshots_per_session <= 62:
        raise ValueError('Mixtral public snapshot limit must be between one and 62')
    raw = dict(config.to_dict() if hasattr(config, 'to_dict') else config)
    settings, capacity = configuration(raw, context_tokens)
    raw = dict(settings)
    path = Path(library).resolve(strict=True)
    if not path.is_file():
        raise ValueError('explicit Mixtral native library required')
    weights = MappingProxyType(dict(tensors))
    limits = ExecutionLimits(settings['vocab_size'], capacity, max_new_tokens,
        max_sessions, prefill_chunk_tokens, max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)

    def create_executor():
        decoder = NativeMixtralDecoder(raw, weights, library=str(path),
            max_bytes=runtime_max_bytes, max_context=capacity,
            max_snapshots=max_snapshots_per_session+2, expert_provider=expert_provider)
        return _StructuredNativeSessionExecutor(decoder)

    from .executor_budget import BudgetedExecutorFactory
    factory=BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    return NativeExecutionService(factory, limits,factory_cleanup=factory.retry_cleanup)


def create_streamed_mixtral_directory_service(directory, *, library, expert_provider,
        max_weight_bytes, runtime_max_bytes, context_tokens=None,
        max_new_tokens=256, max_sessions=1, prefill_chunk_tokens=512,
        max_snapshots_per_session=4, max_logit_buffer_bytes=256*1024**2,
        read_limits=None,max_total_model_bytes=None):
    """Admit local source metadata and load only dense/router banks per session.

    The caller keeps the borrowed provider alive through service shutdown.
    Session caches remain independent; the provider serializes selected expert
    calls over its shared bounded store. This factory never converts source
    weights or chooses a quantizer implicitly.
    """
    from contextlib import ExitStack
    from .native_checkpoint import _open_local, _json, _stamp
    from .native_mixtral_decoder import (NativeMixtralDecoder, configuration,
                                         mixtral_metadata_fingerprint)
    from .mixtral_streamed_experts import MixtralStreamedExperts
    if type(expert_provider) is not MixtralStreamedExperts:
        raise TypeError('explicit MixtralStreamedExperts required')
    for value in (max_weight_bytes, runtime_max_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError('explicit positive checkpoint and decoder byte budgets required')
    if type(prefill_chunk_tokens) is not int or not 1 <= prefill_chunk_tokens <= 4096:
        raise ValueError('Mixtral prefill chunks must contain one to 4096 tokens')
    if type(max_snapshots_per_session) is not int or not 1 <= max_snapshots_per_session <= 62:
        raise ValueError('Mixtral public snapshot limit must be between one and 62')
    root = Path(directory).resolve(strict=True)
    path = Path(library).resolve(strict=True)
    if not root.is_dir() or not path.is_file():
        raise ValueError('explicit local checkpoint directory and native library required')
    with ExitStack() as owned:
        source, stamp = _open_local(root, root/'config.json', owned)
        if stamp[2] > 1024*1024:
            raise MemoryError('Mixtral service configuration exceeds admission')
        raw = _json(source, stamp)
        if _stamp(source) != stamp:
            raise ValueError('Mixtral configuration changed during admission')
    settings, capacity = configuration(raw, context_tokens)
    metadata_identity = mixtral_metadata_fingerprint(raw)
    if expert_provider.runtime.manifest.fingerprint != metadata_identity:
        raise ValueError('Mixtral provider and source architecture differ')
    provider_identity = expert_provider.identity
    pending = {}
    limits = ExecutionLimits(settings['vocab_size'], capacity, max_new_tokens,
        max_sessions, prefill_chunk_tokens, max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)

    def create_executor():
        if pending:
            raise RuntimeError('Mixtral factory cleanup is pending; close the service')
        if expert_provider.identity != provider_identity:
            raise ValueError('Mixtral provider changed after service admission')
        decoder = NativeMixtralDecoder.from_streamed_directory(root,
            library=str(path), expert_provider=expert_provider,
            max_weight_bytes=max_weight_bytes, max_bytes=runtime_max_bytes,
            max_context=capacity, max_snapshots=max_snapshots_per_session+2,
            read_limits=read_limits)
        try:
            if mixtral_metadata_fingerprint(decoder.config) != metadata_identity:
                raise ValueError('Mixtral checkpoint configuration changed after service admission')
        except BaseException as error:
            try:
                decoder.close()
            except BaseException as failure:
                pending[id(decoder)] = decoder
                error.cleanup = decoder.close
                error.add_note('Mixtral factory cleanup remains pending: '+str(failure))
            raise
        return _StructuredNativeSessionExecutor(decoder)

    def cleanup():
        errors = []
        for key, decoder in tuple(pending.items()):
            try:
                decoder.close()
            except BaseException as error:
                errors.append(error)
            else:
                del pending[key]
        if errors:
            raise RuntimeError('Mixtral factory cleanup remains pending') from errors[0]
        factory.retry_cleanup()

    from .executor_budget import BudgetedExecutorFactory
    factory=BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    return NativeExecutionService(factory, limits, factory_cleanup=cleanup)


def create_glm_dsa_execution_service(config, tensors, *, library, runtime_max_bytes,
                                    max_new_tokens=256, max_sessions=1,
                                    prefill_chunk_tokens=512, max_snapshots_per_session=4,
                                    max_logit_buffer_bytes=256*1024**2,
                                    retain_diagnostics=False,max_total_model_bytes=None):
    """Create resident GLM-MoE-DSA sessions with native compressed-cache execution.

    This accepts only the decoder's exact unquantized text configuration and
    tensor inventory. It does not certify a checkpoint or enable expert paging.
    """
    import json
    from types import MappingProxyType
    from .native_glm_dsa import NativeGlmDsaDecoder

    if type(runtime_max_bytes) is not int or runtime_max_bytes <= 0:
        raise ValueError('positive GLM decoder memory budget required')
    if type(max_snapshots_per_session) is not int or not 1 <= max_snapshots_per_session <= 62:
        raise ValueError('GLM public snapshot limit must be between one and 62')
    raw = json.loads(json.dumps(dict(config), allow_nan=False))
    if raw.get('model_type') != 'glm_moe_dsa':
        raise ValueError('exact glm_moe_dsa configuration required')
    limits = ExecutionLimits(raw.get('vocab_size'), raw.get('max_position_embeddings'),
        max_new_tokens, max_sessions, prefill_chunk_tokens, max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)
    path = Path(library).resolve(strict=True)
    if not path.is_file():
        raise ValueError('explicit GLM native library required')
    weights = MappingProxyType(dict(tensors))

    def create_executor():
        decoder = NativeGlmDsaDecoder(raw, weights, library=str(path),
            max_bytes=runtime_max_bytes, max_snapshots=max_snapshots_per_session+2,
            retain_diagnostics=retain_diagnostics)
        return _StructuredNativeSessionExecutor(decoder)

    from .executor_budget import BudgetedExecutorFactory
    factory=BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    return NativeExecutionService(factory, limits,factory_cleanup=factory.retry_cleanup)


def open_native_glm_dsa_directory_service(directory, *, library, max_weight_bytes,
        runtime_max_bytes, max_total_model_bytes, context_tokens,
        max_new_tokens=256, max_sessions=1, prefill_chunk_tokens=32,
        max_snapshots_per_session=4, max_logit_buffer_bytes=256*1024**2,
        read_limits=None, expert_provider=None):
    """Admit exact local original DSA tensors with independently owned sessions."""
    import json
    import hashlib
    from .native_checkpoint import load_native_text_decoder, _pairs
    from .executor_budget import BudgetedExecutorFactory
    root = Path(directory).resolve(strict=True)
    path = Path(library).resolve(strict=True)
    if not root.is_dir() or not path.is_file():
        raise ValueError('local DSA checkpoint and native library required')
    with (root/'config.json').open('rb') as stream:
        encoded = stream.read(1024**2 + 1)
    if len(encoded) > 1024**2:
        raise MemoryError('DSA configuration exceeds admission')
    raw = json.loads(encoded, object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if (type(raw) is not dict or raw.get('model_type') != 'glm_moe_dsa'
            or context_tokens != raw.get('max_position_embeddings')):
        raise ValueError('exact DSA family and checkpoint context required')
    if type(max_snapshots_per_session) is not int or not 1 <= max_snapshots_per_session <= 62:
        raise ValueError('bounded DSA snapshot inventory required')
    if type(prefill_chunk_tokens) is not int or not 1 <= prefill_chunk_tokens <= 4096:
        raise ValueError('bounded DSA prefill chunk required')
    limits = ExecutionLimits(raw.get('vocab_size'), context_tokens, max_new_tokens,
        max_sessions, prefill_chunk_tokens, max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)
    def construct():
        if expert_provider is None:
            decoder = load_native_text_decoder(root, model_type='glm_moe_dsa', library=path,
                max_weight_bytes=max_weight_bytes, max_bytes=runtime_max_bytes,
                max_snapshots=max_snapshots_per_session + 2, retain_diagnostics=False,
                read_limits=read_limits)
        else:
            from .native_glm_dsa import NativeGlmDsaDecoder
            decoder = NativeGlmDsaDecoder.from_streamed_directory(root,library=path,
                expert_provider=expert_provider,max_weight_bytes=max_weight_bytes,
                max_bytes=runtime_max_bytes,max_snapshots=max_snapshots_per_session+2,
                retain_diagnostics=False,read_limits=read_limits)
        try:
            expected = hashlib.sha256(json.dumps(raw, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if decoder.source_config_digest != expected:
                raise ValueError('DSA configuration changed after service admission')
            return _StructuredNativeSessionExecutor(decoder)
        except BaseException as error:
            try:
                decoder.close()
            except BaseException as failure:
                error.cleanup = decoder.close
                error.add_note('DSA constructor cleanup retained: ' + str(failure))
            raise
    factory = BudgetedExecutorFactory(construct, executor_bytes=runtime_max_bytes,
        max_total_bytes=max_total_model_bytes, max_executors=max_sessions)
    return NativeExecutionService(factory, limits, factory_cleanup=factory.retry_cleanup)


def create_qwen4_execution_service(config, tensors, *, library, runtime_max_bytes,
        max_new_tokens=256, max_sessions=1, prefill_chunk_tokens=512,
        max_snapshots_per_session=4, max_logit_buffer_bytes=256*1024**2,native_composition=False,
        max_total_model_bytes=None):
    """Create independent native Qwen4 text generation sessions.

    The caller supplies the exact resident FP32 tensor inventory and explicit
    native library. Per-session admission includes retained snapshots and all
    recurrent/QSA/PLE state. Configuration is copied at service admission;
    source tensor buffers remain caller-owned until a decoder copies them.
    """
    import json
    from types import MappingProxyType
    from .native_qwen4 import NativeQwen4TextDecoder
    if type(native_composition) is not bool:raise ValueError('boolean complete native composition policy required')
    if type(runtime_max_bytes) is not int or runtime_max_bytes <= 0:
        raise ValueError('positive Qwen4 decoder byte budget required')
    if type(max_snapshots_per_session) is not int or not 1 <= max_snapshots_per_session <= 62:
        raise ValueError('Qwen4 public snapshot inventory must be between one and 62')
    if type(prefill_chunk_tokens) is not int or not 1 <= prefill_chunk_tokens <= 4096:
        raise ValueError('bounded Qwen4 prefill chunk required')
    raw = json.loads(json.dumps(dict(config), allow_nan=False))
    if raw.get('model_type') != 'qwen4_exp_text':
        raise ValueError('exact qwen4_exp_text configuration required')
    limits = ExecutionLimits(raw.get('vocab_size'), raw.get('max_position_embeddings'),
        max_new_tokens, max_sessions, prefill_chunk_tokens, max_snapshots_per_session,
        max_logit_buffer_bytes=max_logit_buffer_bytes)
    path = Path(library).resolve(strict=True)
    if not path.is_file():
        raise ValueError('explicit Qwen4 native library required')
    weights = MappingProxyType(dict(tensors))

    def create_executor():
        decoder = NativeQwen4TextDecoder(raw, weights, library=str(path),
            max_bytes=runtime_max_bytes, max_snapshots=max_snapshots_per_session+2,
            native_composition=native_composition)
        return _StructuredNativeSessionExecutor(decoder)

    from .executor_budget import BudgetedExecutorFactory
    factory=BudgetedExecutorFactory(create_executor,executor_bytes=runtime_max_bytes,
        max_total_bytes=runtime_max_bytes*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    return NativeExecutionService(factory, limits,factory_cleanup=factory.retry_cleanup)


def create_publisher_v41_execution_service(payload, arithmetic_factory, *,
                                            library, runtime_max_bytes,
                                            max_new_tokens=256, max_sessions=1,
                                            prefill_chunk_tokens=512,
                                            max_executor_bytes=None,max_total_model_bytes=None):
    """Wire a publisher-shaped decoder to a supplied concrete arithmetic factory.

    The factory must return a new arithmetic facade on every invocation; the
    service owns it. Checkpoint/table providers borrowed by that facade remain
    caller-owned unless the facade explicitly documents different ownership.
    This function neither supplies a missing arithmetic backend nor certifies
    it. It never replaces quantized projections with float32 products.
    """
    from .publisher_checkpoint.v41_adapter import V41ExecutionPayload
    from .v41_execution.publisher import PublisherV41Execution

    if type(payload) is not V41ExecutionPayload or not callable(arithmetic_factory):
        raise ValueError('publisher V4.1 payload and fresh arithmetic factory required')
    if type(runtime_max_bytes) is not int or runtime_max_bytes <= 0:
        raise ValueError('positive publisher decoder memory budget required')
    path = Path(library).resolve(strict=True)
    if not path.is_file():
        raise ValueError('explicit publisher native library file required')
    limits = ExecutionLimits(payload.profile.vocabulary, payload.profile.context,
                             max_new_tokens, max_sessions, prefill_chunk_tokens)
    pending_cleanup = {}
    live_arithmetic = {}

    def cleanup_failed_arithmetic():
        errors = []
        for identity, arithmetic in tuple(pending_cleanup.items()):
            try:
                arithmetic.close()
            except BaseException as error:
                errors.append(error)
            else:
                pending_cleanup.pop(identity, None)
                live_arithmetic.pop(identity, None)
        if errors:
            raise RuntimeError('publisher arithmetic cleanup is incomplete') from errors[0]

    def create_executor():
        if pending_cleanup:
            raise RuntimeError('publisher arithmetic cleanup is pending; close the service')
        arithmetic = arithmetic_factory()
        if id(arithmetic) in live_arithmetic:
            raise ValueError('arithmetic factory reused a live mutable publisher facade')
        close = getattr(arithmetic, 'close', None)
        if not callable(close):
            raise ValueError('owned publisher arithmetic must provide close')
        live_arithmetic[id(arithmetic)] = arithmetic
        try:
            model = PublisherV41Execution.from_payload(
                payload, arithmetic, library=str(path), max_bytes=runtime_max_bytes)
        except BaseException as error:
            try:
                close()
            except BaseException as cleanup_error:
                pending_cleanup[id(arithmetic)] = arithmetic
                error.add_note('publisher arithmetic cleanup is pending: '+str(cleanup_error))
                error.cleanup = cleanup_failed_arithmetic
            else:
                live_arithmetic.pop(id(arithmetic), None)
            raise
        from .owned_resources import OwnedResources
        tracking=OwnedResources()
        tracking.callback(live_arithmetic.pop,id(arithmetic),None)
        return _OwnedPublisherExecutor(model, arithmetic, tracking)

    from .executor_budget import BudgetedExecutorFactory
    reservation=runtime_max_bytes if max_executor_bytes is None else max_executor_bytes
    if type(reservation) is not int or reservation<runtime_max_bytes:
        raise ValueError('publisher executor reservation must include its decoder budget')
    factory=BudgetedExecutorFactory(create_executor,executor_bytes=reservation,
        max_total_bytes=reservation*max_sessions if max_total_model_bytes is None else max_total_model_bytes,
        max_executors=max_sessions)
    def cleanup():
        cleanup_failed_arithmetic()
        factory.retry_cleanup()
    return NativeExecutionService(factory, limits, factory_cleanup=cleanup)


def publisher_v41_dspark_session(payload, arithmetic_factory, *, library,
                                runtime_max_bytes, draft_max_bytes,
                                evidence_max_bytes, sampling_max_bytes,
                                temperature=1.0, target_rng=None, proposal_rng=None):
    """Return a context manager owning one backbone and its arithmetic handles.

    Packed checkpoint providers remain caller-owned. No libraries are built and
    no model files are downloaded by this factory.
    """
    from contextlib import ExitStack, contextmanager
    from .publisher_checkpoint.v41_adapter import V41ExecutionPayload
    from .v41_execution.publisher import PublisherV41Execution
    from .v41_execution.publisher_auxiliary import PublisherV41Auxiliary
    from .v41_execution.dspark import create_dspark_decoder

    if type(payload) is not V41ExecutionPayload or not callable(arithmetic_factory):
        raise ValueError('publisher payload and fresh arithmetic factory required')
    for budget in (runtime_max_bytes, draft_max_bytes, evidence_max_bytes, sampling_max_bytes):
        if type(budget) is not int or budget <= 0:
            raise ValueError('explicit positive session memory budgets required')
    path = Path(library).resolve(strict=True)
    if not path.is_file():
        raise ValueError('explicit native execution library required')

    @contextmanager
    def session():
        with ExitStack() as owned:
            arithmetic = arithmetic_factory()
            close = getattr(arithmetic, 'close', None)
            if not callable(close):
                raise ValueError('owned arithmetic must provide close')
            owned.callback(close)
            model = PublisherV41Execution.from_payload(payload, arithmetic,
                library=str(path), max_bytes=runtime_max_bytes)
            owned.callback(model.close)
            draft = PublisherV41Auxiliary(arithmetic).mtp(max_bytes=draft_max_bytes)
            decoder = create_dspark_decoder(model, draft, temperature=temperature,
                max_bytes=sampling_max_bytes, evidence_max_bytes=evidence_max_bytes,
                target_rng=target_rng, proposal_rng=proposal_rng)
            yield decoder

    return session()


def open_publisher_v41_service(directory, *, execution_library=None, mxfp4_library=None,
                              fp8_library=None, context_tokens, max_weight_bytes,
                              runtime_max_bytes, arithmetic_max_bytes,
                              max_expert_bytes, max_projection_bytes,
                              compressed_ids=None, tokenizer=None,
                              max_new_tokens=256, prefill_chunk_tokens=32,
                              projection_backend=None, sharded_projections=(),
                              bf16_projection_max_bytes=None, distributed=None,
                              max_metadata_bytes=256*1024**2,
                              max_engram_bytes=4*1024**2,max_total_bytes=None,
                              device=None,torch_version=None):
    """Own an admitted local V4.1 checkpoint and serial generation sessions.

    Shared packed providers remain open until decoder cleanup succeeds. This
    entry point does not download files, build libraries, or certify publisher
    numerical equivalence. Aggregate reservations cover declared engine
    allocation budgets; interpreter and system allocations remain separate.

    A projection bank is borrowed and must already contain the exact selected
    source-name shards on every rank. Its owner keeps it and its process group
    alive until this service closes. Every rank must drive matching sessions.
    """
    from .publisher_checkpoint.service import PublisherCheckpoint
    from .publisher_checkpoint.v41_adapter import load_v41
    from .v41_execution.publisher_arithmetic import NativePublisherV41Arithmetic
    from .v41_execution.publisher import PublisherV41Execution
    from .v41_execution.bf16_projections import PublisherBF16Projections

    if bf16_projection_max_bytes is None:
        bf16_projection_max_bytes=64*1024**2 if projection_backend is None else 0
    if type(bf16_projection_max_bytes) is not int or bf16_projection_max_bytes<0:
        raise ValueError('nonnegative BF16 row streaming budget required')
    if (bf16_projection_max_bytes or distributed is not None) and projection_backend is not None:
        raise ValueError('choose borrowed projections or service-owned streamed projections')
    if (bf16_projection_max_bytes or distributed is not None) and sharded_projections:
        raise ValueError('service-owned banks derive the exact omitted projection inventory')
    if distributed is not None:
        from .v41_distributed_session import V41DistributedOptions
        if type(distributed) is not V41DistributedOptions:
            raise TypeError('explicit caller-owned distributed transport required')
        if distributed.bf16 and not bf16_projection_max_bytes:
            raise ValueError('distributed BF16 requires an admitted row-streaming budget')

    for name, value in (
        ('context_tokens', context_tokens), ('max_weight_bytes', max_weight_bytes),
        ('runtime_max_bytes', runtime_max_bytes),
        ('arithmetic_max_bytes', arithmetic_max_bytes),
        ('max_expert_bytes', max_expert_bytes),
        ('max_projection_bytes', max_projection_bytes),
        ('max_metadata_bytes',max_metadata_bytes),('max_engram_bytes',max_engram_bytes),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError('positive integer budget required: '+name)
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('local checkpoint directory required')
    paths = []
    if device is None and any(path is None for path in (execution_library,mxfp4_library,fp8_library)):
        raise ValueError('native V4.1 requires explicit execution, FP4 and FP8 libraries')
    if device is not None:
        if type(torch_version) is not str or not torch_version:
            raise ValueError('V4.1 device execution requires an explicit Torch version pin')
        if distributed is not None and distributed.torch_version!=torch_version:
            raise ValueError('V4.1 collective and arithmetic Torch pins differ')
        paths=[None,None,None]
    for filename in (() if device is not None else (execution_library, mxfp4_library, fp8_library)):
        path = Path(filename).resolve(strict=True)
        if not path.is_file():
            raise ValueError('explicit native library file required')
        paths.append(str(path))

    owned, failed_executors = [], {}

    def own(resource):
        if resource is not None and not any(item is resource for item in owned):
            if not callable(getattr(resource, 'close', None)):
                raise ValueError('owned checkpoint resource must provide close')
            owned.append(resource)
        return resource

    def cleanup():
        for key, executor in tuple(failed_executors.items()):
            executor.close()
            del failed_executors[key]
        # Stop on failure: an earlier owner may depend on this resource.
        while owned:
            owned[-1].close()
            owned.pop()

    try:
        from .v41_resource_admission import reserve_components
        class FactoryOwnership:
            def callback(self,method):
                class Release:
                    def close(self):method()
                own(Release())
        components=dict(metadata=max_metadata_bytes,resident=max_weight_bytes,
            experts=max_expert_bytes,packed_projections=max_projection_bytes,
            engram_and_token_rows=2*max_engram_bytes,bf16=bf16_projection_max_bytes,
            arithmetic=arithmetic_max_bytes,runtime=runtime_max_bytes,
            sampling=128*1024**2,logits=256*1024**2)
        if distributed is not None:components['collective']=distributed.max_collective_bytes
        ledger=reserve_components(components,max_total_bytes=max_total_bytes,owned=FactoryOwnership())
        checkpoint = PublisherCheckpoint.open(root, family='deepseek_v41',
            max_metadata_bytes=max_metadata_bytes)
        if callable(getattr(checkpoint,'close',None)):own(checkpoint)
        if bf16_projection_max_bytes:
            inventory=PublisherBF16Projections(checkpoint,max_bytes=bf16_projection_max_bytes)
            try:sharded_projections=tuple(sorted(inventory.sources))
            finally:inventory.close()
        payload = load_v41(checkpoint, capacity=context_tokens,
            max_weight_bytes=max_weight_bytes, max_expert_bytes=max_expert_bytes,
            max_projection_bytes=max_projection_bytes, use_providers=True,
            max_engram_bytes=max_engram_bytes,
            sharded_projections=sharded_projections)
        for provider in (payload.engram_provider, payload.expert_provider, payload.fp8_provider,
                         payload.embedding_provider):
            own(provider)
        fp4_rows=None
        if device is not None and payload.expert_provider is not None:
            from .publisher_checkpoint.v41_provider import V41FP4ProjectionProvider
            fp4_rows=own(V41FP4ProjectionProvider(checkpoint,max_projection_bytes=max_projection_bytes))
        limits = ExecutionLimits(payload.profile.vocabulary, payload.profile.context,
                                 max_new_tokens, 1, prefill_chunk_tokens)
    except BaseException as error:
        try:
            cleanup()
        except BaseException as cleanup_error:
            # Retain a callable owner on the error so failed release can be
            # retried without losing handles to mapped files or native state.
            error.cleanup = cleanup
            error.add_note('checkpoint cleanup remains pending: '+str(cleanup_error))
        raise

    def create_executor():
        if failed_executors:
            raise RuntimeError('failed publisher executor cleanup is pending; close the service')
        from .owned_resources import OwnedResources
        resources=OwnedResources();bank=projection_backend;bf16=None;arithmetic=None
        try:
            if bf16_projection_max_bytes:
                bf16=PublisherBF16Projections(checkpoint,max_bytes=bf16_projection_max_bytes)
                resources.callback(bf16.close);bank=bf16
            if distributed is not None:
                from .v41_distributed_session import make_projection_backend
                bank=make_projection_backend(distributed,checkpoint=checkpoint,payload=payload,
                    bf16_provider=bf16,fp8_library=paths[2],fp4_library=paths[1],
                    max_arithmetic_bytes=arithmetic_max_bytes,owned=resources,arithmetic_device=device)
            if device is None:
                arithmetic = NativePublisherV41Arithmetic(payload,
                    execution_library=paths[0], mxfp4_library=paths[1], fp8_library=paths[2],
                    tokenizer=tokenizer, compressed_ids=compressed_ids,
                    max_bytes=arithmetic_max_bytes, projection_backend=bank)
                model_type=PublisherV41Execution
            else:
                from .v41_execution.device_arithmetic import DevicePublisherV41Arithmetic
                from .v41_execution.device_execution import DevicePublisherV41Execution
                arithmetic=DevicePublisherV41Arithmetic(payload,device=device,torch_version=torch_version,
                    max_bytes=arithmetic_max_bytes,tokenizer=tokenizer,compressed_ids=compressed_ids,
                    projection_backend=bank,fp4_provider=fp4_rows)
                model_type=DevicePublisherV41Execution
            if bf16 is not None:bf16.bind_arithmetic(arithmetic)
            model = model_type.from_payload(payload, arithmetic,
                library=paths[0], max_bytes=runtime_max_bytes)
        except BaseException as error:
            try:
                if arithmetic is not None:arithmetic.close()
                resources.close()
            except BaseException as cleanup_error:
                class PendingRelease:
                    def close(self):
                        if arithmetic is not None:arithmetic.close()
                        resources.close()
                pending=PendingRelease()
                failed_executors[id(pending)] = pending
                error.add_note('native arithmetic cleanup remains pending: '+str(cleanup_error))
            raise
        return _OwnedPublisherExecutor(model, arithmetic, resources)

    try:
        service=NativeExecutionService(create_executor, limits, factory_cleanup=cleanup)
        service.resource_ledger=ledger
        service.publisher_payload=payload
        return service
    except BaseException as error:
        try:
            cleanup()
        except BaseException as cleanup_error:
            error.cleanup = cleanup
            error.add_note('checkpoint cleanup remains pending: '+str(cleanup_error))
        raise
