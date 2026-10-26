"""Source-reference catalog for selected Phase 1 routes.

This module records claimed source paths; populated strings alone are not
evidence that a path exists or is wired. Runtime route certification belongs
to the pinned observation gate. In particular, the A02 native text service
factories below are programmatic APIs, not proof of family CLI, tensor-parallel,
or portable recovery integration.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable, Any


@dataclass
class LoaderTrace:
    """Trace: config → checkpoint → loader → consumer."""
    config_validator: str  # Function that validates config.model_type
    checkpoint_type: str   # Publisher checkpoint or native directory
    loader_function: str   # Module.load_from_checkpoint(...)
    tensor_source: str     # Source dtype / format from checkpoint

    def validate(self) -> bool:
        """Ensure loader has source dtype check."""
        assert self.config_validator, "Config validator required"
        assert self.loader_function, "Loader function required"
        return True


@dataclass
class ConsumerTrace:
    """Trace: loader output → tensor consumer → logits."""
    tensor_consumer: str   # Class that consumes loaded tensors
    execution_backend: str  # native, device, public
    logits_producer: str   # forward(...) produces logits

    def validate(self) -> bool:
        """Ensure consumer produces logits."""
        assert self.tensor_consumer, "Tensor consumer required"
        assert self.logits_producer, "Logits producer required"
        return True


@dataclass
class TPTrace:
    """Trace: logits → cache/TP owner → next layer."""
    cache_owner: str       # Module that owns cache state
    tp_owner: str          # tensor_parallel module
    state_advancement: str # How KV cache advances per token

    def validate(self) -> bool:
        """Ensure cache owns KV state."""
        assert self.cache_owner, "Cache owner required"
        assert self.tp_owner, "TP owner required"
        return True


@dataclass
class ABITrace:
    """Trace: execution state → ABI → factory/CLI/gateway."""
    abi_symbols: str       # native_*.py module with symbols/ranges
    continuation_format: str  # How session state is captured
    factory_function: str  # execution_factories.create_*_session(...)
    cli_entry: str         # native_*_cli.PublisherSession
    factory_model_type: str = ""

    def validate(self) -> bool:
        """Check that the trace declares its ABI/factory identity fields."""
        assert self.abi_symbols, "ABI symbols required"
        assert self.factory_function, "Factory function required"
        assert self.cli_entry, "CLI entry required"
        assert self.factory_model_type, "Exact factory model_type declaration required"
        return True


@dataclass
class FamilyRouteTrace:
    """Complete route from checkpoint → generation for one family/backend/format."""
    family: str
    model_type: str
    backend: str  # native, device, public
    arithmetic: str
    source_format: str
    persisted_format: str

    loader: LoaderTrace = field(default_factory=lambda: LoaderTrace("", "", "", ""))
    consumer: ConsumerTrace = field(default_factory=lambda: ConsumerTrace("", "", ""))
    cache_tp: TPTrace = field(default_factory=lambda: TPTrace("", "", ""))
    abi: ABITrace = field(default_factory=lambda: ABITrace("", "", "", ""))

    recovery_writer: str = ""  # service_generation_checkpoint writer
    recovery_reader: str = ""  # cache_recovery reader
    tooling: str = ""          # wholemodel_validation adapter

    def validate_complete_path(self) -> bool:
        """Validate catalog completeness, not source existence or runtime closure."""
        self.loader.validate()
        self.consumer.validate()
        self.cache_tp.validate()
        self.abi.validate()
        assert self.recovery_writer, "Recovery writer required"
        assert self.recovery_reader, "Recovery reader required"
        assert self.tooling, "Tooling adapter required"
        return True


# ============================================================================
# PRIORITY MODEL ROUTES - A02
# ============================================================================

QWEN35_NATIVE_TRACE = FamilyRouteTrace(
    family='qwen35',
    model_type='qwen3_5_moe',
    backend='native',
    arithmetic='fp32',
    source_format='F32',
    persisted_format='float32',

    loader=LoaderTrace(
        config_validator='publisher_family_contracts.component_contract(config)',
        checkpoint_type='publisher conditional SafeTensors checkpoint with media component inventory',
        loader_function='publisher_conditional_loader.load_publisher_conditional_decoder(...)',
        tensor_source='F32 tensors admitted as float32 before conditional decoder construction',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='qwen35_multimodal.Qwen35MultimodalDecoder via multimodal_session.MultimodalSessionExecutor',
        execution_backend='native',
        logits_producer='MultimodalSessionExecutor.forward/forward_media -> batch-one logits',
    ),

    cache_tp=TPTrace(
        cache_owner='Qwen35MultimodalDecoder snapshot/cache owner wrapped by MultimodalSessionExecutor',
        tp_owner='not wired by native conditional service; public Segment/Edge is a distinct backend route',
        state_advancement='Text and media-expanded token positions advance in the conditional decoder transaction',
    ),

    abi=ABITrace(
        abi_symbols='native_qwen35_vision.NativeQwen35Vision kernel ABI; public Segment/Edge adapter is not this native route',
        continuation_format='Qwen35MultimodalDecoder snapshot/cache transport through MultimodalSessionExecutor',
        factory_function='execution_factories.open_native_conditional_directory_service(family="qwen3_5_moe", ...)',
        cli_entry='publisher_provider.parse_publisher_operator -> native_conditional_cli.PublisherSession',
        factory_model_type='qwen3_5_moe',
    ),

    recovery_writer='NativeExecutionService.save_session_cache(...) when the conditional executor exposes cache transport',
    recovery_reader='NativeExecutionService.restore_session_cache(...); exact decoder identity required',
    tooling='wholemodel_validation.qwen35_media and qwen35_publisher_caches source adapters; runtime receipts remain required',
)

QWEN4_NATIVE_TRACE = FamilyRouteTrace(
    family='qwen4',
    model_type='qwen4_exp',
    backend='native',
    arithmetic='fp32',
    source_format='F32',
    persisted_format='float32',

    loader=LoaderTrace(
        config_validator='publisher_family_contracts.component_contract(config)',
        checkpoint_type='publisher conditional SafeTensors checkpoint with media and QSA component inventory',
        loader_function='publisher_conditional_loader.load_publisher_conditional_decoder(...)',
        tensor_source='F32 tensors admitted as float32 before conditional decoder construction',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='qwen4_multimodal.Qwen4MultimodalDecoder via multimodal_session.MultimodalSessionExecutor',
        execution_backend='native',
        logits_producer='MultimodalSessionExecutor.forward/forward_media -> batch-one logits',
    ),

    cache_tp=TPTrace(
        cache_owner='Qwen4MultimodalDecoder snapshot/cache owner wrapped by MultimodalSessionExecutor',
        tp_owner='not wired by native conditional service; public Segment/Edge is a distinct backend route',
        state_advancement='QSA/gated attention and text/media token position advance in conditional decoder state',
    ),

    abi=ABITrace(
        abi_symbols='native_qwen4.NativeQwen4TextDecoder native kernel bindings; public Segment/Edge adapter is not this native route',
        continuation_format='Qwen4MultimodalDecoder snapshot/cache transport through MultimodalSessionExecutor',
        factory_function='execution_factories.open_native_conditional_directory_service(family="qwen4_exp", ...)',
        cli_entry='publisher_provider.parse_publisher_operator -> native_conditional_cli.PublisherSession',
        factory_model_type='qwen4_exp',
    ),

    recovery_writer='NativeExecutionService.save_session_cache(...) when the conditional executor exposes cache transport',
    recovery_reader='NativeExecutionService.restore_session_cache(...); exact decoder identity required',
    tooling='wholemodel_validation.qwen4_media and qwen4_publisher_caches source adapters; runtime receipts remain required',
)

GLM5_NATIVE_TRACE = FamilyRouteTrace(
    family='glm5-next',
    model_type='glm5_next',
    backend='native',
    arithmetic='fp32',
    source_format='F32',
    persisted_format='float32',

    loader=LoaderTrace(
        config_validator='publisher_glm5_config.validate_glm5_identity(config)',
        checkpoint_type='publisher conditional SafeTensors checkpoint with media component inventory',
        loader_function='publisher_conditional_loader.load_publisher_conditional_decoder(...)',
        tensor_source='F32 tensors admitted as float32 before conditional decoder construction',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='glm5_multimodal.Glm5MultimodalDecoder via multimodal_session.MultimodalSessionExecutor',
        execution_backend='native',
        logits_producer='MultimodalSessionExecutor.forward/forward_media -> batch-one logits',
    ),

    cache_tp=TPTrace(
        cache_owner='Glm5MultimodalDecoder snapshot/cache owner wrapped by MultimodalSessionExecutor',
        tp_owner='not wired by native conditional service; public Segment/Edge is a distinct backend route',
        state_advancement='GLM5 media-expanded text state advances within the conditional decoder transaction',
    ),

    abi=ABITrace(
        abi_symbols='native_glm5_next native kernel bindings; public Segment/Edge adapter is not this native conditional route',
        continuation_format='Glm5MultimodalDecoder snapshot/cache transport through MultimodalSessionExecutor',
        factory_function='execution_factories.open_native_conditional_directory_service(family="glm5_next", ...)',
        cli_entry='publisher_provider.parse_publisher_operator -> native_conditional_cli.PublisherSession when media library selected',
        factory_model_type='glm5_next',
    ),

    recovery_writer='NativeExecutionService.save_session_cache(...) when the conditional executor exposes cache transport',
    recovery_reader='NativeExecutionService.restore_session_cache(...); exact decoder identity required',
    tooling='wholemodel_validation.glm5_media and publisher-family source adapters; runtime receipts remain required',
)


def _native_text_factory_trace(family, model_type, decoder, factory, context_description):
    """Trace the separate direct text-service APIs without aliasing wrappers."""
    return FamilyRouteTrace(
        family=family, model_type=model_type, backend='native', arithmetic='fp32',
        source_format='F32', persisted_format='float32',
        loader=LoaderTrace(
            config_validator='native_checkpoint exact config.model_type check for ' + model_type,
            checkpoint_type='text-only native SafeTensors checkpoint',
            loader_function='native_checkpoint.load_native_text_decoder(...) after full inventory admission',
            tensor_source='float32 text weights',
        ),
        consumer=ConsumerTrace(
            tensor_consumer=decoder,
            execution_backend='native',
            logits_producer=decoder.rsplit('.', 1)[-1] + '.forward(tokens) -> logits',
        ),
        cache_tp=TPTrace(
            cache_owner='execution_service.NativeExecutionService with the exact text decoder snapshot',
            tp_owner='not wired by this direct text-service factory; no tensor-parallel owner is claimed',
            state_advancement=context_description,
        ),
        abi=ABITrace(
            abi_symbols='direct NativeExecutionService API; no public Segment/Edge ABI is wired by this factory',
            continuation_format='ExecutionSessionSnapshot; process-local handle plus decoder-owned snapshot',
            factory_function='execution_factories.' + factory + '(model_path, config_path, library=...)',
            cli_entry='not wired by this direct factory; family CLI/gateway routes are separately traced',
            factory_model_type=model_type,
        ),
        recovery_writer='NativeExecutionService.save_session_cache(...) when decoder cache transport is available',
        recovery_reader='NativeExecutionService.restore_session_cache(...) with exact executor identity',
        tooling='no dedicated pinned comparison adapter for this direct factory; P13 evidence binding remains open',
    )


QWEN35_TEXT_NATIVE_TRACE = _native_text_factory_trace(
    'qwen35', 'qwen3_5_moe_text', 'native_qwen35.NativeQwen35TextDecoder',
    'create_qwen35_native_session', 'Qwen3.5 text decoder position advances per committed token')
QWEN4_TEXT_NATIVE_TRACE = _native_text_factory_trace(
    'qwen4', 'qwen4_exp_text', 'native_qwen4.NativeQwen4TextDecoder',
    'create_qwen4_native_session', 'Qwen4 text decoder position advances per committed token')
GLM5_TEXT_NATIVE_TRACE = _native_text_factory_trace(
    'glm5-next', 'glm5_next_text', 'native_glm5_next.NativeGlm5NextTextDecoder',
    'create_glm5_native_session', 'GLM5 text decoder position advances per committed token')

GLM_DSA_NATIVE_TRACE = FamilyRouteTrace(
    family='glm-dsa',
    model_type='glm_moe_dsa',
    backend='native',
    arithmetic='fp32',
    source_format='F32',
    persisted_format='float32',

    loader=LoaderTrace(
        config_validator='independent_glm_dsa.validate_glm_dsa_config(config)',
        checkpoint_type='native checkpoint safetensors',
        loader_function='native_checkpoint.read_native_text_checkpoint(...)',
        tensor_source='float32 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_glm_dsa.NativeGlmDsaDecoder',
        execution_backend='native',
        logits_producer='NativeGlmDsaDecoder.forward(tokens) -> (logits, cache)',
    ),

    cache_tp=TPTrace(
        cache_owner='execution_service.NativeExecutionService session state and NativeGlmDsaDecoder snapshot',
        tp_owner='not wired by create_glm_dsa_native_session; tensor parallel is outside this factory',
        state_advancement='Per-token executor forward; service history commits transactionally',
    ),

    abi=ABITrace(
        abi_symbols='glm_dsa_public_adapters.GlmDsaPublicAdapters',
        continuation_format='ExecutionSessionSnapshot (service-owned); portable cache export is a separate opt-in service operation',
        factory_function='execution_factories.create_glm_dsa_native_session(model_path, config_path, library=compiled_glm_dsa_library)',
        cli_entry='not wired by this direct factory; programmatic NativeExecutionService only',
        factory_model_type='glm_moe_dsa',
    ),

    recovery_writer='NativeExecutionService.save_session_cache(...) / write_session_cache(...) when explicitly configured',
    recovery_reader='NativeExecutionService.restore_session_cache(...); executor identity and cache transport required',
    tooling='wholemodel_validation.glm_dsa_caches.compare_dsa_cache_state(...)',
)

DEEPSEEK_V41_NATIVE_BF16_TRACE = FamilyRouteTrace(
    family='deepseek-v41',
    model_type='deepseek_v41',
    backend='native',
    arithmetic='publisher-bf16-fp8-fp4',
    source_format='BF16',
    persisted_format='bf16',

    loader=LoaderTrace(
        config_validator='publisher_checkpoint.contracts.validate_v41_identity(config)',
        checkpoint_type='publisher v41 checkpoint with bf16 bank',
        loader_function='publisher_checkpoint.v41_adapter.load_v41_bf16(...)',
        tensor_source='bf16 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='v41_execution.publisher.PublisherV41Execution',
        execution_backend='native',
        logits_producer='PublisherV41Execution.forward(tokens) -> (logits, target_cache, draft_cache)',
    ),

    cache_tp=TPTrace(
        cache_owner='v41_execution.publisher.PublisherV41Execution and optional SpeculativeExecution target/draft snapshot owners',
        tp_owner='not part of this single-rank native trace; distributed service is a separate gated route',
        state_advancement='Publisher target KV plus optional MTP/DSpark speculative state; exact stages depend on admitted options',
    ),

    abi=ABITrace(
        abi_symbols='v41_execution.publisher_arithmetic.NativePublisherV41Arithmetic',
        continuation_format='PublisherV41Snapshot / SpeculativeSnapshot; DSparkBackboneSnapshot only for an enabled DSpark stage',
        factory_function='execution_factories.open_publisher_v41_service(...) or create_public_publisher_v41_execution_service(...)',
        cli_entry='publisher_v41_cli.PublisherSession',
        factory_model_type='deepseek_v41',
    ),

    recovery_writer='NativeExecutionService.save_session_cache(...) and caller checkpoint sink for the selected execution service',
    recovery_reader='NativeExecutionService.restore_session_cache(...); distributed recovery is separate and remains restricted',
    tooling='wholemodel_validation.v41 replay/cache readers; named speculative comparison adapter is not registered',
)


# ============================================================================
# REMAINING FAMILIES - A03
# ============================================================================

MIXTRAL_NATIVE_TRACE = FamilyRouteTrace(
    family='mixtral',
    model_type='mixtral',
    backend='native',
    arithmetic='fp32',
    source_format='F32',
    persisted_format='float32',

    loader=LoaderTrace(
        config_validator='native_family_cli.PublisherSession exact config.model_type and require_checkpoint_inventory_admission(...)',
        checkpoint_type='local Mixtral SafeTensors checkpoint and optional packed-expert manifest',
        loader_function='native_family_cli.PublisherSession -> NativeMixtralPublicAdapters; source-bound checkpoint admission precedes registration',
        tensor_source='source-bound resident dense/router tensors; optional packed experts use explicit provider and no-fallback kernel',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_mixtral_decoder.NativeMixtralDecoder',
        execution_backend='native',
        logits_producer='NativeMixtralDecoder.forward(tokens) -> logits',
    ),

    cache_tp=TPTrace(
        cache_owner='native_mixtral_decoder.NativeMixtralDecoder snapshot/restore (MixtralSnapshot)',
        tp_owner='not established by the native family CLI route; this is a single-model Segment/Edge registration, not tensor-parallel execution',
        state_advancement='Decoder-owned per-token KV position; snapshot receipts are identity/owner bound',
    ),

    abi=ABITrace(
        abi_symbols='mixtral_public_adapters.NativeMixtralPublicAdapters',
        continuation_format='NativeExecutionService session cache/checkpoint only when the selected public owner exposes the cache transport',
        factory_function='native_family_cli.PublisherSession -> NativeMixtralPublicAdapters.register() -> Segment/Edge service',
        cli_entry='native_family_cli.PublisherSession',
        factory_model_type='mixtral',
    ),

    recovery_writer='caller-owned NativeExecutionService checkpoint sink when supported by the selected execution service',
    recovery_reader='caller-owned NativeExecutionService restore path when supported; no MixtralCacheOwner implementation is present',
    tooling='no dedicated wholemodel_validation Mixtral adapter is registered; P13 comparison binding remains open',
)

OLMOE_NATIVE_TRACE = FamilyRouteTrace(
    family='olmoe',
    model_type='olmoe',
    backend='native',
    arithmetic='fp32',
    source_format='F32',
    persisted_format='float32',

    loader=LoaderTrace(
        config_validator='olmoe.validate_olmoe_config(config)',
        checkpoint_type='OLMoE segment manifest',
        loader_function='native_leased_olmoe_segments.NativeLeasedOlmoeSegmentEngine.load(...)',
        tensor_source='float32 segments',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_olmoe_segments.NativeOlmoeSegmentEngine',
        execution_backend='native',
        logits_producer='NativeOlmoeSegmentEngine.forward(tokens) -> logits',
    ),

    cache_tp=TPTrace(
        cache_owner='universal_adapters.OlmoeNativeOwner decoder snapshot_state/restore_state',
        tp_owner='Segment/Edge layer ranges and expert manifest are not tensor-parallel ownership',
        state_advancement='Decoder-owned token position; leased Segment/Edge owner tracks bounded ranges and expert leases',
    ),

    abi=ABITrace(
        abi_symbols='segment_edge_adapters.NativeOlmoePublicAdapters',
        continuation_format='NativeOlmoeDecoderSession state receipt (owner-local); open_olmoe_owner does not expose portable cache transport',
        factory_function='universal_adapters.open_olmoe_owner(...)',
        cli_entry='olmoe_cli.main(...) (standalone CLI; no PublisherSession class)',
        factory_model_type='olmoe',
    ),

    recovery_writer='UniversalSession owner snapshot receipt; durable cache/checkpoint is not exposed by OlmoeNativeOwner',
    recovery_reader='UniversalSession owner restore receipt; portable cross-process recovery is not established by this owner',
    tooling='wholemodel_validation.olmoe_native.create_olmoe_native_adapter(...) and phase1_capture route observations',
)

K3_NATIVE_TRACE = FamilyRouteTrace(
    family='kimi-k3',
    model_type='kimi_k3',
    backend='native',
    arithmetic='fp32-approximate',
    source_format='F32',
    persisted_format='float32',

    loader=LoaderTrace(
        config_validator='publisher_checkpoint.contracts.validate_k3_identity(config)',
        checkpoint_type='publisher k3 checkpoint',
        loader_function='publisher_checkpoint.k3_adapter.load_k3_checkpoint(...)',
        tensor_source='float32 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='k3_execution.executor.K3Execution',
        execution_backend='native',
        logits_producer='K3Execution.execute(tokens) -> (logits, video_state)',
    ),

    cache_tp=TPTrace(
        cache_owner='k3_execution.executor.K3Execution K3Snapshot and owned per-layer attention/linear state',
        tp_owner='not provided by create_k3_execution_service; optional sharded projections are borrowed inputs, not a collective owner',
        state_advancement='K3Execution advances MLA/KDA and temporal state; batch path has separate releasable snapshots',
    ),

    abi=ABITrace(
        abi_symbols='publisher_public_adapters.NativePublisherPublicAdapters',
        continuation_format='k3_execution.executor.K3Snapshot; serialized cache uses K3Execution.write_cache/restore_cache',
        factory_function='execution_factories.create_public_k3_execution_service(...)',
        cli_entry='publisher_k3_cli.PublisherSession',
        factory_model_type='kimi_k3',
    ),

    recovery_writer='K3Execution.write_cache(...) plus NativeExecutionService checkpoint publication when composed',
    recovery_reader='K3Execution.restore_cache(...) plus NativeExecutionService restore when composed',
    tooling='wholemodel_validation.k3.K3CandidateFactory and cache readers; route-bound comparison binding remains open',
)

INKLING_NATIVE_TRACE = FamilyRouteTrace(
    family='inkling',
    model_type='inkling_mm_model',
    backend='native',
    arithmetic='fp32',
    source_format='F32',
    persisted_format='float32',

    loader=LoaderTrace(
        config_validator='publisher_family_contracts.component_contract(config) for inkling_mm_model',
        checkpoint_type='publisher conditional SafeTensors checkpoint with media/shared component inventory',
        loader_function='execution_factories.open_native_conditional_directory_service(...) -> publisher_conditional_loader.load_publisher_conditional_decoder(...)',
        tensor_source='publisher FP32 source tensors admitted against the complete conditional inventory',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='inkling_multimodal.InklingMultimodalDecoder via multimodal_session.MultimodalSessionExecutor',
        execution_backend='native',
        logits_producer='MultimodalSessionExecutor.forward/forward_media -> logits; publisher MTP drafts remain disabled until target verification',
    ),

    cache_tp=TPTrace(
        cache_owner='inkling_multimodal.InklingMultimodalDecoder continuation; GroupedContinuation exists only when optional MTP is enabled',
        tp_owner='not wired by this native conditional factory; optional public range is a separate backend route',
        state_advancement='Inkling sliding/hybrid attention and FIR history; media placeholders and geometry are caller supplied',
    ),

    abi=ABITrace(
        abi_symbols='native_inkling.NativeInklingDecoder C callback binding; public family Segment/Edge adapter is not this publisher identity',
        continuation_format='grouped_continuation.GroupedSnapshot, composed by MultimodalSessionExecutor',
        factory_function='execution_factories.open_native_conditional_directory_service(family="inkling_mm_model", ...)',
        cli_entry='no direct native_family_cli identity for inkling_mm_model; its inkling_text CLI is a distinct text route',
        factory_model_type='inkling_mm_model',
    ),

    recovery_writer='MultimodalSessionExecutor.write_cache(...) through NativeExecutionService when explicitly enabled',
    recovery_reader='MultimodalSessionExecutor.restore_cache(...) through NativeExecutionService; caller media replay is not encoded as portable history',
    tooling='wholemodel_validation.public_publisher_candidate.load_public_publisher(...); dedicated Inkling route comparison remains open',
)


# ============================================================================
# ROUTE REGISTRY
# ============================================================================

ALL_PRIORITY_ROUTES = [
    QWEN35_NATIVE_TRACE,
    QWEN4_NATIVE_TRACE,
    GLM5_NATIVE_TRACE,
    GLM_DSA_NATIVE_TRACE,
    DEEPSEEK_V41_NATIVE_BF16_TRACE,
]

ALL_REMAINING_ROUTES = [
    QWEN35_TEXT_NATIVE_TRACE,
    QWEN4_TEXT_NATIVE_TRACE,
    GLM5_TEXT_NATIVE_TRACE,
    MIXTRAL_NATIVE_TRACE,
    OLMOE_NATIVE_TRACE,
    K3_NATIVE_TRACE,
    INKLING_NATIVE_TRACE,
]

ALL_ROUTES = ALL_PRIORITY_ROUTES + ALL_REMAINING_ROUTES


def validate_all_routes_complete() -> bool:
    """Validate every catalog row is populated; this is not route certification."""
    for route in ALL_ROUTES:
        route.validate_complete_path()
    return True


def get_route_for_tuple(family: str, model_type: str, backend: str, arithmetic: str,
                        source_format: str, persisted_format: str) -> Optional[FamilyRouteTrace]:
    """Look up a route by exact tuple."""
    for route in ALL_ROUTES:
        if (route.family == family and
            route.model_type == model_type and
            route.backend == backend and
            route.arithmetic == arithmetic and
            route.source_format == source_format and
            route.persisted_format == persisted_format):
            return route
    return None


if __name__ == '__main__':
    validate_all_routes_complete()
    print(f"PASS: {len(ALL_ROUTES)} source-trace rows are structurally populated (not runtime certified)")
    print(f"  Priority: {len(ALL_PRIORITY_ROUTES)}")
    print(f"  Remaining: {len(ALL_REMAINING_ROUTES)}")
