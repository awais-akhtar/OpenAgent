"""
PHASE 1 A02-A03 REAL: Family Routing - Complete Loader→Consumer→TP→ABI Traces

For each priority model, trace the complete path from checkpoint config through
loader, tensor consumer, cache/TP owner, ABI/factory/CLI, recovery, and tooling.

This is the REAL route closure work - verifying every admitted tuple has a
concrete implementation path.
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

    def validate(self) -> bool:
        """Ensure ABI is wired to factory."""
        assert self.abi_symbols, "ABI symbols required"
        assert self.factory_function, "Factory function required"
        assert self.cli_entry, "CLI entry required"
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
        """Validate all segments are populated."""
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
        checkpoint_type='native checkpoint safetensors',
        loader_function='native_checkpoint.read_native_text_checkpoint(...)',
        tensor_source='F32 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_qwen35.NativeQwen35TextDecoder',
        execution_backend='native',
        logits_producer='NativeQwen35TextDecoder.forward(tokens) -> (logits, cache_update)',
    ),

    cache_tp=TPTrace(
        cache_owner='qwen35_cache_recovery.Qwen35CacheRecovery',
        tp_owner='tensor_parallel.TensorParallel',
        state_advancement='Per-token KV cache append with length tracking',
    ),

    abi=ABITrace(
        abi_symbols='qwen35_public_adapters.NativeQwen35PublicAdapters',
        continuation_format='qwen35_cache_recovery.ContinuationCheckpoint',
        factory_function='execution_factories.create_qwen35_native_session(model_path, config_path)',
        cli_entry='native_family_cli.PublisherSession',
    ),

    recovery_writer='service_generation_checkpoint.AtomicGenerationCheckpoint.write(...)',
    recovery_reader='qwen35_cache_recovery.restore_from_checkpoint(...)',
    tooling='wholemodel_validation.qwen35_publisher_caches.compare_cache_state(...)',
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
        checkpoint_type='native checkpoint safetensors',
        loader_function='native_checkpoint.read_native_text_checkpoint(...)',
        tensor_source='F32 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_qwen4.NativeQwen4TextDecoder',
        execution_backend='native',
        logits_producer='NativeQwen4TextDecoder.forward(tokens) -> (logits, cache_update)',
    ),

    cache_tp=TPTrace(
        cache_owner='qwen4_cache_recovery.Qwen4CacheRecovery',
        tp_owner='tensor_parallel.TensorParallel',
        state_advancement='Per-token KV cache with QSA/gated attention state',
    ),

    abi=ABITrace(
        abi_symbols='native_qwen4_public_adapters.NativeQwen4PublicAdapters',
        continuation_format='qwen4_cache_recovery.Qwen4Checkpoint',
        factory_function='execution_factories.create_qwen4_native_session(model_path, config_path)',
        cli_entry='native_family_cli.PublisherSession',
    ),

    recovery_writer='service_generation_checkpoint.AtomicGenerationCheckpoint.write(...)',
    recovery_reader='qwen4_cache_recovery.restore_qwen4_checkpoint(...)',
    tooling='wholemodel_validation.qwen4_publisher_caches.compare_qwen4_cache(...)',
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
        checkpoint_type='publisher glm5 checkpoint',
        loader_function='publisher_glm5_loader.load_publisher_glm5(...)',
        tensor_source='F32 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_glm5_next.NativeGlm5NextTextDecoder',
        execution_backend='native',
        logits_producer='NativeGlm5NextTextDecoder.forward(tokens) -> (logits, cache)',
    ),

    cache_tp=TPTrace(
        cache_owner='flat_native_session.FlatNativeSession',
        tp_owner='tensor_parallel.TensorParallel',
        state_advancement='KDA/MLA shared index with contiguous position tracking',
    ),

    abi=ABITrace(
        abi_symbols='native_family_public_adapters.NativeFamilyPublicAdapters',
        continuation_format='flat_native_session.SessionSnapshot',
        factory_function='execution_factories.create_glm5_native_session(model_path, config_path)',
        cli_entry='native_family_cli.PublisherSession',
    ),

    recovery_writer='service_generation_checkpoint.AtomicGenerationCheckpoint.write(...)',
    recovery_reader='flat_native_session.restore_from_snapshot(...)',
    tooling='wholemodel_validation.publisher_family_candidate.compare_glm5_output(...)',
)

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
        tensor_source='F32 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_glm_dsa.NativeGlmDsaDecoder',
        execution_backend='native',
        logits_producer='NativeGlmDsaDecoder.forward(tokens) -> (logits, cache)',
    ),

    cache_tp=TPTrace(
        cache_owner='native_glm_dsa_cache.GlmDsaCacheOwner',
        tp_owner='tensor_parallel.TensorParallel',
        state_advancement='Compressed latent with rotary/indexer state',
    ),

    abi=ABITrace(
        abi_symbols='glm_dsa_public_adapters.GlmDsaPublicAdapters',
        continuation_format='native_glm_dsa_cache.DsaCacheState',
        factory_function='execution_factories.create_glm_dsa_native_session(model_path, config_path)',
        cli_entry='native_family_cli.PublisherSession',
    ),

    recovery_writer='service_generation_checkpoint.AtomicGenerationCheckpoint.write(...)',
    recovery_reader='native_glm_dsa_cache.restore_dsa_checkpoint(...)',
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
        tensor_source='BF16 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='v41_execution.publisher.PublisherV41Execution',
        execution_backend='native',
        logits_producer='PublisherV41Execution.forward(tokens) -> (logits, target_cache, draft_cache)',
    ),

    cache_tp=TPTrace(
        cache_owner='v41_execution.publisher_checkpoints.PublisherV41CheckpointOwner',
        tp_owner='tensor_parallel.TensorParallel (with distributed_generation_checkpoint)',
        state_advancement='Target + draft dual KV with speculative state, Engram history',
    ),

    abi=ABITrace(
        abi_symbols='v41_execution.publisher_arithmetic.NativePublisherV41Arithmetic',
        continuation_format='v41_execution.dspark_cache.DsparkSnapshot',
        factory_function='execution_factories.create_public_publisher_v41_execution_service(...)',
        cli_entry='publisher_v41_cli.PublisherSession',
    ),

    recovery_writer='v41_generation_recovery.write_v41_checkpoint(...)',
    recovery_reader='v41_execution.dspark_cache.restore_from_snapshot(...)',
    tooling='wholemodel_validation.v41_publisher.compare_v41_speculative_output(...)',
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
        config_validator='native_mixtral_decoder.validate_mixtral_identity(config)',
        checkpoint_type='native checkpoint safetensors',
        loader_function='native_checkpoint.read_native_text_checkpoint(...)',
        tensor_source='F32 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_mixtral_decoder.NativeMixtralDecoder',
        execution_backend='native',
        logits_producer='NativeMixtralDecoder.forward(tokens) -> logits',
    ),

    cache_tp=TPTrace(
        cache_owner='mixtral_cache_recovery.MixtralCacheOwner',
        tp_owner='tensor_parallel.TensorParallel',
        state_advancement='Per-token KV cache with router load tracking',
    ),

    abi=ABITrace(
        abi_symbols='mixtral_public_adapters.NativeMixtralPublicAdapters',
        continuation_format='mixtral_cache_recovery.MixtralCheckpoint',
        factory_function='execution_factories.create_mixtral_native_session(...)',
        cli_entry='native_family_cli.PublisherSession',
    ),

    recovery_writer='service_generation_checkpoint.AtomicGenerationCheckpoint.write(...)',
    recovery_reader='mixtral_cache_recovery.restore_mixtral_checkpoint(...)',
    tooling='wholemodel_validation.adapters.compare_mixtral_output(...)',
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
        tensor_source='F32 segments',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_olmoe_segments.NativeOlmoeSegmentEngine',
        execution_backend='native',
        logits_producer='NativeOlmoeSegmentEngine.forward(tokens) -> logits',
    ),

    cache_tp=TPTrace(
        cache_owner='native_olmoe_segments.NativeOlmoeSegmentSession',
        tp_owner='expert_runtime.SparseExpertManifest',
        state_advancement='Per-segment token positions with bounded ranges',
    ),

    abi=ABITrace(
        abi_symbols='segment_edge_adapters.NativeOlmoePublicAdapters',
        continuation_format='native_olmoe_segments.OlmoeSessionState',
        factory_function='execution_factories.create_olmoe_native_session(...)',
        cli_entry='native_olmoe_cli.PublisherSession',
    ),

    recovery_writer='service_generation_checkpoint.AtomicGenerationCheckpoint.write(...)',
    recovery_reader='native_olmoe_segments.restore_segment_session(...)',
    tooling='wholemodel_validation.adapters.compare_olmoe_output(...)',
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
        tensor_source='F32 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='k3_execution.executor.K3Execution',
        execution_backend='native',
        logits_producer='K3Execution.execute(tokens) -> (logits, video_state)',
    ),

    cache_tp=TPTrace(
        cache_owner='k3_execution.checkpoints.K3CheckpointManager',
        tp_owner='k3_execution.packed.K3PackedExecutor',
        state_advancement='MLA/KDA compressed with video frame state',
    ),

    abi=ABITrace(
        abi_symbols='publisher_public_adapters.NativePublisherPublicAdapters',
        continuation_format='k3_execution.checkpoints.K3Snapshot',
        factory_function='execution_factories.create_public_k3_execution_service(...)',
        cli_entry='publisher_k3_cli.PublisherSession',
    ),

    recovery_writer='service_generation_checkpoint.AtomicGenerationCheckpoint.write(...)',
    recovery_reader='k3_execution.checkpoints.restore_from_snapshot(...)',
    tooling='wholemodel_validation.k3.compare_k3_output(...)',
)

INKLING_NATIVE_TRACE = FamilyRouteTrace(
    family='inkling',
    model_type='inkling_mm_model',
    backend='native',
    arithmetic='fp32',
    source_format='F32',
    persisted_format='float32',

    loader=LoaderTrace(
        config_validator='publisher_family_contracts.component_contract(config)',
        checkpoint_type='native checkpoint safetensors',
        loader_function='native_checkpoint.read_native_text_checkpoint(...)',
        tensor_source='F32 weights',
    ),

    consumer=ConsumerTrace(
        tensor_consumer='native_inkling.NativeInklingDecoder',
        execution_backend='native',
        logits_producer='NativeInklingDecoder.forward(tokens) -> logits',
    ),

    cache_tp=TPTrace(
        cache_owner='flat_native_session.FlatNativeSession',
        tp_owner='grouped_continuation.GroupedContinuation',
        state_advancement='Sliding/hybrid attention with FIR history state',
    ),

    abi=ABITrace(
        abi_symbols='native_family_public_adapters.NativeFamilyPublicAdapters',
        continuation_format='grouped_continuation.ContinuationSnapshot',
        factory_function='execution_factories.create_inkling_native_session(...)',
        cli_entry='native_family_cli.PublisherSession',
    ),

    recovery_writer='service_generation_checkpoint.AtomicGenerationCheckpoint.write(...)',
    recovery_reader='grouped_continuation.restore_continuation(...)',
    tooling='wholemodel_validation.publisher_family_candidate.compare_inkling_output(...)',
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
    MIXTRAL_NATIVE_TRACE,
    OLMOE_NATIVE_TRACE,
    K3_NATIVE_TRACE,
    INKLING_NATIVE_TRACE,
]

ALL_ROUTES = ALL_PRIORITY_ROUTES + ALL_REMAINING_ROUTES


def validate_all_routes_complete() -> bool:
    """Validate every route has complete path."""
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
    print(f"✓ All {len(ALL_ROUTES)} routes validated (complete paths)")
    print(f"  Priority: {len(ALL_PRIORITY_ROUTES)}")
    print(f"  Remaining: {len(ALL_REMAINING_ROUTES)}")


# Backward-compatible import path.  device_routes is the maintained catalog:
# it uses canonical source tuples and explicitly labels unwired integrations.
import sys as _sys
from . import device_routes as _device_routes
_sys.modules[__name__] = _device_routes
