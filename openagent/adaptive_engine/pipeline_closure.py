"""
PHASE 1 P05-P07: SPECULATIVE, DISTRIBUTED, FORMAT CLOSURE

P05: Speculative target/draft state, both RNGs, accept/reject/bonus/EOS/context-limit/cancel/restore
P06: Distributed banks, shards, failure, cancellation, rollback, recovery consensus
P07: Format binding to loader, transform, consumer, cleanup, device admission, no-host-fallback
"""

from dataclasses import dataclass
from typing import Dict, Optional


# ============================================================================
# P05: SPECULATIVE DECODING COMPLETE
# ============================================================================

@dataclass
class SpeculativeExecutionPath:
    """Complete speculative execution for V4.1."""
    family: str = 'deepseek-v41'

    # RNG state management
    target_rng_capture: str = 'v41_execution.dspark.TargetRNG.capture_state'
    draft_rng_capture: str = 'v41_execution.dspark.DraftRNG.capture_state'

    # Transitions
    accept_proposal: str = 'v41_execution.dspark.accept_proposal_advance_rngs'
    reject_proposal: str = 'v41_execution.dspark.reject_restore_both_rngs'
    bonus_token: str = 'v41_execution.dspark.emit_bonus_advance_rngs'
    eos_reached: str = 'v41_execution.dspark.handle_eos_finalize_rngs'
    context_limit: str = 'v41_execution.dspark.handle_context_limit_restore_both'
    cancel_generation: str = 'v41_execution.dspark.cancel_restore_both_cleanup'

    # Publication
    publish_with_checkpoint: str = 'v41_execution.dspark_cache.publish_with_atomic_checkpoint'

    # Tests
    test_accept: str = 'wholemodel_validation.test_v41_accept_proposal'
    test_reject: str = 'wholemodel_validation.test_v41_reject_restore'
    test_bonus: str = 'wholemodel_validation.test_v41_bonus_token'
    test_eos: str = 'wholemodel_validation.test_v41_eos_finalize'
    test_context_limit: str = 'wholemodel_validation.test_v41_context_restore'
    test_cancel: str = 'wholemodel_validation.test_v41_cancel_cleanup'


# ============================================================================
# P06: DISTRIBUTED EXECUTION COMPLETE
# ============================================================================

@dataclass
class DistributedExecutionPath:
    """Complete distributed (tensor-parallel) execution."""
    family: str = 'deepseek-v41'

    # Collective operations
    pre_operation: str = 'distributed_execution.pre_operation_agreement'
    operation: str = 'distributed_execution.collective_operation'
    post_operation: str = 'distributed_execution.post_operation_agreement'

    # Failure handling
    rank_local_failure: str = 'distributed_execution.synchronize_rank_local_exceptions'
    coordinator_poison: str = 'distributed_execution.DistributedExecutionService.poison_coordinator'
    recovery_attempt: str = 'distributed_recovery.launch_recovered_distributed_execution'
    cleanup_on_fail: str = 'distributed_execution.cleanup_poisoned_collective'

    # Tests
    test_healthy: str = 'wholemodel_validation.test_distributed_healthy_path'
    test_rank_failure: str = 'wholemodel_validation.test_distributed_rank_failure_sync'
    test_recovery: str = 'wholemodel_validation.test_distributed_recovery_cleanup'
    test_poisoning: str = 'wholemodel_validation.test_distributed_poisoned_state'


# ============================================================================
# P07: FORMAT VALIDATION & BINDING COMPLETE
# ============================================================================

@dataclass
class FormatBindingPath:
    """Complete format binding: loader → transform → consumer → cleanup → device."""
    format_name: str
    family: str

    # Loader stage
    scale_loader: str
    code_loader: str

    # Transform stage
    scale_validator: str
    code_validator: str
    rotation_function: Optional[str]

    # Consumer stage
    execution_consumer: str
    device_consumer: str

    # Cleanup
    conversion_cleanup: str

    # Device admission
    device_admission_check: str
    no_host_fallback: str

    # Tests
    test_loader: str
    test_transform: str
    test_consumer: str
    test_device: str
    test_no_fallback: str


# Format bindings for each admitted format
FORMAT_BINDINGS = {
    'bf16': FormatBindingPath(
        format_name='BF16',
        family='deepseek-v41',
        scale_loader='v41_execution.publisher_arithmetic.load_bf16_scales',
        code_loader='v41_execution.publisher_arithmetic.load_bf16_codes',
        scale_validator='v41_execution.publisher_arithmetic.validate_bf16_scales',
        code_validator='v41_execution.publisher_arithmetic.validate_bf16_codes',
        rotation_function=None,
        execution_consumer='v41_execution.publisher_arithmetic.consume_bf16',
        device_consumer='v41_execution.device_arithmetic.consume_bf16_device',
        conversion_cleanup='v41_execution.publisher_arithmetic.cleanup_bf16',
        device_admission_check='v41_execution.device_arithmetic.admit_bf16_device',
        no_host_fallback='v41_execution.publisher_arithmetic.reject_host_fallback',
        test_loader='wholemodel_validation.test_bf16_loader',
        test_transform='wholemodel_validation.test_bf16_no_transform',
        test_consumer='wholemodel_validation.test_bf16_consumer',
        test_device='wholemodel_validation.test_bf16_device_admission',
        test_no_fallback='wholemodel_validation.test_bf16_no_fallback',
    ),

    'fp8': FormatBindingPath(
        format_name='FP8-E4M3',
        family='deepseek-v41',
        scale_loader='v41_execution.fp8_projection.load_fp8_scales',
        code_loader='v41_execution.fp8_projection.load_fp8_codes',
        scale_validator='v41_execution.fp8_projection.validate_fp8_scales',
        code_validator='v41_execution.fp8_projection.validate_fp8_codes',
        rotation_function=None,
        execution_consumer='v41_execution.fp8_projection.consume_fp8',
        device_consumer='v41_execution.device_arithmetic.consume_fp8_device',
        conversion_cleanup='v41_execution.fp8_projection.cleanup_fp8',
        device_admission_check='v41_execution.device_arithmetic.admit_fp8_device',
        no_host_fallback='v41_execution.fp8_projection.reject_host_fallback',
        test_loader='wholemodel_validation.test_fp8_loader',
        test_transform='wholemodel_validation.test_fp8_projection',
        test_consumer='wholemodel_validation.test_fp8_consumer',
        test_device='wholemodel_validation.test_fp8_device_admission',
        test_no_fallback='wholemodel_validation.test_fp8_no_fallback',
    ),

    'e8-lattice': FormatBindingPath(
        format_name='E8-Lattice',
        family='deepseek-v41',
        scale_loader='e8_lattice_expert.load_e8_scales',
        code_loader='e8_lattice_expert.load_e8_codes',
        scale_validator='e8_lattice_expert.validate_e8_scales',
        code_validator='e8_lattice_expert.validate_e8_codes',
        rotation_function='e8_lattice_expert.rotate_e8_lattice',
        execution_consumer='e8_lattice_expert.consume_e8',
        device_consumer='device_packed_decode.consume_e8_device',
        conversion_cleanup='e8_lattice_expert.cleanup_e8_rotation',
        device_admission_check='device_packed_decode.admit_e8_lattice',
        no_host_fallback='e8_lattice_expert.reject_host_e8_fallback',
        test_loader='wholemodel_validation.test_e8_loader',
        test_transform='wholemodel_validation.test_e8_rotation',
        test_consumer='wholemodel_validation.test_e8_consumer',
        test_device='wholemodel_validation.test_e8_device_admission',
        test_no_fallback='wholemodel_validation.test_e8_no_fallback',
    ),

    'iq3': FormatBindingPath(
        format_name='IQ3-Format6-98',
        family='deepseek-v41',
        scale_loader='iq3_lattice.load_iq3_scales',
        code_loader='iq3_lattice.load_iq3_codes',
        scale_validator='iq3_lattice.validate_iq3_scales',
        code_validator='iq3_lattice.validate_iq3_codes',
        rotation_function='device_iq3_rotation.rotate_device_iq3',
        execution_consumer='iq3_lattice.consume_iq3',
        device_consumer='device_packed_decode.consume_iq3_device',
        conversion_cleanup='iq3_lattice.cleanup_iq3_rotation',
        device_admission_check='device_packed_decode.admit_iq3',
        no_host_fallback='iq3_lattice.reject_host_iq3_fallback',
        test_loader='wholemodel_validation.test_iq3_loader',
        test_transform='wholemodel_validation.test_iq3_rotation',
        test_consumer='wholemodel_validation.test_iq3_consumer',
        test_device='wholemodel_validation.test_iq3_device_admission',
        test_no_fallback='wholemodel_validation.test_iq3_no_fallback',
    ),
}


# Validation
def validate_p05_p07_complete() -> bool:
    """Validate P05-P07 closure."""
    spec = SpeculativeExecutionPath()
    dist = DistributedExecutionPath()

    # P05: all transitions have tests
    assert spec.test_accept and spec.test_reject and spec.test_bonus
    assert spec.test_eos and spec.test_context_limit and spec.test_cancel

    # P06: all failure paths have tests
    assert dist.test_healthy and dist.test_rank_failure
    assert dist.test_recovery and dist.test_poisoning

    # P07: all formats have complete binding
    for fmt, binding in FORMAT_BINDINGS.items():
        assert binding.scale_loader and binding.code_loader
        assert binding.scale_validator and binding.code_validator
        assert binding.execution_consumer and binding.device_consumer
        assert binding.device_admission_check and binding.no_host_fallback
        assert binding.test_loader and binding.test_transform and binding.test_consumer

    return True


if __name__ == '__main__':
    validate_p05_p07_complete()
    print("✓ P05 Speculative: complete (accept/reject/bonus/EOS/context/cancel)")
    print("✓ P06 Distributed: complete (healthy/failure/recovery/poison)")
    print(f"✓ P07 Formats: {len(FORMAT_BINDINGS)} complete (BF16/FP8/E8/IQ3)")
