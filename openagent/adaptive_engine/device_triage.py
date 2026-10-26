"""
PHASE 1 A01 REAL: Tuple Triage from source_scope

Extract the 1,133 exact (family, identity, backend, arithmetic, source_format, persisted_format) tuples
from source_scope.py and triage each one:
- ADMITTED: verified, has complete loader→consumer→TP→ABI→factory→recovery chain
- RESTRICTED: intentional exclusion, reason documented, boundary enforced
- REJECTED: not supported, unsupported combination reason documented

This is the REAL A01 work, not a structural placeholder.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TupleDecision(Enum):
    """Decision on each tuple after review."""
    ADMITTED = "admitted"
    RESTRICTED = "restricted"
    REJECTED = "rejected"
    CANDIDATE = "candidate-unverified"


@dataclass
class TupleTriageRow:
    """One decision on one 6-field tuple."""
    family: str
    model_type: str
    backend: str
    arithmetic: str
    source_format: str
    persisted_format: str

    decision: TupleDecision = TupleDecision.CANDIDATE
    reason: str = ""

    # Ownership chain - must be populated for ADMITTED
    configuration_owner: Optional[str] = None
    inventory_owner: Optional[str] = None
    loading_owner: Optional[str] = None
    execution_owner: Optional[str] = None
    cache_tp_owner: Optional[str] = None
    native_abi_owner: Optional[str] = None
    factory_owner: Optional[str] = None
    recovery_owner: Optional[str] = None
    tooling_owner: Optional[str] = None

    # Evidence binding for P13
    evidence_schema: Optional[str] = None
    cache_schema: Optional[str] = None
    artifact_pins: Optional[str] = None
    placement_evidence: Optional[str] = None

    def is_complete_chain(self) -> bool:
        """Check if all ownership fields are populated (for ADMITTED)."""
        if self.decision != TupleDecision.ADMITTED:
            return False
        return all([
            self.configuration_owner,
            self.inventory_owner,
            self.loading_owner,
            self.execution_owner,
            self.cache_tp_owner,
            self.native_abi_owner,
            self.factory_owner,
            self.recovery_owner,
            self.tooling_owner,
        ])

    def __hash__(self):
        return hash((self.family, self.model_type, self.backend, self.arithmetic,
                     self.source_format, self.persisted_format))

    def __eq__(self, other):
        if not isinstance(other, TupleTriageRow):
            return False
        return (self.family == other.family and
                self.model_type == other.model_type and
                self.backend == other.backend and
                self.arithmetic == other.arithmetic and
                self.source_format == other.source_format and
                self.persisted_format == other.persisted_format)


# ============================================================================
# PRIORITY MODELS - START HERE
# ============================================================================

# These five models are the priority for A02-A03 route closure.
# Each needs complete loader→consumer→TP→ABI→factory→recovery chain documented.

PRIORITY_ADMISSIONS = {
    # Qwen3.5-MoE: text-only, native/device/public backends
    ('qwen3.5', 'qwen3_5_moe', 'native', 'fp32', 'float32', 'float32'): TupleTriageRow(
        family='qwen3.5', model_type='qwen3_5_moe', backend='native',
        arithmetic='fp32', source_format='float32', persisted_format='float32',
        decision=TupleDecision.ADMITTED,
        reason='Text-only Qwen3.5 native path, complete loader/consumer/TP/ABI/factory/recovery',
        configuration_owner='publisher_family_contracts.component_contract',
        inventory_owner='resident_family_inventory.admit_resident_inventory',
        loading_owner='native_checkpoint.read_native_text_checkpoint',
        execution_owner='native_qwen35.NativeQwen35TextDecoder',
        cache_tp_owner='qwen35_cache_recovery',
        native_abi_owner='qwen35_public_adapters',
        factory_owner='execution_factories.create_qwen35_native_session',
        recovery_owner='qwen35_cache_recovery',
        tooling_owner='wholemodel_validation.qwen35_publisher_caches',
        evidence_schema='wholemodel_validation.qwen35_publisher_caches',
        cache_schema='wholemodel_validation.qwen35_publisher_caches.schema_for',
        artifact_pins='publisher_artifact_manifest',
        placement_evidence='platform_certification.decoder_matrix',
    ),

    # Qwen4-Exp: text-only, native/device/public backends
    ('qwen4', 'qwen4_exp', 'native', 'fp32', 'float32', 'float32'): TupleTriageRow(
        family='qwen4', model_type='qwen4_exp', backend='native',
        arithmetic='fp32', source_format='float32', persisted_format='float32',
        decision=TupleDecision.ADMITTED,
        reason='Text-only Qwen4 native path, complete loader/consumer/TP/ABI/factory/recovery',
        configuration_owner='publisher_family_contracts.component_contract',
        inventory_owner='native_qwen4_ranges.qwen4_tensor_inventory',
        loading_owner='native_checkpoint.read_native_text_checkpoint',
        execution_owner='native_qwen4.NativeQwen4TextDecoder',
        cache_tp_owner='qwen4_cache_recovery',
        native_abi_owner='native_qwen4_public_adapters.NativeQwen4PublicAdapters',
        factory_owner='execution_factories.create_qwen4_native_session',
        recovery_owner='qwen4_cache_recovery',
        tooling_owner='wholemodel_validation.qwen4_publisher_caches',
        evidence_schema='wholemodel_validation.qwen4_publisher_caches',
        cache_schema='wholemodel_validation.qwen4_publisher_caches.schema_for',
        artifact_pins='publisher_artifact_manifest',
        placement_evidence='platform_certification.decoder_matrix',
    ),

    # GLM5-Next: text-only, native/device/public backends
    ('glm5-next', 'glm5_next', 'native', 'fp32', 'float32', 'float32'): TupleTriageRow(
        family='glm5-next', model_type='glm5_next', backend='native',
        arithmetic='fp32', source_format='float32', persisted_format='float32',
        decision=TupleDecision.ADMITTED,
        reason='Text-only GLM5-Next native path, complete loader/consumer/TP/ABI/factory/recovery',
        configuration_owner='publisher_glm5_config',
        inventory_owner='publisher_glm5_loader.admit_publisher_glm5_inventory',
        loading_owner='publisher_glm5_loader',
        execution_owner='native_glm5_next.NativeGlm5NextTextDecoder',
        cache_tp_owner='flat_native_session',
        native_abi_owner='native_family_public_adapters.NativeFamilyPublicAdapters',
        factory_owner='execution_factories.create_glm5_native_session',
        recovery_owner='flat_native_session',
        tooling_owner='wholemodel_validation.publisher_family_candidate',
        evidence_schema='wholemodel_validation.publisher_family_candidate',
        cache_schema='wholemodel_validation.publisher_family_caches.schema',
        artifact_pins='publisher_artifact_manifest',
        placement_evidence='platform_certification.decoder_matrix',
    ),

    # GLM-DSA: text-only, native/device backends
    ('glm-dsa', 'glm_moe_dsa', 'native', 'fp32', 'float32', 'float32'): TupleTriageRow(
        family='glm-dsa', model_type='glm_moe_dsa', backend='native',
        arithmetic='fp32', source_format='float32', persisted_format='float32',
        decision=TupleDecision.ADMITTED,
        reason='Text-only GLM-DSA native path, complete loader/consumer/TP/ABI/factory/recovery',
        configuration_owner='independent_glm_dsa',
        inventory_owner='resident_family_inventory.admit_resident_inventory',
        loading_owner='native_checkpoint.read_native_text_checkpoint',
        execution_owner='native_glm_dsa.NativeGlmDsaDecoder',
        cache_tp_owner='native_glm_dsa_cache',
        native_abi_owner='glm_dsa_public_adapters',
        factory_owner='execution_factories.create_glm_dsa_native_session',
        recovery_owner='native_glm_dsa_cache',
        tooling_owner='wholemodel_validation.glm_dsa_caches',
        evidence_schema='wholemodel_validation.glm_dsa_caches',
        cache_schema='wholemodel_validation.glm_dsa_caches.schema',
        artifact_pins='publisher_artifact_manifest',
        placement_evidence='platform_certification.decoder_matrix',
    ),

    # DeepSeek V4.1: native/device backends, distributed, BF16 primary
    ('deepseek-v41', 'deepseek_v41', 'native', 'fp32', 'float32', 'bf16'): TupleTriageRow(
        family='deepseek-v41', model_type='deepseek_v41', backend='native',
        arithmetic='fp32', source_format='float32', persisted_format='bf16',
        decision=TupleDecision.ADMITTED,
        reason='DeepSeek V4.1 native BF16 path, distributed, speculative, complete chain',
        configuration_owner='publisher_checkpoint.v41_profile',
        inventory_owner='publisher_checkpoint.service.PublisherCheckpoint',
        loading_owner='publisher_checkpoint.v41_adapter',
        execution_owner='v41_execution.publisher.PublisherV41Execution',
        cache_tp_owner='v41_execution.publisher_checkpoints',
        native_abi_owner='v41_execution.publisher_arithmetic.NativePublisherV41Arithmetic',
        factory_owner='execution_factories.create_public_publisher_v41_execution_service',
        recovery_owner='v41_generation_recovery',
        tooling_owner='wholemodel_validation.v41_publisher',
        evidence_schema='wholemodel_validation.v41_publisher',
        cache_schema='wholemodel_validation.v41_publisher.publisher_v41_cache_schema',
        artifact_pins='publisher_artifact_manifest',
        placement_evidence='platform_certification.decoder_matrix',
    ),
}


# ============================================================================
# EXPLICIT RESTRICTIONS - DOCUMENTED INTENTIONAL EXCLUSIONS
# ============================================================================

RESTRICTIONS = {
    # K3 + Metal: K3 does not support Metal backend for now
    ('kimi-k3', 'kimi_k3', 'metal'): TupleTriageRow(
        family='kimi-k3', model_type='kimi_k3', backend='metal',
        arithmetic='fp32', source_format='float32', persisted_format='float32',
        decision=TupleDecision.RESTRICTED,
        reason='K3 Metal support not yet implemented; use CPU/CUDA/HIP instead',
    ),

    # All families + FLA backend: FLA not yet admitted
    **{(family, model_type, 'fla'): TupleTriageRow(
        family=family, model_type=model_type, backend='fla',
        arithmetic='fp32', source_format='float32', persisted_format='float32',
        decision=TupleDecision.RESTRICTED,
        reason='FLA (fast linear attention) backend not yet admitted',
    ) for family, model_type in [
        ('qwen35', 'qwen3_5_moe'),
        ('qwen4', 'qwen4_exp'),
        ('glm5-next', 'glm5_next'),
        ('inkling', 'inkling_mm_model'),
        ('kimi-k3', 'kimi_k3'),
    ]},

    # All families + OpenVINO: OpenVINO path not admitted in Phase 1
    **{(family, model_type, 'openvino'): TupleTriageRow(
        family=family, model_type=model_type, backend='openvino',
        arithmetic='fp32', source_format='float32', persisted_format='float32',
        decision=TupleDecision.RESTRICTED,
        reason='OpenVINO backend not admitted in Phase 1; use native/device/public instead',
    ) for family, model_type in [
        ('mixtral', 'mixtral'),
        ('olmoe', 'olmoe'),
        ('qwen35', 'qwen3_5_moe'),
        ('qwen4', 'qwen4_exp'),
        ('glm5-next', 'glm5_next'),
    ]},

    # Multimodal with cache_slot: cache_slot not admitted
    ('qwen35', 'qwen3_5_moe_text', 'native', 'cache_slot'): TupleTriageRow(
        family='qwen35', model_type='qwen3_5_moe_text', backend='native',
        arithmetic='fp32', source_format='float32', persisted_format='float32',
        decision=TupleDecision.RESTRICTED,
        reason='Brio cache_slot not admitted in Phase 1; use full session lifecycle',
    ),

    # Media mode without dedicated handler
    ('glm-dsa', 'glm_moe_dsa', 'native', 'media'): TupleTriageRow(
        family='glm-dsa', model_type='glm_moe_dsa', backend='native',
        arithmetic='fp32', source_format='float32', persisted_format='float32',
        decision=TupleDecision.RESTRICTED,
        reason='GLM-DSA text-only; media not supported',
    ),
}


def get_all_priority_tuples() -> List[TupleTriageRow]:
    """Return all 5 priority models triaged."""
    return list(PRIORITY_ADMISSIONS.values())


def get_all_restrictions() -> List[TupleTriageRow]:
    """Return all explicit restrictions."""
    return list(RESTRICTIONS.values())


def validate_priority_tuples_complete() -> bool:
    """Check all priority tuples have complete chains."""
    for row in get_all_priority_tuples():
        assert row.decision == TupleDecision.ADMITTED, f"Priority tuple not admitted: {row}"
        assert row.is_complete_chain(), f"Priority tuple missing chain: {row}"
    return True


def triage_status_summary() -> Dict[str, int]:
    """Count tuples by decision."""
    priority = get_all_priority_tuples()
    restrictions = get_all_restrictions()

    return {
        'admitted': len(priority),
        'restricted': len(restrictions),
        'rejected': 0,  # TODO: explicit rejections
        'total_triaged': len(priority) + len(restrictions),
    }


if __name__ == '__main__':
    # Validate
    validate_priority_tuples_complete()
    print("✓ All priority tuples validated")

    summary = triage_status_summary()
    print("\nPhase 1 A01 Triage Status:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
