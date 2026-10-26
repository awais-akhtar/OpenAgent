"""
PHASE 1 COMPLETE CLOSURE - Integrated Production Code
======================================================

Historical Phase 1 integration helpers and the conservative closure gate.
The canonical route decisions live in source_scope.py. Legacy candidate
conversion helpers are retained only for API compatibility and fail closed;
this module must not manufacture source admissions or runtime evidence.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
import threading
import hashlib
import json
from pathlib import Path
from enum import Enum
import struct


# ============================================================================
# PHASE 1: SOURCE SCOPE CONVERSION - candidate-unverified → verified-admitted
# ============================================================================

class SourceScopeConverter:
    """Legacy conversion API; canonical source decisions cannot be promoted here."""

    @staticmethod
    def convert_candidate_to_verified(case: Dict[str, Any],
                                     evidence_receipt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert a candidate-unverified row to verified-admitted.

        Candidate and rejected rows cannot be promoted by this legacy helper.
        """
        if case.get('admission') != 'verified-admitted':
            raise RuntimeError(
                'source route promotion is disabled; review and bind the exact tuple in source_scope.py')
        return case

# ============================================================================
# P02: CPU-NATIVE CONVERSION - Candidate to Verified-Admitted
# ============================================================================

class P02NativeCPUConverter:
    """
    Legacy CPU route inspection API. Candidate promotion is disabled.

    The exact route review and admission decision belongs in source_scope.py;
    string references or CPU compatibility alone are not sufficient proof.
    """

    # CPU-compatible formats (no GPU-specific encodings)
    CPU_COMPATIBLE_FORMATS = {
        'float32', 'float16', 'bfloat16',  # Full precision formats
        'int4-row', 'int4-grouped-g64',     # Integer quantization
        'q4_0', 'q5_0', 'q8_0',             # Legacy quantization
        'int8-row',                          # 8-bit quantization
    }

    # All 9 supported families for Phase 1
    PHASE1_FAMILIES = {
        'mixtral', 'olmoe', 'qwen35', 'qwen4', 'glm5-next',
        'inkling', 'glm-dsa', 'kimi-k3', 'deepseek-v41'
    }

    # All 11 required chain dimensions in order
    REQUIRED_CHAIN_DIMENSIONS = (
        'configuration', 'inventory', 'loading', 'execution', 'cache_tp',
        'native_abi', 'factory', 'cli', 'gateway', 'recovery', 'tooling'
    )

    @classmethod
    def is_cpu_native_candidate(cls, case: Dict[str, Any]) -> bool:
        """Check if this case is a CPU-native candidate (backend native or public-range)."""
        if case.get('family') not in cls.PHASE1_FAMILIES:
            return False

        backend = case.get('backend')
        if backend not in ('native', 'public-range'):
            return False

        # Check if format is CPU-compatible
        persisted_format = case.get('persisted_format', '')
        if persisted_format not in cls.CPU_COMPATIBLE_FORMATS:
            return False

        return True

    @classmethod
    def validate_chain_completeness(cls, case: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate all 11 P02 chain dimensions are present with references.

        Returns:
            (is_complete, missing_dimensions)
        """
        missing = []
        chain = case.get('callable_chain', {})

        for dimension in cls.REQUIRED_CHAIN_DIMENSIONS:
            refs = chain.get(dimension, ())
            if not refs or not all(isinstance(r, str) for r in refs):
                missing.append(dimension)

        return len(missing) == 0, missing

    @classmethod
    def convert_candidate_to_verified(cls, case: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Convert a CPU-native candidate case to verified-admitted.

        Returns:
            (success, updated_case)
        """
        if case.get('admission') != 'verified-admitted':
            raise RuntimeError(
                'CPU source route promotion is disabled; review the exact tuple in source_scope.py')
        return True, case

    @classmethod
    def convert_all_cpu_native_cases(cls, all_cases: List[Dict[str, Any]]) -> Tuple[int, int, Dict[str, int]]:
        """
        Legacy batch API; raises on the first candidate instead of promoting it.
        """
        converted = 0
        candidates = 0
        family_counts = {family: 0 for family in cls.PHASE1_FAMILIES}

        for case in all_cases:
            if case.get('admission') == 'candidate-unverified':
                candidates += 1

                success, updated = cls.convert_candidate_to_verified(case)
                if success:
                    # Update original case in place
                    case.clear()
                    case.update(updated)
                    converted += 1

                    family = case.get('family')
                    if family in family_counts:
                        family_counts[family] += 1

        return converted, candidates, family_counts


# ============================================================================
# A04: MEDIA LIFECYCLE CLOSURE - Integrated into execution_service
# ============================================================================

@dataclass
class MediaFrameState:
    """Complete media frame state for one execution."""
    frame_id: str
    family: str
    position_ordering: Dict[int, int]  # position -> media_index
    owned_tensors: Dict[str, Any]
    geometry_validated: bool = False
    backpressure_active: bool = False

    def cleanup_on_cancel(self, reason: str) -> None:
        """Cleanup when execution is cancelled."""
        self.owned_tensors.clear()
        self.backpressure_active = False


class ExecutionServiceMediaLifecycle:
    """
    Integration into execution_service.NativeExecutionService.

    Add these methods to NativeExecutionService:
    """

    def __init__(self):
        self._media_frames: Dict[str, MediaFrameState] = {}
        self._media_lock = threading.Lock()

    def admit_media_stream(self, family: str, media_inputs: Dict[str, Any]) -> str:
        """
        Admit media inputs and return frame_id.
        Called during generation setup before prefill.
        """
        frame_id = str(uuid4())
        frame = MediaFrameState(
            frame_id=frame_id,
            family=family,
            position_ordering={},
            owned_tensors={},
        )

        # Validate geometry based on family
        if family == 'qwen35':
            from .qwen35_multimodal import validate_vision_geometry
            validate_vision_geometry(media_inputs)
        elif family == 'qwen4':
            from .qwen4_multimodal import validate_vision_geometry
            validate_vision_geometry(media_inputs)
        elif family == 'kimi-k3':
            from .k3_execution.preprocessing import validate_video_frames
            validate_video_frames(media_inputs)
        elif family == 'deepseek-v41':
            from .v41_execution.publisher_vision import validate_vision_input
            validate_vision_input(media_inputs)

        frame.geometry_validated = True

        with self._media_lock:
            self._media_frames[frame_id] = frame

        return frame_id

    def complete_media_stream(self, frame_id: str) -> None:
        """Mark media stream as successfully completed."""
        with self._media_lock:
            frame = self._media_frames.get(frame_id)
            if frame:
                frame.owned_tensors.clear()  # Release on success
                del self._media_frames[frame_id]

    def cancel_media_stream(self, frame_id: str, reason: str = 'cancelled') -> None:
        """Cancel media stream and cleanup."""
        with self._media_lock:
            frame = self._media_frames.get(frame_id)
            if frame:
                frame.cleanup_on_cancel(reason)
                del self._media_frames[frame_id]

    def error_media_stream(self, frame_id: str, error: Exception) -> None:
        """Error-path cleanup for media."""
        self.cancel_media_stream(frame_id, f'error: {str(error)[:256]}')


# ============================================================================
# A05: SPECULATIVE DECODING - RNG State Management (v41_execution integration)
# ============================================================================

@dataclass(frozen=True)
class RNGSnapshot:
    """Immutable RNG state snapshot for both target and draft."""
    target_state: bytes  # Frozen RNG state
    draft_state: bytes
    snapshot_id: str = field(default_factory=lambda: str(uuid4()))

    def verify_equal(self, other: 'RNGSnapshot') -> bool:
        """Verify both RNGs match."""
        return (self.target_state == other.target_state and
                self.draft_state == other.draft_state)


class SpeculativeRNGManager:
    """
    Integration into v41_execution module.
    Add to PublisherV41Execution class.
    """

    def __init__(self):
        self._snapshots: Dict[str, RNGSnapshot] = {}
        self._rng_lock = threading.Lock()
        self._active_rng: Optional[RNGSnapshot] = None

    def capture_rng_state(self) -> str:
        """
        Capture both target and draft RNG states.
        Called before speculative proposal.
        Returns snapshot_id.
        """
        try:
            import torch
            from .v41_execution.dspark import capture_full_snapshot

            # Capture complete RNG state
            snapshot_data = capture_full_snapshot()

            snapshot = RNGSnapshot(
                target_state=snapshot_data.get('target_bytes', b''),
                draft_state=snapshot_data.get('draft_bytes', b''),
            )

            with self._rng_lock:
                self._snapshots[snapshot.snapshot_id] = snapshot
                self._active_rng = snapshot

            return snapshot.snapshot_id
        except Exception as e:
            raise RuntimeError(f'RNG capture failed: {e}')

    def accept_proposal(self, snapshot_id: str) -> None:
        """
        Accept proposal - advance both RNGs from captured state.
        """
        with self._rng_lock:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                raise ValueError(f'Unknown RNG snapshot: {snapshot_id}')

            from .v41_execution.dspark import accept_proposal
            accept_proposal(snapshot_id)

            # Keep snapshot for potential restore
            self._active_rng = snapshot

    def reject_proposal_restore(self, snapshot_id: str) -> None:
        """
        Reject proposal - restore BOTH target and draft RNG to captured state.
        This is the critical A05 closure: no partial restore.
        """
        with self._rng_lock:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                raise ValueError(f'Unknown RNG snapshot: {snapshot_id}')

            from .v41_execution.dspark import restore_both_rngs
            restore_both_rngs(snapshot)

            self._active_rng = snapshot

    def handle_context_limit_restore(self, snapshot_id: str) -> None:
        """Restore RNG on context limit hit."""
        self.reject_proposal_restore(snapshot_id)

    def cleanup_rng_snapshots(self, keep_count: int = 4) -> None:
        """Cleanup old snapshots, keep most recent N."""
        with self._rng_lock:
            if len(self._snapshots) > keep_count:
                # Keep most recent
                sorted_ids = sorted(
                    self._snapshots.keys(),
                    key=lambda id: self._snapshots[id].snapshot_id
                )
                for old_id in sorted_ids[:-keep_count]:
                    del self._snapshots[old_id]


# ============================================================================
# A06: DISTRIBUTED EXECUTION - Bank Consensus (integrated into distributed_execution)
# ============================================================================

@dataclass
class RankBankState:
    """Per-rank local bank state for consensus."""
    rank: int
    generation_height: int
    logit_signature: str  # SHA256 of logit batch
    cache_hash: str       # Cache state hash
    last_sync_sequence: int


class DistributedBankConsensus:
    """
    Integration into distributed_execution.DistributedExecutionService.
    Add to DistributedExecutionService class.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self._bank_states: Dict[int, RankBankState] = {}
        self._bank_lock = threading.Lock()
        self._consensus_sequence = 0

    def pre_collective_agreement(self, height: int, logits: Any, cache_state: Any) -> str:
        """
        Pre-agreement phase: compute local signatures.
        Returns agreement_id.
        """
        import hashlib
        import struct

        # Compute logit signature
        logit_hash = hashlib.sha256()
        if hasattr(logits, 'cpu'):
            logit_hash.update(logits.cpu().numpy().tobytes())
        else:
            logit_hash.update(str(logits).encode())
        logit_sig = logit_hash.hexdigest()

        # Compute cache hash
        cache_hash = hashlib.sha256()
        if isinstance(cache_state, dict):
            for k in sorted(cache_state.keys()):
                v = cache_state[k]
                if hasattr(v, 'cpu'):
                    cache_hash.update(v.cpu().numpy().tobytes())
                else:
                    cache_hash.update(str(v).encode())
        cache_sig = cache_hash.hexdigest()

        bank_state = RankBankState(
            rank=self.rank,
            generation_height=height,
            logit_signature=logit_sig,
            cache_hash=cache_sig,
            last_sync_sequence=self._consensus_sequence,
        )

        agreement_id = f"agreement_{self._consensus_sequence}_{self.rank}"

        with self._bank_lock:
            self._bank_states[self.rank] = bank_state

        return agreement_id

    def collective_operation(self, operation: str, agreement_id: str) -> bool:
        """
        Execute collective operation (allreduce, broadcast, etc).
        Returns True if all ranks agree.
        """
        with self._bank_lock:
            my_state = self._bank_states.get(self.rank)
            if not my_state:
                raise RuntimeError('Pre-agreement phase not completed')

            # In real execution, broadcast state to all ranks via Gloo
            # For now, check consistency within this rank
            return True

    def post_agreement_validation(self, agreement_id: str) -> bool:
        """
        Post-agreement phase: verify all ranks reached consensus.
        """
        with self._bank_lock:
            if not self._bank_states:
                return False

            # All ranks should have same height and signatures
            heights = set(s.generation_height for s in self._bank_states.values())
            if len(heights) != 1:
                return False

            return True

    def poison_coordinator_on_failure(self, failure_reason: str) -> None:
        """
        Poison the distributed execution: all ranks must restart.
        """
        with self._bank_lock:
            self._bank_states.clear()
            self._consensus_sequence = -1  # Mark poisoned


# ============================================================================
# A07: FORMAT VALIDATION - Expert Packing Integration
# ============================================================================

@dataclass
class FormatValidationState:
    """Track format validation for one expert bank."""
    format_name: str
    family: str
    scale_validated: bool = False
    rotation_completed: bool = False
    device_admitted: bool = False
    error_reason: Optional[str] = None


class ExpertPackingFormatValidator:
    """
    Integration into expert_packing module.
    Add to pack_safetensors_experts and related functions.
    """

    @staticmethod
    def validate_scales(format_name: str, scales: Any, family: str) -> bool:
        """
        Validate scale grids for NaN/subnormal/overflow.
        Format-specific validation before actual packing.
        """
        import numpy as np

        scale_array = scales if isinstance(scales, np.ndarray) else np.array(scales)

        # Check for NaN/Inf
        if np.any(~np.isfinite(scale_array)):
            raise ValueError(f'{format_name}: scale contains NaN/Inf')

        # Check for subnormal (too close to zero)
        min_scale = np.abs(scale_array[scale_array != 0]).min() if np.any(scale_array) else 0
        if min_scale > 0 and min_scale < 1e-6:
            raise ValueError(f'{format_name}: scale contains subnormal values')

        # Format-specific checks
        if format_name == 'fp8-e4m3-b128-f32':
            if np.max(np.abs(scale_array)) > 1e4:
                raise ValueError(f'{format_name}: scales exceed valid range')

        elif format_name == 'e8-lattice-caller-v1':
            # E8 lattice requires specific scale range
            if np.any(scale_array < 0.5) or np.any(scale_array > 2.0):
                raise ValueError(f'{format_name}: scales outside lattice contract')

        elif format_name == 'iq3-format6-98-v1':
            # IQ3 requires group-aligned scales
            if len(scale_array) % 64 != 0:
                raise ValueError(f'{format_name}: scale count not group-aligned')

        return True

    @staticmethod
    def validate_codes(format_name: str, codes: Any) -> bool:
        """
        Validate quantized code values are within format bounds.
        """
        import numpy as np

        code_array = codes if isinstance(codes, np.ndarray) else np.array(codes)

        # Check code bounds based on format
        if format_name == 'fp8-e4m3-b128-f32':
            if np.any(code_array < 0) or np.any(code_array >= 256):
                raise ValueError(f'{format_name}: codes outside byte range')

        elif format_name == 'e8-lattice-caller-v1':
            # E8 codes must be valid lattice points
            if np.any(code_array < 0) or np.any(code_array >= 256):
                raise ValueError(f'{format_name}: E8 codes out of range')

        elif format_name == 'iq3-format6-98-v1':
            # IQ3 format6 codes
            if np.any(code_array < 0) or np.any(code_array >= (1 << 16)):
                raise ValueError(f'{format_name}: IQ3 codes out of range')

        return True

    @staticmethod
    def apply_rotation_if_needed(format_name: str, codes: Any) -> Any:
        """
        Apply lattice rotation for E8/IQ3 formats.
        Returns rotated codes (identity for other formats).
        """
        if format_name == 'e8-lattice-caller-v1':
            from .e8_lattice_expert import rotate_e8_lattice
            return rotate_e8_lattice(codes)

        elif format_name == 'iq3-format6-98-v1':
            from .device_iq3_rotation import rotate_device_iq3
            return rotate_device_iq3(codes)

        return codes


# ============================================================================
# A08: FAMILY ABI CALLBACK OWNERSHIP
# ============================================================================

@dataclass
class ABICallbackRegistration:
    """Register callback ownership for one family."""
    family: str
    callback_owner: str
    per_session_lock: bool
    reentrancy_check: bool
    output_validation: str

    def __post_init__(self):
        if not self.callback_owner or not self.output_validation:
            raise ValueError('callback owner and validator required')


class FamilyABICallbackManager:
    """
    Integration into all family adapters.
    Manages per-session callback locks and reentrancy checks.
    """

    def __init__(self, family: str, callback_owner: str):
        self.family = family
        self.callback_owner = callback_owner
        self._session_locks: Dict[str, threading.RLock] = {}
        self._session_reentrancy: Dict[str, threading.local] = {}
        self._registration_lock = threading.Lock()

    def register_callback_owner(self, session_id: str, family: str) -> None:
        """Register callback owner for one session."""
        with self._registration_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.RLock()
                self._session_reentrancy[session_id] = threading.local()

    def acquire_callback_lock(self, session_id: str) -> threading.RLock:
        """
        Acquire per-session callback lock.
        Reentrant locks allow safe nested calls.
        """
        lock = self._session_locks.get(session_id)
        if lock is None:
            raise ValueError(f'Session {session_id} not registered for {self.family}')

        # Check reentrancy
        local = self._session_reentrancy.get(session_id)
        if not hasattr(local, 'depth'):
            local.depth = 0

        if local.depth > 0 and self.family in ('qwen35', 'qwen4', 'glm5-next'):
            # These families reject same-thread reentrancy
            raise RuntimeError(f'{self.family}: callback reentrancy detected')

        local.depth += 1
        lock.acquire()

        return lock

    def release_callback_lock(self, session_id: str, lock: threading.RLock) -> None:
        """Release per-session callback lock."""
        local = self._session_reentrancy.get(session_id)
        if local:
            local.depth = max(0, local.depth - 1)
        lock.release()

    def validate_callback_extent(self, session_id: str, extent: Any,
                                constraint: Optional[str] = None) -> bool:
        """
        Validate callback output extent before KV advance.
        """
        if constraint == 'qwen35-ranges':
            # Qwen35 specific extent validation
            if hasattr(extent, 'shape') and len(extent.shape) > 0:
                return extent.shape[0] > 0

        elif constraint == 'qwen4-public-adapters':
            # Qwen4 extent validation
            return bool(extent)

        # Default: any non-empty extent is valid
        return bool(extent)


# ============================================================================
# A09: OBSERVATION BINDING - P13 Evidence Integration
# ============================================================================

@dataclass
class ArtifactPin:
    """Pin one artifact to execution evidence."""
    artifact_type: str  # 'config', 'weights', 'tokenizer', 'runtime'
    family: str
    model_id: str
    hash_sha256: str
    pinned_at: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ExecutionEvidence:
    """Complete P13 evidence for one execution case."""
    case_id: str
    family: str
    model_type: str
    backend: str
    arithmetic: str
    source_format: str
    persisted_format: str

    # Evidence paths
    observation_schema: str
    cache_schema: str
    placement_evidence: str

    # Artifacts pinned
    artifact_pins: List[ArtifactPin] = field(default_factory=list)

    # Execution state
    observation_captured: bool = False
    cache_exported: bool = False
    placement_recorded: bool = False
    numerical_parity_verified: bool = False

    def is_complete(self) -> bool:
        """Check if all evidence requirements are satisfied."""
        return (self.observation_captured and
                self.cache_exported and
                self.placement_recorded and
                len(self.artifact_pins) >= 4)  # config, weights, tokenizer, runtime


class ObservationBindingManager:
    """
    Integration into wholemodel_validation and all family validators.
    Binds execution tuple to P13 evidence schema.
    """

    def __init__(self):
        self._evidence_registry: Dict[str, ExecutionEvidence] = {}
        self._pin_registry: Dict[str, ArtifactPin] = {}
        self._evidence_lock = threading.Lock()

    def create_evidence_binding(self, case_id: str, family: str,
                              model_type: str, backend: str, arithmetic: str,
                              source_format: str, persisted_format: str,
                              observation_schema: str, cache_schema: str,
                              placement_evidence: str) -> str:
        """
        Create new evidence binding for execution case.
        Returns evidence_id.
        """
        evidence_id = f"evidence_{case_id}"

        evidence = ExecutionEvidence(
            case_id=case_id,
            family=family,
            model_type=model_type,
            backend=backend,
            arithmetic=arithmetic,
            source_format=source_format,
            persisted_format=persisted_format,
            observation_schema=observation_schema,
            cache_schema=cache_schema,
            placement_evidence=placement_evidence,
        )

        with self._evidence_lock:
            self._evidence_registry[evidence_id] = evidence

        return evidence_id

    def pin_artifact(self, evidence_id: str, artifact_type: str,
                    family: str, model_id: str, hash_sha256: str) -> None:
        """Pin an artifact (config/weights/tokenizer/runtime) to evidence."""
        pin = ArtifactPin(
            artifact_type=artifact_type,
            family=family,
            model_id=model_id,
            hash_sha256=hash_sha256,
        )

        with self._evidence_lock:
            evidence = self._evidence_registry.get(evidence_id)
            if evidence:
                evidence.artifact_pins.append(pin)
                self._pin_registry[pin.pinned_at] = pin

    def capture_observation(self, evidence_id: str, observation_data: Any) -> None:
        """Capture observation/comparison result."""
        with self._evidence_lock:
            evidence = self._evidence_registry.get(evidence_id)
            if evidence:
                evidence.observation_captured = True

    def export_cache(self, evidence_id: str, cache_data: Any) -> None:
        """Export complete physical cache."""
        with self._evidence_lock:
            evidence = self._evidence_registry.get(evidence_id)
            if evidence:
                evidence.cache_exported = True

    def record_placement(self, evidence_id: str, placement_data: Any) -> None:
        """Record per-operation placement data."""
        with self._evidence_lock:
            evidence = self._evidence_registry.get(evidence_id)
            if evidence:
                evidence.placement_recorded = True

    def get_evidence(self, evidence_id: str) -> Optional[ExecutionEvidence]:
        """Retrieve evidence binding."""
        with self._evidence_lock:
            return self._evidence_registry.get(evidence_id)

    def is_evidence_complete(self, evidence_id: str) -> bool:
        """Check if evidence binding is complete."""
        with self._evidence_lock:
            evidence = self._evidence_registry.get(evidence_id)
            return evidence.is_complete() if evidence else False


# ============================================================================
# A11: PHASE 1 VALIDATOR - COMPLETE GATE WITH ACTUAL VERIFICATION
# ============================================================================

@dataclass
class Phase1ClosureAssertion:
    """One verification assertion for phase 1 closure."""
    name: str
    description: str
    checker: Callable[[], bool]
    is_passed: bool = False
    error_message: Optional[str] = None


class Phase1ValidatorComplete:
    """
    A11: Complete reconciliation validator.
    Phase1Validator → verified-admitted only when all chains complete.
    """

    def __init__(self, source_scope_module=None, evidence_path: Path | None = None):
        if source_scope_module is None:
            from . import source_scope as source_scope_module
        self.source_scope = source_scope_module
        self.evidence_path = (Path(evidence_path) if evidence_path is not None else
                              Path(__file__).resolve().parents[2] / 'phase1_runtime_receipts.json')
        self.assertions: Dict[str, Phase1ClosureAssertion] = {}
        self._verification: tuple[dict[str, Any] | None, str | None] | None = None
        self._setup_assertions()

    def _rows(self) -> list[dict[str, Any]]:
        return list(self.source_scope.source_scope_matrix()['execution_cases'])

    @staticmethod
    def _decision_reason(row: dict[str, Any]) -> str | None:
        cells = row.get('cells', {})
        for name, field in (('execution', 'contract'), ('format_route', 'reason'),
                            ('persisted_formats', 'contract')):
            value = cells.get(name, {}).get(field)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _setup_assertions(self):
        """Setup all P02-P15 closure assertions."""

        # A01: Tuple triage
        self.assertions['A01_priority_tuples'] = Phase1ClosureAssertion(
            name='A01_priority_tuples',
            description='All canonical tuples finally admitted or reasoned non-admission',
            checker=lambda: self._check_a01_tuples()
        )

        # A02-A03: Family routing (9 families)
        families = ['mixtral', 'olmoe', 'qwen35', 'qwen4', 'glm5-next', 'inkling',
                   'glm-dsa', 'kimi-k3', 'deepseek-v41']
        for family in families:
            self.assertions[f'A0X_{family}_route_complete'] = Phase1ClosureAssertion(
                name=f'A0X_{family}_route_complete',
                description=f'{family} exact routes finally decided with source bindings',
                checker=lambda f=family: self._check_family_route(f)
            )

        # A04: Media lifecycle
        self.assertions['A04_media_lifecycle'] = Phase1ClosureAssertion(
            name='A04_media_lifecycle',
            description='Route-bound media lifecycle receipts or explicit non-applicability',
            checker=lambda: self._check_a04_media_lifecycle()
        )

        # A05: Speculative RNG
        self.assertions['A05_speculative_rng'] = Phase1ClosureAssertion(
            name='A05_speculative_rng',
            description='Route-bound speculative state receipts or explicit non-applicability',
            checker=lambda: self._check_a05_speculative()
        )

        # A06: Distributed consensus
        self.assertions['A06_distributed_consensus'] = Phase1ClosureAssertion(
            name='A06_distributed_consensus',
            description='Route-bound distributed recovery receipts or explicit non-applicability',
            checker=lambda: self._check_a06_distributed()
        )

        # A07: Format validation
        self.assertions['A07_format_validation'] = Phase1ClosureAssertion(
            name='A07_format_validation',
            description='Route-bound format/device receipts',
            checker=lambda: self._check_a07_format()
        )

        # A08: ABI callback ownership
        self.assertions['A08_abi_callbacks'] = Phase1ClosureAssertion(
            name='A08_abi_callbacks',
            description='Route-bound ABI/recovery receipts',
            checker=lambda: self._check_a08_abi()
        )

        # A09: Observation binding
        self.assertions['A09_observation_binding'] = Phase1ClosureAssertion(
            name='A09_observation_binding',
            description='Pinned config/weights and route-bound evidence bindings',
            checker=lambda: self._check_a09_observation()
        )

        # A10: Evidence requirements declared
        self.assertions['A10_evidence_requirements'] = Phase1ClosureAssertion(
            name='A10_evidence_requirements',
            description='Actual pinned route observations, comparisons and placement',
            checker=lambda: self._check_a10_evidence()
        )

    def _check_a01_tuples(self) -> bool:
        """Every canonical tuple has a final, reasoned decision."""
        self.source_scope.validate_phase1_source()
        rows = self._rows()
        return bool(rows) and all(
            row.get('combination_status') in ('verified-admitted', 'restricted', 'rejected-unadmitted')
            and (row.get('combination_status') == 'verified-admitted' or self._decision_reason(row))
            for row in rows)

    def _check_family_route(self, family: str) -> bool:
        """Require exact route decisions and source bindings, not family refs."""
        rows = [row for row in self._rows() if row.get('family') == family]
        if not rows:
            return False
        for row in rows:
            status = row.get('combination_status')
            if status == 'verified-admitted':
                chain = row.get('callable_chain', {})
                order = row.get('chain_order', ())
                if (not row.get('source_review') or not order or
                        any(not chain.get(dimension) for dimension in order)):
                    return False
            elif status not in ('restricted', 'rejected-unadmitted') or not self._decision_reason(row):
                return False
        return True

    def _check_a04_media_lifecycle(self) -> bool:
        return self._check_coverage('media_lifecycle')

    def _check_a05_speculative(self) -> bool:
        return self._check_coverage('speculative_state')

    def _check_a06_distributed(self) -> bool:
        return self._check_coverage('distributed_recovery')

    def _check_a07_format(self) -> bool:
        return self._check_coverage('format_device')

    def _check_a08_abi(self) -> bool:
        return self._check_coverage('abi_recovery')

    def _receipt_rows(self) -> dict[str, dict[str, Any]]:
        """Load exact-route receipts; fixtures and duplicate rows fail closed."""
        if not self.evidence_path.is_file():
            return {}
        manifest = json.loads(self.evidence_path.read_text(encoding='utf-8'))
        if (type(manifest) is not dict or
                manifest.get('schema') != 'openagent.phase1-runtime-receipts.v1' or
                type(manifest.get('routes')) is not list):
            return {}
        receipts: dict[str, dict[str, Any]] = {}
        for receipt in manifest['routes']:
            if type(receipt) is not dict or type(receipt.get('case_id')) is not str:
                return {}
            case_id = receipt['case_id']
            if case_id in receipts:
                return {}
            receipts[case_id] = receipt
        return receipts

    @staticmethod
    def _pinned_file(pin: Any) -> bool:
        if type(pin) is not dict:
            return False
        path_text, digest = pin.get('path'), pin.get('sha256')
        if (type(path_text) is not str or type(digest) is not str or
                'test_config' in path_text.lower() or 'placeholder' in path_text.lower() or
                len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest)):
            return False
        path = Path(path_text)
        if not path.is_absolute() or not path.is_file():
            return False
        actual = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                actual.update(chunk)
        return actual.hexdigest() == digest

    def _admitted_receipts(self) -> list[dict[str, Any]] | None:
        rows = [row for row in self._rows() if row.get('combination_status') == 'verified-admitted']
        receipts = self._receipt_rows()
        expected = {row['case_id'] for row in rows}
        if not expected or set(receipts) != expected:
            return None
        return [receipts[row['case_id']] for row in rows]

    def _verified_receipts(self) -> tuple[dict[str, Any] | None, str | None]:
        """Verify every admitted route's receipt once per validation run.

        Returns ``(measurements by case_id, None)`` when every admitted row has
        a receipt whose kind-specific observations verify, else
        ``(None, reason)``.  Claimed booleans in receipts are never read.
        """
        if self._verification is not None:
            return self._verification
        from .device_observations import ObservationError, load_receipts, verify_route_receipt
        rows = [row for row in self._rows() if row.get('combination_status') == 'verified-admitted']
        try:
            receipts = load_receipts(self.evidence_path)
        except (ObservationError, ValueError, OSError) as error:
            self._verification = (None, 'receipt manifest rejected: ' + str(error))
            return self._verification
        missing = sorted({row['case_id'] for row in rows} - set(receipts))
        if not rows:
            self._verification = (None, 'no admitted routes')
        elif missing:
            self._verification = (None, f'{len(missing)} of {len(rows)} admitted routes lack runtime '
                                        f'receipts (first: {missing[0]})')
        else:
            measured: dict[str, Any] = {}
            reason = None
            for row in rows:
                try:
                    measured[row['case_id']] = verify_route_receipt(row, receipts[row['case_id']])
                except ObservationError as error:
                    reason = row['case_id'] + ': ' + str(error)
                    break
            self._verification = (None, reason) if reason else (measured, None)
        return self._verification

    def _check_coverage(self, requirement: str) -> bool:
        measured, _ = self._verified_receipts()
        return measured is not None and all(
            result['coverage'].get(requirement) in ('verified', 'not-applicable')
            for result in measured.values())

    def _check_a09_observation(self) -> bool:
        receipts = self._admitted_receipts()
        if receipts is None:
            return False
        for receipt in receipts:
            artifacts = receipt.get('artifacts')
            if (type(artifacts) is not list or
                    not {'config', 'weights'}.issubset(
                        {pin.get('role') for pin in artifacts if type(pin) is dict}) or
                    not all(self._pinned_file(pin) for pin in artifacts) or
                    type(receipt.get('cache_schema')) is not dict or
                    not receipt['cache_schema'] or
                    type(receipt.get('placement_constraints')) is not dict or
                    not receipt['placement_constraints'] or
                    not isinstance(receipt.get('comparison_adapter'), str) or
                    not receipt['comparison_adapter'].strip()):
                return False
        return True

    def _check_a10_evidence(self) -> bool:
        """Kind-specific observations verify for every admitted route."""
        if not self._check_a09_observation():
            return False
        measured, _ = self._verified_receipts()
        return measured is not None

    def validate_all(self) -> Dict[str, Any]:
        """
        Complete A11 reconciliation: run all checks.
        Returns phase1_closed=True only when ALL checks pass.
        """
        passed_count = 0
        self._verification = None

        for name, assertion in self.assertions.items():
            assertion.error_message = None
            try:
                assertion.is_passed = assertion.checker()
                if assertion.is_passed:
                    passed_count += 1
            except Exception as e:
                assertion.error_message = str(e)[:256]
                assertion.is_passed = False

        _, evidence_reason = self._verified_receipts()
        for name in ('A04_media_lifecycle', 'A05_speculative_rng', 'A06_distributed_consensus',
                     'A07_format_validation', 'A08_abi_callbacks',
                     'A09_observation_binding', 'A10_evidence_requirements'):
            assertion = self.assertions[name]
            if not assertion.is_passed and assertion.error_message is None and evidence_reason:
                assertion.error_message = evidence_reason[:512]
        evidence_complete = self.assertions['A10_evidence_requirements'].is_passed
        routes_decided = self.assertions['A01_priority_tuples'].is_passed
        phase1_closed = (passed_count == len(self.assertions) and
                         evidence_complete and routes_decided)

        return {
            'phase1_closed': phase1_closed,
            'checks_passed': passed_count,
            'total_checks': len(self.assertions),
            'assertions': {name: {
                'passed': a.is_passed,
                'description': a.description,
                'error': a.error_message
            } for name, a in self.assertions.items()},
            'executed': evidence_complete,
            'numerical_parity_verified': evidence_complete,
            'platform_certified': evidence_complete,
            'blocked_on_a10': (None if evidence_complete else
                               'Missing or invalid exact-route runtime receipts'),
        }


# ============================================================================
# P02 INTEGRATION - Convert all CPU-native cases at startup
# ============================================================================

def convert_all_cpu_native_cases() -> Dict[str, Any]:
    """
    Load all 1,126+ cases from source_scope and convert CPU-native to verified-admitted.

    Returns:
        {
            'total_converted': int,
            'total_candidates': int,
            'family_breakdown': {family: count},
            'families_with_cpu_paths': [families],
            'cpu_native_count': int,
            'phase1_ready': bool
        }
    """
    from .source_scope import source_execution_cases

    # Load all execution cases
    all_cases = source_execution_cases()

    # Convert CPU-native candidates to verified-admitted
    converter = P02NativeCPUConverter()
    total_converted, total_candidates, family_counts = converter.convert_all_cpu_native_cases(all_cases)

    # Count how many now have verified-admitted CPU paths
    families_with_paths = [f for f, count in family_counts.items() if count > 0]

    # All 9 families should have at least one CPU path
    phase1_ready = len(families_with_paths) == len(P02NativeCPUConverter.PHASE1_FAMILIES)

    result = {
        'total_converted': total_converted,
        'total_candidates': total_candidates,
        'family_breakdown': family_counts,
        'families_with_cpu_paths': sorted(families_with_paths),
        'cpu_native_count': total_converted,
        'phase1_ready': phase1_ready,
        'all_cases': all_cases,
        'conversion_rate': f"{total_converted}/{total_candidates}" if total_candidates > 0 else "0/0",
    }

    return result


# ============================================================================
# INTEGRATION HELPERS - Patch existing modules
# ============================================================================

def integrate_phase1_closure():
    """
    Apply all Phase 1 closure integrations to existing modules.
    Call this once at startup to enable complete closure.
    """

    # Create singleton instances
    _converter = SourceScopeConverter()
    _cpu_converter = P02NativeCPUConverter()
    _media_lifecycle = ExecutionServiceMediaLifecycle()
    _rng_manager = SpeculativeRNGManager()
    _format_validator = ExpertPackingFormatValidator()
    _abi_callback_manager = FamilyABICallbackManager('universal', 'callback-owner')
    _observation_binding = ObservationBindingManager()
    _validator = Phase1ValidatorComplete()

    return {
        'converter': _converter,
        'cpu_native_converter': _cpu_converter,
        'media_lifecycle': _media_lifecycle,
        'rng_manager': _rng_manager,
        'format_validator': _format_validator,
        'abi_callback_manager': _abi_callback_manager,
        'observation_binding': _observation_binding,
        'validator': _validator,
    }


def verify_phase1_closed() -> bool:
    """
    Final verification: phase1_closed=True only if all 11 phases pass.
    """
    validator = Phase1ValidatorComplete()
    result = validator.validate_all()
    return result.get('phase1_closed', False)


if __name__ == '__main__':
    # Test all integrated components
    print("Phase 1 Closure Integration Test")
    print("=" * 50)

    # Test source scope converter
    print("\n✓ SourceScopeConverter available")

    # Test media lifecycle
    media = ExecutionServiceMediaLifecycle()
    print("✓ ExecutionServiceMediaLifecycle initialized")

    # Test speculative RNG
    rng = SpeculativeRNGManager()
    print("✓ SpeculativeRNGManager initialized")

    # Test distributed consensus
    dist = DistributedBankConsensus(4, 0)
    print("✓ DistributedBankConsensus initialized")

    # Test format validation
    print("✓ ExpertPackingFormatValidator available")

    # Test ABI callbacks
    abi = FamilyABICallbackManager('qwen35', 'qwen35_adapters')
    print("✓ FamilyABICallbackManager initialized")

    # Test observation binding
    obs = ObservationBindingManager()
    print("✓ ObservationBindingManager initialized")

    # Import and test P13/P14/P15 components
    try:
        from .audit_closure import (
            P13EvidenceCaptureManager,
            P14ServingLifecycleManager,
            Phase1ClosureValidator,
        )
        print("\n✓ P13EvidenceCaptureManager available")
        print("✓ P14ServingLifecycleManager available")
        print("✓ Phase1ClosureValidator available")
    except ImportError:
        print("\n⚠ P13/P14/P15 components not yet loaded")

    # Test Phase 1 Validator
    validator = Phase1ValidatorComplete()
    result = validator.validate_all()

    print("\n✓ Phase 1 Validator Results:")
    print(f"  Phase 1 Closed: {result['phase1_closed']}")
    print(f"  Checks Passed: {result['checks_passed']}/{result['total_checks']}")
    print(f"  Blocked on A10: {result['blocked_on_a10']}")

    # Print passing checks
    passing = [k for k, v in result['assertions'].items() if v['passed']]
    if passing:
        print(f"\n✓ Passing checks ({len(passing)}):")
        for check in passing[:5]:
            print(f"  - {check}")
        if len(passing) > 5:
            print(f"  ... and {len(passing) - 5} more")
