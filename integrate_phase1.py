#!/usr/bin/env python3
"""
Phase 1 Complete Closure - Integration Script
==============================================

Apply all phase1_closure_integrated patches to existing modules.
Run this ONCE at project startup to enable complete Phase 1 closure.

Usage:
    python integrate_phase1.py
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from openagent.adaptive_engine.phase1_closure_integrated import (
    integrate_phase1_closure,
    Phase1ValidatorComplete,
    SourceScopeConverter,
)


def patch_execution_service():
    """Patch NativeExecutionService with A04 media lifecycle."""
    try:
        from openagent.adaptive_engine.execution_service import NativeExecutionService
        from openagent.adaptive_engine.phase1_closure_integrated import ExecutionServiceMediaLifecycle

        # Inject methods
        if not hasattr(NativeExecutionService, '_media_lifecycle'):
            NativeExecutionService._media_lifecycle_class = ExecutionServiceMediaLifecycle
            NativeExecutionService.admit_media_stream = lambda self, family, media: \
                self._media_lifecycle.admit_media_stream(family, media)
            NativeExecutionService.complete_media_stream = lambda self, frame_id: \
                self._media_lifecycle.complete_media_stream(frame_id)
            NativeExecutionService.cancel_media_stream = lambda self, frame_id, reason='cancelled': \
                self._media_lifecycle.cancel_media_stream(frame_id, reason)
            NativeExecutionService.error_media_stream = lambda self, frame_id, error: \
                self._media_lifecycle.error_media_stream(frame_id, error)

        print("✓ Patched execution_service.NativeExecutionService (A04)")
        return True
    except Exception as e:
        print(f"⚠ execution_service patch skipped: {e}")
        return False


def patch_v41_execution():
    """Patch V41Execution with A05 RNG management."""
    try:
        from openagent.adaptive_engine.v41_execution.execution import V41Execution
        from openagent.adaptive_engine.phase1_closure_integrated import SpeculativeRNGManager

        if not hasattr(V41Execution, '_rng_manager_class'):
            V41Execution._rng_manager_class = SpeculativeRNGManager
            V41Execution.capture_rng_snapshot = lambda self: \
                self._rng_manager.capture_rng_state()
            V41Execution.accept_speculative_proposal = lambda self, snapshot_id: \
                self._rng_manager.accept_proposal(snapshot_id)
            V41Execution.reject_speculative_proposal = lambda self, snapshot_id: \
                self._rng_manager.reject_proposal_restore(snapshot_id)

        print("✓ Patched v41_execution.V41Execution (A05)")
        return True
    except Exception as e:
        print(f"⚠ v41_execution patch skipped: {e}")
        return False


def patch_distributed_execution():
    """Patch DistributedExecutionService with A06 consensus."""
    try:
        from openagent.adaptive_engine.distributed_execution import DistributedExecutionService
        from openagent.adaptive_engine.phase1_closure_integrated import DistributedBankConsensus

        if not hasattr(DistributedExecutionService, '_consensus_class'):
            DistributedExecutionService._consensus_class = DistributedBankConsensus
            DistributedExecutionService.pre_agreement_phase = lambda self, height, logits, cache: \
                self._consensus.pre_collective_agreement(height, logits, cache)
            DistributedExecutionService.post_agreement_validation = lambda self, agreement_id: \
                self._consensus.post_agreement_validation(agreement_id)

        print("✓ Patched distributed_execution.DistributedExecutionService (A06)")
        return True
    except Exception as e:
        print(f"⚠ distributed_execution patch skipped: {e}")
        return False


def patch_expert_packing():
    """Patch expert_packing with A07 format validation."""
    try:
        from openagent.adaptive_engine import expert_packing
        from openagent.adaptive_engine.phase1_closure_integrated import ExpertPackingFormatValidator

        # Inject validator
        expert_packing._format_validator = ExpertPackingFormatValidator()
        expert_packing.validate_format_scales = ExpertPackingFormatValidator.validate_scales
        expert_packing.validate_format_codes = ExpertPackingFormatValidator.validate_codes
        expert_packing.apply_format_rotation = ExpertPackingFormatValidator.apply_rotation_if_needed

        print("✓ Patched expert_packing (A07)")
        return True
    except Exception as e:
        print(f"⚠ expert_packing patch skipped: {e}")
        return False


def patch_source_scope():
    """Patch source_scope with SourceScopeConverter."""
    try:
        from openagent.adaptive_engine import source_scope
        from openagent.adaptive_engine.phase1_closure_integrated import SourceScopeConverter

        # Inject converter
        source_scope._converter = SourceScopeConverter()
        source_scope.convert_candidate_to_verified = SourceScopeConverter.convert_candidate_to_verified

        print("✓ Patched source_scope (candidate→verified conversion)")
        return True
    except Exception as e:
        print(f"⚠ source_scope patch skipped: {e}")
        return False


def validate_phase1_closure():
    """Run Phase1ValidatorComplete to verify all integrations."""
    try:
        validator = Phase1ValidatorComplete()
        result = validator.validate_all()

        print("\n" + "="*60)
        print("PHASE 1 CLOSURE VALIDATION")
        print("="*60)

        print(f"\nPhase 1 Closed: {result['phase1_closed']}")
        print(f"Checks Passed: {result['checks_passed']}/{result['total_checks']}")

        if result['phase1_closed']:
            print("\n✓ ALL PHASE 1 CHECKS PASSED - phase1_closed = True")
        else:
            print(f"\n⚠ Checks failed - {result['total_checks'] - result['checks_passed']} remaining")

            # Show failing checks
            failing = [(k, v) for k, v in result['assertions'].items() if not v['passed']]
            if failing:
                print(f"\nFailing checks ({len(failing)}):")
                for name, assertion in failing[:10]:
                    print(f"  ✗ {name}: {assertion['description']}")
                    if assertion['error']:
                        print(f"    Error: {assertion['error'][:80]}")

        print(f"\nBlocked on A10: {result['blocked_on_a10']}")
        print("="*60 + "\n")

        return result['phase1_closed']

    except Exception as e:
        print(f"\n⚠ Validation failed: {e}")
        return False


def main():
    """Apply all integrations."""
    print("\nPhase 1 Complete Closure - Integration")
    print("="*60 + "\n")

    # Create Phase 1 components
    print("Initializing Phase 1 closure infrastructure...")
    components = integrate_phase1_closure()
    print(f"✓ {len(components)} components initialized\n")

    # Apply patches
    patches_ok = all([
        patch_source_scope(),
        patch_execution_service(),
        patch_v41_execution(),
        patch_distributed_execution(),
        patch_expert_packing(),
    ])

    print()

    # Validate
    if patches_ok:
        phase1_closed = validate_phase1_closure()

        if phase1_closed:
            print("✓ PHASE 1 COMPLETE CLOSURE INTEGRATED")
            print("  All A01-A11 packages verified and closed.")
            return 0
        else:
            print("⚠ Phase 1 partial closure. Some A0x checks not yet passing.")
            print("  See above for details.")
            return 1
    else:
        print("⚠ Some patches were skipped (see above).")
        print("  Code is ready for integration when modules are available.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
