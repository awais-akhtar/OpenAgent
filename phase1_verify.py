#!/usr/bin/env python3
"""
Phase 1 Complete Closure - Final Verification
==============================================

Verify that Phase 1 closure is complete and ready for production.

This script:
1. Loads source_scope data
2. Runs the canonical source validator (no rows are converted in memory)
3. Validates all P02-P15 packages
4. Reports phase1_closed status
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ``phase1_closure_integrated.py`` was a duplicate legacy implementation and
# has been removed. Keep the command-line verifier on the canonical gate that
# remains in the package.
from openagent.adaptive_engine.integrated_closure import Phase1ValidatorComplete  # noqa: E402


def report_canonical_matrix():
    """Report the canonical source validator and matrix exactly as computed.

    No row is converted or reclassified here: candidate rows stay candidates
    until a reviewed source change admits or rejects them.
    """
    from collections import Counter
    from openagent.adaptive_engine.source_scope import source_scope_matrix, validate_phase1_source
    from openagent.adaptive_engine.phase1_p02_route_bindings import (
        source_reviewed_route_bindings, validate_source_reviewed_route_bindings,
        validate_callable_owner_sources,
    )
    from openagent.adaptive_engine.public_abi_audit import audit_public_abi

    try:
        validator_ok = validate_phase1_source()
        print("  Canonical source validator: PASS")
    except Exception as error:
        validator_ok = False
        print(f"  Canonical source validator: FAIL ({error})")
    try:
        validate_source_reviewed_route_bindings()
        bindings = source_reviewed_route_bindings()
        print(f"  P02 exact source-route bindings: PASS ({len(bindings)} unique admitted tuples)")
        route_bindings_ok = True
    except Exception as error:
        print(f"  P02 exact source-route bindings: FAIL ({error})")
        route_bindings_ok = False
    try:
        owner_sources = validate_callable_owner_sources(project_root)
        print(f"  P02 callable owner source references: PASS ({owner_sources} unique owners)")
        owner_sources_ok = True
    except Exception as error:
        print(f"  P02 callable owner source references: FAIL ({error})")
        owner_sources_ok = False
    try:
        abi = audit_public_abi(project_root).require_ok()
        print("  Public ABI source declarations/definitions/ctypes names: PASS "
              f"({len(abi.exported_symbols)} exports, {len(abi.ctypes_symbols)} ctypes names)")
        abi_ok = True
    except Exception as error:
        print(f"  Public ABI source declarations/definitions/ctypes names: FAIL ({error})")
        abi_ok = False
    print("  ABI note: source declarations and definitions include calling-convention "
          "token checks; this does not certify platform ABI compatibility, native "
          "compilation, or execution.")
    matrix = source_scope_matrix()
    counts = Counter(case["combination_status"] for case in matrix["execution_cases"])
    print(f"  Execution cases: {len(matrix['execution_cases'])}")
    for status in ("verified-admitted", "candidate-unverified", "restricted", "rejected-unadmitted"):
        print(f"    {status}: {counts.get(status, 0)}")
    return validator_ok and route_bindings_ok and owner_sources_ok and abi_ok, counts


def validate_all_packages():
    """Run complete Phase1 validation."""
    print("\nValidating Phase 1 packages...")

    validator = Phase1ValidatorComplete()
    result = validator.validate_all()

    print(f"  Phase 1 Closed: {result['phase1_closed']}")
    print(f"  Checks Passed: {result['checks_passed']}/{result['total_checks']}")

    # Show passing checks
    passing = [k for k, v in result['assertions'].items() if v['passed']]
    if passing:
        print(f"\n  Passing ({len(passing)}):")
        for check in passing[:15]:
            desc = result['assertions'][check]['description']
            print(f"    PASS {check}: {desc}")

    # Show failing checks
    failing = [k for k, v in result['assertions'].items() if not v['passed']]
    if failing:
        print(f"\n  Failing ({len(failing)}):")
        for check in failing:
            desc = result['assertions'][check]['description']
            print(f"    FAIL {check}: {desc}")
            error = result['assertions'][check].get('error')
            if error:
                print(f"         reason: {error}")

    return result


def main():
    # Windows terminals may default to cp1252 while check descriptions contain
    # arrows and other Unicode. Reporting must not crash before the gate result.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE CLOSURE - VERIFICATION")
    print("="*70 + "\n")

    print("Canonical source matrix (no in-memory conversion):")
    validator_ok, counts = report_canonical_matrix()

    # Validate all packages
    result = validate_all_packages()

    # Final status
    print("\n" + "="*70)
    if result['phase1_closed'] and validator_ok and not counts.get("candidate-unverified"):
        print("PHASE 1 COMPLETE CLOSURE VERIFIED")
        print(f"  All {result['total_checks']} checks passed")
        print("  All P02-P15 packages CLOSED")
        print("="*70)
        return 0
    else:
        remaining = result['total_checks'] - result['checks_passed']
        print("PHASE 1 PARTIAL CLOSURE")
        print(f"  {result['checks_passed']}/{result['total_checks']} checks passed")
        print(f"  {remaining} checks still failing (see above)")
        print(f"  Blocked on: {result.get('blocked_on_a10', 'Unknown')}")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
