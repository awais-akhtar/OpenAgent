#!/usr/bin/env python3
"""
Phase 1 legacy route audit and CPU product scope guard
======================================================

Audit historical family routes while refusing unproven CPU product closure.

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


def report_cpu_product_scope():
    """Keep the legacy full-family gate from certifying the CPU product goal.

    The current matrix is pinned to flagship checkpoints. The CPU baseline
    instead requires pinned production-caller observations and a real run
    under an enforced 8 GiB Windows job memory limit. This qualifies the
    measured process budget, not every physical 16 GiB host or CPU speed.
    """
    print("\nCPU product scope: bounded local baseline")
    print("  Target: useful local CPU inference with a 16 GiB RAM baseline, "
          "32 GiB higher tier, and ordinary local storage; no GPU prerequisite.")
    print("  Local smoke: the small OpenVINO model exercised generation, streaming, "
          "cancellation/reload, chat, coding, research, agents and read-only CodeQ on one 32 GiB host.")
    observation = project_root / "docs/audits/phase1/cpu-product-observation-2026-11-01.json"
    from openagent.cpu_product_gate import validate_cpu_product_observation
    model_dir = project_root / "models/openvino/qwen2.5-1.5b-instruct-int4-ov"
    measured = validate_cpu_product_observation(project_root, observation, model_dir)
    if measured["valid"]:
        print(f"  Local CPU observation gate: PASS ({measured['caller_count']} production callers, "
              f"{measured['sampled_process_tree_peak_rss_bytes']} bytes sampled peak).")
        print("  This is not a publisher-family route receipt or 16 GiB host qualification.")
    else:
        print("  Local CPU observation gate: FAIL (" + "; ".join(measured["issues"]) + ")")
    budget_observation = project_root / "docs/audits/phase1/cpu-product-budget-observation.json"
    budget = validate_cpu_product_observation(
        project_root, budget_observation, model_dir,
        require_job_max_bytes=8 * 1024**3)
    if budget["valid"]:
        print(f"  8 GiB enforced CPU job gate: PASS ({budget['sampled_app_process_tree_peak_rss_bytes']} "
              "bytes sampled app peak).")
    else:
        print("  8 GiB enforced CPU job gate: OPEN (" + "; ".join(budget["issues"]) + ")")
    print("  Autonomous CodeQ edits remain explicitly restricted on the small CPU baseline.")
    print("  The full-family publisher receipt matrix remains a separate research "
          "audit and does not certify this CPU product scope.")
    return measured["valid"] and budget["valid"]


def main():
    # Windows terminals may default to cp1252 while check descriptions contain
    # arrows and other Unicode. Reporting must not crash before the gate result.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    print("\n" + "="*70)
    print("PHASE 1 ROUTE AUDIT AND CPU PRODUCT SCOPE")
    print("="*70 + "\n")

    print("Canonical source matrix (no in-memory conversion):")
    report_canonical_matrix()

    # Validate all packages
    result = validate_all_packages()
    cpu_scope_ok = report_cpu_product_scope()

    # Final status
    print("\n" + "="*70)
    if cpu_scope_ok:
        print("PHASE 1 CPU BASELINE VERIFIED")
        print("  Bounded CPU product observations passed, including an enforced 8 GiB job.")
        print("  This qualifies the measured process budget; latency on other CPUs is not claimed.")
        print(f"  Separate publisher research audit: {result['checks_passed']}/{result['total_checks']} checks passed")
        print("="*70)
        return 0
    remaining = result['total_checks'] - result['checks_passed']
    print("PHASE 1 CPU BASELINE OPEN")
    print("  An enforced 8 GiB CPU job observation is required before CPU closure.")
    print(f"  Separate publisher research audit: {result['checks_passed']}/{result['total_checks']} checks passed")
    print(f"  Publisher checks still failing: {remaining} (see above)")
    print("="*70)
    return 1


if __name__ == '__main__':
    sys.exit(main())
