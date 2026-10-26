#!/usr/bin/env python
"""Phase 1 P0/P1 proof: source catalog, universal production path, transactions, receipts, cleanup and Phase 1B.

Runs the real test suites (temporary directories only, no files outside them,
no leaked threads).  Exit status is non-zero if any test fails.

    python test_p0_p1_fixes.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SUITES = [
    "tests.test_source_scope_inventory",
    "tests.test_device_e8_rotation",
    "tests.test_distributed_checkpoint_errors",
    "tests.test_universal_engine",
    "tests.test_universal_adapters",
    "tests.test_universal_production_path",
    "tests.test_phase1b_lifecycle",
    "tests.test_sparse_expert_runtime",
    "tests.test_olmoe_manifest_selection",
    "tests.test_phase1_closure_gate",
    "tests.test_phase1_capture",
    "tests.test_p02_candidate_triage",
]


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromNames(SUITES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
