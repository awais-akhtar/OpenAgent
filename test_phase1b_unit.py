#!/usr/bin/env python
"""Phase 1B component checks: real expert runtime counters, lifecycle and shutdown.

Runs the real test suites (temporary directories only, no files outside them,
no leaked threads).  Exit status is non-zero if any test fails.

    python test_phase1b_unit.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SUITES = [
    "tests.test_phase1b_lifecycle",
    "tests.test_sparse_expert_runtime",
]


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromNames(SUITES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
