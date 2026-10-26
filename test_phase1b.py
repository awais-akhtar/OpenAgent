#!/usr/bin/env python
"""Phase 1B integration: a real compiled OLMoE checkpoint opened through AdaptiveExecutionEngine.

Runs the real test suites (temporary directories only, no files outside them,
no leaked threads).  Exit status is non-zero if any test fails.

    python test_phase1b.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SUITES = [
    "tests.test_universal_production_path",
]


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromNames(SUITES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
