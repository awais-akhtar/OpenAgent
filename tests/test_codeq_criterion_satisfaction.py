"""Phase 4: a goal criterion is satisfied only by a current passing check for a linked task."""

from __future__ import annotations

import unittest

from openagent.codebase.evidence import criterion_satisfied

FP = "f" * 64
CRITERION = {"id": "c1", "linked_task_ids": ["T1"]}


def _check(**overrides) -> dict:
    base = {"evidence_id": "evidence-1", "kind": "test", "status": "passed", "exit_code": 0, "source_fingerprint": FP, "task_ids": ["T1"]}
    base.update(overrides)
    return base


class CriterionSatisfactionTests(unittest.TestCase):
    def test_passing_check_for_linked_task_satisfies(self) -> None:
        self.assertTrue(criterion_satisfied(CRITERION, [_check()], FP))

    def test_no_linked_task_never_satisfies(self) -> None:
        self.assertFalse(criterion_satisfied({"id": "c2", "linked_task_ids": []}, [_check()], FP))

    def test_check_for_other_task_does_not_satisfy(self) -> None:
        self.assertFalse(criterion_satisfied(CRITERION, [_check(task_ids=["T9"])], FP))

    def test_stale_check_does_not_satisfy(self) -> None:
        self.assertFalse(criterion_satisfied(CRITERION, [_check(source_fingerprint="old")], FP))

    def test_failed_or_nonzero_check_does_not_satisfy(self) -> None:
        self.assertFalse(criterion_satisfied(CRITERION, [_check(status="failed", exit_code=1)], FP))
        self.assertFalse(criterion_satisfied(CRITERION, [_check(status="passed", exit_code=2)], FP))

    def test_artifact_scan_does_not_satisfy(self) -> None:
        self.assertFalse(criterion_satisfied(CRITERION, [_check(kind="artifacts")], FP))

    def test_unpersisted_or_unidentified_check_cannot_satisfy(self) -> None:
        self.assertFalse(criterion_satisfied(CRITERION, [_check(evidence_id="")], FP))

    def test_returns_stable_evidence_identifiers(self) -> None:
        from openagent.codebase.evidence import criterion_evidence_ids

        self.assertEqual(criterion_evidence_ids(CRITERION, [_check(), _check(evidence_id="evidence-2")], FP), ["evidence-1", "evidence-2"])


if __name__ == "__main__":
    unittest.main()
