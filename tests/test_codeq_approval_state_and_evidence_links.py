"""Phase 4 gaps: approval-needed state and a queryable evidence-link chain."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from openagent.codebase.evidence import completion_gate
from openagent.codebase.service import CodebaseService
from openagent.storage import OpenAgentStore
from tests.test_codeq_failure_gates_e2e import ScriptedGateway, _writes_feature

FP = "b" * 64


def _passing_validation() -> dict:
    return {"passed": True, "source_fingerprint": FP, "checks": [
        {"evidence_id": "test-evidence-1", "kind": "test", "status": "passed", "exit_code": 0, "source_fingerprint": FP, "task_ids": []},
    ]}


class ApprovalStateTests(unittest.TestCase):
    def test_pending_approval_is_awaiting_not_blocked_or_completed(self) -> None:
        gate = completion_gate(_passing_validation(), FP, pending_approvals=1)
        self.assertEqual("awaiting_approval", gate["status"])
        self.assertFalse(gate["passed"])

    def test_no_pending_approval_still_completes(self) -> None:
        self.assertEqual("completed", completion_gate(_passing_validation(), FP)["status"])

    def test_awaiting_run_is_persisted_open_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), ScriptedGateway(_writes_feature))  # type: ignore[arg-type]
            project = service.create_project(name="Feature", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add a feature", max_iterations=12, completion={})
            run = service.wait_for_run(run["id"], timeout=60)
            service._block_run(run["id"], {"status": "awaiting_approval", "reason": "1 approval(s) pending",
                                           "next_action": "resolve the pending approval and resume", "evidence": [], "stale": False})
            awaiting = service.get_run(run["id"])
            self.assertEqual("awaiting_approval", awaiting["status"])


class EvidenceChainTests(unittest.TestCase):
    def test_completed_run_links_test_snapshot_and_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("existing project\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), ScriptedGateway(_writes_feature))  # type: ignore[arg-type]
            project = service.create_project(name="Feature", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Add a feature", max_iterations=12,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}},
            )
            run = service.wait_for_run(run["id"], timeout=60)
            self.assertEqual("completed", run["status"], run.get("error"))
            kinds = {link["source_kind"] for link in service.evidence_links(run["id"])}
            self.assertIn("test", kinds)
            self.assertIn("snapshot", kinds)
            self.assertIn("review", kinds)
            review_event_ids = {event["id"] for event in service.events(run["id"]) if event["type"] == "review.completed"}
            review_links = [link for link in service.evidence_links(run["id"]) if link["source_kind"] == "review"]
            self.assertTrue(review_links)
            self.assertTrue(all(link["source_id"] in review_event_ids for link in review_links))


if __name__ == "__main__":
    unittest.main()
