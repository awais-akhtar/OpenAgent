"""Phase 4 exit: an unsatisfied user-stated criterion blocks completion; a satisfied one does not."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from openagent.codebase.service import CodebaseService
from openagent.storage import OpenAgentStore
from tests.test_codeq_failure_gates_e2e import ScriptedGateway, _writes_notes


class UserCriterionGateTests(unittest.TestCase):
    def test_user_criterion_gate_is_truthful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), ScriptedGateway(_writes_notes))  # type: ignore[arg-type]
            project = service.create_project(name="Notes", source_path=str(source))
            goal = service.create_goal(codebase_project_id=project["id"], goal="Write notes",
                                       acceptance_criteria=["notes.txt contains the greeting"])
            run = service.create_run(codebase_project_id=project["id"], prompt="Write notes", max_iterations=8,
                                     completion={}, goal_id=goal["id"])
            run = service.wait_for_run(run["id"], timeout=60)
            fingerprint = service._source_fingerprint_for(run)
            with service.store.session() as connection:
                criterion = connection.execute("SELECT id FROM codebase_goal_criteria WHERE goal_id = ?", (goal["id"],)).fetchone()["id"]
                task_id = connection.execute("SELECT id FROM codebase_tasks WHERE run_id = ? ORDER BY created_at LIMIT 1", (run["id"],)).fetchone()["id"]

            # No evidence: the user-stated criterion is unresolved and blocks.
            unresolved_before = service._evaluate_goal_criteria(run, {"checks": [], "passed": True})
            self.assertEqual(1, len(unresolved_before))

            # A passing check on current sources for the linked task satisfies it.
            service.link_task_criteria(run["id"], task_id, [criterion])
            # Evidence must be a persisted run, as the service records it, not a claim in memory.
            result = {"command": ["check"], "stdout": "", "stderr": "", "exit_code": 0, "duration_ms": 1,
                      "status": "passed", "source_fingerprint": fingerprint, "task_ids": [task_id]}
            service._record_test(run["id"], "test", result)
            evidence = {"passed": True, "source_fingerprint": fingerprint, "checks": [
                {"kind": "test", "status": "passed", "exit_code": 0, "source_fingerprint": fingerprint,
                 "task_ids": [task_id], "evidence_id": result["evidence_id"]},
            ]}
            self.assertEqual([], service._evaluate_goal_criteria(run, evidence))
            with service.store.session() as connection:
                status = connection.execute("SELECT status FROM codebase_goal_criteria WHERE id = ?", (criterion,)).fetchone()["status"]
            self.assertEqual("satisfied", status)


if __name__ == "__main__":
    unittest.main()
