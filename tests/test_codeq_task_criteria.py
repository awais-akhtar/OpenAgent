"""Phase 3 slice 1: task-to-criterion links persist across restart and stay run-scoped."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openagent.codebase.service import CodebaseService
from openagent.storage import OpenAgentStore
from tests.test_codeq_failure_gates_e2e import ScriptedGateway, _writes_notes


class TaskCriteriaLinkTests(unittest.TestCase):
    def test_links_persist_across_restart_and_are_run_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            data = base / "data"
            service = CodebaseService(OpenAgentStore(data), ScriptedGateway(_writes_notes))  # type: ignore[arg-type]
            project = service.create_project(name="Notes", source_path=str(source))
            goal = service.create_goal(codebase_project_id=project["id"], goal="Write notes",
                                       acceptance_criteria=["notes.txt exists", "notes.txt has a greeting"])
            run = service.create_run(codebase_project_id=project["id"], prompt="Write notes", max_iterations=8, completion={}, goal_id=goal["id"])
            run = service.wait_for_run(run["id"], timeout=60)
            with service.store.session() as connection:
                task_ids = [row["id"] for row in connection.execute("SELECT id FROM codebase_tasks WHERE run_id = ? ORDER BY created_at", (run["id"],)).fetchall()]
                criterion_ids = sorted(row["id"] for row in connection.execute("SELECT id FROM codebase_goal_criteria WHERE goal_id = ?", (goal["id"],)).fetchall())
            self.assertTrue(task_ids)
            self.assertEqual(2, len(criterion_ids))
            task_id = task_ids[0]

            linked = service.link_task_criteria(run["id"], task_id, [criterion_ids[1], criterion_ids[0], criterion_ids[0]])
            self.assertEqual(criterion_ids, linked)

            restarted = CodebaseService(OpenAgentStore(data), ScriptedGateway(_writes_notes))  # type: ignore[arg-type]
            self.assertEqual(criterion_ids, restarted.task_criteria(run["id"], task_id))

    def test_task_from_another_run_cannot_be_linked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), ScriptedGateway(_writes_notes))  # type: ignore[arg-type]
            project = service.create_project(name="Notes", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Write notes", max_iterations=8, completion={})
            run = service.wait_for_run(run["id"], timeout=60)
            with service.store.session() as connection:
                task_id = connection.execute("SELECT id FROM codebase_tasks WHERE run_id = ? LIMIT 1", (run["id"],)).fetchone()["id"]
            with self.assertRaises(ValueError):
                service.link_task_criteria("some-other-run", task_id, ["crit-a"])


if __name__ == "__main__":
    unittest.main()
