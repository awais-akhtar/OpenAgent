"""Phase 3: ready-task selection and resume that never reruns verified work."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from openagent.codebase.service import CodebaseService
from openagent.storage import OpenAgentStore
from tests.test_codeq_failure_gates_e2e import ScriptedGateway, _writes_notes


def _insert(connection, run_id: str, task_id: str, status: str, depends: list[str]) -> None:
    now = "2026-11-11T00:00:00+00:00"
    connection.execute(
        "INSERT INTO codebase_tasks (id, run_id, title, assigned_agent, status, depends_on_json, result_json, created_at, updated_at) VALUES (?, ?, ?, 'programmer', ?, ?, '{}', ?, ?)",
        (task_id, run_id, task_id, status, __import__("json").dumps(depends), now, now),
    )


class ReadyTaskSelectionTests(unittest.TestCase):
    def test_only_pending_tasks_with_completed_dependencies_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), ScriptedGateway(_writes_notes))  # type: ignore[arg-type]
            project = service.create_project(name="Notes", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Write notes", max_iterations=8, completion={})
            service.wait_for_run(run["id"], timeout=60)
            with service.store.session() as connection:
                connection.execute("DELETE FROM codebase_tasks WHERE run_id = ?", (run["id"],))
                _insert(connection, run["id"], "A", "completed", [])
                _insert(connection, run["id"], "B", "pending", ["A"])        # ready
                _insert(connection, run["id"], "C", "pending", ["B"])        # waits for B
                _insert(connection, run["id"], "D", "running", ["A"])        # in flight: not ready
                _insert(connection, run["id"], "E", "completed", ["A"])      # done: never ready
                _insert(connection, run["id"], "F", "needs_fix", ["A"])      # ready for repair
            ready = [task["id"] for task in service._ready_tasks(run["id"])]
            self.assertEqual(["B", "F"], ready)

    def test_resume_requeues_interrupted_task_and_keeps_completed_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), ScriptedGateway(_writes_notes))  # type: ignore[arg-type]
            project = service.create_project(name="Notes", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Write notes", max_iterations=8, completion={})
            service.wait_for_run(run["id"], timeout=60)
            with service.store.session() as connection:
                connection.execute("DELETE FROM codebase_tasks WHERE run_id = ?", (run["id"],))
                _insert(connection, run["id"], "DONE", "completed", [])
                _insert(connection, run["id"], "LOST", "running", [])
                connection.execute("UPDATE codebase_runs SET status = 'interrupted' WHERE id = ?", (run["id"],))
            requeued = service._requeue_interrupted_tasks(run["id"])
            self.assertEqual(["LOST"], requeued)
            with service.store.session() as connection:
                statuses = {row["id"]: row["status"] for row in connection.execute("SELECT id, status FROM codebase_tasks WHERE run_id = ?", (run["id"],)).fetchall()}
            self.assertEqual({"DONE": "completed", "LOST": "pending"}, statuses)


if __name__ == "__main__":
    unittest.main()
