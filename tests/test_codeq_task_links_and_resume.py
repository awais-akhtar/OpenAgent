"""Phase 3: tasks record the code they touch, and resume never reruns completed work."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from openagent.codebase.service import CodebaseService
from openagent.storage import OpenAgentStore
from tests.test_codeq_failure_gates_e2e import ScriptedGateway, _writes_feature


class TaskCodeAndResumeTests(unittest.TestCase):
    def _service(self, base: Path) -> CodebaseService:
        return CodebaseService(OpenAgentStore(base / "data"), ScriptedGateway(_writes_feature))  # type: ignore[arg-type]

    def test_task_records_the_code_it_touched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("existing project\n", encoding="utf-8")
            service = self._service(base)
            project = service.create_project(name="Feature", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Add a feature", max_iterations=12,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}},
            )
            run = service.wait_for_run(run["id"], timeout=60)
            self.assertEqual("completed", run["status"], run.get("error"))
            with service.store.session() as connection:
                task_ids = [row["id"] for row in connection.execute("SELECT id FROM codebase_tasks WHERE run_id = ?", (run["id"],)).fetchall()]
            touched = {path for task_id in task_ids for path in service.task_code_paths(run["id"], task_id)}
            self.assertIn("feature.py", touched)

    def test_resume_does_not_rerun_completed_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("existing project\n", encoding="utf-8")
            service = self._service(base)
            project = service.create_project(name="Feature", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Add a feature", max_iterations=12,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}},
            )
            run = service.wait_for_run(run["id"], timeout=60)
            self.assertEqual("completed", run["status"], run.get("error"))

            def writes() -> int:
                with service.store.session() as connection:
                    return int(connection.execute("SELECT COUNT(*) FROM codebase_tool_calls WHERE run_id = ? AND tool = 'write_file'", (run["id"],)).fetchone()[0])

            before = writes()
            with service.store.session() as connection:
                connection.execute("UPDATE codebase_runs SET status = 'interrupted' WHERE id = ?", (run["id"],))
            service.resume_run(run["id"])
            service.wait_for_run(run["id"], timeout=60)
            self.assertEqual(before, writes(), "resume must not repeat a verified write")


if __name__ == "__main__":
    unittest.main()
