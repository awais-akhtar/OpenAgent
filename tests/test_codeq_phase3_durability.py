"""Phase 3 durability: persisted task contracts, attempts, graph links and crash recovery."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from openagent.codebase.service import CodebaseService
from openagent.codebase.work_graph import file_node_id
from openagent.adaptive_engine.execution_service import ExecutionCancelled
from openagent.model_gateway import ModelResult
from openagent.storage import OpenAgentStore


class _Gateway:
    def complete(self, messages, **kwargs):
        system = str(messages[0].get("content", "")).lower()
        prompt = json.dumps(messages).casefold()
        if "project manager" in system:
            return ModelResult("fixture", "fixture-code", json.dumps({"tasks": [
                {"id": "T1", "title": "Implement feature", "agent": "programmer", "depends_on": [], "files": ["feature.py"]},
                # Testers may list a file to inspect; this must not turn a
                # validation-only task into a required write to that file.
                {"id": "T2", "title": "Run tests", "agent": "tester", "depends_on": ["T1"], "files": ["test_feature.py"]},
            ]}), {})
        if "independent code reviewer" in system:
            return ModelResult("fixture", "fixture-code", json.dumps({"status": "passed", "findings": []}), {})
        if "final acceptance auditor" in system:
            return ModelResult("fixture", "fixture-code", json.dumps({"status": "passed", "missing_tasks": [], "evidence": []}), {})
        if "current task: implement feature" in prompt:
            answer = {"summary": "Created the feature", "done": True, "actions": [
                {"tool": "write_file", "arguments": {"path": "feature.py", "content": "value = 1\n"}},
            ]}
        elif "current task: run tests" in prompt:
            check = "import unittest; suite=unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]); result=unittest.TextTestRunner().run(suite); raise SystemExit(not result.wasSuccessful())"
            answer = {"summary": "Test discovery passed", "done": True, "actions": [
                {"tool": "run_shell", "arguments": {"command": [sys.executable, "-c", check]}},
            ]}
        else:
            answer = {"summary": "No action", "done": False, "actions": []}
        return ModelResult("fixture", "fixture-code", json.dumps(answer), {})


class Phase3DurabilityTests(unittest.TestCase):
    def test_successful_full_file_write_survives_malformed_redundant_patch(self) -> None:
        class WriteThenBadDiffGateway:
            def complete(self, messages, **kwargs):
                system = str(messages[0].get("content", "")).lower()
                prompt = json.dumps(messages).casefold()
                if "project manager" in system:
                    answer = {"tasks": [
                        {"id": "T1", "title": "Implement feature", "agent": "programmer", "depends_on": [], "files": ["feature.py"]},
                        {"id": "T2", "title": "Run tests", "agent": "tester", "depends_on": ["T1"], "files": []},
                    ]}
                elif "independent code reviewer" in system:
                    answer = {"status": "passed", "findings": []}
                elif "final acceptance auditor" in system:
                    answer = {"status": "passed", "missing_tasks": [], "evidence": []}
                elif "current task: implement feature" in prompt:
                    answer = {"summary": "Wrote the complete file", "done": False, "actions": [
                        {"tool": "write_file", "arguments": {"path": "feature.py", "content": "value = 1\n"}},
                        {"tool": "patch_file", "arguments": {"path": "feature.py", "patch": "--- a/feature.py\n+++ b/feature.py\n@@ -1,1 +1,1 @@\n-value = 0\n+value = 1\n+extra = True\n"}},
                    ]}
                elif "current task: run tests" in prompt:
                    answer = {"summary": "Ran the existing tests", "done": True, "actions": [
                        {"tool": "run_tests", "arguments": {"command": [sys.executable, "-m", "unittest", "discover", "-v"]}},
                    ]}
                else:
                    answer = {"summary": "No action", "done": False, "actions": []}
                return ModelResult("fixture", "fixture-code", json.dumps(answer), {})

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            (source / "tests").mkdir(parents=True)
            (source / "tests" / "__init__.py").write_text("", encoding="utf-8")
            (source / "tests" / "test_core.py").write_text(
                "import unittest\n\nclass ExistingTests(unittest.TestCase):\n    def test_existing(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            service = CodebaseService(OpenAgentStore(base / "data"), WriteThenBadDiffGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Redundant patch", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Update existing application with a small feature and test it",
                max_iterations=8, completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('runtime ok')"]}},
            )
            finished = service.wait_for_run(run["id"], timeout=60)
            self.assertEqual(
                "completed", finished["status"],
                f"{finished.get('error')}\n" + "\n".join(
                    f"{event.get('type')}: {json.dumps(event, default=str)[:600]}"
                    for event in service.events(run["id"])
                ),
            )
            workspace = Path(service._workspace(finished["workspace_id"], finished["owner"])["workspace_path"])
            self.assertEqual("value = 1\n", (workspace / "feature.py").read_text(encoding="utf-8"))
            events = service.events(run["id"])
            self.assertTrue(any(event["type"] == "tool.action_skipped" for event in events))
            self.assertTrue(any(event["type"] == "test.passed" for event in events))

    def test_test_file_declaration_is_owned_by_the_dependent_test_authoring_task(self) -> None:
        tasks = [
            {"title": "Implement the feature", "assigned_agent": "programmer", "files": ["app.py", "tests/test_app.py"]},
            {"title": "Update tests for the feature", "assigned_agent": "programmer", "files": ["tests/test_app.py"]},
            {"title": "Run tests", "assigned_agent": "tester", "files": ["tests/test_app.py"]},
        ]
        self.assertEqual(
            [["app.py"], ["tests/test_app.py"], []],
            CodebaseService._planned_task_files(tasks),
        )

    def test_incidental_test_file_on_implementation_task_is_not_a_write_obligation(self) -> None:
        tasks = [
            {"title": "Implement fixes to normalize_email function", "assigned_agent": "programmer", "files": ["app.py", "tests/test_normalize_email.py"]},
            {"title": "Run tests to ensure no regressions", "assigned_agent": "tester", "files": ["tests/test_normalize_email.py"]},
        ]
        self.assertEqual([["app.py"], []], CodebaseService._planned_task_files(tasks))

    def _finished(self, base: Path):
        source = base / "source"
        source.mkdir()
        (source / "README.md").write_text("Existing local project\n", encoding="utf-8")
        service = CodebaseService(OpenAgentStore(base / "data"), _Gateway())  # type: ignore[arg-type]
        project = service.create_project(name="Phase 3", source_path=str(source))
        run = service.create_run(
            codebase_project_id=project["id"], prompt="Add a feature and test it", max_iterations=12,
            completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}},
        )
        run = service.wait_for_run(run["id"], timeout=60)
        self.assertEqual("completed", run["status"], run.get("error"))
        return service, run

    def test_service_restart_preserves_runs_owned_by_a_live_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store = OpenAgentStore(base / "data")
            service = CodebaseService(store, _Gateway())  # type: ignore[arg-type]
            source = base / "source"
            source.mkdir()
            project = service.create_project(name="Recovery", source_path=str(source))
            workspace_id = project["workspace"]["id"]
            now = "2026-11-12T00:00:00+00:00"
            with store.session() as connection:
                for run_id, status, worker_pid in (
                    ("live-run", "running", os.getpid()),
                    ("dead-run", "analysing", 2_147_483_647),
                    ("legacy-run", "validating", 0),
                ):
                    connection.execute(
                        """INSERT INTO codebase_runs
                           (id, codebase_project_id, workspace_id, project, owner, user_prompt,
                            status, started_at, updated_at, worker_pid)
                           VALUES (?, ?, ?, ?, 'local', 'recovery test', ?, ?, ?, ?)""",
                        (run_id, project["id"], workspace_id, project["project"], status, now, now, worker_pid),
                    )

            # The application can construct another service object in the
            # same process; that must not mistake its active worker for a crash.
            CodebaseService(OpenAgentStore(base / "data"), _Gateway())  # type: ignore[arg-type]
            with store.session() as connection:
                statuses = {
                    str(row["id"]): (str(row["status"]), int(row["worker_pid"]))
                    for row in connection.execute(
                        "SELECT id, status, worker_pid FROM codebase_runs WHERE id IN ('live-run','dead-run','legacy-run')"
                    ).fetchall()
                }
            self.assertEqual(("running", os.getpid()), statuses["live-run"])
            self.assertEqual(("interrupted", 0), statuses["dead-run"])
            self.assertEqual(("interrupted", 0), statuses["legacy-run"])

    def test_task_contract_attempt_count_and_code_node_links_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, run = self._finished(Path(directory))
            with service.store.session() as connection:
                row = connection.execute("SELECT * FROM codebase_tasks WHERE run_id = ? AND title = 'Implement feature'", (run["id"],)).fetchone()
                self.assertIsNotNone(row)
                task_id = str(row["id"])
                predicates = json.loads(row["done_predicates_json"])
                self.assertEqual([{"kind": "declared_files_touched", "paths": ["feature.py"]}], predicates)
                self.assertEqual(["feature.py"], json.loads(row["code_paths_json"]))
                self.assertEqual([file_node_id("feature.py")], json.loads(row["code_node_ids_json"]))
                self.assertEqual(1, int(row["attempt_count"]))
                tester = connection.execute("SELECT declared_files_json, done_predicates_json, status FROM codebase_tasks WHERE run_id = ? AND title = 'Run tests'", (run["id"],)).fetchone()
                self.assertIsNotNone(tester)
                self.assertEqual([], json.loads(tester["declared_files_json"]))
                self.assertEqual([{"kind": "successful_validation", "tools": ["run_tests", "run_linter", "run_typechecker", "run_build", "browser_test"]}], json.loads(tester["done_predicates_json"]))
                self.assertEqual("completed", tester["status"])
            self.assertEqual(2, service._begin_task_attempt(task_id))
            service._set_task(task_id, "completed", {"summary": "test only"})
            with service.store.session() as connection:
                row = connection.execute("SELECT attempt_count, retry_count FROM codebase_tasks WHERE id = ?", (task_id,)).fetchone()
            self.assertEqual((2, 1), (int(row["attempt_count"]), int(row["retry_count"])))

    def test_interrupted_task_restores_checkpoint_and_previous_graph_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, run = self._finished(Path(directory))
            with service.store.session() as connection:
                task = connection.execute("SELECT id FROM codebase_tasks WHERE run_id = ? AND title = 'Implement feature'", (run["id"],)).fetchone()
            task_id = str(task["id"])
            workspace = Path(service._workspace(run["workspace_id"], run["owner"])["workspace_path"])
            (workspace / "feature.py").write_text("checkpoint value\n", encoding="utf-8")
            service._checkpoint_tasks_before_execution(run, [task_id], label="test recovery checkpoint")
            (workspace / "feature.py").write_text("partial uncommitted attempt\n", encoding="utf-8")
            (workspace / "partial.py").write_text("partial\n", encoding="utf-8")
            service._record_task_code(run["id"], task_id, ["partial.py"])
            with service.store.session() as connection:
                connection.execute("UPDATE codebase_tasks SET status = 'running' WHERE id = ?", (task_id,))

            self.assertEqual([task_id], service._requeue_interrupted_tasks(run["id"]))

            self.assertEqual("checkpoint value\n", (workspace / "feature.py").read_text(encoding="utf-8"))
            self.assertFalse((workspace / "partial.py").exists())
            with service.store.session() as connection:
                row = connection.execute("SELECT status, code_paths_json, code_node_ids_json, checkpoint_snapshot_id FROM codebase_tasks WHERE id = ?", (task_id,)).fetchone()
            self.assertEqual("pending", row["status"])
            self.assertEqual(["feature.py"], json.loads(row["code_paths_json"]))
            self.assertEqual([file_node_id("feature.py")], json.loads(row["code_node_ids_json"]))
            self.assertEqual("", row["checkpoint_snapshot_id"])

    def test_interrupted_parallel_batch_reopens_completed_siblings_after_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, run = self._finished(Path(directory))
            with service.store.session() as connection:
                tasks = connection.execute(
                    "SELECT id FROM codebase_tasks WHERE run_id = ? ORDER BY created_at, id LIMIT 2", (run["id"],)
                ).fetchall()
            task_ids = [str(row["id"]) for row in tasks]
            self.assertEqual(2, len(task_ids))
            workspace = Path(service._workspace(run["workspace_id"], run["owner"])["workspace_path"])
            (workspace / "feature.py").write_text("pre batch feature\n", encoding="utf-8")
            (workspace / "test_feature.py").write_text("pre batch test\n", encoding="utf-8")
            service._record_task_code(run["id"], task_ids[0], ["feature.py"])
            service._record_task_code(run["id"], task_ids[1], ["test_feature.py"])
            service._checkpoint_tasks_before_execution(run, task_ids, label="test parallel recovery checkpoint")
            (workspace / "feature.py").write_text("partial feature write\n", encoding="utf-8")
            (workspace / "test_feature.py").write_text("partial test write\n", encoding="utf-8")
            with service.store.session() as connection:
                connection.execute("UPDATE codebase_tasks SET status = 'completed' WHERE id = ?", (task_ids[0],))
                connection.execute("UPDATE codebase_tasks SET status = 'running' WHERE id = ?", (task_ids[1],))

            self.assertEqual(task_ids, service._requeue_interrupted_tasks(run["id"]))
            self.assertEqual("pre batch feature\n", (workspace / "feature.py").read_text(encoding="utf-8"))
            self.assertEqual("pre batch test\n", (workspace / "test_feature.py").read_text(encoding="utf-8"))
            with service.store.session() as connection:
                statuses = [row["status"] for row in connection.execute(
                    "SELECT status FROM codebase_tasks WHERE id IN (?, ?) ORDER BY id", task_ids
                ).fetchall()]
            self.assertEqual(["pending", "pending"], statuses)

    def test_declared_task_done_predicates_reject_unverified_completion(self) -> None:
        contract = {"done_predicates": [{"kind": "declared_files_touched", "paths": ["a.py", "b.py"]}], "code_paths": ["a.py"]}
        self.assertIn("b.py", CodebaseService._task_done_predicate_failure(contract, [("write_file", {})]))
        validation = {"done_predicates": [{"kind": "successful_validation", "tools": ["run_tests"]}], "code_paths": []}
        self.assertTrue(CodebaseService._task_done_predicate_failure(validation, [("run_shell", {"command": [sys.executable, "-c", "print('ok')"], "exit_code": 0})]))
        self.assertEqual("", CodebaseService._task_done_predicate_failure(validation, [("run_shell", {"command": [sys.executable, "-m", "unittest", "discover"], "exit_code": 0})]))
        evidence = {"title": "Inspect repository", "assigned_agent": "architect", "done_predicates": [{"kind": "successful_task_evidence"}], "code_paths": []}
        self.assertIn("successful tool evidence", CodebaseService._task_done_predicate_failure(evidence, []))
        self.assertEqual("", CodebaseService._task_done_predicate_failure(evidence, [("project_map", {"exit_code": 0})]))
        unknown = {"done_predicates": [{"kind": "model_says_done"}], "code_paths": []}
        self.assertIn("unknown predicate", CodebaseService._task_done_predicate_failure(unknown, []))

    def test_repeated_invalid_model_tool_arguments_stop_before_iteration_limit(self) -> None:
        class InvalidActionGateway:
            task_calls = 0

            def complete(self, messages, **kwargs):
                system = str(messages[0].get("content", "")).lower()
                if "project manager" in system:
                    payload = {"tasks": [
                        {"id": "T1", "title": "Implement feature", "agent": "programmer", "depends_on": [], "files": ["feature.py"]},
                    ]}
                elif "independent code reviewer" in system:
                    payload = {"status": "passed", "findings": []}
                elif "final acceptance auditor" in system:
                    payload = {"status": "passed", "missing_tasks": [], "evidence": []}
                else:
                    self.task_calls += 1
                    if self.task_calls % 2:
                        payload = {"summary": "attempted a patch", "done": True, "actions": [
                            {"tool": "patch_file", "arguments": {"file_path": "feature.py", "patch": "value = 1"}},
                        ]}
                    else:
                        # A different tool failure rewrites result_json. The
                        # invalid-action counter must still survive separately.
                        payload = {"summary": "inspected a missing file", "done": False, "actions": [
                            {"tool": "read_file", "arguments": {"path": f"missing-{self.task_calls}.txt"}},
                        ]}
                return ModelResult("fixture", "fixture-code", json.dumps(payload), {})

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("Existing local project\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), InvalidActionGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Invalid action", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Implement a feature", max_iterations=20, completion={})
            finished = service.wait_for_run(run["id"], timeout=20)

            self.assertEqual("failed", finished["status"])
            self.assertIn("invalid tool arguments three times", finished.get("error", ""))
            task = finished["tasks"][0]
            self.assertEqual(3, task["result"]["invalid_action_attempts"])
            self.assertEqual(3, task["invalid_action_attempts"])
            with service.store.session() as connection:
                row = connection.execute("SELECT invalid_action_attempts, attempt_count FROM codebase_tasks WHERE id = ?", (task["id"],)).fetchone()
            self.assertEqual(3, int(row["invalid_action_attempts"]))
            self.assertEqual(5, int(row["attempt_count"]))
            self.assertLess(finished["iteration"], 20)
            events = service.events(run["id"])
            self.assertTrue(any(event["type"] == "run.failed" and event["payload"].get("reason") == "invalid_model_actions" for event in events))
            workspace = Path(service._workspace(finished["workspace_id"], finished["owner"])["workspace_path"])
            self.assertFalse((workspace / "feature.py").exists(), "invalid tool arguments must never execute")

    def test_cancelling_blocked_model_call_stays_cancelled_and_prevents_late_actions(self) -> None:
        started = threading.Event()

        class BlockingGateway:
            def complete(self, messages, **kwargs):
                system = str(messages[0].get("content", "")).lower()
                prompt = json.dumps(messages).casefold()
                if "project manager" in system:
                    plan = {"tasks": [
                        {"id": "T1", "title": "Implement feature", "agent": "programmer", "depends_on": [], "files": ["feature.py"]},
                        {"id": "T2", "title": "Run tests", "agent": "tester", "depends_on": ["T1"], "files": []},
                    ]}
                    return ModelResult("fixture", "fixture-code", json.dumps(plan), {})
                if "independent code reviewer" in system:
                    return ModelResult("fixture", "fixture-code", json.dumps({"status": "passed", "findings": []}), {})
                if "final acceptance auditor" in system:
                    return ModelResult("fixture", "fixture-code", json.dumps({"status": "passed", "missing_tasks": [], "evidence": []}), {})
                if "current task: implement feature" in prompt:
                    started.set()
                    cancelled = kwargs.get("cancelled")
                    while not cancelled():
                        time.sleep(0.005)
                    raise ExecutionCancelled("test model call observed cancellation")
                return ModelResult("fixture", "fixture-code", json.dumps({"actions": [], "done": False}), {})

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("Existing project\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), BlockingGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Cancellation", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Implement a feature", max_iterations=8, completion={})
            self.assertTrue(started.wait(5), "the model call should start before cancellation")
            cancelled = service.cancel_run(run["id"])
            finished = service.wait_for_run(run["id"], timeout=10)
            self.assertEqual("cancelled", cancelled["status"])
            self.assertEqual("cancelled", finished["status"])
            workspace = Path(service._workspace(run["workspace_id"], run["owner"])["workspace_path"])
            self.assertFalse((workspace / "feature.py").exists(), "cancelled model output must not execute late file writes")


if __name__ == "__main__":
    unittest.main()
