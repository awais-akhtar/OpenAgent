"""Phase 3: disjoint tasks run as a batch, merges are validated, and vertical slices are enforced."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from openagent.codebase.service import CodebaseService
from openagent.codebase.task_graph import layer_of, missing_slice_layers
from openagent.model_gateway import ModelResult
from openagent.storage import OpenAgentStore


def _model(content: dict) -> ModelResult:
    return ModelResult("fake", "fake-code", json.dumps(content), {})


class PlanGateway:
    """Planner returns the given tasks; reviewer and auditor pass; implementer is scripted by title."""

    def __init__(self, tasks: list[dict], implementer) -> None:
        self.tasks = tasks
        self.implementer = implementer

    def select_provider(self, provider_id: str, *, allow_external: bool = False):
        # These tests use a fake remote route unless a test overrides the selector.
        return type("Provider", (), {"id": "openai", "local": False})()

    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        if "project manager" in system:
            return _model({"goal": "change", "tasks": self.tasks})
        if "independent code reviewer" in system:
            return _model({"status": "passed", "findings": []})
        if "final acceptance auditor" in system:
            return _model({"status": "passed", "missing_tasks": [], "evidence": []})
        return _model(self.implementer(json.dumps(messages)))


def _write(path: str, content: str = "x = 1\n", *, done: bool = True) -> dict:
    return {
        "summary": f"wrote {path}",
        "actions": [{"tool": "write_file", "arguments": {"path": path, "content": content}}],
        "done": done,
    }


def _validate() -> dict:
    """Run a bounded local test-discovery command for tester tasks."""
    check = "import unittest; suite=unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]); result=unittest.TextTestRunner().run(suite); raise SystemExit(not result.wasSuccessful())"
    return {
        "summary": "Local unittest smoke check passed",
        "actions": [{"tool": "run_shell", "arguments": {"command": [sys.executable, "-c", check]}}],
        "done": True,
    }


def _by_title(mapping: dict[str, dict]):
    def implementer(prompt: str) -> dict:
        for marker, response in mapping.items():
            if f"Current task: {marker}" in prompt:
                return response
        return {"summary": "nothing", "actions": [], "done": False}
    return implementer


class LayerClassificationTests(unittest.TestCase):
    def test_paths_map_to_their_layer(self) -> None:
        self.assertEqual("ui", layer_of("frontend/src/screens/Workspace.tsx"))
        self.assertEqual("api", layer_of("openagent/routes/codebase_routes.py"))
        self.assertEqual("persistence", layer_of("openagent/codebase/repository.py"))
        self.assertEqual("persistence", layer_of("db/migrations/001_init.sql"))
        self.assertEqual("tests", layer_of("tests/test_codeq_parallel_execution.py"))
        self.assertIsNone(layer_of("README.md"))

    def test_single_layer_task_has_no_slice_requirement(self) -> None:
        self.assertEqual([], missing_slice_layers(["ui/App.tsx", "ui/Panel.tsx"], []))

    def test_multi_layer_task_must_change_every_declared_layer(self) -> None:
        declared = ["ui/App.tsx", "openagent/routes/items.py", "db/migrations/002.sql"]
        self.assertEqual(["api", "persistence"], missing_slice_layers(declared, ["ui/App.tsx"]))
        self.assertEqual([], missing_slice_layers(declared, ["ui/App.tsx", "openagent/routes/items.py", "db/migrations/002.sql"]))


class ParallelBatchEndToEndTests(unittest.TestCase):
    def _run(self, gateway, *, max_iterations: int = 12) -> tuple[dict, list[dict]]:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), gateway)  # type: ignore[arg-type]
            project = service.create_project(name="Batch", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Add two modules", max_iterations=max_iterations,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}},
            )
            run = service.wait_for_run(run["id"], timeout=60)
            return run, service.events(run["id"])

    @staticmethod
    def _two_disjoint_tasks() -> list[dict]:
        return [
            {"id": "T1", "title": "Implement alpha", "agent": "programmer", "depends_on": [], "files": ["alpha.py"]},
            {"id": "T2", "title": "Implement beta", "agent": "programmer", "depends_on": [], "files": ["beta.py"]},
            {"id": "T3", "title": "Validate alpha and beta", "agent": "tester", "depends_on": ["T1", "T2"], "files": []},
        ]

    def test_disjoint_tasks_run_as_one_batch_and_complete(self) -> None:
        implementer = _by_title({
            "Implement alpha": _write("alpha.py"),
            "Implement beta": _write("beta.py"),
            "Validate alpha and beta": _validate(),
        })
        run, events = self._run(PlanGateway(self._two_disjoint_tasks(), implementer))
        self.assertEqual("completed", run["status"], run.get("error"))
        started = [event for event in events if event["type"] == "batch.started"]
        self.assertTrue(started, "two disjoint ready tasks should run as a batch")
        self.assertEqual(2, len(started[0]["payload"]["task_ids"]))
        merged = [event for event in events if event["type"] == "batch.merged"]
        self.assertEqual({}, merged[0]["payload"]["rejected"])

    def test_auto_routed_local_model_serializes_disjoint_tasks(self) -> None:
        class LocalPlanGateway(PlanGateway):
            def select_provider(self, provider_id: str, *, allow_external: bool = False):
                return type("Provider", (), {"id": "ollama", "local": True})()

        implementer = _by_title({
            "Implement alpha": _write("alpha.py"),
            "Implement beta": _write("beta.py"),
            "Validate alpha and beta": _validate(),
        })
        run, events = self._run(LocalPlanGateway(self._two_disjoint_tasks(), implementer))
        self.assertEqual("completed", run["status"], run.get("error"))
        self.assertFalse(
            [event for event in events if event["type"] == "batch.started"],
            "Auto resolving to a local model must not dispatch task generations concurrently",
        )
        decisions = [event for event in events if event["type"] == "agents.parallel.started"]
        self.assertTrue(decisions)
        self.assertTrue(all(len(event["payload"]["task_ids"]) == 1 for event in decisions))

    def test_failed_check_hands_cross_task_file_to_bounded_repair(self) -> None:
        tasks = [
            {"id": "T1", "title": "Implement alpha helper", "agent": "programmer", "depends_on": [], "files": ["src/alpha.js"]},
            {"id": "T2", "title": "Modify beta consumer", "agent": "programmer", "depends_on": ["T1"], "files": ["src/beta.js"]},
            {"id": "T3", "title": "Validate alpha and beta", "agent": "tester", "depends_on": ["T2"], "files": []},
        ]
        failing_check = (
            "import sys; print('AssertionError: src/alpha.js and src/beta.js failed', file=sys.stderr); "
            "raise SystemExit(1)"
        )
        beta_attempts = {"count": 0}

        def implementer(prompt: str) -> dict:
            if "Current task: Implement alpha helper" in prompt:
                return _write("src/alpha.js", "export const alpha = 1;\n")
            if "Current task: Modify beta consumer" in prompt:
                beta_attempts["count"] += 1
                if beta_attempts["count"] == 1:
                    return {
                        "summary": "Edit beta and run its focused check",
                        "actions": [
                            {"tool": "write_file", "arguments": {"path": "src/beta.js", "content": "export const beta = 1;\n"}},
                            {"tool": "run_tests", "arguments": {"command": [sys.executable, "-c", failing_check]}},
                        ],
                        "done": False,
                    }
                return _write("src/beta.js", "export const beta = 2;\n")
            if "Current task: Repair failing check for: Modify beta consumer" in prompt:
                return _write("src/alpha.js", "export const alpha = 2;\n")
            if "Current task: Validate alpha and beta" in prompt:
                return _validate()
            return {"summary": "no action", "actions": [], "done": False}

        run, events = self._run(PlanGateway(tasks, implementer), max_iterations=12)
        self.assertEqual("completed", run["status"], run.get("error"))
        replans = [event for event in events if event["type"] == "task.replanned"]
        self.assertTrue(replans, "a failed check naming a prior task's file must be handed to a repair task")
        self.assertIn("src/alpha.js", replans[0]["payload"]["files"])
        self.assertIn("src/beta.js", replans[0]["payload"]["files"])

    def test_member_writing_an_undeclared_file_is_rejected_at_merge(self) -> None:
        # alpha declares only alpha.py but also writes beta.py, which the other member owns.
        implementer = _by_title({
            "Implement alpha": {
                "summary": "wrote both",
                "actions": [
                    {"tool": "write_file", "arguments": {"path": "alpha.py", "content": "a = 1\n"}},
                    {"tool": "write_file", "arguments": {"path": "beta.py", "content": "b = 1\n"}},
                ],
                "done": True,
            },
            "Implement beta": _write("beta.py"),
            "Validate alpha and beta": _validate(),
        })
        run, events = self._run(PlanGateway(self._two_disjoint_tasks(), implementer), max_iterations=2)
        merged = [event for event in events if event["type"] == "batch.merged"]
        self.assertTrue(merged, "the batch must be merge-validated")
        rejected = merged[0]["payload"]["rejected"]
        self.assertEqual(1, len(rejected))
        self.assertTrue(next(iter(rejected)).endswith(":T1"))
        self.assertEqual(["beta.py"], next(iter(rejected.values())))
        self.assertNotEqual("completed", run["status"])


class VerticalSliceEndToEndTests(unittest.TestCase):
    def test_task_spanning_two_layers_cannot_complete_after_one(self) -> None:
        tasks = [
            {"id": "T1", "title": "Implement item feature", "agent": "programmer", "depends_on": [],
             "files": ["ui/ItemPanel.tsx", "openagent/routes/items.py"]},
            {"id": "T2", "title": "Validate item feature", "agent": "tester", "depends_on": ["T1"], "files": []},
        ]
        implementer = _by_title({
            "Implement item feature": _write("ui/ItemPanel.tsx", "export const Item = 1;\n"),
            "Validate item feature": _write("tests/test_item.py"),
        })
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), PlanGateway(tasks, implementer))  # type: ignore[arg-type]
            project = service.create_project(name="Slice", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Add an item feature", max_iterations=2,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}},
            )
            run = service.wait_for_run(run["id"], timeout=60)
            events = service.events(run["id"])
        self.assertNotEqual("completed", run["status"])
        incomplete = [event for event in events if event["type"] == "task.slice.incomplete"]
        self.assertTrue(incomplete, "a one-layer completion of a two-layer task must be refused")
        self.assertEqual(["api"], incomplete[0]["payload"]["missing_layers"])


class ProgressPastFailureTests(unittest.TestCase):
    def test_other_criteria_progress_while_one_method_keeps_failing(self) -> None:
        tasks = [
            {"id": "T1", "title": "Implement alpha", "agent": "programmer", "depends_on": [], "files": []},
            {"id": "T2", "title": "Implement beta", "agent": "programmer", "depends_on": [], "files": []},
            {"id": "T3", "title": "Validate alpha and beta", "agent": "tester", "depends_on": ["T1", "T2"], "files": []},
        ]
        failing = {"summary": "alpha", "actions": [{"tool": "run_shell", "arguments": {"command": [sys.executable, "-c", "raise SystemExit(9)"]}}], "done": False}
        implementer = _by_title({"Implement alpha": failing, "Implement beta": _write("beta.py")})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), PlanGateway(tasks, implementer))  # type: ignore[arg-type]
            project = service.create_project(name="Progress", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add two modules", max_iterations=24, completion={})
            run = service.wait_for_run(run["id"], timeout=90)
            events = service.events(run["id"])
        types = [event["type"] for event in events]
        self.assertIn("task.deferred", types, "the failing task must be parked, not fail the run")
        parked_at = types.index("task.deferred")
        completed_after_parking = [
            event for event in events[parked_at:]
            if event["type"] == "task.transition" and event["payload"].get("status") == "completed"
        ]
        self.assertTrue(completed_after_parking, "other criteria must keep progressing after the failing task is parked")
        # Once nothing else can move, the deferred task gets one retry round and the run ends honestly.
        self.assertIn("tasks.deferred_retry", types)
        self.assertEqual("failed", run["status"])
        self.assertIn("same command failed", run["error"])


class ControllerTests(unittest.TestCase):
    @staticmethod
    def _tasks() -> list[dict]:
        return [
            {"id": "T1", "title": "Implement alpha", "agent": "programmer", "depends_on": [], "files": []},
            {"id": "T2", "title": "Implement beta", "agent": "programmer", "depends_on": [], "files": []},
            {"id": "T3", "title": "Validate alpha and beta", "agent": "tester", "depends_on": ["T1", "T2"], "files": []},
        ]

    def _service(self, base: Path, gateway) -> CodebaseService:
        return CodebaseService(OpenAgentStore(base / "data"), gateway)  # type: ignore[arg-type]

    def test_pause_stops_at_a_task_boundary_and_resume_completes(self) -> None:
        holder: dict[str, str] = {}

        def implementer(prompt: str) -> dict:
            if "Current task: Implement alpha" in prompt:
                # Wait until the run id is known, then ask to pause once, before this task's write lands.
                while "id" not in holder:
                    time.sleep(0.01)
                if not holder.get("paused"):
                    holder["paused"] = "yes"
                    holder["service"].pause_run(holder["id"])
                return _write("alpha.py")
            if "Current task: Implement beta" in prompt:
                return _write("beta.py")
            if "Current task: Validate alpha and beta" in prompt:
                return _validate()
            return {"summary": "nothing", "actions": [], "done": False}

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = self._service(base, PlanGateway(self._tasks(), implementer))
            holder["service"] = service
            project = service.create_project(name="Pause", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Add two modules", max_iterations=12,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}},
            )
            holder["id"] = run["id"]
            paused = service.wait_for_run(run["id"], timeout=60)
            self.assertEqual("paused", paused["status"], paused.get("error"))
            events = service.events(run["id"])
            self.assertTrue(any(event["type"] == "run.paused" for event in events))
            self.assertTrue(any(event["type"] == "run.pause_requested" for event in events))

            service.resume_run(run["id"])
            finished = service.wait_for_run(run["id"], timeout=60)
            self.assertEqual("completed", finished["status"], finished.get("error"))

    def test_substituted_model_stops_the_run(self) -> None:
        class FallbackGateway(PlanGateway):
            def complete(self, messages, **kwargs):
                if "Current task: Implement alpha" in json.dumps(messages):
                    return ModelResult("stub", "offline", json.dumps(_write("alpha.py")), {"fallback_from": "ollama"})
                return super().complete(messages, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = self._service(base, FallbackGateway(self._tasks(), lambda _p: {"summary": "", "actions": [], "done": False}))
            project = service.create_project(name="Fallback", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add two modules", max_iterations=12, completion={})
            run = service.wait_for_run(run["id"], timeout=60)
            self.assertEqual("failed", run["status"])
            self.assertIn("offline stub", run["error"])
            events = service.events(run["id"])
            self.assertTrue(any(event["type"] == "run.failed" and event["payload"].get("reason") == "model_fallback" for event in events))

    def test_replanning_resets_only_dependants_of_the_invalidated_task(self) -> None:
        implementer = _by_title({
            "Implement alpha": _write("alpha.py"),
            "Implement beta": _write("beta.py"),
            "Validate alpha and beta": _validate(),
        })
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = self._service(base, PlanGateway(self._tasks(), implementer))
            project = service.create_project(name="Replan", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Add two modules", max_iterations=12,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}},
            )
            run = service.wait_for_run(run["id"], timeout=60)
            self.assertEqual("completed", run["status"], run.get("error"))
            with service.store.session() as connection:
                rows = {row["title"]: row["id"] for row in connection.execute("SELECT id, title FROM codebase_tasks WHERE run_id = ?", (run["id"],)).fetchall()}
            alpha, beta, validate = rows["Implement alpha"], rows["Implement beta"], rows["Validate alpha and beta"]

            reset = service._replan_downstream(run["id"], [alpha])

            self.assertEqual([validate], reset)
            with service.store.session() as connection:
                statuses = {row["id"]: row["status"] for row in connection.execute("SELECT id, status FROM codebase_tasks WHERE run_id = ?", (run["id"],)).fetchall()}
            self.assertEqual("pending", statuses[validate])
            self.assertEqual("completed", statuses[beta])
            self.assertEqual("completed", statuses[alpha])
            events = service.events(run["id"])
            self.assertTrue(any(event["type"] == "task.transition" for event in events))


class OpenItemsTests(unittest.TestCase):
    def _service(self, base: Path, gateway) -> CodebaseService:
        return CodebaseService(OpenAgentStore(base / "data"), gateway)  # type: ignore[arg-type]

    def test_unclaimed_shell_write_during_a_batch_is_rejected(self) -> None:
        tasks = [
            {"id": "T1", "title": "Implement alpha", "agent": "programmer", "depends_on": [], "files": ["alpha.py"]},
            {"id": "T2", "title": "Implement beta", "agent": "programmer", "depends_on": [], "files": ["beta.py"]},
            {"id": "T3", "title": "Validate alpha and beta", "agent": "tester", "depends_on": ["T1", "T2"], "files": []},
        ]
        sneaky = {
            "summary": "alpha via shell",
            "actions": [
                {"tool": "write_file", "arguments": {"path": "alpha.py", "content": "a = 1\n"}},
                {"tool": "run_shell", "arguments": {"command": [sys.executable, "-c", "open('stray.txt','w').write('x')"]}},
            ],
            "done": True,
        }
        implementer = _by_title({"Implement alpha": sneaky, "Implement beta": _write("beta.py"), "Validate alpha and beta": _write("tests/test_ab.py")})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = self._service(base, PlanGateway(tasks, implementer))
            project = service.create_project(name="Unclaimed", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add two modules", max_iterations=2, completion={})
            run = service.wait_for_run(run["id"], timeout=60)
            events = service.events(run["id"])
        unclaimed = [event for event in events if event["type"] == "batch.unclaimed_writes"]
        self.assertTrue(unclaimed, "a write no member recorded must be caught at batch merge")
        self.assertIn("stray.txt", unclaimed[0]["payload"]["files"])
        self.assertNotEqual("completed", run["status"])

    def test_failing_check_reopens_only_the_task_whose_code_it_names(self) -> None:
        tasks = [
            {"id": "T1", "title": "Implement alpha", "agent": "programmer", "depends_on": [], "files": []},
            {"id": "T2", "title": "Implement beta", "agent": "programmer", "depends_on": [], "files": []},
            {"id": "T3", "title": "Validate alpha and beta", "agent": "tester", "depends_on": ["T1", "T2"], "files": []},
        ]
        implementer = _by_title({
            "Implement alpha": _write("alpha.py"),
            "Implement beta": _write("beta.py"),
            "Validate alpha and beta": _validate(),
        })
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = self._service(base, PlanGateway(tasks, implementer))
            project = service.create_project(name="Attribution", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add two modules", max_iterations=12,
                                     completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}})
            run = service.wait_for_run(run["id"], timeout=60)
            self.assertEqual("completed", run["status"], run.get("error"))
            with service.store.session() as connection:
                rows = {row["title"]: row["id"] for row in connection.execute("SELECT id, title FROM codebase_tasks WHERE run_id = ?", (run["id"],)).fetchall()}
            alpha, beta, validate = rows["Implement alpha"], rows["Implement beta"], rows["Validate alpha and beta"]

            self.assertEqual([], service._tasks_implicated_by_failure(run["id"], "Traceback: unrelated module"))
            self.assertEqual([alpha], service._tasks_implicated_by_failure(run["id"], 'File "alpha.py", line 3, in <module>'))
            self.assertEqual([], service._tasks_implicated_by_failure(run["id"], "alpha.pyc is not a file we wrote"))

            service._reopen_implicated(run["id"], [alpha], reason="test")
            with service.store.session() as connection:
                statuses = {row["id"]: row["status"] for row in connection.execute("SELECT id, status FROM codebase_tasks WHERE run_id = ?", (run["id"],)).fetchall()}
            # The implicated task stays completed (the repair task fixes its code); only its dependant is replanned.
            self.assertEqual("completed", statuses[alpha])
            self.assertEqual("completed", statuses[beta])
            self.assertEqual("pending", statuses[validate])

    def test_model_stall_parks_the_task_while_other_work_remains(self) -> None:
        tasks = [
            {"id": "T1", "title": "Implement alpha", "agent": "programmer", "depends_on": [], "files": []},
            {"id": "T2", "title": "Implement beta", "agent": "programmer", "depends_on": [], "files": []},
            {"id": "T3", "title": "Validate alpha and beta", "agent": "tester", "depends_on": ["T1", "T2"], "files": []},
        ]
        implementer = _by_title({
            "Implement alpha": {"summary": "nothing", "actions": [], "done": False},
            "Implement beta": _write("beta.py"),
        })
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = self._service(base, PlanGateway(tasks, implementer))
            project = service.create_project(name="Stall", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add two modules", max_iterations=24, completion={})
            run = service.wait_for_run(run["id"], timeout=90)
            events = service.events(run["id"])
        types = [event["type"] for event in events]
        self.assertIn("task.deferred", types, "a stalled task must be parked while other work is ready")
        parked_at = types.index("task.deferred")
        self.assertTrue(any(event["type"] == "task.transition" and event["payload"].get("status") == "completed" for event in events[parked_at:]))
        self.assertEqual("failed", run["status"])

    def test_each_completed_task_leaves_a_checkpoint_snapshot(self) -> None:
        tasks = [
            {"id": "T1", "title": "Implement alpha", "agent": "programmer", "depends_on": [], "files": []},
            {"id": "T2", "title": "Validate alpha", "agent": "tester", "depends_on": ["T1"], "files": []},
        ]
        implementer = _by_title({"Implement alpha": _write("alpha.py"), "Validate alpha": _validate()})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = self._service(base, PlanGateway(tasks, implementer))
            project = service.create_project(name="Checkpoint", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add alpha", max_iterations=6, completion={})
            run = service.wait_for_run(run["id"], timeout=60)
            events = service.events(run["id"])
        labels = [event["payload"].get("label", "") for event in events if event["type"] == "snapshot.created"]
        self.assertTrue(any(label.startswith("Task checkpoint") for label in labels), labels)


if __name__ == "__main__":
    unittest.main()
