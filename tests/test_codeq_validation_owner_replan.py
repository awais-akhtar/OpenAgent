"""A validation-only task with no owner for its failing check gets a repair task instead of looping."""

from __future__ import annotations

import json
import tempfile
import sys
import unittest
from pathlib import Path

from openagent.codebase.service import CodebaseService
from openagent.model_gateway import ModelResult
from openagent.storage import OpenAgentStore


class _ProjectMapLoopGateway:
    def complete(self, messages, **_kwargs):
        system = str(messages[0].get("content", "")).lower()
        prompt = str(messages[1].get("content", "")).casefold()
        if "specialist in a software engineering team" in system:
            value = {"summary": "The change has a single implementation and local test boundary.", "components": [], "risks": [], "acceptance_criteria": [], "recommended_tasks": []}
        elif "project manager" in system:
            value = {"tasks": [
                {"id": "T1", "title": "Implement a small feature", "agent": "programmer", "depends_on": [], "files": ["feature.py"]},
                {"id": "T2", "title": "Run tests", "agent": "tester", "depends_on": ["T1"], "files": []},
            ]}
        elif "current task: implement a small feature" in prompt:
            value = {"summary": "Implemented the feature", "actions": [
                {"tool": "write_file", "arguments": {"path": "feature.py", "content": "value = 1\n"}},
            ], "done": True}
        elif "current task: run tests" in prompt:
            value = {"summary": "I inspected the repository map", "actions": [
                {"tool": "project_map", "arguments": {}},
            ], "done": True}
        elif "independent code reviewer" in system:
            value = {"status": "passed", "findings": []}
        elif "final acceptance auditor" in system:
            value = {"status": "passed", "missing_tasks": [], "evidence": []}
        else:
            value = {"status": "passed", "findings": []}
        return ModelResult("fixture", "fixture-model", __import__("json").dumps(value), {})


class _ConfiguredCheckGateway:
    def __init__(self) -> None:
        self.roles: list[str] = []
        self.prompts: list[str] = []

    def complete(self, messages, role="", **_kwargs):
        self.roles.append(str(role))
        system = str(messages[0].get("content", "")).lower()
        prompt = str(messages[1].get("content", "")).lower() if len(messages) > 1 else ""
        self.prompts.append(prompt)
        if "specialist in a software engineering team" in system:
            value = {"summary": "The change has a single implementation and local test boundary.", "components": [], "risks": [], "acceptance_criteria": [], "recommended_tasks": []}
        elif "project manager" in system:
            value = {"tasks": [
                {"id": "T1", "title": "Implement add helper and its test", "agent": "programmer", "depends_on": [], "files": ["feature.py", "test_feature.py"]},
                {"id": "T2", "title": "Run tests", "agent": "tester", "depends_on": ["T1"], "files": []},
            ]}
        elif "independent code reviewer" in system:
            value = {"status": "passed", "findings": []}
        elif "final acceptance auditor" in system:
            value = {"status": "passed", "missing_tasks": [], "evidence": []}
        elif "current task" in prompt and "implement" in prompt:
            value = {"summary": "Implemented add and a focused unittest.", "actions": [
                {"tool": "write_file", "arguments": {"path": "feature.py", "content": "def add(left, right):\n    return left + right\n"}},
                {"tool": "write_file", "arguments": {"path": "test_feature.py", "content": "import unittest\nfrom feature import add\n\nclass AddTests(unittest.TestCase):\n    def test_adds_two_values(self):\n        self.assertEqual(add(2, 3), 5)\n\nif __name__ == '__main__':\n    unittest.main()\n"}},
            ], "done": True}
        else:
            raise AssertionError(f"unexpected model call for role {role!r}; system={system[:120]!r}; prompt={prompt[:240]!r}")
        return ModelResult("fixture", "fixture-model", __import__("json").dumps(value), {})


class ValidationOwnerTests(unittest.TestCase):
    def test_tester_fallback_uses_the_run_completion_command_when_map_is_empty(self) -> None:
        action = CodebaseService._configured_validation_action(
            {"completion": {"test": {"enabled": True, "command": ["python", "-m", "unittest", "discover", "-v"], "timeout_seconds": 300}}},
            {"commands": {}},
        )
        self.assertEqual({
            "tool": "run_tests",
            "arguments": {"command": ["python", "-m", "unittest", "discover", "-v"], "timeout_seconds": 300},
        }, action)

    def test_tester_fallback_uses_detected_repository_validation(self) -> None:
        action = CodebaseService._configured_validation_action(
            {"completion": {}},
            {"commands": {"test": "python -m pytest"}},
        )
        self.assertEqual({"tool": "run_tests", "arguments": {"command": "python -m pytest"}}, action)

    def test_tester_fallback_does_not_invent_a_validation_command(self) -> None:
        self.assertIsNone(CodebaseService._configured_validation_action(
            {"completion": {}}, {"commands": {}}
        ))

    def test_validation_only_tester_uses_configured_command_without_model_turn(self) -> None:
        action = CodebaseService._configured_validation_task_action(
            {"completion": {"test": {"enabled": True, "command": ["node", "--test"]}}},
            {"commands": {}},
            {"assigned_agent": "tester", "title": "Run Node tests to validate the feature", "done_predicates": []},
        )
        self.assertEqual({"tool": "run_tests", "arguments": {"command": ["node", "--test"]}}, action)

    def test_tester_file_creation_still_requires_model(self) -> None:
        self.assertIsNone(CodebaseService._configured_validation_task_action(
            {"completion": {"test": {"enabled": True, "command": ["node", "--test"]}}},
            {"commands": {"test": "node --test"}},
            {"assigned_agent": "tester", "title": "Create focused tests for the feature", "done_predicates": []},
        ))

    def test_implementation_map_noop_reads_declared_source_and_requests_a_real_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            (source / "src").mkdir(parents=True)
            (source / "src" / "slug.js").write_text(
                'export function slugify(value) { return String(value).toLowerCase(); }\n',
                encoding="utf-8",
            )
            service = CodebaseService(OpenAgentStore(Path(directory) / "data"), object())  # type: ignore[arg-type]
            project = service.create_project(name="No-op recovery", source_path=str(source))
            service._model_json = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
                "summary": "Mapped the project", "actions": [{"tool": "project_map", "arguments": {}}], "done": False,
            }
            service._event = lambda *_args, **_kwargs: "event"  # type: ignore[method-assign]
            run = {
                "id": "no-op-recovery", "codebase_project_id": project["id"], "user_prompt": "Implement slug normalization",
                "provider": "fixture", "model": "fixture-model", "allow_external": False,
            }
            task = {
                "id": "T1", "title": "Implement slug normalization", "assigned_agent": "programmer",
                "declared_files": ["src/slug.js"], "code_paths": ["src/slug.js"], "result": {},
            }
            decision = service._model_actions(
                run, project, project["map"], [{"path": "src/slug.js", "content": "export function slugify(value) { return String(value).toLowerCase(); }"}], task, "",
            )
            self.assertEqual("read_files", decision["actions"][0]["tool"])
            self.assertEqual(["src/slug.js"], decision["actions"][0]["arguments"]["paths"])
            self.assertIn("do not repeat the inspection action", decision["summary"].lower())

    def test_tester_inspection_response_is_replaced_by_configured_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "app.py").write_text("def app():\n    return True\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(Path(directory) / "data"), object())  # type: ignore[arg-type]
            project = service.create_project(name="tester fallback", source_path=str(source))
            service._model_json = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
                "summary": "I inspected the map", "actions": [{"tool": "project_map", "arguments": {}}], "done": True,
            }
            service._event = lambda *_args, **_kwargs: "event"  # type: ignore[method-assign]
            run = {
                "id": "test-run", "codebase_project_id": project["id"], "user_prompt": "Improve the local application", "completion": {
                    "test": {"enabled": True, "command": ["python", "-m", "unittest", "discover"]},
                },
            }
            task = {"id": "T3", "title": "Run project validation and repair failures", "assigned_agent": "tester"}
            decision = service._model_actions(run, project, {"commands": {}}, [], task, "")
            self.assertEqual("run_tests", decision["actions"][0]["tool"])
            self.assertEqual(["python", "-m", "unittest", "discover"], decision["actions"][0]["arguments"]["command"])
            self.assertTrue(decision["done"])

    def test_tester_without_any_check_is_blocked_instead_of_repeating_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "app.py").write_text("value = 1\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(Path(directory) / "data"), object())  # type: ignore[arg-type]
            project = service.create_project(name="missing check", source_path=str(source))
            service._model_json = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
                "summary": "I inspected the map", "actions": [{"tool": "project_map", "arguments": {}}], "done": True,
            }
            service._event = lambda *_args, **_kwargs: "event"  # type: ignore[method-assign]
            run = {"id": "test-run", "codebase_project_id": project["id"], "user_prompt": "Improve the local application", "completion": {}}
            task = {"id": "T3", "title": "Run project validation and repair failures", "assigned_agent": "tester"}
            decision = service._model_actions(run, project, {"commands": {}}, [], task, "")
            self.assertEqual([], decision["actions"])
            self.assertIn("blocked_reason", decision)

    def test_project_map_only_tester_response_blocks_run_without_repeating_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "app.py").write_text("value = 0\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(Path(directory) / "data"), _ProjectMapLoopGateway())  # type: ignore[arg-type]
            project = service.create_project(name="map-only tester", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Update the existing app with a small feature and verify it.",
                provider="fixture", model="fixture-model", max_iterations=12,
                completion={"test": {"enabled": False}},
            )
            finished = service.wait_for_run(run["id"], timeout=30)
            self.assertEqual("blocked", finished["status"], finished.get("error"))
            self.assertIn("No validation command is configured", finished["error"])
            tool_events = [item for item in service.events(run["id"]) if item.get("type") == "tool.started"]
            self.assertFalse(any(item.get("payload", {}).get("tool") == "project_map" for item in tool_events))
            self.assertEqual(1, sum(item.get("type") == "run.blocked" for item in service.events(run["id"])))

    def test_configured_tester_executes_real_check_without_a_model_noop_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            gateway = _ConfiguredCheckGateway()
            store = OpenAgentStore(Path(directory) / "data")
            service = CodebaseService(store, gateway)  # type: ignore[arg-type]
            project = service.create_project(name="direct validation", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"],
                prompt="Implement an add helper and verify it with a focused test.",
                provider="fixture", model="fixture-model", max_iterations=10,
                completion={"test": {"enabled": True, "command": [sys.executable, "-m", "unittest", "discover", "-v"]}},
                acceptance_criteria=["add returns the sum of two numbers", "the focused unittest passes"],
            )
            finished = service.wait_for_run(run["id"], timeout=30)
            self.assertEqual("completed", finished["status"], finished.get("error"))
            self.assertFalse(any("current task" in prompt and "run tests" in prompt for prompt in gateway.prompts))
            events = service.events(run["id"])
            self.assertTrue(any(item.get("type") == "tester.validation.started" for item in events))
            with store.session() as connection:
                check = connection.execute(
                    "SELECT status, output_json FROM codebase_tool_calls WHERE run_id = ? AND tool = 'run_tests' ORDER BY started_at DESC LIMIT 1",
                    (run["id"],),
                ).fetchone()
            self.assertIsNotNone(check)
            self.assertEqual("completed", check["status"])
            self.assertEqual(0, json.loads(check["output_json"]).get("exit_code"))

    def test_validation_shell_action_is_not_confused_with_project_map(self) -> None:
        self.assertTrue(CodebaseService._is_validation_action({
            "tool": "run_shell", "arguments": {"command": ["python", "-m", "unittest", "discover"]},
        }))
        self.assertFalse(CodebaseService._is_validation_action({"tool": "project_map", "arguments": {}}))

    def test_skipped_check_is_not_successful_validation_evidence(self) -> None:
        task = {"done_predicates": [{"kind": "successful_validation", "tools": ["run_tests"]}]}
        self.assertIn(
            "run and pass",
            CodebaseService._task_done_predicate_failure(task, [("run_tests", {"skipped": True, "exit_code": -1})]),
        )
        self.assertEqual(
            "",
            CodebaseService._task_done_predicate_failure(task, [("run_tests", {"exit_code": 0})]),
        )

    def test_validation_only_task_needs_an_owner(self) -> None:
        task = {"done_predicates": [{"kind": "successful_validation", "tools": ["run_tests"]}]}
        self.assertTrue(CodebaseService._validation_task_needs_owner(task))

    def test_task_that_edits_files_does_not_need_a_separate_owner(self) -> None:
        task = {"done_predicates": [
            {"kind": "declared_files_touched", "paths": ["app.py"]},
            {"kind": "successful_validation", "tools": ["run_tests"]},
        ]}
        self.assertFalse(CodebaseService._validation_task_needs_owner(task))

    def test_task_without_validation_predicate_is_not_affected(self) -> None:
        self.assertFalse(CodebaseService._validation_task_needs_owner({"done_predicates": [{"kind": "successful_task_evidence"}]}))
        self.assertFalse(CodebaseService._validation_task_needs_owner({"done_predicates": "invalid"}))
        self.assertFalse(CodebaseService._validation_task_needs_owner({}))


if __name__ == "__main__":
    unittest.main()
