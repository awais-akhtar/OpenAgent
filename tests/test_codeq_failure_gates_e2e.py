"""Phase 4 exit gate, end to end: runs that must never end as completed.

Each test drives the real CodeQ run lifecycle with a scripted model and a real
local command. These cover the cases the roadmap names: forged done, failed
command, missing layer, and no executed evidence.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from openagent.codebase.service import CodebaseService
from openagent.model_gateway import ModelResult
from openagent.storage import OpenAgentStore


def _model(content: dict) -> ModelResult:
    return ModelResult("fake", "fake-code", json.dumps(content), {})


class ScriptedGateway:
    """Planner and reviewer answer normally; the implementation step is scripted by `implementer`."""

    def __init__(self, implementer, audit_missing: list[dict] | None = None) -> None:
        self.implementer = implementer
        self.audit_missing = audit_missing or []

    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        if "project manager" in system:
            return _model({"goal": "change", "tasks": [{"id": "T1", "title": "Implement the change", "agent": "programmer", "depends_on": []}]})
        if "independent code reviewer" in system:
            return _model({"status": "passed", "findings": []})
        if "final acceptance auditor" in system:
            if self.audit_missing:
                return _model({"status": "needs_work", "missing_tasks": self.audit_missing, "evidence": []})
            return _model({"status": "passed", "missing_tasks": [], "evidence": []})
        return _model(self.implementer(messages))


def _writes_feature(_messages) -> dict:
    return {"summary": "added feature", "actions": [{"tool": "write_file", "arguments": {"path": "feature.py", "content": "VALUE = 42\n"}}], "done": True}


def _writes_notes(_messages) -> dict:
    # A plain text file: no build, test or lint check is discoverable for it.
    return {"summary": "wrote notes", "actions": [{"tool": "write_file", "arguments": {"path": "notes.txt", "content": "hello\n"}}], "done": True}


def _forged_done(_messages) -> dict:
    # The model claims done and writes nothing: no evidence exists for it.
    return {"summary": "all done", "actions": [], "done": True}


class Phase4EndToEndTests(unittest.TestCase):
    def _run(self, gateway, *, completion=None, max_iterations: int = 1) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), gateway)  # type: ignore[arg-type]
            project = service.create_project(name="Feature", source_path=str(source))
            kwargs = {"max_iterations": max_iterations}
            if completion is not None:
                kwargs["completion"] = completion
            run = service.create_run(codebase_project_id=project["id"], prompt="Add a feature", **kwargs)
            return service.wait_for_run(run["id"], timeout=30)

    def test_forged_done_without_evidence_never_completes(self) -> None:
        run = self._run(ScriptedGateway(_forged_done), completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}})
        self.assertNotEqual("completed", run["status"])

    def test_failed_command_never_completes(self) -> None:
        failing = {"runtime": {"enabled": True, "command": [sys.executable, "-c", "import sys; sys.exit(3)"]}}
        run = self._run(ScriptedGateway(_writes_feature), completion=failing)
        self.assertNotEqual("completed", run["status"])
        self.assertIn(run["status"], {"failed", "blocked"})

    def test_missing_layer_never_completes(self) -> None:
        # The audit finds a requested backend layer absent: the run must not complete.
        gateway = ScriptedGateway(_writes_feature, audit_missing=[{"title": "Add the backend API", "agent": "backend"}])
        run = self._run(gateway, completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}})
        self.assertNotEqual("completed", run["status"])

    def test_no_executed_check_never_completes(self) -> None:
        # Files are written but no build, test or runtime command is configured.
        run = self._run(ScriptedGateway(_writes_notes), completion={}, max_iterations=8)
        self.assertNotEqual("completed", run["status"])
        self.assertIn(run["status"], {"blocked", "failed"})
        self.assertTrue(run.get("error"))


if __name__ == "__main__":
    unittest.main()
