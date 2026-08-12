from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from openagent.codebase.repository import RepositoryBoundaryError, RepositoryIntelligence
from openagent.codebase.sandbox import SandboxManager
from openagent.codebase.service import CodebaseService
from openagent.codebase.tools import ApprovalRequired, CodebaseTools, ToolPolicy
from openagent.model_gateway import ModelResult
from openagent.storage import OpenAgentStore


class FakeGateway:
    def complete(self, messages, **kwargs):
        system = messages[0]["content"]
        if "project manager" in system.lower():
            content = json.dumps({"goal": "change", "tasks": [{"id": "T1", "title": "Implement the change", "agent": "programmer", "depends_on": []}]})
        elif "independent code reviewer" in system.lower():
            content = json.dumps({"status": "passed", "findings": []})
        else:
            content = json.dumps({"summary": "added feature", "actions": [{"tool": "write_file", "arguments": {"path": "feature.py", "content": "VALUE = 42\n"}}], "done": True})
        return ModelResult("fake", "fake-code", content, {})


class RepairGateway:
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"]
        if "project manager" in system:
            content = json.dumps({"tasks": [{"id": "T1", "title": "Implement feature", "agent": "programmer", "depends_on": []}]})
        elif "independent code reviewer" in system:
            content = json.dumps({"status": "passed", "findings": []})
        elif "repair failed completion checks" in user.lower():
            content = json.dumps({"summary": "repaired syntax", "actions": [{"tool": "write_file", "arguments": {"path": "feature.py", "content": "VALUE = 42\n"}}], "done": True})
        else:
            content = json.dumps({"summary": "first implementation", "actions": [{"tool": "write_file", "arguments": {"path": "feature.py", "content": "VALUE = (\n"}}], "done": True})
        return ModelResult("fake", kwargs.get("model") or "fake-code", content, {})


class NeverDoneGateway(FakeGateway):
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        if "project manager" in system or "independent code reviewer" in system:
            return super().complete(messages, **kwargs)
        return ModelResult("fake", "fake-code", json.dumps({"summary": "still working", "actions": [{"tool": "list_directory", "arguments": {}}], "done": False}), {})


class InlineCodeRepairGateway(FakeGateway):
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "project manager" in system or "independent code reviewer" in system:
            return super().complete(messages, **kwargs)
        if "inline source code" in user:
            content = json.dumps({"summary": "write the function to a file", "actions": [{"tool": "write_file", "arguments": {"path": "function.py", "content": "def double(value):\n    return value * 2\n"}}], "done": True})
        else:
            content = json.dumps({"summary": "incorrect inline attempt", "actions": [{"tool": "run_shell", "arguments": {"command": "def double(value):\n    return value * 2"}}], "done": True})
        return ModelResult("fake", "fake-code", content, {})


class EmptyLoginPageGateway(FakeGateway):
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "independent code reviewer" in system:
            return ModelResult("fake", "review", json.dumps({"status": "passed", "findings": []}), {})
        if "semantic login page markup" in user:
            if "required argument(s) were missing" not in user:
                payload = {"summary": "inspect the page", "actions": [{"tool": "browser_inspect", "arguments": {}}], "done": True}
            else:
                payload = {"summary": "created login markup", "actions": [{"tool": "write_file", "arguments": {"path": "index.html", "content": "<!doctype html><html><head><link rel='stylesheet' href='styles.css'></head><body><main><form><h1>Sign in</h1><label>Email<input type='email' required></label><label>Password<input type='password' required></label><button>Sign in</button></form></main></body></html>"}}], "done": True}
        elif "responsive styling" in user:
            payload = {"summary": "added responsive styles", "actions": [{"tool": "write_file", "arguments": {"path": "styles.css", "content": "*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;font-family:system-ui;background:#f4f7fb}form{width:min(92vw,420px);display:grid;gap:1rem;padding:2rem;background:white;border:1px solid #d8dee8;border-radius:8px}label{display:grid;gap:.4rem}input,button{min-height:44px;font:inherit}button{color:white;background:#1663d6;border:0;border-radius:6px}@media(max-width:480px){form{padding:1.25rem}}"}}], "done": True}
        else:
            payload = {"summary": "validated generated files", "actions": [{"tool": "read_files", "arguments": {"paths": ["index.html", "styles.css"]}}], "done": True}
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class RepositoryIntelligenceTests(unittest.TestCase):
    def test_scan_context_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "app.py").write_text("from service import login\n\ndef password_reset():\n    return login()\n", encoding="utf-8")
            (root / "service.py").write_text("def login():\n    return True\n", encoding="utf-8")
            (root / "requirements.txt").write_text("fastapi==1.0\n", encoding="utf-8")
            (root / "schema.prisma").write_text("model User {\n  id Int @id\n}\n", encoding="utf-8")
            (root / "test_app.py").write_text("def test_login():\n    assert True\n", encoding="utf-8")
            repo = RepositoryIntelligence(root)
            project_map = repo.scan()
            self.assertIn("python", project_map["languages"])
            self.assertTrue(any(item["name"] == "password_reset" for item in project_map["symbols"]))
            self.assertIn("fastapi", project_map["dependencies"]["python"])
            self.assertTrue(any(item["name"] == "User" for item in project_map["database_models"]))
            self.assertTrue(any(item["symbol"] == "password_reset" and "login" in item["calls"] for item in project_map["call_graph"]))
            self.assertIn("test_app.py", project_map["test_files"])
            self.assertEqual("app.py", repo.select_context("password reset")[0]["path"])
            with self.assertRaises(RepositoryBoundaryError):
                repo.resolve("../outside.txt")


class SandboxAndPolicyTests(unittest.TestCase):
    def test_workspace_is_copied_and_snapshot_restores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "value.txt").write_text("before", encoding="utf-8")
            manager = SandboxManager(base / "sandboxes")
            manager.create("workspace-1", source=source)
            tools = CodebaseTools(manager, "workspace-1")
            tools.execute("write_file", {"path": "value.txt", "content": "after"})
            snapshot = manager.snapshot("workspace-1", "after-change")
            tools.execute("write_file", {"path": "value.txt", "content": "broken"})
            manager.restore("workspace-1", snapshot["path"])
            self.assertEqual("after", (manager.path("workspace-1") / "value.txt").read_text(encoding="utf-8"))
            self.assertEqual("before", (source / "value.txt").read_text(encoding="utf-8"))

    def test_risk_policy_blocks_push_and_asks_for_install(self) -> None:
        policy = ToolPolicy()
        self.assertEqual(3, policy.classify("git_push", {})["risk"])
        inline = policy.classify("run_shell", {"command": "def double(value):\n    return value * 2"})
        self.assertEqual(3, inline["risk"])
        self.assertIn("write_file", inline["reason"])
        self.assertEqual(2, policy.classify("run_shell", {"command": ["npm", "install"]})["risk"])
        missing_browser = policy.classify("browser_inspect", {})
        self.assertEqual(3, missing_browser["risk"])
        self.assertIn("complete HTTP URL", missing_browser["reason"])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            manager = SandboxManager(base / "sandboxes")
            manager.create("workspace-1", source=source)
            tools = CodebaseTools(manager, "workspace-1")
            with self.assertRaises(ApprovalRequired):
                tools.execute("install_dependency", {"command": ["npm", "install", "x"]})
            (manager.path("workspace-1") / ".env").write_text("TOKEN=private", encoding="utf-8")
            with self.assertRaises(PermissionError):
                tools.execute("read_file", {"path": ".env"})
            with self.assertRaisesRegex(PermissionError, "inline source code"):
                tools.execute("run_shell", {"command": "import inspect; print(inspect.getsourcefile())"})

    @unittest.skipUnless(os.name == "nt" and shutil.which("npm.cmd"), "Windows npm command is not available")
    def test_windows_command_runner_resolves_npm_cmd_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            manager = SandboxManager(base / "sandboxes")
            manager.create("npm", source=source)
            result = manager.run("npm", ["npm", "--version"])
            self.assertEqual(0, result.exit_code, result.stderr)

    def test_managed_process_has_status_logs_and_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            manager = SandboxManager(base / "sandboxes")
            manager.create("workspace-1", source=source)
            process = manager.start_process("workspace-1", [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(5)"])
            time.sleep(0.2)
            logs = manager.process_logs("workspace-1", process["id"])
            self.assertIn("ready", logs["stdout"])
            stopped = manager.stop_process("workspace-1", process["id"])
            self.assertEqual("exited", stopped["status"])

    def test_configured_hosted_adapter_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            adapter = base / "adapter.py"
            adapter.write_text(
                "import json,sys\np=json.load(sys.stdin)\nprint(json.dumps({'exit_code':0,'stdout':p['protocol']+' '+p['workspace_id'],'stderr':'','duration_ms':1}))\n",
                encoding="utf-8",
            )
            previous = os.environ.get("OPENAGENT_HOSTED_SANDBOX_COMMAND")
            os.environ["OPENAGENT_HOSTED_SANDBOX_COMMAND"] = f'"{sys.executable}" "{adapter}"'
            try:
                manager = SandboxManager(base / "sandboxes")
                manager.create("workspace-1", source=source, backend="hosted")
                result = manager.run("workspace-1", ["python", "-V"], backend="hosted")
                self.assertEqual(0, result.exit_code)
                self.assertIn("openagent.sandbox.v1", result.stdout)
            finally:
                if previous is None:
                    os.environ.pop("OPENAGENT_HOSTED_SANDBOX_COMMAND", None)
                else:
                    os.environ["OPENAGENT_HOSTED_SANDBOX_COMMAND"] = previous


class CodebaseServiceTests(unittest.TestCase):
    def test_native_folder_picker_returns_resolved_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = CodebaseService(OpenAgentStore(Path(directory) / "data"), FakeGateway())  # type: ignore[arg-type]
            fake_root = MagicMock()
            fake_tk = MagicMock(return_value=fake_root)
            with patch("tkinter.Tk", fake_tk), patch(
                "tkinter.filedialog.askdirectory", return_value=directory,
            ):
                selected = service.pick_source_folder()
            self.assertEqual(str(Path(directory).resolve()), selected)
            fake_root.destroy.assert_called_once()

    def test_run_persists_tasks_events_changes_validation_and_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "main.py").write_text("print('ready')\n", encoding="utf-8")
            store = OpenAgentStore(base / "data")
            service = CodebaseService(store, FakeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Demo", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"],
                prompt="Add a feature",
                completion={"runtime": {"enabled": True, "command": ["python", "-m", "compileall", "-q", "."]}},
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                run = service.get_run(run["id"])
                if run["status"] in {"completed", "failed", "cancelled", "blocked"}:
                    break
                time.sleep(0.05)
            run = service.wait_for_run(run["id"], timeout=5)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertEqual("completed", run["tasks"][0]["status"])
            self.assertTrue(run["validation"])
            self.assertGreaterEqual(len(run["snapshots"]), 2)
            self.assertTrue((Path(project["workspace"]["workspace_path"]) / "feature.py").is_file())
            event_types = {event["type"] for event in service.events(run["id"])}
            self.assertIn("run.completed", event_types)
            self.assertIn("tool.completed", event_types)
            self.assertIn("agent.handoff", event_types)
            self.assertFalse((source / "feature.py").exists())

    def test_failed_validation_creates_repair_task_and_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "main.py").write_text("print('ready')\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), RepairGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Repair", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Add feature",
                max_iterations=8,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-m", "py_compile", "feature.py"]}},
                role_models={"programmer": {"model": "code-model"}, "reviewer": {"model": "review-model"}},
            )
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                run = service.get_run(run["id"])
                if run["status"] in {"completed", "failed", "cancelled", "blocked"}:
                    break
                time.sleep(0.05)
            run = service.wait_for_run(run["id"], timeout=5)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertTrue(any(task["assigned_agent"] == "debugger" for task in run["tasks"]))
            self.assertTrue(any(item["status"] == "failed" for item in run["validation"]))
            self.assertEqual("VALUE = 42\n", (Path(project["workspace"]["workspace_path"]) / "feature.py").read_text(encoding="utf-8"))

    def test_iteration_limit_cannot_complete_unfinished_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "main.py").write_text("print('ready')\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), NeverDoneGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Bounded", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Keep working", max_iterations=1)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                run = service.get_run(run["id"])
                if run["status"] in {"completed", "failed", "cancelled", "blocked"}:
                    break
                time.sleep(0.05)
            run = service.wait_for_run(run["id"], timeout=5)
            self.assertEqual("failed", run["status"])
            self.assertIn("Iteration limit", run["error"])
            self.assertNotEqual("completed", run["tasks"][0]["status"])

    def test_interactive_tool_session_is_created_before_agent_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "main.py").write_text("print('ready')\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), FakeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Interactive", source_path=str(source))
            response = service.invoke_project_tool(project["id"], "run_shell", {"command": [sys.executable, "main.py"]})
            self.assertEqual("interactive", response["run"]["status"])
            self.assertEqual(0, response["result"]["exit_code"])
            self.assertIn("ready", response["result"]["stdout"])
            self.assertIn("tool.completed", {item["type"] for item in service.events(response["run"]["id"])})

    def test_failed_approved_executable_returns_repair_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), FakeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Approval", source_path=str(source))
            pending = service.invoke_project_tool(project["id"], "run_shell", {"command": "missing-openagent-executable"})
            approval_id = pending["result"]["approval_required"]["id"]
            resolution = service.resolve_approval(approval_id, approved=True)
            self.assertEqual("approved", resolution["status"])
            self.assertTrue(resolution["result"]["queued"])
            duplicate = service.resolve_approval(approval_id, approved=True)
            self.assertTrue(duplicate["result"]["already_resolved"])
            deadline = time.monotonic() + 5
            approval = {}
            while time.monotonic() < deadline:
                approval = next(item for item in service.get_run(pending["run"]["id"])["approvals"] if item["id"] == approval_id)
                if approval["status"] == "failed":
                    break
                time.sleep(0.02)
            self.assertEqual("failed", approval["status"])
            with service.store.session() as connection:
                tool_call_count = connection.execute(
                    "SELECT COUNT(*) FROM codebase_tool_calls WHERE run_id = ? AND tool = 'run_shell'",
                    (pending["run"]["id"],),
                ).fetchone()[0]
            # One policy-gated attempt plus one approved execution. The repeated
            # resolve call must not create a third tool invocation.
            self.assertEqual(2, tool_call_count)

    def test_empty_login_project_repairs_missing_browser_url_and_creates_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), EmptyLoginPageGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Login", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Make a responsive login page", max_iterations=8)
            run = service.wait_for_run(run["id"], timeout=15)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertEqual([], run["approvals"])
            workspace = Path(project["workspace"]["workspace_path"])
            self.assertIn("Sign in", (workspace / "index.html").read_text(encoding="utf-8"))
            self.assertIn("@media", (workspace / "styles.css").read_text(encoding="utf-8"))
            event_types = {item["type"] for item in service.events(run["id"])}
            self.assertIn("model.action_replaced", event_types)

    def test_inline_python_is_self_corrected_without_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), InlineCodeRepairGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Function", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"], prompt="Write a Python function that doubles a number", max_iterations=6,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-m", "py_compile", "function.py"]}},
            )
            run = service.wait_for_run(run["id"], timeout=15)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertEqual([], run["approvals"])
            function_file = Path(project["workspace"]["workspace_path"]) / "function.py"
            self.assertIn("def double", function_file.read_text(encoding="utf-8"))
            self.assertIn("repair.requested", {item["type"] for item in service.events(run["id"])})


if __name__ == "__main__":
    unittest.main()
