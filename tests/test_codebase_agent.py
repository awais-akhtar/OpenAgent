from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from openagent.codebase.repository import RepositoryBoundaryError, RepositoryIntelligence
from openagent.codebase.profiles import check_artifact_contract, classify_request, task_needs_file_change
from openagent.codebase.sandbox import SandboxManager
from openagent.codebase.service import CodebaseService
from openagent.codebase.tools import CodebaseTools, ToolPolicy
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


class WritesWithoutDoneGateway(FakeGateway):
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        if "project manager" in system or "independent code reviewer" in system or "final acceptance auditor" in system:
            return super().complete(messages, **kwargs)
        payload = {
            "summary": "wrote the requested implementation",
            "actions": [{"tool": "write_file", "arguments": {"path": "feature.py", "content": "VALUE = 42\n"}}],
            "done": False,
        }
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


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


class StaticSiteGateway(FakeGateway):
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        if "independent code reviewer" in system:
            payload = {"status": "passed", "findings": []}
        elif "final acceptance auditor" in system:
            payload = {"status": "passed", "missing_tasks": [], "evidence": ["four linked pages and assets verified"]}
        else:
            payload = {"summary": "attempted an invalid browser action", "actions": [{"tool": "browser_inspect", "arguments": {}}], "done": True}
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class NarratingPythonGateway(FakeGateway):
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "independent code reviewer" in system:
            return ModelResult("fake", "review", json.dumps({"status": "passed", "findings": []}), {})
        if "agent role: tester" in user:
            payload = {"summary": "run selected files", "actions": [{"tool": "run_tests", "arguments": {"files": ["test_function.py"]}}], "done": False}
        else:
            payload = {
                "summary": "describe work instead of doing it",
                "actions": [{"tool": "start_process", "arguments": {"command": ["echo", "Creating a factorial function"]}}],
                "done": True,
            }
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class ApprovalLoopGateway(FakeGateway):
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "project manager" in system or "independent code reviewer" in system:
            return super().complete(messages, **kwargs)
        if "outside the protected autonomous boundary" in user:
            payload = {"summary": "used a workspace-safe implementation", "actions": [{"tool": "write_file", "arguments": {"path": "feature.py", "content": "VALUE = 42\n"}}], "done": True}
        else:
            payload = {"summary": "attempted an unavailable helper", "actions": [{"tool": "run_shell", "arguments": {"command": ["missing-codeq-helper", "generate"]}}], "done": True}
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class FastApiCreationGateway(FakeGateway):
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "independent code reviewer" in system:
            payload = {"status": "passed", "findings": []}
        elif "foundation" in user:
            payload = {"summary": "created service foundation", "actions": [
                {"tool": "write_file", "arguments": {"path": "pyproject.toml", "content": "[project]\nname='inventory-service'\nversion='0.1.0'\nrequires-python='>=3.11'\n"}},
                {"tool": "write_file", "arguments": {"path": "main.py", "content": "from inventory import create_item\n\nif __name__ == '__main__':\n    print(create_item('demo'))\n"}},
            ], "done": True}
        elif "complete user-requested behavior" in user:
            payload = {"summary": "implemented inventory behavior", "actions": [{"tool": "write_file", "arguments": {"path": "inventory.py", "content": "def create_item(name: str) -> dict[str, str]:\n    value = name.strip()\n    if not value:\n        raise ValueError('name is required')\n    return {'name': value}\n"}}], "done": True}
        elif "add focused automated tests" in user:
            payload = {"summary": "added service tests", "actions": [{"tool": "write_file", "arguments": {"path": "test_inventory.py", "content": "import unittest\nfrom inventory import create_item\n\nclass InventoryTests(unittest.TestCase):\n    def test_create(self):\n        self.assertEqual({'name': 'desk'}, create_item(' desk '))\n\nif __name__ == '__main__':\n    unittest.main()\n"}}], "done": True}
        else:
            payload = {"summary": "validated service", "actions": [{"tool": "run_tests", "arguments": {"command": ["python", "-m", "unittest", "discover", "-v"]}}], "done": True}
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class ModelSelectedNodeGateway(FakeGateway):
    """Chooses a non-React stack for a request that names no framework."""

    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "project manager" in system:
            payload = {"tasks": [
                {"id": "T1", "title": "Create the Node command-line project foundation", "agent": "programmer", "depends_on": []},
                {"id": "T2", "title": "Implement inventory item normalization", "agent": "programmer", "depends_on": ["T1"]},
                {"id": "T3", "title": "Test and validate the command-line behavior", "agent": "tester", "depends_on": ["T2"]},
            ]}
        elif "independent code reviewer" in system:
            payload = {"status": "passed", "findings": []}
        elif "final acceptance auditor" in system:
            payload = {"status": "passed", "missing_tasks": [], "evidence": ["generated tests passed"]}
        elif "project foundation" in user:
            payload = {"summary": "created the Node project", "actions": [
                {"tool": "write_file", "arguments": {"path": "package.json", "content": "{\n  \"name\": \"inventory-cli\",\n  \"private\": true,\n  \"type\": \"module\",\n  \"scripts\": {\"test\": \"node --test\"}\n}\n"}},
                {"tool": "write_file", "arguments": {"path": "src/index.js", "content": "export { normalizeItem } from './inventory.js';\n"}},
            ], "done": True}
        elif "item normalization" in user:
            payload = {"summary": "implemented inventory behavior", "actions": [
                {"tool": "write_file", "arguments": {"path": "src/inventory.js", "content": "export function normalizeItem(name) {\n  const value = String(name).trim();\n  if (!value) throw new Error('name is required');\n  return { name: value };\n}\n"}},
            ], "done": True}
        else:
            payload = {"summary": "tested inventory behavior", "actions": [
                {"tool": "write_file", "arguments": {"path": "test/inventory.test.js", "content": "import test from 'node:test';\nimport assert from 'node:assert/strict';\nimport { normalizeItem } from '../src/inventory.js';\ntest('normalizes an item', () => assert.deepEqual(normalizeItem(' desk '), { name: 'desk' }));\n"}},
                {"tool": "run_tests", "arguments": {"command": ["npm", "test"]}},
            ], "done": True}
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class ParallelPlanningGateway(FakeGateway):
    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "project manager" in system:
            payload = {"tasks": [
                {"id": "T1", "title": "Implement backend module", "agent": "backend", "depends_on": []},
                {"id": "T2", "title": "Implement frontend module", "agent": "frontend", "depends_on": []},
            ]}
        elif "independent code reviewer" in system:
            payload = {"status": "passed", "findings": []}
        elif "final acceptance auditor" in system:
            payload = {"status": "passed", "missing_tasks": [], "evidence": ["validation passed"]}
        elif "backend module" in user:
            payload = {"summary": "backend complete", "actions": [{"tool": "write_file", "arguments": {"path": "backend.py", "content": "VALUE = 'backend'\n"}}], "done": True}
        else:
            payload = {"summary": "frontend complete", "actions": [{"tool": "write_file", "arguments": {"path": "frontend.js", "content": "export const ready = true;\n"}}], "done": True}
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class TitleDependencyPageUpdateGateway(FakeGateway):
    """Planner fixture matching title-based dependencies from a page-update request."""

    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        if "project manager" in system:
            payload = {
                "tasks": [
                    {"id": "T1", "title": "Review existing pages for updates", "agent": "programmer", "depends_on": ["T0"]},
                    {"id": "T2", "title": "Update templates and content for each page", "agent": "programmer", "depends_on": ["Review existing pages for updates"]},
                    {"id": "T3", "title": "Ensure all dependencies are up-to-date and tested", "agent": "tester", "depends_on": ["Update templates and content for each page"]},
                ],
            }
        elif "independent code reviewer" in system:
            payload = {"status": "passed", "findings": []}
        elif "final acceptance auditor" in system:
            payload = {"status": "passed", "missing_tasks": [], "evidence": ["page update and validation completed"]}
        elif "agent role: tester" in messages[-1]["content"].lower():
            payload = {
                "summary": "validated the updated page files",
                "actions": [{"tool": "run_tests", "arguments": {"command": ["python", "-m", "compileall", "-q", "."]}}],
                "done": True,
            }
        else:
            payload = {
                "summary": "updated page content",
                "actions": [{"tool": "write_file", "arguments": {"path": "pages-updated.txt", "content": "Updated page content and data.\n"}}],
                "done": True,
            }
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class MalformedVueMigrationGateway(FakeGateway):
    """Reproduces a local-model reply that previously froze a dependent plan."""

    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "project manager" in system:
            payload = {
                "tasks": [
                    {"id": "T1", "title": "Review architecture and dependencies for changes.", "agent": "programmer", "depends_on": []},
                    {"id": "T2", "title": "Update package.json to include new page files.", "agent": "programmer", "depends_on": ["T1"]},
                    {"id": "T3", "title": "Integrate new page into existing Vue.js template.", "agent": "programmer", "depends_on": ["T2"]},
                    {"id": "T4", "title": "Test new page functionality in both development and production environments.", "agent": "tester", "depends_on": ["T3"]},
                ],
            }
        elif "independent code reviewer" in system:
            payload = {"status": "passed", "findings": []}
        elif "final acceptance auditor" in system:
            payload = {"status": "passed", "missing_tasks": [], "evidence": ["Vue files and local checks completed"]}
        elif "review architecture" in user:
            # This is the malformed response stored in the user's failed run:
            # it describes an edit but names neither a file nor a CodeQ tool.
            payload = {"search": "home.html", "replace": "new_page.html"}
        elif "update package.json" in user:
            payload = {
                "summary": "created the Vue project files",
                "actions": [
                    {"tool": "write_file", "arguments": {"path": "package.json", "content": "{\n  \"name\": \"site-client\",\n  \"private\": true,\n  \"scripts\": {\"build\": \"vite build\"},\n  \"dependencies\": {\"vue\": \"^3.5.0\"}\n}\n"}},
                    {"tool": "write_file", "arguments": {"path": "index.html", "content": "<div id=\"app\"></div><script type=\"module\" src=\"/src/main.js\"></script>\n"}},
                    {"tool": "write_file", "arguments": {"path": "src/main.js", "content": "import { createApp } from 'vue';\nimport App from './App.vue';\ncreateApp(App).mount('#app');\n"}},
                ],
                "done": True,
            }
        elif "integrate new page" in user:
            payload = {
                "summary": "integrated the new Vue page",
                "actions": [
                    {"tool": "write_file", "arguments": {"path": "src/App.vue", "content": "<template><main><h1>Updated site</h1><p>Sample page content.</p></main></template>\n"}},
                    {"tool": "write_file", "arguments": {"path": "src/pages.js", "content": "export const pages = [{ title: 'Home', body: 'Sample home content.' }, { title: 'New page', body: 'Sample page content.' }];\n"}},
                ],
                "done": True,
            }
        else:
            payload = {
                "summary": "validated source files",
                "actions": [{"tool": "run_tests", "arguments": {"command": ["python", "-m", "compileall", "-q", "."]}}],
                "done": True,
            }
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class NewPageGateway(FakeGateway):
    """Model plan with several page preparation steps.

    The service should execute these steps against one generated artifact,
    even when the model gives each step a different implementation title.
    """

    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "project manager" in system:
            payload = {"tasks": [
                {"id": "T1", "title": "Configure the page builder", "agent": "programmer", "depends_on": []},
                {"id": "T2", "title": "Prepare dummy content for the new page", "agent": "programmer", "depends_on": ["T1"]},
                {"id": "T3", "title": "Inspect and validate the generated page", "agent": "tester", "depends_on": ["T2"]},
            ]}
        elif "independent code reviewer" in system:
            payload = {"status": "passed", "findings": []}
        elif "final acceptance auditor" in system:
            payload = {"status": "passed", "missing_tasks": [], "evidence": ["generated page was inspected"]}
        else:
            payload = {"summary": "model attempted a non-writing page action", "actions": [{"tool": "project_map", "arguments": {}}], "done": False}
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class DependencyOnlyPlanGateway(FakeGateway):
    """Return the malformed setup-only plan that previously caused a loop."""

    def complete(self, messages, **kwargs):
        system = messages[0]["content"].lower()
        if "project manager" in system:
            payload = {
                "tasks": [
                    {"id": "T1", "title": "Ensure all dependencies are installed and configured correctly", "agent": "tester", "depends_on": []},
                ],
            }
        else:
            return super().complete(messages, **kwargs)
        return ModelResult("fake", "fake-code", json.dumps(payload), {})


class RepositoryIntelligenceTests(unittest.TestCase):
    def test_task_classifier_keeps_validation_setup_out_of_file_edit_tasks(self) -> None:
        self.assertFalse(task_needs_file_change("Ensure all dependencies are installed and configured correctly", "tester"))
        self.assertTrue(task_needs_file_change("Update templates and content for each page", "programmer"))
        self.assertTrue(task_needs_file_change("Implement frontend module", "frontend"))

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

    def test_project_profiles_cover_multiple_product_and_stack_types(self) -> None:
        empty = {"file_count": 0, "languages": [], "framework": ""}
        cases = {
            "Build a FastAPI inventory service": ("service", "fastapi", "python"),
            "Create a Rust command line backup tool": ("command-line application", "rust", "rust"),
            "Make a Flutter mobile app": ("mobile application", "flutter", "dart"),
            "Build a desktop application with Electron": ("desktop application", "electron", "javascript"),
            "Create a Go web service": ("service", "go", "go"),
            "Create a Solidity smart contract with Foundry": ("application", "solidity", "solidity"),
            "Build a Ruby on Rails portal": ("web application", "rails", "ruby"),
            "Create a C++ command line tool with CMake": ("command-line application", "cmake", "cpp"),
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                profile = classify_request(prompt, empty)
                self.assertEqual(expected, (profile["kind"], profile["framework"], profile["language"]))

        default = classify_request("Build an inventory management product", empty)
        self.assertEqual(("application", "model-selected"), (default["kind"], default["framework"]))
        self.assertFalse(default["explicit_stack"])
        full_stack = classify_request("Build a login system with PostgreSQL, Google login, and a dashboard", empty)
        self.assertEqual("full-stack", full_stack["architecture"])
        misspelled_django = classify_request(
            "make application with djnaog backend and frontend with bootstrap",
            empty,
        )
        self.assertEqual(("web application", "django", "python"), (
            misspelled_django["kind"], misspelled_django["framework"], misspelled_django["language"],
        ))
        self.assertTrue(misspelled_django["explicit_stack"])

    def test_greenfield_contract_rejects_single_file_react_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text("<div id='root'></div>", encoding="utf-8")
            profile = classify_request("Build a React application", {"file_count": 0, "languages": [], "framework": ""})
            missing = check_artifact_contract(root, profile)
            self.assertTrue(any("package.json" in item for item in missing))
            self.assertTrue(any("src/App" in item for item in missing))

    def test_full_stack_contract_requires_a_server_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "package.json").write_text("{}", encoding="utf-8")
            (root / "index.html").write_text("<div id='root'></div>", encoding="utf-8")
            (root / "src" / "App.jsx").write_text("export default function App() {}", encoding="utf-8")
            (root / "src" / "main.jsx").write_text("", encoding="utf-8")
            profile = classify_request("Build a login system with PostgreSQL and a dashboard", {"file_count": 0, "languages": [], "framework": ""})
            self.assertTrue(any("server implementation" in item for item in check_artifact_contract(root, profile)))
            (root / "server").mkdir()
            (root / "server" / "app.py").write_text("print('ready')", encoding="utf-8")
            self.assertFalse(any("server implementation" in item for item in check_artifact_contract(root, profile)))


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

    def test_risk_policy_keeps_workspace_changes_autonomous_and_blocks_push(self) -> None:
        policy = ToolPolicy()
        self.assertEqual(3, policy.classify("git_push", {})["risk"])
        self.assertEqual(1, policy.classify("delete_file", {"path": "generated.py"})["risk"])
        self.assertEqual(1, policy.classify("install_dependency", {"command": ["python", "-m", "pip", "install", "pytest"]})["risk"])
        self.assertEqual(3, policy.classify("install_dependency", {"command": ["unknown-installer", "package"]})["risk"])
        inline = policy.classify("run_shell", {"command": "def double(value):\n    return value * 2"})
        self.assertEqual(3, inline["risk"])
        self.assertIn("write_file", inline["reason"])
        self.assertEqual(1, policy.classify("run_shell", {"command": ["npm", "install"]})["risk"])
        self.assertEqual(3, policy.classify("run_shell", {"command": ["npm", "publish"]})["risk"])
        missing_browser = policy.classify("browser_inspect", {})
        self.assertEqual(3, missing_browser["risk"])
        self.assertIn("complete HTTP URL", missing_browser["reason"])
        self.assertEqual(1, policy.classify("browser_automate", {"url": "http://localhost:8000"})["risk"])
        self.assertEqual(2, policy.classify("browser_automate", {"url": "https://example.com"})["risk"])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            manager = SandboxManager(base / "sandboxes")
            manager.create("workspace-1", source=source)
            tools = CodebaseTools(manager, "workspace-1")
            (manager.path("workspace-1") / ".env").write_text("TOKEN=private", encoding="utf-8")
            with self.assertRaises(PermissionError):
                tools.execute("read_file", {"path": ".env"})
            with self.assertRaisesRegex(PermissionError, "inline source code"):
                tools.execute("run_shell", {"command": "import inspect; print(inspect.getsourcefile())"})

    def test_preview_readiness_probe_is_loopback_only_bounded_and_read_only(self) -> None:
        class ReadyHandler(BaseHTTPRequestHandler):
            def do_HEAD(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), ReadyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                source = base / "source"
                source.mkdir()
                manager = SandboxManager(base / "sandboxes")
                manager.create("workspace-1", source=source)
                tools = CodebaseTools(manager, "workspace-1")
                url = f"http://127.0.0.1:{server.server_port}/health"
                process = manager.start_process("workspace-1", [sys.executable, "-c", f"import time; print({url!r}, flush=True); time.sleep(30)"])
                deadline = time.monotonic() + 5
                logs = manager.process_logs("workspace-1", process["id"])
                while url not in logs["stdout"] and time.monotonic() < deadline:
                    time.sleep(0.05)
                    logs = manager.process_logs("workspace-1", process["id"])
                self.assertIn(url, logs["stdout"])
                try:
                    self.assertEqual(0, ToolPolicy().classify("probe_local_url", {"url": url, "process_id": process["id"]})["risk"])
                    self.assertEqual({"url": url, "reachable": True, "http_status": 200, "content_type": "text/html", "redirect": False}, tools.execute("probe_local_url", {"url": url, "process_id": process["id"]}))
                    with self.assertRaisesRegex(PermissionError, "reported by this running preview"):
                        tools.execute("probe_local_url", {"url": f"http://127.0.0.1:{server.server_port + 1}/", "process_id": process["id"]})
                    with self.assertRaisesRegex(ValueError, "loopback"):
                        tools.execute("probe_local_url", {"url": "http://example.com/", "process_id": process["id"]})
                    with self.assertRaisesRegex(ValueError, "loopback"):
                        tools.execute("probe_local_url", {"url": "http://user@localhost/", "process_id": process["id"]})
                finally:
                    manager.stop_process("workspace-1", process["id"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_local_model_argument_aliases_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "module.py").write_text("def calculate_total():\n    return 1\n", encoding="utf-8")
            manager = SandboxManager(base / "sandboxes")
            manager.create("aliases", source=source)
            tools = CodebaseTools(manager, "aliases")
            references = tools.execute("find_references", {"query": "calculate_total"})
            self.assertTrue(references["matches"])
            result = tools.execute("write_file", {"filename": "created.py", "code": "VALUE = 1\n"})
            self.assertEqual("created.py", result["path"])
            test_result = tools.execute("run_tests", {"files": ["created.py"]})
            self.assertTrue(test_result["skipped"])

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
            # Interpreter cold starts can take longer than 200 ms (notably in
            # fresh Windows venvs). Observe readiness, don't assume a timing.
            deadline = time.monotonic() + 5
            logs = manager.process_logs("workspace-1", process["id"])
            while "ready" not in logs["stdout"] and time.monotonic() < deadline:
                time.sleep(0.05)
                logs = manager.process_logs("workspace-1", process["id"])
            self.assertIn("ready", logs["stdout"])
            stopped = manager.stop_process("workspace-1", process["id"])
            self.assertEqual("exited", stopped["status"])

    def test_managed_preview_process_is_recovered_and_stopped_after_manager_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            root = base / "sandboxes"
            first_manager = SandboxManager(root)
            first_manager.create("workspace-1", source=source)
            process = first_manager.start_process(
                "workspace-1",
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); print(f'recovered-preview-ready child={child.pid}',flush=True); time.sleep(60)",
                ],
            )
            original_record = first_manager._managed_process("workspace-1", process["id"])
            try:
                deadline = time.monotonic() + 8
                logs = first_manager.process_logs("workspace-1", process["id"])
                while "recovered-preview-ready" not in logs["stdout"] and time.monotonic() < deadline:
                    time.sleep(0.05)
                    logs = first_manager.process_logs("workspace-1", process["id"])
                self.assertIn("recovered-preview-ready", logs["stdout"])
                child_pid = int(logs["stdout"].strip().split("child=")[-1])
                self.assertIsNotNone(SandboxManager._process_identity(child_pid))

                restarted_manager = SandboxManager(root)
                self.assertEqual("running", restarted_manager.process_status("workspace-1", process["id"])["status"])
                recovered_logs = restarted_manager.process_logs("workspace-1", process["id"])
                self.assertIn("recovered-preview-ready", recovered_logs["stdout"])

                stopped = restarted_manager.stop_process("workspace-1", process["id"])
                self.assertEqual("exited", stopped["status"])
                self.assertIsNone(SandboxManager._process_identity(child_pid))
                deadline = time.monotonic() + 8
                while first_manager.process_status("workspace-1", process["id"])["status"] != "exited" and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertEqual("exited", first_manager.process_status("workspace-1", process["id"])["status"])
            finally:
                if original_record["process"].poll() is None:
                    first_manager.stop_process("workspace-1", process["id"])
                first_manager._close_process_logs(original_record)
                original_record["process"].wait(timeout=8)

    def test_recovered_process_identity_mismatch_never_stops_a_reused_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            root = base / "sandboxes"
            first_manager = SandboxManager(root)
            first_manager.create("workspace-1", source=source)
            process = first_manager.start_process(
                "workspace-1",
                [sys.executable, "-c", "import time; time.sleep(60)"],
            )
            original_record = first_manager._managed_process("workspace-1", process["id"])
            try:
                restarted_manager = SandboxManager(root)
                recovered = restarted_manager._managed_process("workspace-1", process["id"])
                recovered["identity"] = {**recovered["identity"], "created": "different-process"}
                self.assertEqual("exited", restarted_manager.process_status("workspace-1", process["id"])["status"])
                self.assertEqual("exited", restarted_manager.stop_process("workspace-1", process["id"])["status"])
                self.assertIsNone(original_record["process"].poll())
            finally:
                if original_record["process"].poll() is None:
                    first_manager.stop_process("workspace-1", process["id"])
                first_manager._close_process_logs(original_record)
                original_record["process"].wait(timeout=8)

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
    def test_git_read_tools_do_not_walk_into_parent_repository(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            manager = SandboxManager(base / "sandboxes")
            manager.create("workspace", source=source)
            tools = CodebaseTools(manager, "workspace")

            result = tools.tool_git_diff()

            self.assertFalse(result["repository"])
            self.assertEqual("", result["stdout"])
            self.assertNotIn("web/sw.js", result["stderr"])

    def test_new_page_dummy_content_is_idempotent_across_plan_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "index.html").write_text("<h1>Home</h1>\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), NewPageGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Page builder", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Implement a new page with dummy content", max_iterations=12)
            run = service.wait_for_run(run["id"], timeout=20)

            self.assertEqual("completed", run["status"], run.get("error"))
            generated = sorted(source.glob("new-page*.html"))
            self.assertEqual(1, len(generated))
            self.assertIn('data-codeq-generated="page"', generated[0].read_text(encoding="utf-8"))
            self.assertEqual(3, len([task for task in run["tasks"] if task["status"] == "completed"]))

    def test_dependency_only_model_plan_falls_back_without_iteration_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("existing project\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), DependencyOnlyPlanGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Dependency plan", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Update pages with content and data", max_iterations=8)
            run = service.wait_for_run(run["id"], timeout=20)

            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertTrue(any(event["type"] == "plan.fallback.selected" for event in service.events(run["id"])))
            self.assertLessEqual(run["iteration"], 8)

    def test_title_based_page_update_dependencies_are_resolved_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "index.html").write_text("<h1>Before</h1>\n", encoding="utf-8")
            (source / "about.html").write_text("<h1>About</h1>\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), TitleDependencyPageUpdateGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Pages", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Update pages with content and data", max_iterations=8)
            run = service.wait_for_run(run["id"], timeout=20)

            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertTrue((source / "pages-updated.txt").is_file())
            completed_ids = {task["id"] for task in run["tasks"] if task["status"] == "completed"}
            for task in run["tasks"]:
                self.assertTrue(set(task["depends_on"]).issubset(completed_ids))
            repairs = next(event["payload"]["dependency_repairs"] for event in service.events(run["id"]) if event["type"] == "plan.created")
            self.assertTrue(any(item["action"] == "resolved" for item in repairs))
            self.assertTrue(any(item["action"] == "removed_unknown" and item["dependency"].endswith(":T0") for item in repairs))

    def test_malformed_review_action_does_not_deadlock_a_vue_migration_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "selected-django-project"
            source.mkdir()
            (source / "manage.py").write_text("print('Django entrypoint')\n", encoding="utf-8")
            templates = source / "templates"
            templates.mkdir()
            (templates / "home.html").write_text("<h1>Home</h1>\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), MalformedVueMigrationGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Vue migration", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"],
                prompt="read whole code and add one more page and also in all pages add dummy content also frontend convert in vue js",
                max_iterations=12,
            )
            run = service.wait_for_run(run["id"], timeout=20)

            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertTrue((source / "package.json").is_file())
            self.assertTrue((source / "src" / "App.vue").is_file())
            self.assertTrue(all(task["status"] == "completed" for task in run["tasks"]))
            self.assertFalse(any(task["status"] == "blocked" for task in run["tasks"]))
            events = service.events(run["id"])
            self.assertTrue(any(
                item["type"] == "model.action_replaced"
                and item["payload"].get("reason") == "analysis task received no executable action"
                for item in events
            ))

    def test_resume_recovers_legacy_malformed_action_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "main.py").write_text("print('ready')\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), FakeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Legacy malformed action", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Update the existing application", max_iterations=1)
            run = service.wait_for_run(run["id"], timeout=10)
            self.assertEqual("failed", run["status"])
            task_id = run["tasks"][0]["id"]
            with service.store.session() as connection:
                connection.execute(
                    "UPDATE codebase_tasks SET status = 'blocked', result_json = ? WHERE id = ?",
                    (json.dumps({"reason": "model returned no executable actions"}), task_id),
                )
                connection.execute("UPDATE codebase_runs SET max_iterations = ? WHERE id = ?", (8, run["id"]))
            resumed = service.resume_run(run["id"])
            resumed = service.wait_for_run(resumed["id"], timeout=15)

            self.assertEqual("completed", resumed["status"], resumed.get("error"))
            self.assertTrue(any(item["type"] == "tasks.recovered" for item in service.events(run["id"])))

    def test_resume_repairs_legacy_invalid_dependencies_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "main.py").write_text("print('ready')\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), NeverDoneGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Legacy graph", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Repair existing page content", max_iterations=1)
            run = service.wait_for_run(run["id"], timeout=10)
            self.assertEqual("failed", run["status"])
            task_id = run["tasks"][0]["id"]
            with service.store.session() as connection:
                connection.execute(
                    "UPDATE codebase_tasks SET depends_on_json = ? WHERE id = ?",
                    (json.dumps([f"{run['id']}:Review existing pages for updates"]), task_id),
                )
                connection.execute("UPDATE codebase_runs SET max_iterations = ? WHERE id = ?", (8, run["id"]))
            service.gateway = FakeGateway()  # type: ignore[assignment]
            resumed = service.resume_run(run["id"])
            resumed = service.wait_for_run(resumed["id"], timeout=15)

            self.assertEqual("completed", resumed["status"], resumed.get("error"))
            repairs = [event for event in service.events(run["id"]) if event["type"] == "plan.dependencies_repaired"]
            self.assertTrue(repairs)
            self.assertTrue(any(item["action"] == "removed_unknown" for item in repairs[-1]["payload"]["repairs"]))

    def test_independent_specialist_tasks_generate_in_parallel_and_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("existing project\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), ParallelPlanningGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Mixed", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add backend and worker modules", max_iterations=8)
            run = service.wait_for_run(run["id"], timeout=25)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertTrue((source / "backend.py").is_file())
            self.assertTrue((source / "frontend.js").is_file())
            events = {item["type"] for item in service.events(run["id"])}
            self.assertIn("agents.parallel.started", events)
            self.assertIn("agents.parallel.completed", events)
            self.assertIn("plan.model.accepted", events)

    def test_explorer_operations_sync_workspace_and_selected_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("existing project\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), FakeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Explorer", source_path=str(source))
            service.mutate_path(project["id"], action="create_directory", path="src")
            service.mutate_path(project["id"], action="create_file", path="src/app.py")
            service.write_file(project["id"], "src/app.py", "print('ready')\n")
            service.mutate_path(project["id"], action="rename", path="src/app.py", destination="src/main.py")
            workspace = Path(project["workspace"]["workspace_path"])
            self.assertEqual("print('ready')\n", (source / "src" / "main.py").read_text(encoding="utf-8"))
            self.assertTrue((workspace / "src" / "main.py").is_file())
            service.mutate_path(project["id"], action="delete", path="src/main.py")
            self.assertFalse((source / "src" / "main.py").exists())
            self.assertFalse((workspace / "src" / "main.py").exists())
            with self.assertRaises(PermissionError):
                service.mutate_path(project["id"], action="create_file", path=".env")

    def test_review_material_contains_new_untracked_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("existing project\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), FakeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Review", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add a feature", max_iterations=4)
            run = service.wait_for_run(run["id"], timeout=20)
            self.assertEqual("completed", run["status"], run.get("error"))
            material = service._review_material(run)
            self.assertIn("recorded file: feature.py", material)
            self.assertIn("+VALUE = 42", material)

    def test_greenfield_non_react_service_builds_tests_and_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), FastApiCreationGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Inventory", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Build a FastAPI inventory service", max_iterations=12)
            run = service.wait_for_run(run["id"], timeout=25)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertTrue((source / "inventory.py").is_file())
            self.assertTrue((source / "test_inventory.py").is_file())
            self.assertTrue(any(item["kind"] == "runtime" and item["status"] == "passed" for item in run["validation"]))
            self.assertIn("source.publish.completed", {item["type"] for item in service.events(run["id"])})

    def test_unspecified_greenfield_stack_is_chosen_by_model_not_forced_to_react(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), ModelSelectedNodeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Inventory product", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Build an inventory management product", max_iterations=10)
            run = service.wait_for_run(run["id"], timeout=25)

            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertTrue((source / "src" / "inventory.js").is_file())
            self.assertTrue((source / "test" / "inventory.test.js").is_file())
            self.assertNotIn("react", (source / "package.json").read_text(encoding="utf-8").lower())
            events = {item["type"] for item in service.events(run["id"])}
            self.assertIn("plan.model.accepted", events)
            self.assertNotIn("plan.fallback.selected", events)

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
            self.assertEqual("VALUE = 42\n", (source / "feature.py").read_text(encoding="utf-8"))
            self.assertIn("source.publish.completed", event_types)

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

    def test_successful_file_evidence_advances_task_when_model_omits_done_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "main.py").write_text("print('ready')\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), WritesWithoutDoneGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Evidence completion", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add a feature", max_iterations=4)
            run = service.wait_for_run(run["id"], timeout=20)

            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertEqual("VALUE = 42\n", (source / "feature.py").read_text(encoding="utf-8"))
            self.assertIn("task.completion.inferred", {item["type"] for item in service.events(run["id"])})

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

    def test_empty_login_project_honours_react_and_publishes_verified_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), EmptyLoginPageGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Login", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="make a login page responsive, with React", max_iterations=8)
            run = service.wait_for_run(run["id"], timeout=15)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertEqual([], run["approvals"])
            workspace = Path(project["workspace"]["workspace_path"])
            self.assertIn('"react"', (workspace / "package.json").read_text(encoding="utf-8"))
            self.assertIn("useState", (workspace / "src" / "App.jsx").read_text(encoding="utf-8"))
            self.assertIn("@media", (workspace / "src" / "styles.css").read_text(encoding="utf-8"))
            self.assertEqual(
                (workspace / "src" / "App.jsx").read_text(encoding="utf-8"),
                (source / "src" / "App.jsx").read_text(encoding="utf-8"),
            )
            self.assertTrue((source / "package.json").is_file())
            event_types = {item["type"] for item in service.events(run["id"])}
            self.assertIn("model.action_replaced", event_types)
            self.assertIn("source.publish.completed", event_types)

    def test_static_four_page_site_is_verified_and_published_to_selected_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "selected-folder"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), StaticSiteGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Static site", source_path=str(source))
            run = service.create_run(
                codebase_project_id=project["id"],
                prompt="make a web appliation with 4 pages static",
                max_iterations=8,
            )
            run = service.wait_for_run(run["id"], timeout=20)
            self.assertEqual("completed", run["status"], run.get("error"))
            page_names = {"index.html", "about.html", "services.html", "contact.html"}
            self.assertEqual(page_names, {path.name for path in source.glob("*.html")})
            self.assertTrue((source / "styles.css").is_file())
            self.assertTrue((source / "app.js").is_file())
            self.assertFalse((source / "package.json").exists())
            for page_name in page_names:
                content = (source / page_name).read_text(encoding="utf-8")
                for linked_page in page_names:
                    self.assertIn(f'href="{linked_page}"', content)
            self.assertTrue(any(item["kind"] == "artifacts" and item["status"] == "passed" for item in run["validation"]))
            self.assertFalse(any("sign-in" in task["title"].lower() for task in run["tasks"]))
            self.assertIn("source.publish.completed", {item["type"] for item in service.events(run["id"])})

    def test_django_three_page_request_keeps_framework_and_publishes_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "selected-django-folder"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), FakeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Django site", source_path=str(source))
            stale_workspace = Path(project["workspace"]["workspace_path"])
            (stale_workspace / "index.html").write_text("<h1>Rejected login attempt</h1>\n", encoding="utf-8")
            (stale_workspace / "styles.css").write_text("body{}\n", encoding="utf-8")
            prompt = "make a django application with intergtaed frontend like with jguery 3 pages, like home contact and about"
            self.assertFalse(service._standalone_web_request(prompt, {"file_count": 0, "total_bytes": 0, "framework": "", "languages": []}))
            with patch.dict(os.environ, {"OPENAGENT_CODEQ_INSTALL_DEPENDENCIES": "0"}):
                run = service.create_run(codebase_project_id=project["id"], prompt=prompt, max_iterations=12)
                run = service.wait_for_run(run["id"], timeout=25)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertTrue((source / "manage.py").is_file())
            self.assertTrue((source / "config" / "settings.py").is_file())
            self.assertTrue((source / "pages" / "urls.py").is_file())
            self.assertEqual(
                {"home.html", "about.html", "contact.html", "base.html"},
                {path.name for path in (source / "templates").glob("*.html")},
            )
            self.assertIn("jquery-3.7.1.min.js", (source / "templates" / "base.html").read_text(encoding="utf-8"))
            self.assertIn("window.jQuery", (source / "static" / "js" / "site.js").read_text(encoding="utf-8"))
            self.assertTrue(any(item["kind"] == "test" and item["status"] == "passed" for item in run["validation"]))
            self.assertTrue(any(item["kind"] == "artifacts" and item["status"] == "passed" for item in run["validation"]))
            self.assertFalse(any("login" in task["title"].lower() or "sign-in" in task["title"].lower() for task in run["tasks"]))
            self.assertIn("source.publish.completed", {item["type"] for item in service.events(run["id"])})

    def test_misspelled_django_bootstrap_request_builds_four_named_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "selected-bootstrap-folder"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), FakeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Bootstrap site", source_path=str(source))
            prompt = "make application with djnaog backend and frontend with bootstrap, 4 pages index, contact about and careers"
            with patch.dict(os.environ, {"OPENAGENT_CODEQ_INSTALL_DEPENDENCIES": "0"}):
                run = service.create_run(codebase_project_id=project["id"], prompt=prompt, max_iterations=12)
                run = service.wait_for_run(run["id"], timeout=25)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertEqual(
                {"base.html", "home.html", "contact.html", "about.html", "careers.html"},
                {path.name for path in (source / "templates").glob("*.html")},
            )
            base_template = (source / "templates" / "base.html").read_text(encoding="utf-8")
            self.assertIn("bootstrap@5.3.3", base_template)
            self.assertIn("bootstrap.bundle.min.js", base_template)
            self.assertIn("name='careers'", (source / "pages" / "urls.py").read_text(encoding="utf-8"))
            profile = classify_request(prompt, {"file_count": 0, "languages": [], "framework": ""})
            self.assertFalse(check_artifact_contract(source, profile))
            self.assertTrue(any(item["kind"] == "artifacts" and item["status"] == "passed" for item in run["validation"]))
            self.assertIn("framework.scaffold.selected", {item["type"] for item in service.events(run["id"])})

    def test_django_runtime_validation_uses_project_local_environment(self) -> None:
        with patch.dict(os.environ, {"OPENAGENT_CODEQ_INSTALL_DEPENDENCIES": "1"}):
            actions = CodebaseService._django_validation_actions(["manage.py"])
        commands = [item["arguments"]["command"] for item in actions if "command" in item.get("arguments", {})]
        self.assertIn(["python", "-m", "venv", ".venv"], commands)
        self.assertTrue(any(command[-2:] == ["manage.py", "check"] for command in commands))
        self.assertTrue(any(command[-2:] == ["manage.py", "test"] for command in commands))
        self.assertEqual("install_dependency", actions[1]["tool"])

    def test_identical_command_failure_stops_without_spending_iteration_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "selected-folder"
            source.mkdir()
            service = CodebaseService(OpenAgentStore(base / "data"), FakeGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Failure guard", source_path=str(source))
            failed_action = [{"tool": "run_shell", "arguments": {"command": [sys.executable, "-c", "raise SystemExit(9)"]}}]
            with patch.object(CodebaseService, "_django_validation_actions", return_value=failed_action):
                run = service.create_run(
                    codebase_project_id=project["id"],
                    prompt="Build a Django site with three pages",
                    max_iterations=12,
                )
                run = service.wait_for_run(run["id"], timeout=25)
            self.assertEqual("failed", run["status"])
            self.assertLess(run["iteration"], run["max_iterations"])
            self.assertIn("same command failed twice", run["error"].lower())
            failed_events = [item for item in service.events(run["id"]) if item["type"] == "run.failed"]
            self.assertEqual("repeated_command_failure", failed_events[-1]["payload"]["reason"])

    def test_running_process_without_exit_code_is_not_treated_as_a_failed_command(self) -> None:
        self.assertIsNone(CodebaseService._exit_code(None))
        self.assertIsNone(CodebaseService._exit_code("not-finished"))
        self.assertEqual(0, CodebaseService._exit_code(0))
        self.assertEqual(1, CodebaseService._exit_code("1"))

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
            self.assertIn("model.action_replaced", {item["type"] for item in service.events(run["id"])})

    def test_narration_process_is_replaced_by_files_and_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            for index in range(25):
                (source / f"empty-{index}").mkdir()
            (source / "app.js").write_text("console.log('existing web asset');\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), NarratingPythonGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Function", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="write a Python function that calculates factorial and test it", max_iterations=6)
            run = service.wait_for_run(run["id"], timeout=20)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertEqual([], run["approvals"])
            workspace = Path(project["workspace"]["workspace_path"])
            self.assertIn("def factorial", (workspace / "function.py").read_text(encoding="utf-8"))
            self.assertTrue((workspace / "test_function.py").is_file())
            self.assertTrue(any(item["status"] == "passed" for item in run["validation"]))
            self.assertIn("model.action_replaced", {item["type"] for item in service.events(run["id"])})

    def test_autonomous_run_redirects_approval_action_without_blocking_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "main.py").write_text("print('ready')\n", encoding="utf-8")
            service = CodebaseService(OpenAgentStore(base / "data"), ApprovalLoopGateway())  # type: ignore[arg-type]
            project = service.create_project(name="Autonomous", source_path=str(source))
            run = service.create_run(codebase_project_id=project["id"], prompt="Add the requested feature", max_iterations=6)
            run = service.wait_for_run(run["id"], timeout=20)
            self.assertEqual("completed", run["status"], run.get("error"))
            self.assertEqual([], run["approvals"])
            self.assertEqual("VALUE = 42\n", (Path(project["workspace"]["workspace_path"]) / "feature.py").read_text(encoding="utf-8"))
            self.assertIn("autonomy.redirected", {item["type"] for item in service.events(run["id"])})

    def test_iteration_budget_is_selected_from_task_and_repository_scale(self) -> None:
        self.assertEqual(24, CodebaseService._automatic_iteration_budget("write a function", {"file_count": 2}))
        self.assertEqual(64, CodebaseService._automatic_iteration_budget("build a feature", {"file_count": 100}))
        self.assertEqual(120, CodebaseService._automatic_iteration_budget("build a platform", {"file_count": 900}))
        self.assertEqual(200, CodebaseService._automatic_iteration_budget("build a platform", {"file_count": 3000}))


if __name__ == "__main__":
    unittest.main()
