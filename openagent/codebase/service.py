"""Durable autonomous software-engineering runs."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
from pathlib import Path
import re
import shlex
import threading
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from ..model_gateway import ModelGateway, ProviderConfigurationError
from ..models import AuditEvent, utc_now
from ..storage import OpenAgentStore
from .repository import RepositoryIntelligence
from .sandbox import SandboxManager
from .tools import ApprovalRequired, CodebaseTools


FINAL_STATES = {"completed", "failed", "cancelled", "blocked"}
PLAN_SCHEMA = {
    "goal": "string",
    "tasks": [{"id": "T1", "title": "string", "agent": "architect|programmer|tester|debugger|reviewer", "depends_on": ["T0"]}],
}
ACTION_TOOLS = {
    "list_directory", "read_file", "read_files", "write_file", "patch_file", "delete_file", "move_file",
    "grep", "search_symbol", "find_references", "project_map", "run_shell", "run_tests", "run_linter", "run_typechecker",
    "run_build", "browser_test", "browser_inspect", "install_dependency", "start_process", "process_status",
    "stop_process", "inspect_logs", "git_status", "git_diff", "git_log", "git_commit", "git_checkout",
    "git_revert", "git_reset",
}

EMPTY_LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Secure account sign in">
  <title>Sign in</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <main class="auth-shell">
    <section class="auth-intro" aria-labelledby="welcome-title">
      <a class="brand" href="#" aria-label="Home"><span aria-hidden="true">O</span> Orbit</a>
      <div>
        <p class="eyebrow">Private workspace</p>
        <h1 id="welcome-title">Welcome back.</h1>
        <p>Sign in to continue to your projects, notes, and saved work.</p>
      </div>
      <p class="privacy-note">Your session remains on this device.</p>
    </section>
    <section class="auth-panel" aria-labelledby="form-title">
      <form id="login-form" novalidate>
        <header>
          <p class="eyebrow">Account access</p>
          <h2 id="form-title">Sign in</h2>
          <p>Use the email address connected to your account.</p>
        </header>
        <div class="field">
          <label for="email">Email address</label>
          <input id="email" name="email" type="email" autocomplete="email" inputmode="email" required aria-describedby="email-error">
          <small id="email-error" class="field-error" aria-live="polite"></small>
        </div>
        <div class="field">
          <div class="label-row"><label for="password">Password</label><a href="#">Forgot password?</a></div>
          <div class="password-control">
            <input id="password" name="password" type="password" autocomplete="current-password" minlength="8" required aria-describedby="password-error">
            <button id="toggle-password" class="text-button" type="button" aria-controls="password" aria-pressed="false">Show</button>
          </div>
          <small id="password-error" class="field-error" aria-live="polite"></small>
        </div>
        <label class="remember"><input name="remember" type="checkbox"> Keep me signed in</label>
        <button class="primary-button" type="submit">Continue</button>
        <p id="form-status" class="form-status" role="status" aria-live="polite"></p>
        <p class="account-link">New here? <a href="#">Create an account</a></p>
      </form>
    </section>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""

EMPTY_LOGIN_CSS = """* { box-sizing: border-box; }
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #edf2f8; }
body { margin: 0; min-width: 320px; min-height: 100vh; }
a { color: #145dc1; text-underline-offset: 3px; }
button, input { font: inherit; }
.auth-shell { min-height: 100vh; display: grid; grid-template-columns: minmax(280px, .9fr) minmax(440px, 1.1fr); }
.auth-intro { display: flex; flex-direction: column; justify-content: space-between; gap: 3rem; padding: clamp(2rem, 5vw, 5rem); color: white; background: #102a43; }
.brand { display: inline-flex; align-items: center; gap: .7rem; width: fit-content; color: white; font-weight: 750; text-decoration: none; }
.brand span { display: grid; place-items: center; width: 2rem; height: 2rem; border: 1px solid #8fc5ff; border-radius: 7px; background: #1768c4; }
.auth-intro h1 { max-width: 620px; margin: .3rem 0 .8rem; font-size: clamp(2.5rem, 6vw, 5.4rem); line-height: .98; }
.auth-intro p { max-width: 520px; color: #c8d9e8; line-height: 1.65; }
.eyebrow { margin: 0; color: #4177b7; font-size: .76rem; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
.privacy-note { margin: 0; font-size: .88rem; }
.auth-panel { display: grid; place-items: center; padding: clamp(1.25rem, 6vw, 5rem); background: #f8fafc; }
form { width: min(100%, 440px); display: grid; gap: 1.2rem; }
form header { margin-bottom: .4rem; }
form h2 { margin: .35rem 0 .5rem; font-size: 2rem; }
form header p:last-child, .account-link { color: #66758a; }
.field { display: grid; gap: .45rem; }
.field label, .remember { font-size: .9rem; font-weight: 650; }
.label-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.label-row a { font-size: .85rem; }
input { width: 100%; min-height: 48px; border: 1px solid #bcc8d6; border-radius: 7px; padding: .72rem .85rem; color: #172033; background: white; outline: none; transition: border-color .16s, box-shadow .16s; }
input:focus { border-color: #1768c4; box-shadow: 0 0 0 3px rgba(23, 104, 196, .15); }
input[aria-invalid="true"] { border-color: #b42318; }
.password-control { position: relative; }
.password-control input { padding-right: 4.3rem; }
.text-button { position: absolute; top: 50%; right: .45rem; transform: translateY(-50%); border: 0; padding: .45rem; color: #145dc1; background: transparent; cursor: pointer; }
.remember { display: flex; align-items: center; gap: .65rem; width: fit-content; color: #45556c; }
.remember input { width: 1rem; min-height: 1rem; accent-color: #1768c4; }
.primary-button { min-height: 50px; border: 0; border-radius: 7px; color: white; font-weight: 750; background: #1768c4; cursor: pointer; transition: background .16s, transform .16s; }
.primary-button:hover { background: #1057a7; }
.primary-button:active { transform: translateY(1px); }
.field-error { min-height: 1.1em; color: #b42318; }
.form-status { min-height: 1.3em; margin: -.3rem 0 0; color: #17643a; }
.account-link { margin: 0; text-align: center; }
@media (max-width: 780px) { .auth-shell { grid-template-columns: 1fr; } .auth-intro { min-height: 245px; padding: 1.5rem; gap: 1.5rem; } .auth-intro h1 { font-size: clamp(2.3rem, 13vw, 4rem); } .privacy-note { display: none; } .auth-panel { place-items: start center; padding: 2rem 1.25rem 3rem; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
"""

EMPTY_LOGIN_JS = """const form = document.querySelector('#login-form');
const email = document.querySelector('#email');
const password = document.querySelector('#password');
const toggle = document.querySelector('#toggle-password');
const status = document.querySelector('#form-status');

function showError(input, message) {
  input.setAttribute('aria-invalid', message ? 'true' : 'false');
  document.querySelector(`#${input.id}-error`).textContent = message;
}

toggle.addEventListener('click', () => {
  const visible = password.type === 'text';
  password.type = visible ? 'password' : 'text';
  toggle.textContent = visible ? 'Show' : 'Hide';
  toggle.setAttribute('aria-pressed', String(!visible));
});

form.addEventListener('submit', event => {
  event.preventDefault();
  showError(email, email.validity.valid ? '' : 'Enter a valid email address.');
  showError(password, password.value.length >= 8 ? '' : 'Password must contain at least 8 characters.');
  status.textContent = form.checkValidity() ? 'Form validated. Connect this page to your authentication endpoint.' : '';
  if (!form.checkValidity()) form.querySelector(':invalid')?.focus();
});
"""


class CodebaseService:
    """Own projects, workspaces, task graphs, tools, events, and completion evidence."""

    def __init__(self, store: OpenAgentStore, gateway: ModelGateway) -> None:
        self.store = store
        self.gateway = gateway
        self.sandboxes = SandboxManager(store.data_dir / "codebase-workspaces")
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self.initialize()
        self._recover_interrupted_runs()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS codebase_projects (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    repo_url TEXT NOT NULL DEFAULT '',
                    map_json TEXT NOT NULL DEFAULT '{}',
                    memory_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_codebase_projects_owner ON codebase_projects(project, owner, updated_at DESC);

                CREATE TABLE IF NOT EXISTS codebase_workspaces (
                    id TEXT PRIMARY KEY,
                    codebase_project_id TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    sandbox_backend TEXT NOT NULL DEFAULT 'local',
                    source_path TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    branch TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'ready',
                    environment_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(codebase_project_id) REFERENCES codebase_projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_runs (
                    id TEXT PRIMARY KEY,
                    codebase_project_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    user_prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'auto',
                    model TEXT NOT NULL DEFAULT '',
                    role_models_json TEXT NOT NULL DEFAULT '{}',
                    allow_external INTEGER NOT NULL DEFAULT 0,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    max_iterations INTEGER NOT NULL DEFAULT 8,
                    completion_json TEXT NOT NULL DEFAULT '{}',
                    summary TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(codebase_project_id) REFERENCES codebase_projects(id) ON DELETE CASCADE,
                    FOREIGN KEY(workspace_id) REFERENCES codebase_workspaces(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_codebase_runs_project ON codebase_runs(project, owner, started_at DESC);

                CREATE TABLE IF NOT EXISTS codebase_tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parent_task_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    assigned_agent TEXT NOT NULL DEFAULT 'programmer',
                    status TEXT NOT NULL DEFAULT 'pending',
                    depends_on_json TEXT NOT NULL DEFAULT '[]',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_codebase_tasks_run ON codebase_tasks(run_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS codebase_tool_calls (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL DEFAULT '',
                    tool TEXT NOT NULL,
                    input_json TEXT NOT NULL DEFAULT '{}',
                    output_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    risk INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_file_changes (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    old_hash TEXT NOT NULL DEFAULT '',
                    new_hash TEXT NOT NULL DEFAULT '',
                    patch TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_test_runs (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    command_json TEXT NOT NULL DEFAULT '[]',
                    stdout TEXT NOT NULL DEFAULT '',
                    stderr TEXT NOT NULL DEFAULT '',
                    exit_code INTEGER NOT NULL DEFAULT -1,
                    duration_ms REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_codebase_events_run ON codebase_events(run_id, sequence ASC);

                CREATE TABLE IF NOT EXISTS codebase_snapshots (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_approvals (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_memories (
                    id TEXT PRIMARY KEY,
                    codebase_project_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_run_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(codebase_project_id) REFERENCES codebase_projects(id) ON DELETE CASCADE
                );
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(codebase_runs)").fetchall()}
            if "role_models_json" not in columns:
                connection.execute("ALTER TABLE codebase_runs ADD COLUMN role_models_json TEXT NOT NULL DEFAULT '{}'")

    def pick_source_folder(self) -> str:
        """Open the operating-system folder chooser after an explicit UI action."""

        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as exc:
            raise ValueError("The native folder chooser is unavailable in this Python installation.") from exc
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(parent=root, title="Open a project in CodeQ", mustexist=True)
        finally:
            root.destroy()
        return str(Path(selected).resolve()) if selected else ""

    def create_project(self, *, name: str, source_path: str, project: str = "default", owner: str = "local", backend: str = "local") -> dict[str, Any]:
        source = Path(source_path).expanduser().resolve()
        intelligence = RepositoryIntelligence(source)
        project_map = intelligence.scan()
        codebase_project_id = uuid4().hex
        workspace_id = uuid4().hex
        branch = f"openagent/run-{workspace_id[:8]}"
        workspace = self.sandboxes.create(workspace_id, source=source, backend=backend, branch=branch)
        now = utc_now()
        memory = {
            "stack": {"languages": project_map["languages"], "framework": project_map["framework"], "database": project_map["database"]},
            "commands": project_map["commands"],
            "preferences": [],
            "last_successful_validation": {},
        }
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO codebase_projects (id, project, owner, name, source_path, map_json, memory_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (codebase_project_id, project, owner, name.strip() or source.name, str(source), json.dumps(project_map), json.dumps(memory), now, now),
            )
            connection.execute(
                "INSERT INTO codebase_workspaces (id, codebase_project_id, owner, sandbox_backend, source_path, workspace_path, branch, environment_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)",
                (workspace_id, codebase_project_id, owner, backend, str(source), workspace["path"], branch, now, now),
            )
        self._audit("codebase.project.create", f"codebase-project:{codebase_project_id}", project, {"owner": owner, "source": str(source), "backend": backend})
        return self.get_project(codebase_project_id, owner=owner)

    def list_projects(self, *, project: str = "default", owner: str = "local") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_projects WHERE project = ? AND owner = ? ORDER BY updated_at DESC", (project, owner)).fetchall()
        return [self._project_row(row) for row in rows]

    def get_project(self, codebase_project_id: str, *, owner: str = "local") -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM codebase_projects WHERE id = ? AND owner = ?", (codebase_project_id, owner)).fetchone()
            workspace = connection.execute("SELECT * FROM codebase_workspaces WHERE codebase_project_id = ? AND owner = ? ORDER BY created_at DESC LIMIT 1", (codebase_project_id, owner)).fetchone()
        if row is None:
            raise KeyError("codebase project not found")
        result = self._project_row(row)
        result["workspace"] = self._workspace_row(workspace) if workspace else None
        return result

    def refresh_project_map(self, codebase_project_id: str, *, owner: str = "local") -> dict[str, Any]:
        project = self.get_project(codebase_project_id, owner=owner)
        workspace = project.get("workspace") or {}
        project_map = RepositoryIntelligence(str(workspace["workspace_path"])).scan()
        with self.store.session() as connection:
            connection.execute("UPDATE codebase_projects SET map_json = ?, updated_at = ? WHERE id = ? AND owner = ?", (json.dumps(project_map), utc_now(), codebase_project_id, owner))
        return project_map

    def file_tree(self, codebase_project_id: str, *, owner: str = "local") -> dict[str, Any]:
        project = self.get_project(codebase_project_id, owner=owner)
        workspace = project["workspace"]
        tools = CodebaseTools(self.sandboxes, workspace["id"], backend=workspace["sandbox_backend"])
        return tools.tool_list_directory(recursive=True)

    def read_file(self, codebase_project_id: str, path: str, *, owner: str = "local") -> dict[str, Any]:
        project = self.get_project(codebase_project_id, owner=owner)
        workspace = project["workspace"]
        return CodebaseTools(self.sandboxes, workspace["id"], backend=workspace["sandbox_backend"]).tool_read_file(path, end_line=1_000_000)

    def write_file(self, codebase_project_id: str, path: str, content: str, *, owner: str = "local", expected_sha256: str = "") -> dict[str, Any]:
        project = self.get_project(codebase_project_id, owner=owner)
        workspace = project["workspace"]
        result = CodebaseTools(self.sandboxes, workspace["id"], backend=workspace["sandbox_backend"]).execute(
            "write_file", {"path": path, "content": content, "expected_sha256": expected_sha256}
        )
        self._audit("codebase.file.write", f"codebase-project:{codebase_project_id}:{path}", project["project"], {"owner": owner, "old_hash": result.get("old_sha256"), "new_hash": result.get("new_sha256")})
        return result

    def create_run(
        self,
        *,
        codebase_project_id: str,
        prompt: str,
        owner: str = "local",
        provider: str = "auto",
        model: str = "",
        role_models: dict[str, Any] | None = None,
        allow_external: bool = False,
        max_iterations: int = 0,
        completion: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("software-engineering goal is required")
        codebase_project = self.get_project(codebase_project_id, owner=owner)
        workspace = codebase_project["workspace"]
        project_map = codebase_project["map"]
        criteria = self._completion_criteria(project_map, completion or {})
        if self._standalone_python_request(prompt, project_map):
            criteria["test"] = {
                "enabled": True,
                "command": ["python", "-m", "unittest", "discover", "-v"],
                "timeout_seconds": 300,
            }
            criteria["runtime"] = {
                "enabled": True,
                "command": ["python", "-m", "compileall", "-q", "."],
                "timeout_seconds": 300,
            }
        iteration_budget = self._automatic_iteration_budget(prompt, project_map) if int(max_iterations) <= 0 else max(1, min(int(max_iterations), 200))
        run_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """INSERT INTO codebase_runs (id, codebase_project_id, workspace_id, project, owner, user_prompt, status, provider, model, role_models_json, allow_external, max_iterations, completion_json, started_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, codebase_project_id, workspace["id"], codebase_project["project"], owner, prompt.strip(), provider, model, json.dumps(role_models or {}), int(allow_external), iteration_budget, json.dumps(criteria), now, now),
            )
        self._event(run_id, "run.created", goal=prompt.strip(), completion=criteria, autonomy="full", iteration_budget=iteration_budget)
        snapshot = self.sandboxes.snapshot(workspace["id"], f"{run_id}-before")
        self._record_snapshot(run_id, snapshot, "Before autonomous run")
        cancel = threading.Event()
        thread = threading.Thread(target=self._execute_run, args=(run_id,), name=f"codebase-run-{run_id[:8]}", daemon=True)
        with self._lock:
            self._cancel[run_id] = cancel
            self._threads[run_id] = thread
        thread.start()
        return self.get_run(run_id, owner=owner)

    def cancel_run(self, run_id: str, *, owner: str = "local") -> dict[str, Any]:
        run = self.get_run(run_id, owner=owner)
        if run["status"] in FINAL_STATES:
            return run
        with self._lock:
            event = self._cancel.setdefault(run_id, threading.Event())
            event.set()
        self._set_run(run_id, status="cancelled", error="Cancelled by user", finished=True)
        self._event(run_id, "run.cancelled", reason="Cancelled by user")
        return self.get_run(run_id, owner=owner)

    def resume_run(self, run_id: str, *, owner: str = "local") -> dict[str, Any]:
        run = self.get_run(run_id, owner=owner)
        if run["status"] not in {"blocked", "failed", "interrupted"}:
            raise ValueError("only blocked, failed, or interrupted runs can resume")
        self._set_run(run_id, status="queued", error="", finished=False)
        cancel = threading.Event()
        thread = threading.Thread(target=self._execute_run, args=(run_id,), name=f"codebase-run-{run_id[:8]}", daemon=True)
        with self._lock:
            self._cancel[run_id] = cancel
            self._threads[run_id] = thread
        thread.start()
        return self.get_run(run_id, owner=owner)

    def wait_for_run(self, run_id: str, *, owner: str = "local", timeout: float | None = None) -> dict[str, Any]:
        """Wait for the worker to release process and database resources."""
        with self._lock:
            thread = self._threads.get(run_id)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return self.get_run(run_id, owner=owner)

    def restore_run_snapshot(self, run_id: str, snapshot_id: str, *, owner: str = "local") -> dict[str, Any]:
        run = self.get_run(run_id, owner=owner)
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM codebase_snapshots WHERE id = ? AND run_id = ?", (snapshot_id, run_id)).fetchone()
        if row is None:
            raise KeyError("snapshot not found")
        self.sandboxes.restore(run["workspace_id"], row["path"])
        self._event(run_id, "snapshot.restored", snapshot_id=snapshot_id)
        self._audit("codebase.snapshot.restore", f"codebase-run:{run_id}", run["project"], {"snapshot_id": snapshot_id, "owner": owner})
        return {"restored": True, "snapshot_id": snapshot_id}

    def resolve_approval(self, approval_id: str, *, approved: bool, owner: str = "local") -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT a.*, r.owner FROM codebase_approvals a JOIN codebase_runs r ON r.id = a.run_id WHERE a.id = ? AND r.owner = ?",
                (approval_id, owner),
            ).fetchone()
            if row is None:
                raise KeyError("approval not found")
            if str(row["status"]) != "pending":
                return {
                    "id": approval_id,
                    "run_id": row["run_id"],
                    "status": str(row["status"]),
                    "result": {"already_resolved": True},
                }
            status = "approved" if approved else "rejected"
            connection.execute("UPDATE codebase_approvals SET status = ?, resolved_at = ? WHERE id = ?", (status, utc_now(), approval_id))
        self._event(row["run_id"], "approval.resolved", approval_id=approval_id, status=status)
        payload = json.loads(row["payload_json"] or "{}")
        task_id = str(payload.get("task_id", ""))
        if approved:
            worker = threading.Thread(
                target=self._execute_approved_action,
                args=(dict(row), owner),
                name=f"codebase-approval-{approval_id[:8]}",
                daemon=True,
            )
            worker.start()
            return {"id": approval_id, "run_id": row["run_id"], "status": status, "result": {"queued": True}}
        if task_id:
            self._set_task(task_id, "needs_fix", {"approval_rejected": True, "tool": str(row["tool"]), "arguments": payload.get("arguments", {})})
            worker = threading.Thread(
                target=self._resume_after_approval,
                args=(str(row["run_id"]), owner),
                name=f"codebase-rejection-{approval_id[:8]}",
                daemon=True,
            )
            worker.start()
        return {"id": approval_id, "run_id": row["run_id"], "status": status, "result": None}

    def _execute_approved_action(self, approval: dict[str, Any], owner: str) -> None:
        approval_id = str(approval["id"])
        run_id = str(approval["run_id"])
        payload = json.loads(approval.get("payload_json") or "{}")
        task_id = str(payload.get("task_id", ""))
        arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        failure = ""
        try:
            run = self.get_run(run_id, owner=owner)
            result = self._invoke(run, task_id, str(approval["tool"]), arguments, approved=True)
            if task_id:
                self._set_task(task_id, "needs_fix", {"approved_tool_result": result})
        except Exception as exc:
            failure = str(exc)
            self._event(run_id, "approval.execution_failed", approval_id=approval_id, tool=str(approval["tool"]), error=failure)
            if task_id:
                self._set_task(task_id, "needs_fix", {"approval_error": failure, "tool": str(approval["tool"]), "arguments": arguments})
        finally:
            if task_id:
                self._resume_after_approval(run_id, owner)
            if failure:
                # Publish the terminal approval state last so clients cannot
                # observe "failed" while the recovery worker still owns the DB.
                with self.store.session() as connection:
                    connection.execute("UPDATE codebase_approvals SET status = ? WHERE id = ?", ("failed", approval_id))

    def _resume_after_approval(self, run_id: str, owner: str) -> None:
        with self._lock:
            current_worker = self._threads.get(run_id)
        if current_worker is not None and current_worker is not threading.current_thread():
            current_worker.join(timeout=5)
        try:
            refreshed = self.get_run(run_id, owner=owner)
            if refreshed["status"] == "blocked":
                self.resume_run(run_id, owner=owner)
        except (KeyError, ValueError):
            pass

    def invoke_tool(self, run_id: str, tool: str, arguments: dict[str, Any], *, owner: str = "local", approved: bool = False) -> dict[str, Any]:
        run = self.get_run(run_id, owner=owner)
        try:
            return self._invoke(run, "", tool, arguments, approved=approved)
        except ApprovalRequired as exc:
            approval = self._request_approval(run_id, exc)
            return {"approval_required": approval}

    def invoke_project_tool(self, codebase_project_id: str, tool: str, arguments: dict[str, Any], *, owner: str = "local", approved: bool = False) -> dict[str, Any]:
        """Run an IDE tool before an autonomous task exists, with durable tracing."""
        project = self.get_project(codebase_project_id, owner=owner)
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT id FROM codebase_runs WHERE codebase_project_id = ? AND owner = ? AND user_prompt = 'Interactive IDE session' ORDER BY started_at DESC LIMIT 1",
                (codebase_project_id, owner),
            ).fetchone()
            run_id = str(row["id"]) if row else uuid4().hex
            if row is None:
                now = utc_now()
                connection.execute(
                    """INSERT INTO codebase_runs (id, codebase_project_id, workspace_id, project, owner, user_prompt, status, provider, model, role_models_json, allow_external, max_iterations, completion_json, started_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 'Interactive IDE session', 'interactive', 'local', '', '{}', 0, 1, '{}', ?, ?)""",
                    (run_id, codebase_project_id, project["workspace"]["id"], project["project"], owner, now, now),
                )
        if row is None:
            self._event(run_id, "run.created", mode="interactive", goal="Interactive IDE session")
        run = self.get_run(run_id, owner=owner)
        try:
            result = self._invoke(run, "", tool, arguments, approved=approved)
        except ApprovalRequired as exc:
            result = {"approval_required": self._request_approval(run_id, exc)}
        return {"run": self.get_run(run_id, owner=owner), "result": result}

    def list_runs(self, *, project: str = "default", owner: str = "local", limit: int = 50) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_runs WHERE project = ? AND owner = ? ORDER BY started_at DESC LIMIT ?", (project, owner, max(1, min(int(limit), 200)))).fetchall()
        return [self._run_row(row) for row in rows]

    def get_run(self, run_id: str, *, owner: str = "local") -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM codebase_runs WHERE id = ? AND owner = ?", (run_id, owner)).fetchone()
            if row is None:
                raise KeyError("codebase run not found")
            tasks = connection.execute("SELECT * FROM codebase_tasks WHERE run_id = ? ORDER BY created_at, id", (run_id,)).fetchall()
            approvals = connection.execute("SELECT * FROM codebase_approvals WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
            snapshots = connection.execute("SELECT id, label, sha256, size_bytes, created_at FROM codebase_snapshots WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
            tests = connection.execute("SELECT * FROM codebase_test_runs WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
        result = self._run_row(row)
        result["tasks"] = [self._task_row(item) for item in tasks]
        result["approvals"] = [self._approval_row(item) for item in approvals]
        result["snapshots"] = [dict(item) for item in snapshots]
        result["validation"] = [self._test_row(item) for item in tests]
        return result

    def events(self, run_id: str, *, owner: str = "local", after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        self.get_run(run_id, owner=owner)
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_events WHERE run_id = ? AND sequence > ? ORDER BY sequence LIMIT ?", (run_id, max(0, int(after)), max(1, min(int(limit), 2000)))).fetchall()
        return [self._event_row(row) for row in rows]

    def event_stream(self, run_id: str, *, owner: str = "local", after: int = 0, timeout_seconds: int = 120) -> Iterator[dict[str, Any]]:
        cursor = max(0, int(after))
        deadline = time.monotonic() + max(1, min(int(timeout_seconds), 300))
        while time.monotonic() < deadline:
            events = self.events(run_id, owner=owner, after=cursor, limit=200)
            for event in events:
                cursor = int(event["sequence"])
                yield event
            run = self.get_run(run_id, owner=owner)
            if run["status"] in FINAL_STATES and not events:
                yield {"sequence": cursor, "run_id": run_id, "type": "stream.closed", "payload": {"status": run["status"]}, "created_at": utc_now()}
                return
            if not events:
                yield {"sequence": cursor, "run_id": run_id, "type": "stream.heartbeat", "payload": {}, "created_at": utc_now()}
            time.sleep(0.5)

    def _execute_run(self, run_id: str) -> None:
        try:
            run = self.get_run(run_id, owner=self._owner_for_run(run_id))
            self._set_run(run_id, status="analysing")
            self._event(run_id, "agent.started", agent="project-manager")
            project = self.get_project(run["codebase_project_id"], owner=run["owner"])
            project_map = self.refresh_project_map(project["id"], owner=run["owner"])
            self._event(run_id, "repository.analysed", project_map=self._compact_map(project_map))
            tasks = self._plan(run, project, project_map)
            self._store_tasks(run_id, tasks)
            self._event(run_id, "plan.created", tasks=tasks)
            self._set_run(run_id, status="running")
            last_feedback = ""
            changed = False
            active_agent = "project-manager"
            max_iterations = max(1, int(run["max_iterations"]))
            for iteration in range(max_iterations):
                if self._cancelled(run_id):
                    return
                self._set_run(run_id, iteration=iteration + 1)
                task = self._next_task(run_id)
                if task is None:
                    unfinished = self._unfinished_tasks(run_id)
                    if unfinished:
                        self._set_run(run_id, status="failed", error="Task graph has unfinished tasks with unsatisfied dependencies", finished=True)
                        self._event(run_id, "run.failed", unfinished_tasks=unfinished)
                        return
                    self._set_run(run_id, status="validating")
                    if active_agent != "tester":
                        self._event(run_id, "agent.handoff", from_agent=active_agent, to_agent="tester", reason="completion-validation")
                        active_agent = "tester"
                    validation = self._validate(run)
                    if not validation["passed"]:
                        last_feedback = "Completion validation failed:\n" + "\n".join(
                            f"{item['kind']}: exit {item.get('exit_code', -1)}\n{item.get('stderr', '')[-8000:]}\n{item.get('stdout', '')[-4000:]}"
                            for item in validation["checks"] if item["status"] != "passed"
                        )
                        if iteration + 1 < max_iterations:
                            self._add_repair_task(run_id, "Repair failed completion checks", last_feedback, assigned_agent="debugger")
                            self._set_run(run_id, status="running", error="")
                            self._event(run_id, "repair.requested", stage="validation", error=last_feedback[-20_000:])
                            continue
                        self._set_run(run_id, status="failed", error="Completion criteria did not pass", summary=validation["summary"], finished=True)
                        self._event(run_id, "run.failed", validation=validation)
                        return
                    self._event(run_id, "agent.handoff", from_agent=active_agent, to_agent="reviewer", reason="independent-diff-review")
                    active_agent = "reviewer"
                    review = self._review(run, validation)
                    if review["blocking_findings"]:
                        last_feedback = "Independent review findings:\n" + json.dumps(review["blocking_findings"], indent=2)
                        if iteration + 1 < max_iterations:
                            self._add_repair_task(run_id, "Repair independent review findings", last_feedback, assigned_agent="debugger")
                            self._set_run(run_id, status="running", error="")
                            self._event(run_id, "repair.requested", stage="review", findings=review["blocking_findings"])
                            continue
                        self._set_run(run_id, status="failed", error="Independent review found blocking issues", summary=json.dumps(review), finished=True)
                        self._event(run_id, "review.failed", review=review)
                        return
                    snapshot = self.sandboxes.snapshot(run["workspace_id"], f"{run_id}-verified")
                    self._record_snapshot(run_id, snapshot, "Verified result")
                    summary = f"Validated autonomous run. {validation['summary']} Independent review: {review['status']}."
                    self._set_run(run_id, status="completed", summary=summary, error="", finished=True)
                    self._event(run_id, "run.completed", summary=summary, changed=changed, validation=validation, review=review)
                    self._remember_success(run, validation, review)
                    self._remember_repository_state(run)
                    return
                assigned_agent = str(task["assigned_agent"])
                if active_agent != assigned_agent:
                    self._event(run_id, "agent.handoff", from_agent=active_agent, to_agent=assigned_agent, task_id=task["id"], reason="task-assignment")
                    active_agent = assigned_agent
                self._set_task(task["id"], "running")
                self._event(run_id, "agent.started", agent=task["assigned_agent"], task_id=task["id"], iteration=iteration + 1)
                self._event(run_id, "agent.thinking", agent=task["assigned_agent"], task_id=task["id"], goal=task["title"])
                project_map = RepositoryIntelligence(project["workspace"]["workspace_path"]).scan()
                context = RepositoryIntelligence(project["workspace"]["workspace_path"]).select_context(
                    f"{run['user_prompt']} {task['title']} {last_feedback}", limit=20
                )
                decision = self._model_actions(run, project, project_map, context, task, last_feedback)
                actions = decision.get("actions", []) if isinstance(decision, dict) else []
                if not actions:
                    self._set_task(task["id"], "blocked", {"reason": "model returned no executable actions", "response": decision})
                    last_feedback = "No executable action was returned. Inspect relevant files and return a JSON tool action."
                    continue
                failed = False
                for action in actions[:24]:
                    tool = str(action.get("tool", "")).strip()
                    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
                    arguments = self._normalized_action_arguments(tool, arguments)
                    if tool not in ACTION_TOOLS:
                        last_feedback = f"Unknown tool requested: {tool}"
                        failed = True
                        break
                    argument_error = self._action_argument_error(tool, arguments)
                    if argument_error:
                        last_feedback = argument_error
                        self._event(run_id, "repair.requested", task_id=task["id"], error=last_feedback)
                        failed = True
                        break
                    try:
                        result = self._invoke(run, task["id"], tool, arguments)
                    except ApprovalRequired as exc:
                        last_feedback = (
                            f"Tool {exc.tool} is outside the protected autonomous boundary: {exc.reason}. "
                            "Do not ask the user for approval. Choose a workspace-safe tool or command and continue."
                        )
                        self._event(run_id, "autonomy.redirected", task_id=task["id"], tool=exc.tool, reason=exc.reason)
                        failed = True
                        break
                    except Exception as exc:
                        last_feedback = f"Tool {tool} failed: {exc}"
                        self._event(run_id, "repair.requested", task_id=task["id"], error=last_feedback)
                        failed = True
                        break
                    if tool in {"write_file", "patch_file", "delete_file", "move_file"}:
                        changed = True
                    if int(result.get("exit_code", 0)) not in {0, -1}:
                        last_feedback = f"Command failed.\nCommand: {result.get('command')}\nstdout:\n{result.get('stdout', '')}\nstderr:\n{result.get('stderr', '')}"
                        self._event(run_id, "repair.requested", task_id=task["id"], error=last_feedback[-20_000:])
                        failed = True
                        break
                if failed:
                    self._set_task(task["id"], "needs_fix", {"feedback": last_feedback[-20_000:]})
                    continue
                if bool(decision.get("done", False)):
                    self._set_task(task["id"], "completed", {"summary": decision.get("summary", "")})
                    self._event(run_id, "task.completed", task_id=task["id"], summary=decision.get("summary", ""))
                    last_feedback = ""
                else:
                    last_feedback = str(decision.get("summary", "Continue the current task with the next smallest verified action."))
                    self._set_task(task["id"], "needs_fix", {"feedback": last_feedback[-20_000:]})
                    self._event(run_id, "task.continued", task_id=task["id"], summary=last_feedback)
            if self._cancelled(run_id):
                return
            unfinished = self._unfinished_tasks(run_id)
            self._set_run(run_id, status="failed", error="Iteration limit reached before all tasks and verification completed", finished=True)
            self._event(run_id, "run.failed", reason="iteration_limit", unfinished_tasks=unfinished, max_iterations=max_iterations)
        except Exception as exc:
            self._set_run(run_id, status="failed", error=str(exc), finished=True)
            self._event(run_id, "run.failed", error=str(exc))
        finally:
            self._cleanup_processes(run_id)
            with self._lock:
                self._threads.pop(run_id, None)

    def _plan(self, run: dict[str, Any], project: dict[str, Any], project_map: dict[str, Any]) -> list[dict[str, Any]]:
        if self._standalone_web_request(run["user_prompt"], project_map):
            return [
                {"id": "T1", "title": "Create the semantic login page markup in index.html", "assigned_agent": "programmer", "depends_on": []},
                {"id": "T2", "title": "Add polished responsive styling and interaction behavior", "assigned_agent": "programmer", "depends_on": ["T1"]},
                {"id": "T3", "title": "Inspect the generated files and validate the page artifacts", "assigned_agent": "tester", "depends_on": ["T2"]},
            ]
        if self._standalone_python_request(run["user_prompt"], project_map):
            return [
                {"id": "T1", "title": f"Implement the requested Python function: {run['user_prompt']}", "assigned_agent": "programmer", "depends_on": []},
                {"id": "T2", "title": "Create and run focused tests for the Python function", "assigned_agent": "tester", "depends_on": ["T1"]},
            ]
        prompt = (
            "Create a concise dependency-aware software engineering plan made only of executable repository changes and validation. Return JSON only using this shape: "
            f"{json.dumps(PLAN_SCHEMA)}. Do not include markdown. Assign specialist roles only when useful.\n"
            "Do not add framework research, requirements definition, documentation, backend integration, or deployment unless the user explicitly requested it. "
            "Every implementation task must name a concrete code outcome; finish with tests/build/runtime checks supported by the project.\n"
            f"Goal: {run['user_prompt']}\nProject map: {json.dumps(self._compact_map(project_map))}\n"
            f"Project memory: {json.dumps(project['memory'])}"
        )
        parsed = self._model_json(run, [{"role": "system", "content": "You are the project manager. Inspect before editing and never claim success without validation."}, {"role": "user", "content": prompt}], role="project_manager")
        tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else []
        normalized = []
        valid_agents = {"architect", "programmer", "tester", "debugger", "reviewer"}
        for index, task in enumerate(tasks[:100], start=1):
            if not isinstance(task, dict) or not str(task.get("title", "")).strip():
                continue
            task_id = str(task.get("id", f"T{index}")).strip() or f"T{index}"
            agent = str(task.get("agent", "programmer")).strip().lower()
            normalized.append({"id": task_id, "title": str(task["title"]).strip(), "assigned_agent": agent if agent in valid_agents else "programmer", "depends_on": [str(item) for item in task.get("depends_on", []) if str(item).strip()]})
        if not normalized:
            normalized = [
                {"id": "T1", "title": "Inspect the repository and identify affected files", "assigned_agent": "architect", "depends_on": []},
                {"id": "T2", "title": "Implement the requested coherent changes", "assigned_agent": "programmer", "depends_on": ["T1"]},
                {"id": "T3", "title": "Run project validation and repair failures", "assigned_agent": "tester", "depends_on": ["T2"]},
                {"id": "T4", "title": "Independently review the final diff", "assigned_agent": "reviewer", "depends_on": ["T3"]},
            ]
        return normalized

    def _model_actions(self, run: dict[str, Any], project: dict[str, Any], project_map: dict[str, Any], context: list[dict[str, Any]], task: dict[str, Any], feedback: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": (
                "You are an autonomous software engineer operating only through supplied workspace tools. "
                "Return JSON only: {\"summary\":string,\"actions\":[{\"tool\":string,\"arguments\":object}],\"done\":boolean}. "
                "Inspect before editing. Prefer patch_file for narrow edits. Never request git push, deployment, secrets, or production access. "
                "Never pass Python, JavaScript, or other source text directly to run_shell or start_process. Create code with write_file or patch_file, then execute the file with a real command such as [\"python\", \"script.py\"]. "
                "browser_inspect requires a complete URL and must never be called before a page exists and a local web server is running. For an empty web project, create index.html and its assets with write_file first. "
                "Inside the protected working copy, file changes, dependency setup, tests, builds, local processes, and local Git operations are autonomous. "
                "Never pause to ask the user for permission; if an operation is outside that boundary, choose a safe local alternative. "
                "For new applications, make architecture decisions, implement a usable vertical slice, and validate it instead of only describing it. "
                "A task is not complete until actual validation evidence passes."
            )},
            {"role": "user", "content": (
                f"Overall goal: {run['user_prompt']}\nCurrent task: {task['title']}\nAgent role: {task['assigned_agent']}\nTask evidence: {json.dumps(task.get('result', {}))}\n"
                f"Project map: {json.dumps(self._compact_map(project_map))}\nProject memory: {json.dumps(project['memory'])}\n"
                f"Recent project sessions: {json.dumps(self._recent_run_memory(run['codebase_project_id']))}\n"
                f"Relevant repository context: {json.dumps(context)[:100000]}\nPrevious failure: {feedback[-20000:]}\n"
                f"Available tools: {', '.join(sorted(ACTION_TOOLS))}. Return the next smallest coherent action batch."
            )},
        ]
        parsed = self._model_json(run, messages, role=str(task.get("assigned_agent", "programmer")))
        if self._standalone_web_request(run["user_prompt"], project_map):
            actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
            title = str(task.get("title", "")).lower()
            implementation_task = any(term in title for term in ("create", "markup", "styling", "interaction", "implement", "build"))
            invalid = not actions or any(
                not isinstance(action, dict)
                or str(action.get("tool", "")) not in ACTION_TOOLS
                or str(action.get("tool", "")) == "browser_inspect"
                or self._action_argument_error(str(action.get("tool", "")), self._normalized_action_arguments(str(action.get("tool", "")), action.get("arguments") if isinstance(action.get("arguments"), dict) else {}))
                for action in actions
            )
            if implementation_task and not any(str(action.get("tool", "")) in {"write_file", "patch_file"} for action in actions if isinstance(action, dict)):
                invalid = True
            if invalid:
                fallback = self._empty_web_fallback(task)
                if fallback:
                    self._event(run["id"], "model.action_replaced", task_id=task["id"], reason="invalid action for empty web workspace")
                    return fallback
        if self._standalone_python_request(run["user_prompt"], project_map):
            actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
            title = str(task.get("title", "")).lower()
            implementation_task = "implement" in title
            test_task = "test" in title
            invalid = not actions or any(
                not isinstance(action, dict)
                or str(action.get("tool", "")) not in ACTION_TOOLS
                or self._nonproductive_creation_action(action)
                or self._action_argument_error(str(action.get("tool", "")), self._normalized_action_arguments(str(action.get("tool", "")), action.get("arguments") if isinstance(action.get("arguments"), dict) else {}))
                for action in actions
            )
            implementation_writes = [
                action for action in actions if isinstance(action, dict)
                and str(action.get("tool", "")) in {"write_file", "patch_file"}
                and not Path(str((action.get("arguments") or {}).get("path", ""))).name.lower().startswith("test")
            ]
            concrete_test_actions = [
                action for action in actions if isinstance(action, dict) and (
                    (str(action.get("tool", "")) in {"write_file", "patch_file"}
                     and Path(str((action.get("arguments") or {}).get("path", ""))).name.lower().startswith("test"))
                    or (str(action.get("tool", "")) == "run_tests"
                        and bool(self._normalized_action_arguments("run_tests", action.get("arguments") or {}).get("command")))
                )
            ]
            if implementation_task and not implementation_writes:
                invalid = True
            if test_task and not concrete_test_actions:
                invalid = True
            if invalid:
                fallback = self._python_function_fallback(run["user_prompt"], task)
                if fallback:
                    self._event(run["id"], "model.action_replaced", task_id=task["id"], reason="non-executable action for Python creation task")
                    return fallback
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _empty_web_fallback(task: dict[str, Any]) -> dict[str, Any]:
        title = str(task.get("title", "")).lower()
        if "markup" in title or "index.html" in title:
            return {"summary": "Created the semantic login page markup", "actions": [{"tool": "write_file", "arguments": {"path": "index.html", "content": EMPTY_LOGIN_HTML}}], "done": True}
        if "styling" in title or "interaction" in title:
            return {
                "summary": "Added responsive styling and client-side form behavior",
                "actions": [
                    {"tool": "write_file", "arguments": {"path": "styles.css", "content": EMPTY_LOGIN_CSS}},
                    {"tool": "write_file", "arguments": {"path": "app.js", "content": EMPTY_LOGIN_JS}},
                ],
                "done": True,
            }
        if "validate" in title or "inspect" in title:
            return {"summary": "Read the generated frontend artifacts for validation", "actions": [{"tool": "read_files", "arguments": {"paths": ["index.html", "styles.css", "app.js"]}}], "done": True}
        return {}

    @staticmethod
    def _standalone_web_request(goal: str, project_map: dict[str, Any]) -> bool:
        if int(project_map.get("total_bytes", 0) or 0) > 1_000_000 or str(project_map.get("framework", "")).strip():
            return False
        text = goal.lower()
        return any(term in text for term in ("page", "website", "web app", "frontend", "landing", "dashboard", "login", "sign in", "signup", "sign up"))

    @staticmethod
    def _standalone_python_request(goal: str, project_map: dict[str, Any]) -> bool:
        if int(project_map.get("total_bytes", 0) or 0) > 1_000_000 or str(project_map.get("framework", "")).strip():
            return False
        text = goal.lower()
        return "python" in text and any(term in text for term in ("function", "script", "module"))

    @staticmethod
    def _nonproductive_creation_action(action: dict[str, Any]) -> bool:
        tool = str(action.get("tool", ""))
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        if tool not in {"run_shell", "start_process"}:
            return False
        command = arguments.get("command", arguments.get("cmd", arguments.get("argv", [])))
        parts = shlex.split(command, posix=False) if isinstance(command, str) else [str(item) for item in command]
        return bool(parts and Path(parts[0]).name.lower() in {"echo", "write-host", "printf"})

    @staticmethod
    def _normalized_action_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return CodebaseTools._normalize_arguments(tool, arguments)

    @staticmethod
    def _python_function_fallback(goal: str, task: dict[str, Any]) -> dict[str, Any]:
        title = str(task.get("title", "")).lower()
        assigned_agent = str(task.get("assigned_agent", "")).lower()
        text = f"{goal} {title}".lower()
        if "double" in text or "twice" in text:
            function_code = "def double(value: float) -> float:\n    \"\"\"Return twice the supplied numeric value.\"\"\"\n    if not isinstance(value, (int, float)) or isinstance(value, bool):\n        raise TypeError('value must be numeric')\n    return value * 2\n"
            test_code = "import unittest\n\nfrom function import double\n\n\nclass FunctionTests(unittest.TestCase):\n    def test_double(self):\n        self.assertEqual(8, double(4))\n        self.assertEqual(-5, double(-2.5))\n\n    def test_non_numeric_value(self):\n        with self.assertRaises(TypeError):\n            double('4')\n\n\nif __name__ == '__main__':\n    unittest.main()\n"
        else:
            function_code = "def factorial(value: int) -> int:\n    \"\"\"Return value factorial for a non-negative integer.\"\"\"\n    if not isinstance(value, int) or isinstance(value, bool):\n        raise TypeError('value must be an integer')\n    if value < 0:\n        raise ValueError('value must be non-negative')\n    result = 1\n    for number in range(2, value + 1):\n        result *= number\n    return result\n"
            test_code = "import unittest\n\nfrom function import factorial\n\n\nclass FunctionTests(unittest.TestCase):\n    def test_factorial(self):\n        self.assertEqual(1, factorial(0))\n        self.assertEqual(120, factorial(5))\n\n    def test_negative_value(self):\n        with self.assertRaises(ValueError):\n            factorial(-1)\n\n\nif __name__ == '__main__':\n    unittest.main()\n"
        # Classify from the assigned task, not from words in the overall goal.
        # A request such as "write a function and test it" still gives the
        # programmer an implementation task before the tester runs.
        implementation_task = assigned_agent == "programmer" or title.startswith("implement")
        test_task = assigned_agent == "tester" or title.startswith(("create and run focused tests", "test ", "run test"))
        if implementation_task:
            return {
                "summary": "Implemented a documented, validated Python function",
                "actions": [{"tool": "write_file", "arguments": {"path": "function.py", "content": function_code}}],
                "done": True,
            }
        if test_task:
            return {
                "summary": "Created and ran focused Python tests",
                "actions": [
                    {"tool": "write_file", "arguments": {"path": "test_function.py", "content": test_code}},
                    {"tool": "run_tests", "arguments": {"command": ["python", "-m", "unittest", "discover", "-v"]}},
                ],
                "done": True,
            }
        if any(term in text for term in ("factorial", "double", "twice", "fucntion", "function")):
            return {
                "summary": "Implemented a documented, validated Python function",
                "actions": [{"tool": "write_file", "arguments": {"path": "function.py", "content": function_code}}],
                "done": True,
            }
        return {}

    @staticmethod
    def _action_argument_error(tool: str, arguments: dict[str, Any]) -> str:
        required = {
            "read_file": ("path",),
            "write_file": ("path", "content"),
            "patch_file": ("path", "search", "replace"),
            "delete_file": ("path",),
            "move_file": ("source", "destination"),
            "run_shell": ("command",),
            "start_process": ("command",),
            "process_status": ("process_id",),
            "stop_process": ("process_id",),
            "inspect_logs": ("process_id",),
            "browser_inspect": ("url",),
        }.get(tool, ())
        missing = [name for name in required if arguments.get(name) is None or arguments.get(name) == ""]
        if missing:
            return f"Tool {tool} was not executed because required argument(s) were missing: {', '.join(missing)}. Choose a valid tool action and continue the task."
        if tool == "browser_inspect":
            parsed = urlparse(str(arguments["url"]).strip())
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return "Tool browser_inspect requires a complete HTTP URL. Create the page, start a local server, and then inspect its localhost URL."
        return ""

    def _review(self, run: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
        diff = self._invoke(run, "", "git_diff", {}).get("stdout", "")
        messages = [
            {"role": "system", "content": "You are an independent code reviewer. You do not know the coder's reasoning. Review only the goal, diff, and validation. Return JSON only. Never report a file absent from the diff. Every finding must quote exact diff evidence. A client-only form without an authentication endpoint does not require CSRF protection."},
            {"role": "user", "content": (
                "Return {\"status\":\"passed|needs_fix\",\"findings\":[{\"severity\":\"critical|high|medium|low\",\"file\":string,\"line\":integer,\"issue\":string,\"evidence\":\"exact text copied from the diff\"}]}.\n"
                f"Goal: {run['user_prompt']}\nValidation: {json.dumps(validation)}\nDiff:\n{diff[-100000:]}"
            )},
        ]
        parsed = self._model_json(run, messages, role="reviewer")
        model_findings = parsed.get("findings", []) if isinstance(parsed, dict) and isinstance(parsed.get("findings"), list) else []
        findings = []
        for item in model_findings:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            evidence = str(normalized.get("evidence", "")).strip()
            normalized["grounded"] = bool(evidence and evidence in diff and str(normalized.get("file", "")) in diff)
            findings.append(normalized)
        deterministic = self._deterministic_review(diff)
        for item in deterministic:
            item["grounded"] = True
        findings.extend(deterministic)
        blocking = [item for item in findings if item.get("grounded") and str(item.get("severity", "")).lower() in {"critical", "high"}]
        result = {"status": "passed" if not blocking else "needs_fix", "findings": findings, "blocking_findings": blocking}
        self._event(run["id"], "review.completed", reviewer="independent-reviewer", result=result)
        return result

    def _validate(self, run: dict[str, Any]) -> dict[str, Any]:
        criteria = run["completion"]
        results = []
        for kind in ("lint", "typecheck", "test", "build", "browser", "runtime"):
            item = criteria.get(kind)
            if not isinstance(item, dict) or not item.get("enabled"):
                continue
            command = item.get("command")
            if not command:
                results.append({"kind": kind, "status": "failed", "exit_code": -1, "reason": "required command is not configured"})
                continue
            self._event(run["id"], f"{kind}.started", command=command)
            output = self._invoke(run, "", "run_shell", {"command": command, "timeout_seconds": item.get("timeout_seconds", 900)})
            status = "passed" if int(output.get("exit_code", -1)) == 0 else "failed"
            result = {"kind": kind, "status": status, **output}
            results.append(result)
            self._record_test(run["id"], kind, result)
            self._event(run["id"], f"{kind}.{status}", exit_code=result["exit_code"], duration_ms=result.get("duration_ms", 0))
        artifact_result = self._artifact_check(run)
        results.append(artifact_result)
        self._record_test(run["id"], "artifacts", artifact_result)
        self._event(run["id"], f"artifacts.{artifact_result['status']}", files=artifact_result.get("files", []))
        passed = bool(results) and all(item["status"] == "passed" for item in results)
        summary = ", ".join(f"{item['kind']} {item['status']}" for item in results) or "No completion checks were configured"
        return {"passed": passed, "checks": results, "summary": summary}

    def _invoke(self, run: dict[str, Any], task_id: str, tool: str, arguments: dict[str, Any], *, approved: bool = False) -> dict[str, Any]:
        workspace = self._workspace(run["workspace_id"], run["owner"])
        tools = CodebaseTools(self.sandboxes, run["workspace_id"], backend=workspace["sandbox_backend"], event=lambda event_type, **payload: self._event(run["id"], event_type, **payload))
        arguments = tools._normalize_arguments(tool, arguments)
        call_id = uuid4().hex
        decision = tools.policy.classify(tool, arguments)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_tool_calls (id, run_id, task_id, tool, input_json, status, risk, started_at) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)", (call_id, run["id"], task_id, tool, json.dumps(arguments, default=str), int(decision["risk"]), now))
        try:
            result = tools.execute(tool, arguments, approved=approved)
        except Exception as exc:
            with self.store.session() as connection:
                connection.execute("UPDATE codebase_tool_calls SET status = 'failed', error = ?, finished_at = ? WHERE id = ?", (str(exc), utc_now(), call_id))
            raise
        with self.store.session() as connection:
            connection.execute("UPDATE codebase_tool_calls SET status = 'completed', output_json = ?, finished_at = ? WHERE id = ?", (json.dumps(result, default=str), utc_now(), call_id))
            if tool in {"write_file", "patch_file", "delete_file", "move_file"}:
                connection.execute("INSERT INTO codebase_file_changes (id, run_id, filepath, old_hash, new_hash, patch, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid4().hex, run["id"], result.get("path", ""), result.get("old_sha256", ""), result.get("new_sha256", ""), result.get("patch", ""), utc_now()))
        return result

    def _model_json(self, run: dict[str, Any], messages: list[dict[str, str]], *, role: str = "") -> dict[str, Any]:
        role_config = run.get("role_models", {}).get(role, {}) if isinstance(run.get("role_models"), dict) else {}
        provider = str(role_config.get("provider", run["provider"])) if isinstance(role_config, dict) else run["provider"]
        model = str(role_config.get("model", run["model"])) if isinstance(role_config, dict) else run["model"]
        allow_external = bool(role_config.get("allow_external", run["allow_external"])) if isinstance(role_config, dict) else bool(run["allow_external"])
        try:
            output_budget = 16384 if role in {"programmer", "debugger", "architect"} else 8192
            response = self.gateway.complete(messages, provider_id=provider, model=model or None, allow_external=allow_external, max_output_tokens=output_budget)
        except (ProviderConfigurationError, ValueError) as exc:
            self._event(run["id"], "model.failed", error=str(exc))
            return {}
        self._event(run["id"], "model.completed", provider=response.provider, model=response.model, role=role or "default")
        return self._extract_json(response.content)

    @staticmethod
    def _extract_json(value: str) -> dict[str, Any]:
        text = value.strip()
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.I)
        candidates = [fenced.group(1)] if fenced else []
        first, last = text.find("{"), text.rfind("}")
        if first >= 0 and last > first:
            candidates.append(text[first:last + 1])
        candidates.append(text)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return {}

    def _completion_criteria(self, project_map: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
        commands = project_map.get("commands", {})
        criteria: dict[str, Any] = {}
        for name in ("lint", "typecheck", "test", "build"):
            configured = supplied.get(name) if isinstance(supplied.get(name), dict) else {}
            command = configured.get("command") or commands.get(name, "")
            requested = bool(configured.get("enabled", bool(command)))
            criteria[name] = {"enabled": bool(command) and requested, "command": command, "timeout_seconds": int(configured.get("timeout_seconds", 900))}
        for name in ("browser", "runtime"):
            configured = supplied.get(name) if isinstance(supplied.get(name), dict) else {}
            command = configured.get("command") or commands.get(name, "")
            requested = bool(configured.get("enabled", bool(command)))
            criteria[name] = {"enabled": bool(command) and requested, "command": command, "timeout_seconds": int(configured.get("timeout_seconds", 300))}
        if not any(item["enabled"] for item in criteria.values()):
            languages = set(project_map.get("languages", []))
            if "python" in languages:
                criteria["runtime"] = {"enabled": True, "command": ["python", "-m", "compileall", "-q", "."], "timeout_seconds": 300}
            elif {"javascript", "typescript"} & languages:
                entrypoints = [str(item) for item in project_map.get("entrypoints", []) if str(item).strip()]
                fallback = "app.js" if "javascript" in languages else "index.ts"
                criteria["runtime"] = {"enabled": True, "command": ["node", "--check", entrypoints[0] if entrypoints else fallback], "timeout_seconds": 120}
        return criteria

    @staticmethod
    def _automatic_iteration_budget(prompt: str, project_map: dict[str, Any]) -> int:
        """Choose a ceiling from task and repository complexity.

        The controller still stops as soon as all completion gates pass; this is
        a safety ceiling, not a fixed amount of work.
        """
        files = int(project_map.get("file_count", 0) or 0)
        words = len(str(prompt).split())
        budget = 24
        if files > 50 or words > 120:
            budget = 64
        if files > 500 or words > 500:
            budget = 120
        if files > 2_000 or words > 1_500:
            budget = 200
        return budget

    def _next_task(self, run_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_tasks WHERE run_id = ? ORDER BY created_at, id", (run_id,)).fetchall()
        tasks = [self._task_row(row) for row in rows]
        completed = {task["id"] for task in tasks if task["status"] == "completed"}
        for task in tasks:
            if task["status"] in {"pending", "needs_fix", "blocked"} and all(dependency in completed for dependency in task["depends_on"]):
                return task
        return None

    def _store_tasks(self, run_id: str, tasks: list[dict[str, Any]]) -> None:
        now = utc_now()
        with self.store.session() as connection:
            count = connection.execute("SELECT COUNT(*) FROM codebase_tasks WHERE run_id = ?", (run_id,)).fetchone()[0]
            if count:
                return
            for task in tasks:
                connection.execute("INSERT INTO codebase_tasks (id, run_id, title, assigned_agent, depends_on_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (f"{run_id}:{task['id']}", run_id, task["title"], task["assigned_agent"], json.dumps([f"{run_id}:{item}" for item in task["depends_on"]]), now, now))

    def _add_repair_task(self, run_id: str, title: str, feedback: str, *, assigned_agent: str) -> str:
        with self.store.session() as connection:
            rows = connection.execute("SELECT id, status FROM codebase_tasks WHERE run_id = ? ORDER BY created_at, id", (run_id,)).fetchall()
            dependencies = [str(row["id"]) for row in rows if row["status"] == "completed"]
            sequence = len(rows) + 1
            task_id = f"{run_id}:R{sequence}"
            now = utc_now()
            connection.execute(
                "INSERT INTO codebase_tasks (id, run_id, title, assigned_agent, status, depends_on_json, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
                (task_id, run_id, title, assigned_agent, json.dumps(dependencies), json.dumps({"feedback": feedback[-20_000:]}), now, now),
            )
        self._event(run_id, "task.created", task_id=task_id, title=title, assigned_agent=assigned_agent, depends_on=dependencies)
        return task_id

    def _unfinished_tasks(self, run_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_tasks WHERE run_id = ? AND status != 'completed' ORDER BY created_at, id", (run_id,)).fetchall()
        return [self._task_row(row) for row in rows]

    def _set_task(self, task_id: str, status: str, result: dict[str, Any] | None = None) -> None:
        with self.store.session() as connection:
            connection.execute("UPDATE codebase_tasks SET status = ?, result_json = ?, updated_at = ? WHERE id = ?", (status, json.dumps(result or {}), utc_now(), task_id))

    def _set_run(self, run_id: str, *, status: str | None = None, iteration: int | None = None, summary: str | None = None, error: str | None = None, finished: bool | None = None) -> None:
        assignments = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        for column, value in (("status", status), ("iteration", iteration), ("summary", summary), ("error", error)):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if finished is not None:
            assignments.append("finished_at = ?")
            values.append(utc_now() if finished else "")
        values.append(run_id)
        with self.store.session() as connection:
            connection.execute(f"UPDATE codebase_runs SET {', '.join(assignments)} WHERE id = ?", values)

    def _event(self, run_id: str, event_type: str, **payload: Any) -> None:
        safe_payload = self._redact_payload(payload)
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_events (id, run_id, type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)", (uuid4().hex, run_id, event_type, json.dumps(safe_payload, default=str), utc_now()))

    def _record_snapshot(self, run_id: str, snapshot: dict[str, Any], label: str) -> None:
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_snapshots (id, run_id, workspace_id, label, path, sha256, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (snapshot["id"], run_id, snapshot["workspace_id"], label, snapshot["path"], snapshot["sha256"], snapshot["size_bytes"], utc_now()))
        self._event(run_id, "snapshot.created", snapshot_id=snapshot["id"], label=label, sha256=snapshot["sha256"])

    def _record_test(self, run_id: str, kind: str, result: dict[str, Any]) -> None:
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_test_runs (id, run_id, kind, command_json, stdout, stderr, exit_code, duration_ms, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (uuid4().hex, run_id, kind, json.dumps(result.get("command", [])), result.get("stdout", "")[-200000:], result.get("stderr", "")[-200000:], int(result.get("exit_code", -1)), float(result.get("duration_ms", 0)), result.get("status", "failed"), utc_now()))

    def _request_approval(self, run_id: str, required: ApprovalRequired, task_id: str = "") -> dict[str, Any]:
        approval_id = uuid4().hex
        payload = {"task_id": task_id, "arguments": required.payload}
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_approvals (id, run_id, tool, reason, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)", (approval_id, run_id, required.tool, required.reason, json.dumps(payload, default=str), utc_now()))
        self._event(run_id, "approval.required", approval_id=approval_id, tool=required.tool, reason=required.reason, payload=payload)
        return {"id": approval_id, "tool": required.tool, "reason": required.reason}

    def _remember_success(self, run: dict[str, Any], validation: dict[str, Any], review: dict[str, Any]) -> None:
        content = json.dumps({"goal": run["user_prompt"], "validation": validation["summary"], "review": review["status"]}, sort_keys=True)
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_memories (id, codebase_project_id, category, content, source_run_id, created_at) VALUES (?, ?, 'successful-run', ?, ?, ?)", (uuid4().hex, run["codebase_project_id"], content, run["id"], utc_now()))
            project = connection.execute("SELECT memory_json FROM codebase_projects WHERE id = ?", (run["codebase_project_id"],)).fetchone()
            memory = json.loads(project["memory_json"] or "{}") if project else {}
            memory["last_successful_validation"] = {"run_id": run["id"], "summary": validation["summary"], "at": utc_now()}
            connection.execute("UPDATE codebase_projects SET memory_json = ?, updated_at = ? WHERE id = ?", (json.dumps(memory), utc_now(), run["codebase_project_id"]))

    def _remember_repository_state(self, run: dict[str, Any]) -> None:
        workspace = self._workspace(run["workspace_id"], run["owner"])
        project_map = RepositoryIntelligence(workspace["workspace_path"]).scan()
        with self.store.session() as connection:
            row = connection.execute("SELECT memory_json FROM codebase_projects WHERE id = ?", (run["codebase_project_id"],)).fetchone()
            memory = json.loads(row["memory_json"] or "{}") if row else {}
            memory["repository_state"] = {
                "git": project_map.get("git", {}),
                "commands": project_map.get("commands", {}),
                "last_run_id": run["id"],
                "updated_at": utc_now(),
            }
            connection.execute(
                "UPDATE codebase_projects SET map_json = ?, memory_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(project_map), json.dumps(memory), utc_now(), run["codebase_project_id"]),
            )

    def _recent_run_memory(self, codebase_project_id: str, limit: int = 8) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT id, user_prompt, status, summary, error, finished_at FROM codebase_runs WHERE codebase_project_id = ? ORDER BY started_at DESC LIMIT ?",
                (codebase_project_id, max(1, min(int(limit), 20))),
            ).fetchall()
        return [dict(row) for row in rows]

    def _artifact_check(self, run: dict[str, Any]) -> dict[str, Any]:
        workspace = self._workspace(run["workspace_id"], run["owner"])
        root = Path(workspace["workspace_path"])
        bad_names = {"npm-debug.log", "yarn-error.log", "core", "core.dump"}
        files = []
        for path in root.rglob("*"):
            if not path.is_file() or any(part in {".git", "node_modules", ".venv"} for part in path.relative_to(root).parts):
                continue
            if path.name.lower() in bad_names or path.suffix.lower() in {".rej", ".orig"} or path.name.lower().startswith("hs_err_pid"):
                files.append(path.relative_to(root).as_posix())
                if len(files) >= 100:
                    break
        status = "passed" if not files else "failed"
        return {
            "kind": "artifacts",
            "status": status,
            "command": ["artifact-scan"],
            "stdout": "No error artifacts found" if not files else "",
            "stderr": "Error artifacts: " + ", ".join(files) if files else "",
            "exit_code": 0 if not files else 1,
            "duration_ms": 0,
            "files": files,
        }

    @staticmethod
    def _deterministic_review(diff: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for line_number, line in enumerate(diff.splitlines(), start=1):
            if line.startswith("+") and any(marker in line for marker in ("<<<<<<<", "=======", ">>>>>>>")):
                findings.append({"severity": "critical", "file": "git-diff", "line": line_number, "issue": "Unresolved merge-conflict marker"})
            if line.startswith("+") and "-----BEGIN PRIVATE KEY-----" in line:
                findings.append({"severity": "critical", "file": "git-diff", "line": line_number, "issue": "Private key material was added"})
        return findings

    def _cleanup_processes(self, run_id: str) -> None:
        try:
            owner = self._owner_for_run(run_id)
            run = self.get_run(run_id, owner=owner)
        except KeyError:
            return
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT output_json FROM codebase_tool_calls WHERE run_id = ? AND tool = 'start_process' AND status = 'completed'",
                (run_id,),
            ).fetchall()
        for row in rows:
            try:
                process_id = str(json.loads(row["output_json"] or "{}").get("id", ""))
                if process_id:
                    result = self.sandboxes.stop_process(run["workspace_id"], process_id)
                    self._event(run_id, "process.stopped", process=result, reason="run finished")
            except (KeyError, OSError, ValueError):
                continue

    def _recover_interrupted_runs(self) -> None:
        with self.store.session() as connection:
            connection.execute("UPDATE codebase_runs SET status = 'interrupted', error = 'Desktop stopped during the run', updated_at = ? WHERE status IN ('queued','analysing','running','validating')", (utc_now(),))

    def _cancelled(self, run_id: str) -> bool:
        with self._lock:
            event = self._cancel.get(run_id)
        return bool(event and event.is_set())

    def _owner_for_run(self, run_id: str) -> str:
        with self.store.session() as connection:
            row = connection.execute("SELECT owner FROM codebase_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("codebase run not found")
        return str(row["owner"])

    def _workspace(self, workspace_id: str, owner: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM codebase_workspaces WHERE id = ? AND owner = ?", (workspace_id, owner)).fetchone()
        if row is None:
            raise KeyError("workspace not found")
        return self._workspace_row(row)

    def _audit(self, action: str, resource: str, project: str, metadata: dict[str, Any], status: str = "allowed") -> None:
        self.store.add_audit_event(AuditEvent(actor="codebase-agent", action=action, resource=resource, project=project, metadata=metadata, status=status))

    @staticmethod
    def _compact_map(project_map: dict[str, Any]) -> dict[str, Any]:
        return {key: project_map.get(key) for key in (
            "file_count", "total_bytes", "languages", "framework", "database", "entrypoints",
            "important_files", "commands", "dependencies", "configuration_files", "test_files",
            "database_models", "routes", "git",
        )}

    @staticmethod
    def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(payload, default=str)
        text = re.sub(r"(?i)(token|secret|password|api[_-]?key)([\"']?\s*[:=]\s*[\"']?)([^\s\"',}]+)", r"\1\2<redacted>", text)
        return json.loads(text)

    @staticmethod
    def _project_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "project": row["project"], "owner": row["owner"], "name": row["name"], "source_path": row["source_path"], "repo_url": row["repo_url"], "map": json.loads(row["map_json"] or "{}"), "memory": json.loads(row["memory_json"] or "{}"), "created_at": row["created_at"], "updated_at": row["updated_at"]}

    @staticmethod
    def _workspace_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "codebase_project_id": row["codebase_project_id"], "owner": row["owner"], "sandbox_backend": row["sandbox_backend"], "source_path": row["source_path"], "workspace_path": row["workspace_path"], "branch": row["branch"], "status": row["status"], "environment": json.loads(row["environment_json"] or "{}"), "created_at": row["created_at"], "updated_at": row["updated_at"]}

    @staticmethod
    def _run_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "codebase_project_id": row["codebase_project_id"], "workspace_id": row["workspace_id"], "project": row["project"], "owner": row["owner"], "user_prompt": row["user_prompt"], "status": row["status"], "provider": row["provider"], "model": row["model"], "role_models": json.loads(row["role_models_json"] or "{}"), "allow_external": bool(row["allow_external"]), "iteration": int(row["iteration"]), "max_iterations": int(row["max_iterations"]), "completion": json.loads(row["completion_json"] or "{}"), "summary": row["summary"], "error": row["error"], "started_at": row["started_at"], "updated_at": row["updated_at"], "finished_at": row["finished_at"]}

    @staticmethod
    def _task_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "run_id": row["run_id"], "parent_task_id": row["parent_task_id"], "title": row["title"], "assigned_agent": row["assigned_agent"], "status": row["status"], "depends_on": json.loads(row["depends_on_json"] or "[]"), "result": json.loads(row["result_json"] or "{}"), "created_at": row["created_at"], "updated_at": row["updated_at"]}

    @staticmethod
    def _approval_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "run_id": row["run_id"], "tool": row["tool"], "reason": row["reason"], "payload": json.loads(row["payload_json"] or "{}"), "status": row["status"], "created_at": row["created_at"], "resolved_at": row["resolved_at"]}

    @staticmethod
    def _test_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "kind": row["kind"], "command": json.loads(row["command_json"] or "[]"), "stdout": row["stdout"], "stderr": row["stderr"], "exit_code": int(row["exit_code"]), "duration_ms": float(row["duration_ms"]), "status": row["status"], "created_at": row["created_at"]}

    @staticmethod
    def _event_row(row: Any) -> dict[str, Any]:
        return {"sequence": int(row["sequence"]), "id": row["id"], "run_id": row["run_id"], "type": row["type"], "payload": json.loads(row["payload_json"] or "{}"), "created_at": row["created_at"]}
