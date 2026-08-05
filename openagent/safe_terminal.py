"""Audited local terminal execution with strict command safety rules."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any
from uuid import uuid4

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


DENIED_RE = re.compile(r"[;&|`$<>]")
REDACT_RE = re.compile(r"(?i)(token|secret|password|apikey|api_key)=([^\s]+)")
SAFE_COMMANDS = {
    ("python", "--version"),
    ("python.exe", "--version"),
    ("node", "--version"),
    ("git", "status"),
    ("git", "status", "--short"),
}


class SafeTerminal:
    """Run local commands without shell interpolation and store stream events."""

    def __init__(self, store: OpenAgentStore, *, workspace_root: str | Path | None = None) -> None:
        self.store = store
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS terminal_runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    cwd TEXT NOT NULL,
                    command_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    exit_code INTEGER NOT NULL DEFAULT -1,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_terminal_runs_project_owner
                    ON terminal_runs(project, owner, started_at DESC);

                CREATE TABLE IF NOT EXISTS terminal_logs (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    stream TEXT NOT NULL DEFAULT 'stdout',
                    chunk TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES terminal_runs(id) ON DELETE CASCADE
                );
                """
            )

    def run(
        self,
        *,
        command: list[str],
        cwd: str = "",
        project: str = "default",
        owner: str = "local",
        timeout_seconds: int = 30,
        approved: bool = False,
    ) -> dict[str, Any]:
        command = [str(part) for part in command if str(part).strip()]
        self._validate_command(command, approved=approved)
        resolved_cwd = self._validate_cwd(cwd)
        run_id = uuid4().hex
        started = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO terminal_runs (id, project, owner, cwd, command_json, status, started_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (run_id, project, owner, str(resolved_cwd), json.dumps(command), started),
            )
        try:
            proc = subprocess.run(
                command,
                cwd=str(resolved_cwd),
                capture_output=True,
                text=True,
                timeout=max(1, min(int(timeout_seconds), 300)),
                shell=False,
                check=False,
            )
            stdout = redact(proc.stdout)
            stderr = redact(proc.stderr)
            status = "completed" if proc.returncode == 0 else "failed"
            self._append_log(run_id, "stdout", stdout)
            self._append_log(run_id, "stderr", stderr)
            self._complete_run(run_id, status=status, exit_code=int(proc.returncode))
        except subprocess.TimeoutExpired as exc:
            stdout = redact(exc.stdout or "")
            stderr = redact(exc.stderr or "command timed out")
            status = "timeout"
            self._append_log(run_id, "stdout", stdout)
            self._append_log(run_id, "stderr", stderr)
            self._complete_run(run_id, status=status, exit_code=124)
        self.store.add_audit_event(
            AuditEvent(
                actor="terminal",
                action="terminal.run",
                resource=f"terminal:{run_id}",
                project=project,
                metadata={"owner": owner, "command": command, "cwd": str(resolved_cwd), "approved": approved},
                status="allowed" if status == "completed" else status,
            )
        )
        return self.get_run(run_id)

    def runs(self, *, project: str = "default", owner: str = "", limit: int = 50) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        sql = "SELECT * FROM terminal_runs WHERE " + " AND ".join(clauses) + " ORDER BY started_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
        return [run_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM terminal_runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError("terminal run not found")
            logs = connection.execute("SELECT * FROM terminal_logs WHERE run_id = ? ORDER BY created_at ASC, rowid ASC", (run_id,)).fetchall()
        result = run_row(row)
        result["events"] = [log_row(log) for log in logs]
        result["streaming_mode"] = "buffered-local-events"
        return result

    def _validate_command(self, command: list[str], *, approved: bool) -> None:
        if not command:
            raise ValueError("command is required")
        for part in command:
            if DENIED_RE.search(part):
                raise ValueError("shell metacharacters are not allowed")
        if tuple(command[:3]) in SAFE_COMMANDS or tuple(command[:2]) in SAFE_COMMANDS:
            return
        if not approved:
            raise ValueError("non-default terminal commands require approved=true")

    def _validate_cwd(self, cwd: str) -> Path:
        base = Path(cwd).expanduser() if cwd else self.workspace_root
        resolved = base.resolve()
        roots = [self.workspace_root, self.store.data_dir.resolve()]
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise ValueError("cwd must stay inside the workspace or OpenAgent data directory")
        return resolved

    def _append_log(self, run_id: str, stream: str, chunk: str) -> None:
        if not chunk:
            return
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO terminal_logs (id, run_id, stream, chunk, created_at) VALUES (?, ?, ?, ?, ?)",
                (uuid4().hex, run_id, stream, chunk[-20000:], utc_now()),
            )

    def _complete_run(self, run_id: str, *, status: str, exit_code: int) -> None:
        with self.store.session() as connection:
            connection.execute(
                "UPDATE terminal_runs SET status = ?, exit_code = ?, completed_at = ? WHERE id = ?",
                (status, exit_code, utc_now(), run_id),
            )


def redact(value: str | bytes) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    return REDACT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)


def run_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "cwd": row["cwd"],
        "command": json.loads(row["command_json"] or "[]"),
        "status": row["status"],
        "exit_code": int(row["exit_code"]),
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
    }


def log_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "stream": row["stream"],
        "chunk": row["chunk"],
        "created_at": row["created_at"],
    }
