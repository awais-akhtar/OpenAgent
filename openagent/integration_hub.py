"""Local integration hub for pairing, diagnostics, preferences, and safe tools."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

from .models import AuditEvent, ContextEvent, utc_now
from .storage import OpenAgentStore


StatusProvider = Callable[[], dict[str, Any]]
ProjectStatusProvider = Callable[[str], dict[str, Any]]
ListProvider = Callable[[], list[dict[str, Any]]]


class IntegrationHubError(RuntimeError):
    """Raised when an integration hub action is invalid or unsafe."""


class IntegrationHub:
    """Original OpenAgent-native capability bridge for desktop and local services.

    The hub exposes operational features often needed by a mature local AI app:
    device pairing, user preferences, diagnostics, backups, and a tiny allowlisted
    local command surface. It is intentionally local-first and auditable.
    """

    SAFE_COMMANDS: dict[str, dict[str, Any]] = {
        "python-version": {
            "label": "Python Version",
            "command": [sys.executable, "--version"],
            "description": "Show the Python runtime used by OpenAgent.",
        },
        "storage-health": {
            "label": "Storage Health",
            "command": [sys.executable, "-c", "from openagent.storage import OpenAgentStore; import tempfile; d=tempfile.mkdtemp(); print(OpenAgentStore(d).health()['status'])"],
            "description": "Create a temporary local store and verify SQLite health.",
        },
        "hard-check": {
            "label": "Hard Check",
            "command": [sys.executable, "-m", "openagent.hard_check"],
            "description": "Run OpenAgent's local feature hard check.",
        },
    }

    def __init__(
        self,
        store: OpenAgentStore,
        *,
        health_provider: StatusProvider,
        module_health_provider: ProjectStatusProvider,
        metrics_provider: StatusProvider,
        model_provider: StatusProvider,
        voice_provider: ProjectStatusProvider,
        developer_provider: StatusProvider,
        tool_provider: ListProvider,
        backup_provider: Callable[[], dict[str, Any]],
        backups_provider: ListProvider,
    ):
        self.store = store
        self.health_provider = health_provider
        self.module_health_provider = module_health_provider
        self.metrics_provider = metrics_provider
        self.model_provider = model_provider
        self.voice_provider = voice_provider
        self.developer_provider = developer_provider
        self.tool_provider = tool_provider
        self.backup_provider = backup_provider
        self.backups_provider = backups_provider
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS integration_pairing_codes (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    label TEXT NOT NULL,
                    code_hash TEXT NOT NULL UNIQUE,
                    scopes_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    redeemed_at TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_integration_pairing_project_created
                    ON integration_pairing_codes(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS integration_preferences (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project, scope, key)
                );

                CREATE INDEX IF NOT EXISTS idx_integration_preferences_project_scope
                    ON integration_preferences(project, scope);

                CREATE TABLE IF NOT EXISTS integration_diagnostics (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_integration_diagnostics_project_created
                    ON integration_diagnostics(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS integration_safe_command_runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    command_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    return_code INTEGER NOT NULL DEFAULT 0,
                    duration_ms REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_integration_safe_runs_project_created
                    ON integration_safe_command_runs(project, created_at DESC);
                """
            )

    def status(self, *, project: str = "default") -> dict[str, Any]:
        return {
            "status": "ready",
            "project": project,
            "pairing": {
                "active_codes": len(self.pairing_codes(project=project, include_expired=False)),
                "privacy_boundary": "Pairing codes are stored as hashes and expire automatically.",
            },
            "preferences": {
                "count": len(self.preferences(project=project)),
                "scopes": sorted({item["scope"] for item in self.preferences(project=project)}),
            },
            "diagnostics": {
                "reports": len(self.diagnostics(project=project, limit=50)),
                "latest": (self.diagnostics(project=project, limit=1) or [None])[0],
            },
            "safe_commands": list(self.SAFE_COMMANDS.keys()),
            "backups": len(self.backups_provider()),
            "privacy_boundary": "Integration hub actions stay local, store audit events, and do not call external providers.",
        }

    def create_pairing_code(
        self,
        *,
        label: str,
        scopes: list[str] | None = None,
        project: str = "default",
        ttl_minutes: int = 15,
    ) -> dict[str, Any]:
        label = label.strip() or "Local companion"
        ttl_minutes = max(1, min(int(ttl_minutes), 1440))
        scopes = scopes or ["api.read", "workspace.read", "chat.write"]
        code = f"OA-{secrets.token_urlsafe(18)}"
        now = utc_now()
        expires_at = iso_from_epoch(epoch_from_iso(now) + ttl_minutes * 60)
        record_id = stable_id("pair", project, label, now, secrets.token_hex(6))
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO integration_pairing_codes (
                    id, project, label, code_hash, scopes_json, status, expires_at, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    project,
                    label,
                    hash_secret(code),
                    json.dumps(scopes, sort_keys=True),
                    "active",
                    expires_at,
                    now,
                    json.dumps({"ttl_minutes": ttl_minutes}, sort_keys=True),
                ),
            )
        self._audit("integration.pairing_code.create", record_id, project=project, metadata={"label": label, "scopes": scopes})
        return {
            **(self.get_pairing_code(record_id) or {}),
            "code": code,
            "expires_in_minutes": ttl_minutes,
            "one_time_display": True,
        }

    def redeem_pairing_code(
        self,
        *,
        code: str,
        device_name: str,
        project: str = "default",
    ) -> dict[str, Any]:
        code_hash = hash_secret(code.strip())
        now = utc_now()
        now_epoch = epoch_from_iso(now)
        with self.store.session() as connection:
            row = connection.execute(
                """
                SELECT * FROM integration_pairing_codes
                WHERE code_hash = ? AND project = ? AND status = 'active'
                """,
                (code_hash, project),
            ).fetchone()
            if not row:
                raise IntegrationHubError("Pairing code was not found or already used.")
            record = pairing_row(row)
            if epoch_from_iso(record["expires_at"]) < now_epoch:
                connection.execute(
                    "UPDATE integration_pairing_codes SET status = 'expired' WHERE id = ?",
                    (record["id"],),
                )
                raise IntegrationHubError("Pairing code has expired.")
            connection.execute(
                """
                UPDATE integration_pairing_codes
                SET status = 'redeemed', redeemed_at = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    now,
                    json.dumps({**record["metadata"], "device_name": device_name.strip() or "Local device"}, sort_keys=True),
                    record["id"],
                ),
            )
        token = f"oa_pair_{secrets.token_urlsafe(24)}"
        token_hint = token[:12] + "..."
        self.store.add_event(
            ContextEvent(
                type="integration",
                source="integration_hub",
                project=project,
                title=f"Paired {device_name.strip() or 'local device'}",
                content="A local companion/device pairing code was redeemed.",
                metadata={"pairing_id": record["id"], "scopes": record["scopes"], "token_hint": token_hint},
            )
        )
        self._audit("integration.pairing_code.redeem", record["id"], project=project, metadata={"device_name": device_name, "token_hint": token_hint})
        return {
            "status": "paired",
            "device_name": device_name.strip() or "Local device",
            "project": project,
            "scopes": record["scopes"],
            "token": token,
            "token_hint": token_hint,
            "privacy_boundary": "The returned token is shown once and is not stored in plaintext.",
        }

    def get_pairing_code(self, pairing_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM integration_pairing_codes WHERE id = ?", (pairing_id,)).fetchone()
        return pairing_row(row) if row else None

    def pairing_codes(self, *, project: str = "default", include_expired: bool = True) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM integration_pairing_codes WHERE project = ? ORDER BY created_at DESC",
                (project,),
            ).fetchall()
        records = [pairing_row(row) for row in rows]
        now_epoch = epoch_from_iso(utc_now())
        for record in records:
            if record["status"] == "active" and epoch_from_iso(record["expires_at"]) < now_epoch:
                record["status"] = "expired"
        if include_expired:
            return records
        return [record for record in records if record["status"] == "active"]

    def set_preference(
        self,
        *,
        key: str,
        value: Any,
        scope: str = "ui",
        project: str = "default",
    ) -> dict[str, Any]:
        key = key.strip()
        scope = scope.strip() or "ui"
        if not key:
            raise IntegrationHubError("Preference key is required.")
        record_id = stable_id("pref", project, scope, key)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO integration_preferences (id, project, scope, key, value_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project, scope, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (record_id, project, scope, key, json.dumps(value, sort_keys=True), now),
            )
        self._audit("integration.preference.set", record_id, project=project, metadata={"scope": scope, "key": key})
        return self.preference(project=project, scope=scope, key=key) or {}

    def preference(self, *, project: str, scope: str, key: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute(
                """
                SELECT * FROM integration_preferences
                WHERE project = ? AND scope = ? AND key = ?
                """,
                (project, scope, key),
            ).fetchone()
        return preference_row(row) if row else None

    def preferences(self, *, project: str = "default", scope: str = "") -> list[dict[str, Any]]:
        if scope:
            sql = "SELECT * FROM integration_preferences WHERE project = ? AND scope = ? ORDER BY scope, key"
            params: tuple[Any, ...] = (project, scope)
        else:
            sql = "SELECT * FROM integration_preferences WHERE project = ? ORDER BY scope, key"
            params = (project,)
        with self.store.session() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [preference_row(row) for row in rows]

    def delete_preference(self, *, project: str, scope: str, key: str) -> dict[str, Any]:
        pref = self.preference(project=project, scope=scope, key=key)
        with self.store.session() as connection:
            connection.execute(
                "DELETE FROM integration_preferences WHERE project = ? AND scope = ? AND key = ?",
                (project, scope, key),
            )
        self._audit("integration.preference.delete", stable_id("pref", project, scope, key), project=project, metadata={"scope": scope, "key": key})
        return {"deleted": bool(pref), "preference": pref}

    def create_diagnostic_report(self, *, project: str = "default") -> dict[str, Any]:
        report = self.diagnostic_snapshot(project=project)
        report_id = stable_id("diagnostic", project, report["created_at"])
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO integration_diagnostics (id, project, report_json, created_at) VALUES (?, ?, ?, ?)",
                (report_id, project, json.dumps(report, sort_keys=True), report["created_at"]),
            )
        self._audit(
            "integration.diagnostics.create",
            report_id,
            project=project,
            metadata={"overall_status": report["status"], "issue_count": len(report["issues"])},
        )
        return {"id": report_id, **report}

    def diagnostic_snapshot(self, *, project: str = "default") -> dict[str, Any]:
        sections = {
            "health": safe_call(self.health_provider),
            "modules": safe_call(lambda: self.module_health_provider(project)),
            "system": safe_call(self.metrics_provider),
            "models": safe_call(self.model_provider),
            "voice": safe_call(lambda: self.voice_provider(project)),
            "developer": safe_call(self.developer_provider),
            "tools": {"tools": safe_call(self.tool_provider)},
            "backups": {"backups": safe_call(self.backups_provider)},
        }
        issues = collect_issues(sections)
        return {
            "status": "needs_attention" if issues else "ok",
            "project": project,
            "created_at": utc_now(),
            "sections": sections,
            "issues": issues,
            "privacy_boundary": "Diagnostics read local status providers only. No provider sync, model download, or external network call is performed.",
        }

    def diagnostics(self, *, project: str = "default", limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM integration_diagnostics
                WHERE project = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project, limit),
            ).fetchall()
        return [diagnostic_row(row) for row in rows]

    def create_backup(self, *, project: str = "default") -> dict[str, Any]:
        backup = self.backup_provider()
        self._audit("integration.backup.create", str(backup.get("id", "backup")), project=project, metadata={"path": backup.get("path", "")})
        return {
            "backup": backup,
            "privacy_boundary": "Backup is written under the local OpenAgent data directory.",
        }

    def safe_commands(self, *, project: str = "default", limit: int = 20) -> dict[str, Any]:
        limit = max(1, min(limit, 100))
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM integration_safe_command_runs
                WHERE project = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project, limit),
            ).fetchall()
        return {
            "commands": [
                {
                    "key": key,
                    "label": spec["label"],
                    "description": spec["description"],
                    "command": redact_command(spec["command"]),
                }
                for key, spec in self.SAFE_COMMANDS.items()
            ],
            "runs": [safe_run_row(row) for row in rows],
            "privacy_boundary": "Only fixed read-only command keys can run. User-provided shell strings are rejected.",
        }

    def run_safe_command(self, *, command_key: str, project: str = "default", timeout_seconds: int = 15) -> dict[str, Any]:
        command_key = command_key.strip()
        spec = self.SAFE_COMMANDS.get(command_key)
        if spec is None:
            raise IntegrationHubError(f"Unknown safe command '{command_key}'.")
        timeout_seconds = max(1, min(int(timeout_seconds), 60))
        started = epoch_from_iso(utc_now())
        status = "completed"
        output = ""
        error = ""
        return_code = 0
        try:
            proc = subprocess.run(
                list(spec["command"]),
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                shell=False,
                timeout=timeout_seconds,
                check=False,
            )
            output = (proc.stdout or "").strip()[:4000]
            error = (proc.stderr or "").strip()[:4000]
            return_code = int(proc.returncode)
            if proc.returncode != 0:
                status = "failed"
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            output = (exc.stdout or "").strip()[:4000] if isinstance(exc.stdout, str) else ""
            error = "Command timed out."
            return_code = 124
        except OSError as exc:
            status = "failed"
            error = str(exc)
            return_code = 1
        duration_ms = round((epoch_from_iso(utc_now()) - started) * 1000, 3)
        run_id = stable_id("safe-run", project, command_key, utc_now(), secrets.token_hex(4))
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO integration_safe_command_runs (
                    id, project, command_key, status, output, error, return_code, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, project, command_key, status, output, error, return_code, duration_ms, utc_now()),
            )
        self._audit(
            "integration.safe_command.run",
            command_key,
            project=project,
            status="allowed" if status == "completed" else status,
            metadata={"run_id": run_id, "return_code": return_code, "duration_ms": duration_ms},
        )
        return {
            "run": {
                "id": run_id,
                "project": project,
                "command_key": command_key,
                "label": spec["label"],
                "status": status,
                "output": output,
                "error": error,
                "return_code": return_code,
                "duration_ms": duration_ms,
                "created_at": utc_now(),
            },
            "privacy_boundary": "Safe command ran locally with shell=False and a fixed allowlist.",
        }

    def feature_matrix(self, *, project: str = "default") -> dict[str, Any]:
        rows = [
            ("local_pairing", "Local companion/device pairing", "implemented", "/api/integrations/pairing-codes"),
            ("diagnostics", "Local diagnostics center", "implemented", "/api/integrations/diagnostics"),
            ("preferences", "Persistent user/app preferences", "implemented", "/api/integrations/preferences"),
            ("safe_shell", "Allowlisted local command actions", "implemented", "/api/integrations/safe-shell"),
            ("backups", "Local backup creation and listing", "implemented", "/api/integrations/backups"),
            ("developer_tools", "Tool, webhook, MCP, and SDK visibility", "implemented", "/api/developer/status"),
            ("voice", "Local voice runtime and model readiness", "implemented", "/api/voice/status"),
            ("models", "Local model lifecycle and quantization", "implemented", "/api/local-ai/status"),
            ("knowledge", "Knowledge/RAG surfaces", "implemented", "/api/knowledge/status"),
            ("automation", "Workflow builder and execution", "implemented", "/api/workflows/builder"),
        ]
        return {
            "project": project,
            "status": "ready",
            "items": [
                {"id": row[0], "label": row[1], "status": row[2], "endpoint": row[3]}
                for row in rows
            ],
            "privacy_boundary": "This matrix is generated from OpenAgent's own local modules and does not inspect external repositories.",
        }

    def _audit(
        self,
        action: str,
        resource: str,
        *,
        project: str,
        status: str = "allowed",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="integration_hub",
                action=action,
                resource=resource,
                project=project,
                status=status,
                metadata=metadata or {},
            )
        )


def safe_call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def collect_issues(sections: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for name, value in sections.items():
        if isinstance(value, dict):
            status = str(value.get("status", "")).lower()
            if status in {"error", "failed", "needs_attention"}:
                issues.append({"section": name, "status": status, "message": str(value.get("error") or value.get("reason") or status)})
            counts = value.get("counts")
            if isinstance(counts, dict) and int(counts.get("error", 0) or 0) > 0:
                issues.append({"section": name, "status": "error", "message": f"{counts.get('error')} module errors"})
    return issues


def pairing_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "label": row["label"],
        "scopes": json.loads(row["scopes_json"]),
        "status": row["status"],
        "expires_at": row["expires_at"],
        "created_at": row["created_at"],
        "redeemed_at": row["redeemed_at"],
        "metadata": json.loads(row["metadata_json"]),
    }


def preference_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "scope": row["scope"],
        "key": row["key"],
        "value": json.loads(row["value_json"]),
        "updated_at": row["updated_at"],
    }


def diagnostic_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "report": json.loads(row["report_json"]),
        "created_at": row["created_at"],
    }


def safe_run_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "command_key": row["command_key"],
        "status": row["status"],
        "output": row["output"],
        "error": row["error"],
        "return_code": row["return_code"],
        "duration_ms": row["duration_ms"],
        "created_at": row["created_at"],
    }


def hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def epoch_from_iso(value: str) -> float:
    from datetime import datetime

    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).timestamp()


def iso_from_epoch(value: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value, tz=timezone.utc).replace(microsecond=0).isoformat()


def redact_command(command: list[str]) -> list[str]:
    return [Path(part).name if index == 0 else part for index, part in enumerate(command)]
