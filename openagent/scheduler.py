"""Local scheduler, reminders, notifications, and due-task runner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .egress import EgressClient
from .models import AuditEvent, utc_now
from .security_privacy import decrypt_text, encrypt_text
from .storage import OpenAgentStore


class Scheduler:
    """Durable local task scheduler with notification records."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.egress = EgressClient()
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    title TEXT NOT NULL,
                    task_type TEXT NOT NULL DEFAULT 'reminder',
                    schedule_json TEXT NOT NULL DEFAULT '{}',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'active',
                    next_run_at TEXT NOT NULL,
                    last_run_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_due
                    ON scheduled_tasks(status, next_run_at);

                CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_project_owner
                    ON scheduled_tasks(project, owner, updated_at DESC);

                CREATE TABLE IF NOT EXISTS task_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES scheduled_tasks(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS task_notifications (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL DEFAULT '',
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    level TEXT NOT NULL DEFAULT 'info',
                    title TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '',
                    read_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_task_notifications_project_owner
                    ON task_notifications(project, owner, created_at DESC);

                CREATE TABLE IF NOT EXISTS task_webhooks (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    label TEXT NOT NULL,
                    target_url TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    secret_ref TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_revisions_task ON task_revisions(task_id, id DESC);

                CREATE TABLE IF NOT EXISTS task_dependencies (
                    task_id TEXT NOT NULL,
                    depends_on_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, depends_on_id)
                );

                CREATE TABLE IF NOT EXISTS task_cache (
                    task_id TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    value_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, cache_key)
                );

                CREATE TABLE IF NOT EXISTS task_delivery_endpoints (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    target_url TEXT NOT NULL,
                    token_sha256 TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_output_destinations (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    destination_type TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    secret_json TEXT NOT NULL DEFAULT '{}',
                    allow_external INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_output_destinations_task
                    ON task_output_destinations(task_id, enabled);

                CREATE TABLE IF NOT EXISTS task_output_deliveries (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    destination_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_output_deliveries_task
                    ON task_output_deliveries(task_id, created_at DESC);
                """
            )

    def create_task(
        self,
        *,
        title: str,
        project: str = "default",
        owner: str = "local",
        task_type: str = "reminder",
        schedule: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        title = title.strip()
        if not title:
            raise ValueError("title is required")
        schedule = schedule or {}
        next_run_at = next_run_from_schedule(schedule)
        task_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO scheduled_tasks (
                    id, project, owner, title, task_type, schedule_json, payload_json,
                    status, next_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    task_id,
                    project,
                    owner,
                    title,
                    task_type or "reminder",
                    json.dumps(schedule, sort_keys=True),
                    json.dumps(payload or {}, sort_keys=True),
                    next_run_at,
                    now,
                    now,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="scheduler",
                action="task.create",
                resource=f"task:{task_id}",
                project=project,
                metadata={"owner": owner, "task_type": task_type, "next_run_at": next_run_at},
            )
        )
        return self.get_task(task_id, owner=owner)

    def list_tasks(self, *, project: str = "default", owner: str = "", status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        if status:
            clauses.append("status = ?")
            args.append(status)
        sql = "SELECT * FROM scheduled_tasks WHERE " + " AND ".join(clauses) + " ORDER BY next_run_at ASC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
        return [task_row(row) for row in rows]

    def get_task(self, task_id: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._task_row(connection, task_id, owner=owner)
        return task_row(row)

    def update_task(self, task_id: str, *, patch: dict[str, Any], owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._task_row(connection, task_id, owner=owner)
            current = task_row(row)
            connection.execute(
                "INSERT INTO task_revisions (task_id, snapshot_json, reason, created_at) VALUES (?, ?, ?, ?)",
                (task_id, json.dumps(current, sort_keys=True), str(patch.get("reason", "update")), utc_now()),
            )
            schedule = patch.get("schedule") if isinstance(patch.get("schedule"), dict) else current["schedule"]
            payload = patch.get("payload") if isinstance(patch.get("payload"), dict) else current["payload"]
            title = str(patch.get("title", current["title"])).strip() or current["title"]
            status = str(patch.get("status", current["status"])).strip() or current["status"]
            next_run_at = str(patch.get("next_run_at") or next_run_from_schedule(schedule))
            connection.execute(
                """
                UPDATE scheduled_tasks
                SET title = ?, schedule_json = ?, payload_json = ?, status = ?, next_run_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (title, json.dumps(schedule, sort_keys=True), json.dumps(payload, sort_keys=True), status, next_run_at, utc_now(), task_id),
            )
        return self.get_task(task_id, owner=owner)

    def stop(self, task_id: str, *, owner: str = "") -> dict[str, Any]:
        task = self._set_status(task_id, "cancelled", owner=owner)
        self._audit("task.stop", task, {})
        return task

    def delete(self, task_id: str, *, owner: str = "") -> dict[str, Any]:
        task = self.get_task(task_id, owner=owner)
        with self.store.session() as connection:
            connection.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM task_notifications WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM task_revisions WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM task_dependencies WHERE task_id = ? OR depends_on_id = ?", (task_id, task_id))
            connection.execute("DELETE FROM task_cache WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM task_delivery_endpoints WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM task_output_deliveries WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM task_output_destinations WHERE task_id = ?", (task_id,))
            connection.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        self._audit("task.delete", task, {})
        return {"deleted": True, "task_id": task_id}

    def run_history(self, task_id: str, *, owner: str = "", limit: int = 100) -> list[dict[str, Any]]:
        self.get_task(task_id, owner=owner)
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM task_runs WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
                (task_id, max(1, min(int(limit), 500))),
            ).fetchall()
        return [
            {
                "id": row["id"], "task_id": row["task_id"], "project": row["project"],
                "owner": row["owner"], "status": row["status"],
                "result": json.loads(row["result_json"] or "{}"), "created_at": row["created_at"],
            }
            for row in rows
        ]

    def revert(self, task_id: str, *, owner: str = "") -> dict[str, Any]:
        self.get_task(task_id, owner=owner)
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM task_revisions WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,)
            ).fetchone()
            if row is None:
                raise ValueError("No task revision is available to revert.")
            snapshot = json.loads(row["snapshot_json"])
            connection.execute(
                """
                UPDATE scheduled_tasks SET title = ?, task_type = ?, schedule_json = ?, payload_json = ?,
                    status = ?, next_run_at = ?, last_run_at = ?, updated_at = ? WHERE id = ?
                """,
                (
                    snapshot["title"], snapshot["task_type"], json.dumps(snapshot["schedule"], sort_keys=True),
                    json.dumps(snapshot["payload"], sort_keys=True), snapshot["status"], snapshot["next_run_at"],
                    snapshot["last_run_at"], utc_now(), task_id,
                ),
            )
            connection.execute("DELETE FROM task_revisions WHERE id = ?", (row["id"],))
        task = self.get_task(task_id, owner=owner)
        self._audit("task.revert", task, {"revision_id": row["id"]})
        return task

    def clear_cache(self, task_id: str, *, owner: str = "") -> dict[str, Any]:
        task = self.get_task(task_id, owner=owner)
        with self.store.session() as connection:
            cursor = connection.execute("DELETE FROM task_cache WHERE task_id = ?", (task_id,))
        self._audit("task.cache.clear", task, {"deleted": cursor.rowcount})
        return {"task_id": task_id, "cleared": cursor.rowcount}

    def set_dependencies(self, task_id: str, dependency_ids: list[str], *, owner: str = "") -> dict[str, Any]:
        task = self.get_task(task_id, owner=owner)
        clean = list(dict.fromkeys(str(item).strip() for item in dependency_ids if str(item).strip()))
        if task_id in clean:
            raise ValueError("A task cannot depend on itself.")
        for dependency_id in clean:
            self.get_task(dependency_id, owner=owner)
        if self._would_cycle(task_id, clean):
            raise ValueError("Task dependencies would create a cycle.")
        with self.store.session() as connection:
            connection.execute("DELETE FROM task_dependencies WHERE task_id = ?", (task_id,))
            for dependency_id in clean:
                connection.execute(
                    "INSERT INTO task_dependencies (task_id, depends_on_id, created_at) VALUES (?, ?, ?)",
                    (task_id, dependency_id, utc_now()),
                )
        return {"task_id": task_id, "dependencies": clean}

    def create_webhook(self, task_id: str, target_url: str, *, owner: str = "") -> dict[str, Any]:
        self.get_task(task_id, owner=owner)
        if not target_url.startswith(("http://127.0.0.1", "http://localhost", "https://")):
            raise ValueError("Webhook target must use HTTPS or an explicit localhost URL.")
        endpoint_id = uuid4().hex
        token = secrets.token_urlsafe(32)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO task_delivery_endpoints (id, task_id, target_url, token_sha256, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (endpoint_id, task_id, target_url, hashlib.sha256(token.encode("utf-8")).hexdigest(), now, now),
            )
        return {
            "id": endpoint_id, "task_id": task_id, "target_url": target_url, "token": token,
            "token_displayed_once": True,
            "signature": "Use HMAC-SHA256 over the raw request body with this token.",
        }

    def add_output_destination(
        self,
        task_id: str,
        destination_type: str,
        *,
        config: dict[str, Any] | None = None,
        owner: str = "",
        allow_external: bool = False,
    ) -> dict[str, Any]:
        task = self.get_task(task_id, owner=owner)
        kind = destination_type.strip().lower().replace("-", "_")
        if kind not in {"local_file", "webhook", "notification", "run_history"}:
            raise ValueError("destination_type must be local_file, webhook, notification, or run_history.")
        config = dict(config or {})
        displayed_secret = ""
        secret_payload: dict[str, Any] = {}
        if kind == "local_file":
            filename = Path(str(config.get("filename", f"{task_id}.jsonl"))).name
            if not filename.lower().endswith((".json", ".jsonl")):
                raise ValueError("Local task outputs must use a .json or .jsonl filename.")
            config = {"filename": filename, "mode": "append"}
        elif kind == "webhook":
            target_url = str(config.get("target_url", "")).strip()
            parsed = urlparse(target_url)
            loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Webhook output requires an HTTP(S) target_url.")
            if not loopback and (parsed.scheme != "https" or not allow_external):
                raise ValueError("External webhook output requires HTTPS and allow_external=true.")
            displayed_secret = secrets.token_urlsafe(32)
            secret_payload = encrypt_text(displayed_secret, self._installation_secret())
            config = {
                "target_url": target_url,
                "timeout_seconds": max(1, min(int(config.get("timeout_seconds", 10)), 30)),
            }
        destination_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO task_output_destinations (
                    id, task_id, destination_type, config_json, secret_json,
                    allow_external, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    destination_id, task_id, kind, json.dumps(config, sort_keys=True),
                    json.dumps(secret_payload, sort_keys=True), 1 if allow_external else 0, now, now,
                ),
            )
        self._audit(
            "task.destination.create", task,
            {"destination_id": destination_id, "destination_type": kind, "allow_external": allow_external},
        )
        destination = next(item for item in self.output_destinations(task_id, owner=owner) if item["id"] == destination_id)
        if displayed_secret:
            destination["signing_secret"] = displayed_secret
            destination["secret_displayed_once"] = True
        return destination

    def output_destinations(self, task_id: str, *, owner: str = "") -> list[dict[str, Any]]:
        self.get_task(task_id, owner=owner)
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM task_output_destinations WHERE task_id = ? ORDER BY created_at DESC", (task_id,)
            ).fetchall()
        return [
            {
                "id": row["id"], "task_id": row["task_id"], "destination_type": row["destination_type"],
                "config": json.loads(row["config_json"] or "{}"), "allow_external": bool(row["allow_external"]),
                "enabled": bool(row["enabled"]), "created_at": row["created_at"], "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def delivery_history(self, task_id: str, *, owner: str = "", limit: int = 100) -> list[dict[str, Any]]:
        self.get_task(task_id, owner=owner)
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM task_output_deliveries WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
                (task_id, max(1, min(int(limit), 500))),
            ).fetchall()
        return [{**dict(row), "detail": json.loads(row["detail_json"] or "{}")} for row in rows]

    def deliver_output(self, task_id: str, run_id: str, result: dict[str, Any], *, owner: str = "") -> list[dict[str, Any]]:
        task = self.get_task(task_id, owner=owner)
        payload = json.dumps(
            {"task": task, "run_id": run_id, "result": result, "delivered_at": utc_now()}, sort_keys=True
        ).encode("utf-8")
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM task_output_destinations WHERE task_id = ? AND enabled = 1 ORDER BY created_at",
                (task_id,),
            ).fetchall()
        deliveries = []
        for row in rows:
            status = "completed"
            try:
                config = json.loads(row["config_json"] or "{}")
                if row["destination_type"] == "local_file":
                    folder = (self.store.data_dir / "task_outputs").resolve()
                    folder.mkdir(parents=True, exist_ok=True)
                    path = (folder / Path(config["filename"]).name).resolve()
                    if folder not in path.parents:
                        raise ValueError("Output path escaped the local task output directory.")
                    with path.open("ab") as handle:
                        handle.write(payload + b"\n")
                    detail = {"path": str(path), "bytes": len(payload)}
                elif row["destination_type"] == "webhook":
                    secret = decrypt_text(json.loads(row["secret_json"] or "{}"), self._installation_secret())
                    signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
                    response = self.egress.fetch(
                        str(config["target_url"]), approved=bool(row["allow_external"]), allow_loopback=True,
                        purpose=f"scheduled task output: {task_id}", body=payload, method="POST",
                        headers={"Content-Type": "application/json", "X-OpenAgent-Signature": f"sha256={signature}"},
                        timeout_seconds=int(config.get("timeout_seconds", 10)), max_response_bytes=1024 * 1024,
                    )
                    detail = {"status_code": response.status, "signature": f"sha256={signature}"}
                elif row["destination_type"] == "notification":
                    notification = self.add_notification(
                        title=task["title"], body=str(result.get("message", "Task completed.")),
                        project=task["project"], owner=task["owner"], task_id=task_id,
                    )
                    detail = {"notification_id": notification["id"]}
                else:
                    detail = {"run_history": True}
            except Exception as exc:
                status = "failed"
                detail = {"error": str(exc)}
            delivery_id = uuid4().hex
            with self.store.session() as connection:
                connection.execute(
                    "INSERT INTO task_output_deliveries (id, task_id, run_id, destination_id, status, detail_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (delivery_id, task_id, run_id, row["id"], status, json.dumps(detail, sort_keys=True), utc_now()),
                )
            delivery = {
                "id": delivery_id, "task_id": task_id, "run_id": run_id,
                "destination_id": row["id"], "status": status, "detail": detail,
            }
            deliveries.append(delivery)
            self._audit(
                "task.output.deliver", task,
                {"delivery_id": delivery_id, "destination_id": row["id"], "status": status},
            )
        return deliveries

    def _installation_secret(self) -> str:
        path = self.store.data_dir / ".scheduler-output-key"
        if not path.exists():
            path.write_text(secrets.token_urlsafe(48), encoding="ascii")
            try:
                path.chmod(0o600)
            except OSError:
                pass
        return path.read_text(encoding="ascii").strip()

    def create_from_text(self, text: str, *, project: str = "default", owner: str = "local") -> dict[str, Any]:
        clean = text.strip()
        if not clean:
            raise ValueError("task text is required")
        schedule: dict[str, Any] = {}
        match = re.search(r"\bin\s+(\d+)\s+(minute|minutes|hour|hours|day|days)\b", clean, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()
            multiplier = 60 if unit.startswith("minute") else 3600 if unit.startswith("hour") else 86400
            schedule["delay_seconds"] = value * multiplier
        elif re.search(r"\btomorrow\b", clean, re.IGNORECASE):
            schedule["delay_seconds"] = 86400
        elif re.search(r"\bdaily\b", clean, re.IGNORECASE):
            schedule["recurrence"] = "@daily"
        title = re.sub(r"\b(in\s+\d+\s+(?:minutes?|hours?|days?)|tomorrow|daily)\b", "", clean, flags=re.IGNORECASE).strip(" ,.-")
        return self.create_task(title=title or clean, project=project, owner=owner, schedule=schedule, payload={"source_text": clean})

    def pause(self, task_id: str, *, owner: str = "") -> dict[str, Any]:
        return self._set_status(task_id, "paused", owner=owner)

    def resume(self, task_id: str, *, owner: str = "") -> dict[str, Any]:
        task = self.get_task(task_id, owner=owner)
        return self.update_task(task_id, patch={"status": "active", "schedule": task["schedule"]}, owner=owner)

    def notifications(self, *, project: str = "default", owner: str = "", unread_only: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        if unread_only:
            clauses.append("read_at = ''")
        sql = "SELECT * FROM task_notifications WHERE " + " AND ".join(clauses) + " ORDER BY created_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
        return [notification_row(row) for row in rows]

    def add_notification(
        self,
        *,
        title: str,
        body: str = "",
        project: str = "default",
        owner: str = "local",
        task_id: str = "",
        level: str = "info",
    ) -> dict[str, Any]:
        notification_id = uuid4().hex
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO task_notifications (
                    id, task_id, project, owner, level, title, body, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (notification_id, task_id, project, owner, level, title, body, utc_now()),
            )
            row = connection.execute("SELECT * FROM task_notifications WHERE id = ?", (notification_id,)).fetchone()
        return notification_row(row)

    def mark_notification_read(
        self, notification_id: str, *, project: str = "default", owner: str = ""
    ) -> dict[str, Any]:
        clauses = ["id = ?", "project = ?"]
        values: list[Any] = [notification_id, project]
        if owner:
            clauses.append("owner = ?")
            values.append(owner)
        read_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "UPDATE task_notifications SET read_at = ? WHERE " + " AND ".join(clauses),
                [read_at, *values],
            )
            row = connection.execute(
                "SELECT * FROM task_notifications WHERE " + " AND ".join(clauses), values
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown notification '{notification_id}'.")
        return notification_row(row)

    def run_due(self, *, project: str = "default", owner: str = "", limit: int = 25) -> dict[str, Any]:
        now = utc_now()
        clauses = ["project = ?", "status = 'active'", "next_run_at <= ?"]
        args: list[Any] = [project, now]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        sql = "SELECT * FROM scheduled_tasks WHERE " + " AND ".join(clauses) + " ORDER BY next_run_at ASC LIMIT ?"
        args.append(max(1, min(int(limit), 100)))
        runs = []
        notifications = []
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
            for row in rows:
                task = task_row(row)
                if not self._dependencies_complete(connection, task["id"]):
                    continue
                run_id = uuid4().hex
                result = {"message": f"Task due: {task['title']}", "task_type": task["task_type"]}
                connection.execute(
                    """
                    INSERT INTO task_runs (id, task_id, project, owner, status, result_json, created_at)
                    VALUES (?, ?, ?, ?, 'completed', ?, ?)
                    """,
                    (run_id, task["id"], task["project"], task["owner"], json.dumps(result, sort_keys=True), now),
                )
                notification_id = uuid4().hex
                connection.execute(
                    """
                    INSERT INTO task_notifications (id, task_id, project, owner, level, title, body, created_at)
                    VALUES (?, ?, ?, ?, 'info', ?, ?, ?)
                    """,
                    (notification_id, task["id"], task["project"], task["owner"], task["title"], result["message"], now),
                )
                next_run = next_run_after(task["schedule"], now)
                new_status = "active" if next_run else "completed"
                connection.execute(
                    """
                    UPDATE scheduled_tasks SET status = ?, last_run_at = ?, next_run_at = ?, updated_at = ? WHERE id = ?
                    """,
                    (new_status, now, next_run or now, now, task["id"]),
                )
                runs.append({"id": run_id, "task_id": task["id"], "status": "completed", "result": result, "created_at": now})
                notifications.append({"id": notification_id, "task_id": task["id"], "title": task["title"], "body": result["message"]})
        deliveries = []
        for run in runs:
            deliveries.extend(self.deliver_output(run["task_id"], run["id"], run["result"], owner=owner))
        return {"runs": runs, "notifications": notifications, "deliveries": deliveries, "count": len(runs)}

    def _dependencies_complete(self, connection: Any, task_id: str) -> bool:
        row = connection.execute(
            """
            SELECT COUNT(*) AS pending FROM task_dependencies d
            JOIN scheduled_tasks t ON t.id = d.depends_on_id
            WHERE d.task_id = ? AND t.status != 'completed'
            """,
            (task_id,),
        ).fetchone()
        return not row or int(row["pending"]) == 0

    def _would_cycle(self, task_id: str, dependencies: list[str]) -> bool:
        with self.store.session() as connection:
            graph: dict[str, set[str]] = {}
            for row in connection.execute("SELECT task_id, depends_on_id FROM task_dependencies").fetchall():
                graph.setdefault(row["task_id"], set()).add(row["depends_on_id"])
        graph[task_id] = set(dependencies)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(visit(child) for child in graph.get(node, ())):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        return visit(task_id)

    def _audit(self, action: str, task: dict[str, Any], metadata: dict[str, Any]) -> None:
        self.store.add_audit_event(
            AuditEvent(actor="scheduler", action=action, resource=f"task:{task['id']}", project=task["project"], metadata={"owner": task["owner"], **metadata})
        )

    def _set_status(self, task_id: str, status: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            self._task_row(connection, task_id, owner=owner)
            connection.execute(
                "UPDATE scheduled_tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), task_id),
            )
        return self.get_task(task_id, owner=owner)

    def _task_row(self, connection: Any, task_id: str, *, owner: str = "") -> Any:
        if owner:
            row = connection.execute("SELECT * FROM scheduled_tasks WHERE id = ? AND owner = ?", (task_id, owner)).fetchone()
        else:
            row = connection.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError("scheduled task not found")
        return row


def task_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "title": row["title"],
        "task_type": row["task_type"],
        "schedule": json.loads(row["schedule_json"] or "{}"),
        "payload": json.loads(row["payload_json"] or "{}"),
        "status": row["status"],
        "next_run_at": row["next_run_at"],
        "last_run_at": row["last_run_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def notification_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "project": row["project"],
        "owner": row["owner"],
        "level": row["level"],
        "title": row["title"],
        "body": row["body"],
        "read_at": row["read_at"],
        "created_at": row["created_at"],
    }


def next_run_from_schedule(schedule: dict[str, Any]) -> str:
    run_at = str(schedule.get("run_at") or schedule.get("once_at") or "").strip()
    if run_at:
        return normalize_iso(run_at)
    delay_seconds = int(schedule.get("delay_seconds", 0) or 0)
    return (datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=max(0, delay_seconds))).isoformat()


def next_run_after(schedule: dict[str, Any], previous: str) -> str:
    interval = int(schedule.get("interval_seconds", 0) or 0)
    if interval > 0:
        base = parse_iso(previous)
        return (base + timedelta(seconds=interval)).isoformat()
    recurrence = str(schedule.get("recurrence", "")).lower()
    base = parse_iso(previous)
    if recurrence == "@hourly":
        return (base + timedelta(hours=1)).isoformat()
    if recurrence == "@daily":
        return (base + timedelta(days=1)).isoformat()
    return ""


def normalize_iso(value: str) -> str:
    return parse_iso(value).isoformat()


def parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)
