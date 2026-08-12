"""SQLite-backed local storage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import AgentResponse, AuditEvent, ChatMessage, ContextEvent, utc_now


class OpenAgentStore:
    """Durable local storage for context events and agent runs."""

    def __init__(self, data_dir: str | Path):
        path = Path(data_dir).expanduser()
        if path.suffix in {".db", ".sqlite", ".sqlite3"}:
            self.data_dir = path.parent
            self.db_path = path
        else:
            self.data_dir = path
            self.db_path = path / "openagent.sqlite3"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    project TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_project_created
                    ON events(project, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_events_source
                    ON events(source);

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    project TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_runs_project_created
                    ON agent_runs(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_project_updated
                    ON conversations(project, updated_at DESC);

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_created
                    ON chat_messages(conversation_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_project_created
                    ON audit_events(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    trigger_type TEXT NOT NULL DEFAULT 'manual',
                    steps_json TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_workflows_project_updated
                    ON workflows(project, updated_at DESC);

                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_runs_project_created
                    ON workflow_runs(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS job_queue (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    state TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    run_after TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    progress_message TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT '',
                    cancelled_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_job_queue_ready
                    ON job_queue(state, run_after, priority DESC, created_at ASC);

                CREATE INDEX IF NOT EXISTS idx_job_queue_project_updated
                    ON job_queue(project, updated_at DESC);

                CREATE TABLE IF NOT EXISTS job_attempts (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(job_id) REFERENCES job_queue(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_job_attempts_job
                    ON job_attempts(job_id, started_at DESC);

                CREATE TABLE IF NOT EXISTS job_locks (
                    job_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    lease_until TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES job_queue(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_job_locks_lease_until
                    ON job_locks(lease_until);

                CREATE TABLE IF NOT EXISTS job_dead_letters (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES job_queue(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_job_dead_letters_project_created
                    ON job_dead_letters(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """INSERT OR IGNORE INTO schema_migrations (version, name, checksum, applied_at)
                   VALUES (1, 'core-sqlite-baseline', 'openagent-core-v1', ?)""",
                (utc_now(),),
            )

    def migration_status(self) -> dict[str, Any]:
        with self.session() as connection:
            rows = connection.execute(
                "SELECT version, name, checksum, applied_at FROM schema_migrations ORDER BY version"
            ).fetchall()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {
            "current_version": rows[-1]["version"] if rows else 0,
            "migrations": [dict(row) for row in rows],
            "integrity": integrity,
            "database": str(self.db_path),
        }

    def add_event(self, event: ContextEvent) -> dict[str, Any]:
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO events (
                    id, type, source, project, title, content, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.type,
                    event.source,
                    event.project,
                    event.title,
                    event.content,
                    json.dumps(event.metadata, sort_keys=True),
                    event.created_at,
                ),
            )
        return event.to_record()

    def list_events(self, limit: int = 50, project: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        if project:
            query = "SELECT * FROM events WHERE project = ? ORDER BY created_at DESC LIMIT ?"
            params: tuple[Any, ...] = (project, limit)
        else:
            query = "SELECT * FROM events ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
        with self.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._event_row_to_dict(row) for row in rows]

    def search_events(
        self,
        query: str,
        limit: int = 20,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return self.list_events(limit=limit, project=project)

        limit = max(1, min(limit, 100))
        terms = [term for term in query.lower().split() if len(term) >= 3]
        patterns = [f"%{term}%" for term in terms] or [f"%{query.lower()}%"]
        conditions = " OR ".join(
            "(lower(title) LIKE ? OR lower(content) LIKE ? OR lower(source) LIKE ?)"
            for _ in patterns
        )
        condition_params = tuple(pattern for pattern in patterns for _ in range(3))
        if project:
            sql = f"""
                SELECT * FROM events
                WHERE project = ?
                  AND ({conditions})
                ORDER BY created_at DESC
                LIMIT ?
            """
            params: tuple[Any, ...] = (project, *condition_params, limit)
        else:
            sql = f"""
                SELECT * FROM events
                WHERE {conditions}
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (*condition_params, limit)

        with self.session() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._event_row_to_dict(row) for row in rows]

    def save_agent_run(self, response: AgentResponse, project: str = "default") -> dict[str, Any]:
        payload = response.to_record()
        with self.session() as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_runs (agent, goal, response_json, project, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    response.agent,
                    response.goal,
                    json.dumps(payload, sort_keys=True),
                    project,
                    response.created_at,
                ),
            )
            run_id = cursor.lastrowid
        return {"id": run_id, "project": project, **payload}

    def list_agent_runs(self, limit: int = 20, project: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        if project:
            query = "SELECT * FROM agent_runs WHERE project = ? ORDER BY created_at DESC LIMIT ?"
            params: tuple[Any, ...] = (project, limit)
        else:
            query = "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
        with self.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._agent_run_row_to_dict(row) for row in rows]

    def create_conversation(
        self,
        title: str,
        *,
        project: str = "default",
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        conversation_id = conversation_id or ContextEvent(title=title, content=title).id
        title = title.strip() or "New conversation"
        project = project.strip() or "default"
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, project, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, project, title, now, now),
            )
        return {
            "id": conversation_id,
            "project": project,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def ensure_conversation(
        self,
        conversation_id: str | None,
        *,
        title: str,
        project: str = "default",
    ) -> dict[str, Any]:
        if conversation_id:
            existing = self.get_conversation(conversation_id)
            if existing:
                return existing
        return self.create_conversation(title, project=project, conversation_id=conversation_id)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.session() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return self._conversation_row_to_dict(row) if row else None

    def list_conversations(
        self,
        *,
        project: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        if project:
            query = "SELECT * FROM conversations WHERE project = ? ORDER BY updated_at DESC LIMIT ?"
            params: tuple[Any, ...] = (project, limit)
        else:
            query = "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?"
            params = (limit,)
        with self.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._conversation_row_to_dict(row) for row in rows]

    def add_chat_message(self, message: ChatMessage) -> dict[str, Any]:
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (
                    id, conversation_id, role, content, provider, model, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.provider,
                    message.model,
                    json.dumps(message.metadata, sort_keys=True),
                    message.created_at,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message.created_at, message.conversation_id),
            )
        return message.to_record()

    def list_chat_messages(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 5000))
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT chat_messages.*, rowid AS _rowid FROM chat_messages
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                ) recent
                ORDER BY created_at ASC, _rowid ASC
                """,
                (conversation_id, limit),
            ).fetchall()
        return [self._chat_message_row_to_dict(row) for row in rows]

    def add_audit_event(self, event: AuditEvent) -> dict[str, Any]:
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    id, actor, action, resource, project, status, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.actor,
                    event.action,
                    event.resource,
                    event.project,
                    event.status,
                    json.dumps(event.metadata, sort_keys=True),
                    event.created_at,
                ),
            )
        return event.to_record()

    def list_audit_events(
        self,
        *,
        project: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        if project:
            query = "SELECT * FROM audit_events WHERE project = ? ORDER BY created_at DESC LIMIT ?"
            params: tuple[Any, ...] = (project, limit)
        else:
            query = "SELECT * FROM audit_events ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
        with self.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._audit_event_row_to_dict(row) for row in rows]

    def create_workflow(
        self,
        name: str,
        steps: list[dict[str, Any]],
        *,
        project: str = "default",
        description: str = "",
        trigger_type: str = "manual",
        workflow_id: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        now = utc_now()
        workflow_id = workflow_id or ContextEvent(title=name, content=description or name).id
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO workflows (
                    id, project, name, description, trigger_type, steps_json,
                    enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    project.strip() or "default",
                    name.strip() or "Untitled workflow",
                    description.strip(),
                    trigger_type.strip() or "manual",
                    json.dumps(steps, sort_keys=True),
                    1 if enabled else 0,
                    now,
                    now,
                ),
            )
        return {
            "id": workflow_id,
            "project": project.strip() or "default",
            "name": name.strip() or "Untitled workflow",
            "description": description.strip(),
            "trigger_type": trigger_type.strip() or "manual",
            "steps": steps,
            "enabled": enabled,
            "created_at": now,
            "updated_at": now,
        }

    def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        with self.session() as connection:
            row = connection.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
        return self._workflow_row_to_dict(row) if row else None

    def list_workflows(
        self,
        *,
        project: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        if project:
            query = "SELECT * FROM workflows WHERE project = ? ORDER BY updated_at DESC LIMIT ?"
            params: tuple[Any, ...] = (project, limit)
        else:
            query = "SELECT * FROM workflows ORDER BY updated_at DESC LIMIT ?"
            params = (limit,)
        with self.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._workflow_row_to_dict(row) for row in rows]

    def save_workflow_run(
        self,
        workflow_id: str,
        *,
        project: str,
        status: str,
        result: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        run_id = run_id or ContextEvent(title=workflow_id, content=status).id
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO workflow_runs (
                    id, workflow_id, project, status, result_json, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    workflow_id,
                    project,
                    status,
                    json.dumps(result, sort_keys=True),
                    now,
                    now,
                ),
            )
        return {
            "id": run_id,
            "workflow_id": workflow_id,
            "project": project,
            "status": status,
            "result": result,
            "created_at": now,
            "completed_at": now,
        }

    def list_workflow_runs(
        self,
        *,
        project: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        if project:
            query = "SELECT * FROM workflow_runs WHERE project = ? ORDER BY created_at DESC LIMIT ?"
            params: tuple[Any, ...] = (project, limit)
        else:
            query = "SELECT * FROM workflow_runs ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
        with self.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._workflow_run_row_to_dict(row) for row in rows]

    def enqueue_job(
        self,
        job_type: str,
        payload: dict[str, Any] | None = None,
        *,
        project: str = "default",
        priority: int = 0,
        run_after: str | None = None,
        max_attempts: int = 3,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        job_id = job_id or ContextEvent(title=job_type, content=json.dumps(payload or {}, sort_keys=True)).id
        normalized_type = job_type.strip()
        if not normalized_type:
            raise ValueError("job_type is required.")
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO job_queue (
                    id, project, job_type, payload_json, state, priority, attempts,
                    max_attempts, run_after, progress, progress_message, result_json,
                    last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    project.strip() or "default",
                    normalized_type,
                    json.dumps(payload or {}, sort_keys=True),
                    "queued",
                    int(priority),
                    0,
                    max(1, int(max_attempts)),
                    run_after or now,
                    0,
                    "Queued.",
                    "{}",
                    "",
                    now,
                    now,
                ),
            )
        return self.get_job(job_id) or {}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.session() as connection:
            row = connection.execute("SELECT * FROM job_queue WHERE id = ?", (job_id,)).fetchone()
            lock = connection.execute("SELECT * FROM job_locks WHERE job_id = ?", (job_id,)).fetchone()
            attempts = connection.execute(
                "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY started_at DESC",
                (job_id,),
            ).fetchall()
        if row is None:
            return None
        return self._job_row_to_dict(row, lock=lock, attempts=attempts)

    def list_jobs(
        self,
        *,
        project: str | None = None,
        state: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        clauses = []
        params: list[Any] = []
        if project:
            clauses.append("project = ?")
            params.append(project)
        if state:
            clauses.append("state = ?")
            params.append(state)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM job_queue {where} ORDER BY updated_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            locks = {
                row["job_id"]: row
                for row in connection.execute("SELECT * FROM job_locks").fetchall()
            }
        return [self._job_row_to_dict(row, lock=locks.get(row["id"])) for row in rows]

    def worker_status(self, *, project: str | None = None) -> dict[str, Any]:
        clauses = []
        params: list[Any] = []
        if project:
            clauses.append("project = ?")
            params.append(project)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.session() as connection:
            counts = {
                row["state"]: row["count"]
                for row in connection.execute(
                    f"SELECT state, count(*) AS count FROM job_queue {where} GROUP BY state",
                    params,
                ).fetchall()
            }
            active_leases = connection.execute(
                """
                SELECT count(*) FROM job_locks
                WHERE lease_until > ?
                """,
                (utc_now(),),
            ).fetchone()[0]
            dead_letters = connection.execute(
                f"SELECT count(*) FROM job_dead_letters {where}",
                params,
            ).fetchone()[0]
        return {
            "status": "ok",
            "project": project or "all",
            "queue_depth": sum(counts.get(state, 0) for state in ("queued", "retrying")),
            "active_leases": active_leases,
            "dead_letters": dead_letters,
            "states": counts,
            "privacy_boundary": "Worker jobs, leases, attempts, and dead letters are stored in the local OpenAgent SQLite database.",
            "checked_at": utc_now(),
        }

    def lease_job(self, worker_id: str, *, lease_seconds: int = 300) -> dict[str, Any] | None:
        now = utc_now()
        lease_until = _utc_after(max(5, int(lease_seconds)))
        worker_id = worker_id.strip() or "openagent-worker"
        with self.session() as connection:
            expired = connection.execute(
                "SELECT job_id FROM job_locks WHERE lease_until <= ?",
                (now,),
            ).fetchall()
            expired_ids = [row["job_id"] for row in expired]
            if expired_ids:
                placeholders = ",".join("?" for _ in expired_ids)
                connection.execute(
                    f"""
                    UPDATE job_queue
                    SET state = 'retrying',
                        updated_at = ?,
                        progress_message = 'Lease expired; job is available for retry.',
                        last_error = CASE WHEN last_error = '' THEN 'lease expired' ELSE last_error END
                    WHERE id IN ({placeholders}) AND state = 'running'
                    """,
                    (now, *expired_ids),
                )
                connection.execute(
                    f"DELETE FROM job_locks WHERE job_id IN ({placeholders})",
                    tuple(expired_ids),
                )
            row = connection.execute(
                """
                SELECT * FROM job_queue
                WHERE state IN ('queued', 'retrying')
                  AND run_after <= ?
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            attempt_id = ContextEvent(title=f"attempt:{row['id']}", content=worker_id).id
            connection.execute(
                """
                UPDATE job_queue
                SET state = 'running',
                    attempts = attempts + 1,
                    progress = CASE WHEN progress < 5 THEN 5 ELSE progress END,
                    progress_message = 'Worker lease acquired.',
                    updated_at = ?,
                    last_error = ''
                WHERE id = ?
                """,
                (now, row["id"]),
            )
            connection.execute(
                """
                INSERT INTO job_attempts (id, job_id, worker_id, state, started_at)
                VALUES (?, ?, ?, 'running', ?)
                """,
                (attempt_id, row["id"], worker_id, now),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO job_locks (
                    job_id, worker_id, lease_until, acquired_at, heartbeat_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (row["id"], worker_id, lease_until, now, now),
            )
            locked = connection.execute("SELECT * FROM job_queue WHERE id = ?", (row["id"],)).fetchone()
            lock = connection.execute("SELECT * FROM job_locks WHERE job_id = ?", (row["id"],)).fetchone()
        job = self._job_row_to_dict(locked, lock=lock)
        job["attempt_id"] = attempt_id
        return job

    def heartbeat_job(self, job_id: str, worker_id: str, *, lease_seconds: int = 300) -> dict[str, Any] | None:
        now = utc_now()
        lease_until = _utc_after(max(5, int(lease_seconds)))
        with self.session() as connection:
            connection.execute(
                """
                UPDATE job_locks
                SET lease_until = ?, heartbeat_at = ?
                WHERE job_id = ? AND worker_id = ?
                """,
                (lease_until, now, job_id, worker_id),
            )
        return self.get_job(job_id)

    def update_job_progress(self, job_id: str, *, progress: int, message: str = "") -> dict[str, Any] | None:
        with self.session() as connection:
            connection.execute(
                """
                UPDATE job_queue
                SET progress = ?, progress_message = ?, updated_at = ?
                WHERE id = ? AND state IN ('queued', 'retrying', 'running')
                """,
                (max(0, min(100, int(progress))), message.strip(), utc_now(), job_id),
            )
        return self.get_job(job_id)

    def complete_job(self, job_id: str, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
        now = utc_now()
        result_json = json.dumps(result or {}, sort_keys=True)
        with self.session() as connection:
            connection.execute(
                """
                UPDATE job_queue
                SET state = 'completed',
                    progress = 100,
                    progress_message = 'Completed.',
                    result_json = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (result_json, now, now, job_id),
            )
            connection.execute(
                """
                UPDATE job_attempts
                SET state = 'completed', finished_at = ?, result_json = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (now, result_json, job_id),
            )
            connection.execute("DELETE FROM job_locks WHERE job_id = ?", (job_id,))
        return self.get_job(job_id)

    def fail_job(
        self,
        job_id: str,
        error: str,
        *,
        retry_delay_seconds: int = 30,
    ) -> dict[str, Any] | None:
        now = utc_now()
        with self.session() as connection:
            row = connection.execute("SELECT * FROM job_queue WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            attempts = int(row["attempts"])
            max_attempts = int(row["max_attempts"])
            if attempts >= max_attempts:
                state = "dead_lettered"
                run_after = row["run_after"]
                completed_at = now
                progress_message = "Moved to dead letters after max attempts."
            else:
                state = "retrying"
                run_after = _utc_after(max(1, int(retry_delay_seconds)))
                completed_at = ""
                progress_message = f"Retry scheduled after failure: {error}"
            connection.execute(
                """
                UPDATE job_queue
                SET state = ?,
                    run_after = ?,
                    progress_message = ?,
                    last_error = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (state, run_after, progress_message, error, now, completed_at, job_id),
            )
            connection.execute(
                """
                UPDATE job_attempts
                SET state = 'failed', finished_at = ?, error = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (now, error, job_id),
            )
            connection.execute("DELETE FROM job_locks WHERE job_id = ?", (job_id,))
            if state == "dead_lettered":
                dead_letter_id = ContextEvent(title=f"dead:{job_id}", content=error).id
                connection.execute(
                    """
                    INSERT INTO job_dead_letters (
                        id, job_id, project, job_type, payload_json, error, attempts, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dead_letter_id,
                        job_id,
                        row["project"],
                        row["job_type"],
                        row["payload_json"],
                        error,
                        attempts,
                        now,
                    ),
                )
        return self.get_job(job_id)

    def cancel_job(self, job_id: str, *, reason: str = "cancelled") -> dict[str, Any] | None:
        now = utc_now()
        with self.session() as connection:
            connection.execute(
                """
                UPDATE job_queue
                SET state = 'cancelled',
                    progress = 100,
                    progress_message = ?,
                    updated_at = ?,
                    cancelled_at = ?
                WHERE id = ? AND state NOT IN ('completed', 'dead_lettered')
                """,
                (reason, now, now, job_id),
            )
            connection.execute(
                """
                UPDATE job_attempts
                SET state = 'cancelled', finished_at = ?, error = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (now, reason, job_id),
            )
            connection.execute("DELETE FROM job_locks WHERE job_id = ?", (job_id,))
        return self.get_job(job_id)

    def retry_job(self, job_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self.session() as connection:
            connection.execute(
                """
                UPDATE job_queue
                SET state = 'queued',
                    run_after = ?,
                    progress = 0,
                    progress_message = 'Retry requested.',
                    last_error = '',
                    updated_at = ?,
                    completed_at = '',
                    cancelled_at = ''
                WHERE id = ? AND state IN ('failed', 'dead_lettered', 'cancelled', 'retrying')
                """,
                (now, now, job_id),
            )
            connection.execute("DELETE FROM job_locks WHERE job_id = ?", (job_id,))
        return self.get_job(job_id)

    def list_dead_letter_jobs(
        self,
        *,
        project: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        if project:
            query = "SELECT * FROM job_dead_letters WHERE project = ? ORDER BY created_at DESC LIMIT ?"
            params: tuple[Any, ...] = (project, limit)
        else:
            query = "SELECT * FROM job_dead_letters ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
        with self.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._dead_letter_row_to_dict(row) for row in rows]

    def health(self) -> dict[str, Any]:
        with self.session() as connection:
            event_count = connection.execute("SELECT count(*) FROM events").fetchone()[0]
            run_count = connection.execute("SELECT count(*) FROM agent_runs").fetchone()[0]
            conversation_count = connection.execute("SELECT count(*) FROM conversations").fetchone()[0]
            workflow_count = connection.execute("SELECT count(*) FROM workflows").fetchone()[0]
            audit_count = connection.execute("SELECT count(*) FROM audit_events").fetchone()[0]
        return {
            "status": "ok",
            "database": str(self.db_path),
            "event_count": event_count,
            "agent_run_count": run_count,
            "conversation_count": conversation_count,
            "workflow_count": workflow_count,
            "audit_event_count": audit_count,
            "checked_at": utc_now(),
        }

    @staticmethod
    def _event_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "type": row["type"],
            "source": row["source"],
            "project": row["project"],
            "title": row["title"],
            "content": row["content"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _agent_run_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["response_json"])
        return {
            "id": row["id"],
            "project": row["project"],
            **payload,
        }

    @staticmethod
    def _conversation_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "project": row["project"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _chat_message_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "role": row["role"],
            "content": row["content"],
            "provider": row["provider"],
            "model": row["model"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _audit_event_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "actor": row["actor"],
            "action": row["action"],
            "resource": row["resource"],
            "project": row["project"],
            "status": row["status"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _workflow_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "project": row["project"],
            "name": row["name"],
            "description": row["description"],
            "trigger_type": row["trigger_type"],
            "steps": json.loads(row["steps_json"]),
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _workflow_run_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "workflow_id": row["workflow_id"],
            "project": row["project"],
            "status": row["status"],
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }

    @staticmethod
    def _job_row_to_dict(
        row: sqlite3.Row,
        *,
        lock: sqlite3.Row | None = None,
        attempts: list[sqlite3.Row] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": row["id"],
            "project": row["project"],
            "job_type": row["job_type"],
            "payload": json.loads(row["payload_json"] or "{}"),
            "state": row["state"],
            "priority": row["priority"],
            "attempts": row["attempts"],
            "max_attempts": row["max_attempts"],
            "run_after": row["run_after"],
            "progress": row["progress"],
            "progress_message": row["progress_message"],
            "result": json.loads(row["result_json"] or "{}"),
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "cancelled_at": row["cancelled_at"],
        }
        if lock is not None:
            record["lease"] = {
                "worker_id": lock["worker_id"],
                "lease_until": lock["lease_until"],
                "acquired_at": lock["acquired_at"],
                "heartbeat_at": lock["heartbeat_at"],
            }
        else:
            record["lease"] = None
        if attempts is not None:
            record["attempt_records"] = [OpenAgentStore._attempt_row_to_dict(attempt) for attempt in attempts]
        return record

    @staticmethod
    def _attempt_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "job_id": row["job_id"],
            "worker_id": row["worker_id"],
            "state": row["state"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": row["error"],
            "result": json.loads(row["result_json"] or "{}"),
        }

    @staticmethod
    def _dead_letter_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "job_id": row["job_id"],
            "project": row["project"],
            "job_type": row["job_type"],
            "payload": json.loads(row["payload_json"] or "{}"),
            "error": row["error"],
            "attempts": row["attempts"],
            "created_at": row["created_at"],
        }


def _utc_after(seconds: int) -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=seconds)).isoformat()
