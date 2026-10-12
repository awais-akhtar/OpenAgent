"""Durable research sessions layered over the local research engine."""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


TERMINAL_RESEARCH_STATES = {"completed", "failed", "cancelled"}


class ResearchLifecycle:
    """Own durable sessions, checkpoints, sources, branches, and retention."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.initialize()

    def initialize(self) -> None:
        self.store.initialize()

    def create(
        self,
        query: str,
        *,
        project: str = "default",
        max_iterations: int = 3,
        allow_external: bool = False,
        provider: str = "auto",
        model: str = "",
        parent_id: str = "",
        branch_label: str = "",
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query is required.")
        if parent_id and self.get(parent_id) is None:
            raise KeyError(parent_id)
        session_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO research_sessions (
                    id, project, query, status, parent_id, branch_label, provider,
                    model, allow_external, max_iterations, max_attempts, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, project, query, parent_id, branch_label, provider or "auto",
                    model, 1 if allow_external else 0, max(1, min(int(max_iterations), 12)),
                    max(1, min(int(max_attempts), 10)), now, now,
                ),
            )
        self.checkpoint(session_id, "queued", 0, "Research session queued.")
        self._audit("research.session.create", session_id, project, {"parent_id": parent_id})
        return self.get(session_id) or {}

    def list(
        self,
        *,
        project: str = "default",
        include_archived: bool = False,
        status: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = ["project = ?"]
        values: list[Any] = [project]
        if not include_archived:
            where.append("archived = 0")
        if status:
            where.append("status = ?")
            values.append(status)
        values.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM research_sessions WHERE {' AND '.join(where)} ORDER BY updated_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._record(row, details=False) for row in rows]

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM research_sessions WHERE id = ?", (session_id,)).fetchone()
        return self._record(row, details=True) if row else None

    def attach_job(self, session_id: str, job_id: str) -> dict[str, Any]:
        self._update(session_id, job_id=job_id)
        return self.get(session_id) or {}

    def start(self, session_id: str) -> dict[str, Any]:
        current = self._require(session_id)
        if current["cancel_requested"]:
            self._update(session_id, status="cancelled", completed_at=utc_now())
            return self.get(session_id) or {}
        self._update(session_id, status="running", attempt=current["attempt"] + 1, error="")
        self.checkpoint(session_id, "planning", 10, "Planning research strategy.")
        self._audit("research.session.start", session_id, current["project"], {"attempt": current["attempt"] + 1})
        return self.get(session_id) or {}

    def complete(self, session_id: str, result: dict[str, Any]) -> dict[str, Any]:
        current = self._require(session_id)
        if current["cancel_requested"]:
            self._update(session_id, status="cancelled", completed_at=utc_now())
            self.checkpoint(session_id, "cancelled", 100, "Research cancelled before results were committed.")
            return self.get(session_id) or {}
        sources = result.get("sources") if isinstance(result.get("sources"), list) else []
        with self.store.session() as connection:
            connection.execute("DELETE FROM research_sources WHERE session_id = ?", (session_id,))
            for source in sources:
                source = source if isinstance(source, dict) else {"title": str(source)}
                connection.execute(
                    """
                    INSERT INTO research_sources (
                        session_id, source_id, title, source_type, snippet, url, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id, str(source.get("id", "")), str(source.get("title", "")),
                        str(source.get("type", "source")), str(source.get("snippet", "")),
                        str(source.get("url", "")), json.dumps(source.get("metadata", {}), sort_keys=True), utc_now(),
                    ),
                )
        self.checkpoint(session_id, "sources", 45, f"Collected {len(sources)} source(s).", {"count": len(sources)})
        self.checkpoint(session_id, "synthesis", 80, "Synthesized source-backed findings.")
        self._update(
            session_id, status="completed", plan=str(result.get("plan", "")),
            report=str(result.get("report", "")), result_json=json.dumps(result, sort_keys=True),
            provider=str(result.get("provider", current["provider"])),
            model=str(result.get("model", current["model"])),
            completed_at=utc_now(), error="",
        )
        self.checkpoint(session_id, "completed", 100, "Research report completed.")
        self._audit("research.session.complete", session_id, current["project"], {"sources": len(sources)})
        return self.get(session_id) or {}

    def fail(self, session_id: str, error: str) -> dict[str, Any]:
        current = self._require(session_id)
        status = "cancelled" if current["cancel_requested"] else "failed"
        self._update(session_id, status=status, error=error, completed_at=utc_now())
        self.checkpoint(session_id, status, 100, error or f"Research {status}.")
        self._audit("research.session.fail", session_id, current["project"], {"status": status, "error": error})
        return self.get(session_id) or {}

    def cancel(self, session_id: str) -> dict[str, Any]:
        current = self._require(session_id)
        status = "cancelled" if current["status"] == "queued" else current["status"]
        completed_at = utc_now() if status == "cancelled" else current["completed_at"]
        self._update(session_id, cancel_requested=1, status=status, completed_at=completed_at)
        self.checkpoint(session_id, "cancel-requested", 100 if status == "cancelled" else 0, "Cancellation requested.")
        self._audit("research.session.cancel", session_id, current["project"], {})
        job_id = current.get("job_id", "")
        if job_id:
            try:
                self.store.cancel_job(job_id)
            except (KeyError, ValueError):
                pass
        return self.get(session_id) or {}

    def retry(self, session_id: str) -> dict[str, Any]:
        current = self._require(session_id)
        if current["status"] not in {"failed", "cancelled"}:
            raise ValueError("Only failed or cancelled research can be retried.")
        if current["attempt"] >= current["max_attempts"]:
            raise ValueError("Research retry limit reached.")
        self._update(session_id, status="queued", cancel_requested=0, error="", completed_at="", job_id="")
        self.checkpoint(session_id, "retry", 0, "Research queued for retry.")
        return self.get(session_id) or {}

    def branch(self, session_id: str, *, query: str = "", label: str = "") -> dict[str, Any]:
        parent = self._require(session_id)
        return self.create(
            query or parent["query"], project=parent["project"], max_iterations=parent["max_iterations"],
            allow_external=parent["allow_external"], provider=parent["provider"], model=parent["model"],
            parent_id=session_id, branch_label=label or "Research branch", max_attempts=parent["max_attempts"],
        )

    def archive(self, session_id: str, *, archived: bool = True) -> dict[str, Any]:
        self._require(session_id)
        self._update(session_id, archived=1 if archived else 0)
        return self.get(session_id) or {}

    def delete(self, session_id: str) -> dict[str, Any]:
        current = self._require(session_id)
        if current["status"] == "running":
            raise ValueError("Cancel running research before deleting it.")
        with self.store.session() as connection:
            connection.execute("DELETE FROM research_checkpoints WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM research_sources WHERE session_id = ?", (session_id,))
            connection.execute("UPDATE research_sessions SET parent_id = '' WHERE parent_id = ?", (session_id,))
            connection.execute("DELETE FROM research_sessions WHERE id = ?", (session_id,))
        self._audit("research.session.delete", session_id, current["project"], {})
        return {"deleted": True, "session_id": session_id}

    def checkpoints(self, session_id: str, *, after_id: int = 0) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM research_checkpoints WHERE session_id = ? AND id > ? ORDER BY id",
                (session_id, max(0, int(after_id))),
            ).fetchall()
        return [
            {
                "id": row["id"], "session_id": row["session_id"], "stage": row["stage"],
                "progress": row["progress"], "message": row["message"],
                "payload": json.loads(row["payload_json"] or "{}"), "created_at": row["created_at"],
            }
            for row in rows
        ]

    def sources(self, session_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM research_sources WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        return [
            {
                "id": row["source_id"], "title": row["title"], "type": row["source_type"],
                "snippet": row["snippet"], "url": row["url"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]

    def source_graph(self, session_id: str) -> dict[str, Any]:
        current = self._require(session_id)
        nodes = [{"id": session_id, "kind": "research", "label": current["query"]}]
        edges = []
        for index, source in enumerate(current["sources"]):
            source_id = source["id"] or f"source-{index}"
            nodes.append({"id": source_id, "kind": source["type"], "label": source["title"] or source["type"]})
            edges.append({"from": session_id, "to": source_id, "relation": "uses"})
        return {"session_id": session_id, "nodes": nodes, "edges": edges}

    def stream_events(
        self,
        session_id: str,
        *,
        after_id: int = 0,
        poll_seconds: float = 0.2,
        timeout_seconds: float = 120.0,
    ):
        """Yield durable progress events until the session reaches a terminal state.

        Every event is recoverable from SQLite. Clients can reconnect with the
        last checkpoint id instead of relying on an in-memory event queue.
        """

        self._require(session_id)
        cursor = max(0, int(after_id))
        poll_seconds = max(0.05, min(float(poll_seconds), 2.0))
        deadline = time.monotonic() + max(1.0, min(float(timeout_seconds), 300.0))
        yield {"type": "session", "session": self.get(session_id), "cursor": cursor}
        while True:
            checkpoints = self.checkpoints(session_id, after_id=cursor)
            for checkpoint in checkpoints:
                cursor = max(cursor, int(checkpoint["id"]))
                yield {"type": "checkpoint", "checkpoint": checkpoint, "cursor": cursor}
            session = self._require(session_id)
            if session["status"] in TERMINAL_RESEARCH_STATES:
                yield {"type": "complete", "session": session, "cursor": cursor}
                return
            if time.monotonic() >= deadline:
                yield {"type": "heartbeat", "session_id": session_id, "status": session["status"], "cursor": cursor}
                return
            time.sleep(poll_seconds)

    def checkpoint(self, session_id: str, stage: str, progress: int, message: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        now = utc_now()
        with self.store.session() as connection:
            cursor = connection.execute(
                """
                INSERT INTO research_checkpoints (session_id, stage, progress, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, stage, max(0, min(int(progress), 100)), message, json.dumps(payload or {}, sort_keys=True), now),
            )
            connection.execute("UPDATE research_sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        return {"id": cursor.lastrowid, "stage": stage, "progress": progress, "message": message, "created_at": now}

    def _record(self, row: Any, *, details: bool) -> dict[str, Any]:
        result = {
            "id": row["id"], "project": row["project"], "query": row["query"], "status": row["status"],
            "parent_id": row["parent_id"], "branch_label": row["branch_label"], "provider": row["provider"],
            "model": row["model"], "allow_external": bool(row["allow_external"]),
            "max_iterations": row["max_iterations"], "attempt": row["attempt"], "max_attempts": row["max_attempts"],
            "cancel_requested": bool(row["cancel_requested"]), "archived": bool(row["archived"]),
            "job_id": row["job_id"], "plan": row["plan"], "report": row["report"], "error": row["error"],
            "created_at": row["created_at"], "updated_at": row["updated_at"], "completed_at": row["completed_at"],
        }
        if details:
            result["result"] = json.loads(row["result_json"] or "{}")
            result["checkpoints"] = self.checkpoints(row["id"])
            result["sources"] = self.sources(row["id"])
        return result

    def _require(self, session_id: str) -> dict[str, Any]:
        current = self.get(session_id)
        if current is None:
            raise KeyError(session_id)
        return current

    def _update(self, session_id: str, **values: Any) -> None:
        allowed = {
            "status", "attempt", "cancel_requested", "archived", "job_id", "plan", "report",
            "provider", "model", "result_json", "error", "completed_at",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = utc_now()
        with self.store.session() as connection:
            cursor = connection.execute(
                f"UPDATE research_sessions SET {', '.join(f'{key} = ?' for key in updates)} WHERE id = ?",
                [*updates.values(), session_id],
            )
        if cursor.rowcount != 1:
            raise KeyError(session_id)

    def _audit(self, action: str, session_id: str, project: str, metadata: dict[str, Any]) -> None:
        self.store.add_audit_event(
            AuditEvent(actor="research-lifecycle", action=action, resource=f"research:{session_id}", project=project, metadata=metadata)
        )
