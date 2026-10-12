"""Unified local agent bridge over memory, calendar, documents, and tasks."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from .calendar_system import CalendarSystem
from .documents import DocumentManager
from .models import AuditEvent, ContextEvent, utc_now
from .scheduler import Scheduler
from .storage import OpenAgentStore


class AgentBridge:
    """Small stable API for agents to read/write core local OpenAgent objects."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.documents = DocumentManager(store)
        self.calendar = CalendarSystem(store)
        self.scheduler = Scheduler(store)
        self.initialize()

    def initialize(self) -> None:
        self.store.initialize()

    def capabilities(self) -> dict[str, Any]:
        return {
            "runtime": "local-agent-bridge",
            "capabilities": [
                "memory.read",
                "memory.write",
                "calendar.read",
                "calendar.write",
                "documents.read",
                "documents.write",
                "tasks.read",
                "tasks.write",
            ],
            "privacy_boundary": "Agent bridge reads and writes only local OpenAgent SQLite/media records.",
        }

    def create_task(
        self,
        *,
        goal: str,
        project: str = "default",
        owner: str = "local",
        agent: str = "assistant",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not goal.strip():
            raise ValueError("goal is required")
        task_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO agent_bridge_tasks (
                    id, project, owner, agent, goal, status, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (task_id, project, owner, agent or "assistant", goal.strip(), json.dumps(payload or {}, sort_keys=True), now, now),
            )
            row = connection.execute("SELECT * FROM agent_bridge_tasks WHERE id = ?", (task_id,)).fetchone()
        self.store.add_audit_event(
            AuditEvent(
                actor="agent-bridge",
                action="agent.task.create",
                resource=f"agent_task:{task_id}",
                project=project,
                metadata={"owner": owner, "agent": agent},
            )
        )
        return task_row(row)

    def tasks(self, *, project: str = "default", owner: str = "", status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        if status:
            clauses.append("status = ?")
            args.append(status)
        sql = "SELECT * FROM agent_bridge_tasks WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
        return [task_row(row) for row in rows]

    def memory(self, *, project: str = "default", query: str = "", limit: int = 50) -> list[dict[str, Any]]:
        if query:
            return self.store.search_events(query, limit=limit, project=project)
        return self.store.list_events(limit=limit, project=project)

    def write_memory(
        self,
        *,
        title: str,
        content: str,
        project: str = "default",
        memory_type: str = "agent_memory",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not title.strip() or not content.strip():
            raise ValueError("title and content are required")
        event = self.store.add_event(
            ContextEvent(
                title=title.strip(),
                content=content.strip(),
                type=memory_type or "agent_memory",
                source="agent-bridge",
                project=project,
                metadata=metadata or {},
            )
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="agent-bridge",
                action="memory.write",
                resource=f"event:{event['id']}",
                project=project,
                metadata={"title": title},
            )
        )
        return event

    def documents_list(self, *, project: str = "default", owner: str = "", query: str = "", limit: int = 50) -> dict[str, Any]:
        return self.documents.list_library(project=project, owner=owner, query=query, limit=limit)

    def create_document(
        self,
        *,
        title: str,
        content: str = "",
        project: str = "default",
        owner: str = "local",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        doc = self.documents.create_document(title=title, content=content, project=project, owner=owner, tags=tags or ["agent"])
        self.store.add_audit_event(
            AuditEvent(
                actor="agent-bridge",
                action="document.write",
                resource=f"document:{doc['doc_id']}",
                project=project,
                metadata={"owner": owner, "title": title},
            )
        )
        return doc

    def calendar_events(self, *, project: str = "default", calendar_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        return self.calendar.list_events(project=project, calendar_id=calendar_id, limit=limit)

    def create_calendar_event(
        self,
        *,
        title: str,
        starts_at: str,
        ends_at: str,
        project: str = "default",
        calendar_id: str = "",
        description: str = "",
    ) -> dict[str, Any]:
        event = self.calendar.create_event(
            title,
            starts_at,
            ends_at,
            project=project,
            calendar_id=calendar_id,
            description=description,
            metadata={"created_by": "agent-bridge"},
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="agent-bridge",
                action="calendar.write",
                resource=f"calendar_event:{event['id']}",
                project=project,
                metadata={"title": title},
            )
        )
        return event


def task_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "agent": row["agent"],
        "goal": row["goal"],
        "status": row["status"],
        "payload": json.loads(row["payload_json"] or "{}"),
        "result": json.loads(row["result_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
