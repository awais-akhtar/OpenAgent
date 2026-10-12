"""Personal assistant, task, planning, meeting, travel, and CRM foundation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import secrets
from typing import Any

from .models import AuditEvent, ContextEvent, utc_now
from .storage import OpenAgentStore


ACTION_RE = re.compile(r"\b(action|todo|follow up|send|schedule|prepare|decide|decision|owner)\b", re.I)


class AssistantManager:
    def __init__(self, store: OpenAgentStore):
        self.store = store
        self.initialize()

    def initialize(self) -> None:
        self.store.initialize()

    def upsert_profile(
        self,
        *,
        user_id: str = "local-user",
        project: str = "default",
        preferences: dict[str, Any] | None = None,
        facts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        project = project.strip() or "default"
        user_id = user_id.strip() or "local-user"
        profile_id = stable_id("profile", project, user_id)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO assistant_profiles (
                    id, project, user_id, preferences_json, facts_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM assistant_profiles WHERE id = ?), ?), ?)
                """,
                (
                    profile_id,
                    project,
                    user_id,
                    json.dumps(preferences or {}, sort_keys=True),
                    json.dumps(facts or {}, sort_keys=True),
                    profile_id,
                    now,
                    now,
                ),
            )
        self._audit("assistant.profile.upsert", profile_id, project=project)
        return self.get_profile(profile_id) or {}

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM assistant_profiles WHERE id = ?", (profile_id,)).fetchone()
        return profile_row(row) if row else None

    def create_goal(
        self,
        *,
        title: str,
        project: str = "default",
        description: str = "",
        horizon: str = "short",
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        title = title.strip()
        if not title:
            raise ValueError("Goal title is required.")
        project = project.strip() or "default"
        goal_id = unique_id("goal", project, title)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO assistant_goals (
                    id, project, title, description, status, horizon,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal_id,
                    project,
                    title,
                    description,
                    status,
                    horizon,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                    now,
                ),
            )
        self._index_context("goal", title, description, project, {"goal_id": goal_id})
        self._audit("assistant.goal.create", goal_id, project=project, metadata={"status": status})
        return self.get_goal(goal_id) or {}

    def get_goal(self, goal_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM assistant_goals WHERE id = ?", (goal_id,)).fetchone()
        return goal_row(row) if row else None

    def goals(self, *, project: str = "default", status: str = "") -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if status:
            clauses.append("status = ?")
            params.append(status)
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM assistant_goals WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC",
                tuple(params),
            ).fetchall()
        return [goal_row(row) for row in rows]

    def create_task(
        self,
        *,
        title: str,
        project: str = "default",
        priority: str = "normal",
        due_at: str = "",
        goal_id: str = "",
        contact_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        title = title.strip()
        if not title:
            raise ValueError("Task title is required.")
        project = project.strip() or "default"
        task_id = unique_id("task", project, title)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO assistant_tasks (
                    id, project, title, status, priority, due_at, goal_id,
                    contact_id, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    project,
                    title,
                    "open",
                    priority,
                    due_at,
                    goal_id,
                    contact_id,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                    now,
                ),
            )
        self._index_context("task", title, f"Priority: {priority}. Due: {due_at or 'unscheduled'}", project, {"task_id": task_id})
        self._audit("assistant.task.create", task_id, project=project, metadata={"priority": priority})
        return self.get_task(task_id) or {}

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM assistant_tasks WHERE id = ?", (task_id,)).fetchone()
        return task_row(row) if row else None

    def tasks(self, *, project: str = "default", status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if status:
            clauses.append("status = ?")
            params.append(status)
        params.append(max(1, min(limit, 500)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM assistant_tasks
                WHERE {' AND '.join(clauses)}
                ORDER BY status, due_at = '', due_at, updated_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [task_row(row) for row in rows]

    def complete_task(self, task_id: str) -> dict[str, Any]:
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Unknown task '{task_id}'.")
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "UPDATE assistant_tasks SET status = 'completed', updated_at = ? WHERE id = ?",
                (now, task_id),
            )
        self._audit("assistant.task.complete", task_id, project=task["project"])
        return self.get_task(task_id) or {}

    def create_meeting_brief(
        self,
        *,
        title: str,
        project: str = "default",
        starts_at: str = "",
        attendees: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        title = title.strip()
        if not title:
            raise ValueError("Meeting title is required.")
        attendees = attendees or []
        project = project.strip() or "default"
        related_context = self.store.search_events(" ".join([title, *attendees]), project=project, limit=8)
        agenda = [
            f"Confirm desired outcome for {title}.",
            "Review open tasks, blockers, and owners.",
            "Capture decisions, next actions, and follow-up dates.",
        ]
        brief = {
            "agenda": agenda,
            "attendees": attendees,
            "context": related_context,
            "preparation": [event["title"] for event in related_context[:5]],
        }
        brief_id = unique_id("meeting-brief", project, title)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO assistant_meeting_briefs (
                    id, project, title, starts_at, attendees_json,
                    notes, brief_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    brief_id,
                    project,
                    title,
                    starts_at,
                    json.dumps(attendees),
                    notes,
                    json.dumps(brief, sort_keys=True),
                    created_at,
                ),
            )
        self._index_context("meeting", title, notes or json.dumps(brief), project, {"brief_id": brief_id})
        self._audit("assistant.meeting.brief", brief_id, project=project)
        return self.get_meeting_brief(brief_id) or {}

    def get_meeting_brief(self, brief_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM assistant_meeting_briefs WHERE id = ?", (brief_id,)).fetchone()
        return meeting_brief_row(row) if row else None

    def summarize_meeting(self, *, transcript: str, project: str = "default", title: str = "Meeting summary") -> dict[str, Any]:
        lines = [line.strip() for line in transcript.splitlines() if line.strip()]
        action_items = [line for line in lines if ACTION_RE.search(line)]
        decisions = [line for line in lines if "decide" in line.lower() or "decision" in line.lower()]
        summary = {
            "title": title,
            "summary": " ".join(lines[:3])[:800],
            "decisions": decisions,
            "action_items": action_items,
            "transcript_lines": len(lines),
        }
        event = self.store.add_event(
            ContextEvent(
                title=f"Meeting summary: {title}",
                content=json.dumps(summary, sort_keys=True),
                type="meeting_summary",
                source="assistant",
                project=project,
                metadata=summary,
            )
        )
        self._audit("assistant.meeting.summary", event["id"], project=project)
        return {"summary": summary, "event": event}

    def create_travel_plan(
        self,
        *,
        destination: str,
        project: str = "default",
        starts_at: str = "",
        ends_at: str = "",
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        destination = destination.strip()
        if not destination:
            raise ValueError("Travel destination is required.")
        preferences = preferences or {}
        checklist = [
            "Confirm passport, visa, and identity documents.",
            "Book transport and lodging inside approved budget.",
            "Create calendar holds for travel, check-in, and meetings.",
            "Prepare local emergency contacts and offline documents.",
            "Review follow-ups after return.",
        ]
        plan = {
            "destination": destination,
            "starts_at": starts_at,
            "ends_at": ends_at,
            "preferences": preferences,
            "checklist": checklist,
            "monitoring": ["flight_status", "weather", "schedule_conflicts"],
        }
        plan_id = unique_id("travel", project, destination)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO assistant_travel_plans (
                    id, project, destination, starts_at, ends_at, plan_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (plan_id, project, destination, starts_at, ends_at, json.dumps(plan, sort_keys=True), created_at),
            )
        self._index_context("travel", f"Travel plan: {destination}", json.dumps(plan), project, {"travel_plan_id": plan_id})
        self._audit("assistant.travel.plan", plan_id, project=project)
        return self.get_travel_plan(plan_id) or {}

    def get_travel_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM assistant_travel_plans WHERE id = ?", (plan_id,)).fetchone()
        return travel_plan_row(row) if row else None

    def upsert_crm_record(
        self,
        *,
        record_type: str,
        name: str,
        project: str = "default",
        status: str = "active",
        value: float = 0,
        contact: str = "",
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record_type = record_type.strip() or "relationship"
        if record_type not in {"relationship", "lead", "opportunity", "contact_intelligence"}:
            raise ValueError(f"Unsupported CRM record type '{record_type}'.")
        name = name.strip()
        if not name:
            raise ValueError("CRM record name is required.")
        project = project.strip() or "default"
        record_id = stable_id("crm", project, record_type, name)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO crm_records (
                    id, project, record_type, name, status, value, contact,
                    notes, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM crm_records WHERE id = ?), ?), ?)
                """,
                (
                    record_id,
                    project,
                    record_type,
                    name,
                    status,
                    float(value),
                    contact,
                    notes,
                    json.dumps(metadata or {}, sort_keys=True),
                    record_id,
                    now,
                    now,
                ),
            )
        self._index_context("crm", f"{record_type}: {name}", notes, project, {"crm_record_id": record_id})
        self._audit("assistant.crm.upsert", record_id, project=project, metadata={"record_type": record_type})
        return self.get_crm_record(record_id) or {}

    def get_crm_record(self, record_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM crm_records WHERE id = ?", (record_id,)).fetchone()
        return crm_record_row(row) if row else None

    def crm_records(self, *, project: str = "default", record_type: str = "") -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if record_type:
            clauses.append("record_type = ?")
            params.append(record_type)
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM crm_records WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC",
                tuple(params),
            ).fetchall()
        return [crm_record_row(row) for row in rows]

    def daily_brief(self, *, project: str = "default") -> dict[str, Any]:
        open_tasks = self.tasks(project=project, status="open", limit=20)
        active_goals = self.goals(project=project, status="active")
        recent_context = self.store.list_events(project=project, limit=8)
        priorities = [task for task in open_tasks if task["priority"] in {"high", "urgent"}]
        return {
            "project": project,
            "open_tasks": open_tasks,
            "active_goals": active_goals,
            "recent_context": recent_context,
            "priority_count": len(priorities),
            "recommended_focus": priorities[:3] or open_tasks[:3],
        }

    def productivity_coaching(self, *, project: str = "default") -> dict[str, Any]:
        tasks = self.tasks(project=project, limit=500)
        goals = self.goals(project=project)
        by_status = Counter(task["status"] for task in tasks)
        by_priority = Counter(task["priority"] for task in tasks)
        suggestions = []
        if by_status.get("open", 0) > 5:
            suggestions.append("Reduce work in progress by choosing the top three open tasks.")
        if not goals:
            suggestions.append("Create at least one active goal so tasks have a clear direction.")
        if by_priority.get("high", 0) == 0 and tasks:
            suggestions.append("Mark one task as high priority to clarify the next move.")
        return {
            "project": project,
            "task_count": len(tasks),
            "goal_count": len(goals),
            "by_status": dict(by_status),
            "by_priority": dict(by_priority),
            "suggestions": suggestions,
        }

    def _index_context(
        self,
        type_name: str,
        title: str,
        content: str,
        project: str,
        metadata: dict[str, Any],
    ) -> None:
        self.store.add_event(
            ContextEvent(
                title=f"Assistant {type_name}: {title}",
                content=content or title,
                type=f"assistant_{type_name}",
                source="assistant",
                project=project,
                metadata=metadata,
            )
        )

    def _audit(
        self,
        action: str,
        resource: str,
        *,
        project: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="assistant",
                action=action,
                resource=resource,
                project=project,
                metadata=metadata or {},
            )
        )


def profile_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "user_id": row["user_id"],
        "preferences": json.loads(row["preferences_json"]),
        "facts": json.loads(row["facts_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def goal_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "title": row["title"],
        "description": row["description"],
        "status": row["status"],
        "horizon": row["horizon"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def task_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "title": row["title"],
        "status": row["status"],
        "priority": row["priority"],
        "due_at": row["due_at"],
        "goal_id": row["goal_id"],
        "contact_id": row["contact_id"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def meeting_brief_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "title": row["title"],
        "starts_at": row["starts_at"],
        "attendees": json.loads(row["attendees_json"]),
        "notes": row["notes"],
        "brief": json.loads(row["brief_json"]),
        "created_at": row["created_at"],
    }


def travel_plan_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "destination": row["destination"],
        "starts_at": row["starts_at"],
        "ends_at": row["ends_at"],
        "plan": json.loads(row["plan_json"]),
        "created_at": row["created_at"],
    }


def crm_record_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "record_type": row["record_type"],
        "name": row["name"],
        "status": row["status"],
        "value": row["value"],
        "contact": row["contact"],
        "notes": row["notes"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:32]


def unique_id(*parts: str) -> str:
    return stable_id(*parts, utc_now(), secrets.token_hex(8))
