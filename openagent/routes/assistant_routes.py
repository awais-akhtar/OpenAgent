"""Personal assistant HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "assistant"
ROUTES = (
    ("GET", "/api/assistant/tasks"),
    ("GET", "/api/assistant/goals"),
    ("GET", "/api/assistant/activity"),
    ("GET", "/api/assistant/daily-brief"),
    ("GET", "/api/assistant/coach"),
    ("GET", "/api/assistant/crm"),
    ("POST", "/api/assistant/profile"),
    ("POST", "/api/assistant/goals"),
    ("POST", "/api/assistant/tasks"),
    ("POST", "/api/assistant/tasks/complete"),
    ("PATCH", "/api/assistant/tasks/{task_id}"),
    ("DELETE", "/api/assistant/tasks/{task_id}"),
    ("PATCH", "/api/assistant/goals/{goal_id}"),
    ("DELETE", "/api/assistant/goals/{goal_id}"),
    ("POST", "/api/assistant/meeting/brief"),
    ("POST", "/api/assistant/meeting/summary"),
    ("POST", "/api/assistant/travel/plan"),
    ("POST", "/api/assistant/crm"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class AssistantRouteDependencies:
    assistant: Any
    first: Callable[..., str | None]
    parse_int: Callable[..., int]
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, query: str, deps: AssistantRouteDependencies) -> JsonRouteResult | None:
    if path not in {route for method, route in ROUTES if method == "GET"}:
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    if path == "/api/assistant/activity":
        explicit_project = str(deps.first(params, "project") or "").strip()
        if not explicit_project:
            return JsonRouteResult({"error": "Project is required for assistant activity."}, HTTPStatus.BAD_REQUEST)
        try:
            return JsonRouteResult({"events": deps.assistant.activity_events(
                project=explicit_project,
                limit=deps.parse_int(deps.first(params, "limit"), default=20),
            )})
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    if path == "/api/assistant/tasks":
        return JsonRouteResult({"tasks": deps.assistant.tasks(
            project=project,
            status=deps.first(params, "status") or "",
            limit=deps.parse_int(deps.first(params, "limit"), default=100),
        )})
    if path == "/api/assistant/goals":
        return JsonRouteResult({"goals": deps.assistant.goals(
            project=project, status=deps.first(params, "status") or "",
        )})
    if path == "/api/assistant/daily-brief":
        return JsonRouteResult(deps.assistant.daily_brief(project=project))
    if path == "/api/assistant/coach":
        return JsonRouteResult(deps.assistant.productivity_coaching(project=project))
    return JsonRouteResult({"records": deps.assistant.crm_records(
        project=project, record_type=deps.first(params, "record_type") or "",
    )})


def handle_post(path: str, deps: AssistantRouteDependencies) -> JsonRouteResult | None:
    if path not in {route for method, route in ROUTES if method == "POST"}:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    assistant = deps.assistant
    try:
        if path == "/api/assistant/profile":
            profile = assistant.upsert_profile(
                user_id=str(payload.get("user_id", "local-user")).strip() or "local-user",
                project=project,
                preferences=payload.get("preferences") if isinstance(payload.get("preferences"), dict) else {},
                facts=payload.get("facts") if isinstance(payload.get("facts"), dict) else {},
            )
            return JsonRouteResult({"profile": profile}, HTTPStatus.CREATED)
        if path == "/api/assistant/goals":
            goal = assistant.create_goal(
                title=str(payload.get("title", "")).strip(), project=project,
                description=str(payload.get("description", "")),
                horizon=str(payload.get("horizon", "short")).strip() or "short",
                status=str(payload.get("status", "active")).strip() or "active",
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"goal": goal}, HTTPStatus.CREATED)
        if path == "/api/assistant/tasks":
            task = assistant.create_task(
                title=str(payload.get("title", "")).strip(), project=project,
                priority=str(payload.get("priority", "normal")).strip() or "normal",
                due_at=str(payload.get("due_at", "")).strip(),
                goal_id=str(payload.get("goal_id", "")).strip(),
                contact_id=str(payload.get("contact_id", "")).strip(),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"task": task}, HTTPStatus.CREATED)
        if path == "/api/assistant/tasks/complete":
            if not str(payload.get("project", "")).strip():
                return JsonRouteResult({"error": "Project is required to complete a task."}, HTTPStatus.BAD_REQUEST)
            task = assistant.complete_task(
                str(payload.get("task_id", "")).strip(),
                project=str(payload["project"]).strip(),
            )
            return JsonRouteResult({"task": task})
        if path == "/api/assistant/meeting/brief":
            brief = assistant.create_meeting_brief(
                title=str(payload.get("title", "")).strip(), project=project,
                starts_at=str(payload.get("starts_at", "")).strip(),
                attendees=[str(item) for item in payload.get("attendees", [])]
                if isinstance(payload.get("attendees"), list) else [],
                notes=str(payload.get("notes", "")),
            )
            return JsonRouteResult({"brief": brief}, HTTPStatus.CREATED)
        if path == "/api/assistant/meeting/summary":
            result = assistant.summarize_meeting(
                transcript=str(payload.get("transcript", "")), project=project,
                title=str(payload.get("title", "Meeting summary")).strip() or "Meeting summary",
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path == "/api/assistant/travel/plan":
            plan = assistant.create_travel_plan(
                destination=str(payload.get("destination", "")).strip(), project=project,
                starts_at=str(payload.get("starts_at", "")).strip(),
                ends_at=str(payload.get("ends_at", "")).strip(),
                preferences=payload.get("preferences") if isinstance(payload.get("preferences"), dict) else {},
            )
            return JsonRouteResult({"plan": plan}, HTTPStatus.CREATED)
        record = assistant.upsert_crm_record(
            record_type=str(payload.get("record_type", "relationship")).strip() or "relationship",
            name=str(payload.get("name", "")).strip(), project=project,
            status=str(payload.get("status", "active")).strip() or "active",
            value=float(payload.get("value", 0) or 0),
            contact=str(payload.get("contact", "")), notes=str(payload.get("notes", "")),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )
        return JsonRouteResult({"record": record}, HTTPStatus.CREATED)
    except (TypeError, ValueError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def _record_route(path: str, kind: str) -> str | None:
    prefix = f"/api/assistant/{kind}/"
    if not path.startswith(prefix):
        return None
    record_id = unquote(path.removeprefix(prefix)).strip()
    return record_id if record_id and "/" not in record_id else None


def handle_patch(path: str, query: str, deps: AssistantRouteDependencies) -> JsonRouteResult | None:
    kind = "tasks" if path.startswith("/api/assistant/tasks/") else "goals" if path.startswith("/api/assistant/goals/") else ""
    if not kind:
        return None
    record_id = _record_route(path, kind)
    if record_id is None:
        return JsonRouteResult({"error": "A record ID is required."}, HTTPStatus.BAD_REQUEST)
    payload = deps.read_json()
    params = parse_qs(query)
    project = str(payload.get("project") or deps.first(params, "project") or "default").strip()
    updates = {key: value for key, value in payload.items() if key != "project"}
    try:
        item = deps.assistant.update_task(record_id, project=project, updates=updates) if kind == "tasks" else deps.assistant.update_goal(record_id, project=project, updates=updates)
        return JsonRouteResult({"task" if kind == "tasks" else "goal": item})
    except KeyError:
        return JsonRouteResult({"error": "Record not found."}, HTTPStatus.NOT_FOUND)
    except (TypeError, ValueError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def handle_delete(path: str, query: str, deps: AssistantRouteDependencies) -> JsonRouteResult | None:
    kind = "tasks" if path.startswith("/api/assistant/tasks/") else "goals" if path.startswith("/api/assistant/goals/") else ""
    if not kind:
        return None
    record_id = _record_route(path, kind)
    if record_id is None:
        return JsonRouteResult({"error": "A record ID is required."}, HTTPStatus.BAD_REQUEST)
    project = str(deps.first(parse_qs(query), "project") or "default").strip()
    try:
        deleted = deps.assistant.delete_task(record_id, project=project) if kind == "tasks" else deps.assistant.delete_goal(record_id, project=project)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    if not deleted:
        return JsonRouteResult({"error": "Record not found."}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult({"deleted": True, "id": record_id, "kind": kind[:-1]})
