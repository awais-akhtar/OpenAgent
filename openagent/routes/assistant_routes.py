"""Personal assistant HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "assistant"
ROUTES = (
    ("GET", "/api/assistant/tasks"),
    ("GET", "/api/assistant/goals"),
    ("GET", "/api/assistant/daily-brief"),
    ("GET", "/api/assistant/coach"),
    ("GET", "/api/assistant/crm"),
    ("POST", "/api/assistant/profile"),
    ("POST", "/api/assistant/goals"),
    ("POST", "/api/assistant/tasks"),
    ("POST", "/api/assistant/tasks/complete"),
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
