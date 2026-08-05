"""Agent bridge route contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs

from ..agent_bridge import AgentBridge
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "agent_bridge"
ROUTES = (
    ("GET", "/api/agent-bridge/capabilities"),
    ("GET", "/api/agent-bridge/tasks"),
    ("POST", "/api/agent-bridge/tasks"),
    ("GET", "/api/agent-bridge/memory"),
    ("POST", "/api/agent-bridge/memory"),
    ("GET", "/api/agent-bridge/calendar/events"),
    ("POST", "/api/agent-bridge/calendar/events"),
    ("GET", "/api/agent-bridge/documents"),
    ("POST", "/api/agent-bridge/documents"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class AgentBridgeRouteDependencies:
    bridge: AgentBridge
    first: Any
    parse_int: Any
    query_owner: Any
    payload_owner: Any
    read_json: Any


def handle_get(path: str, query: str, deps: AgentBridgeRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/agent-bridge"):
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    owner = deps.query_owner(params)
    if path == "/api/agent-bridge/capabilities":
        return JsonRouteResult(deps.bridge.capabilities())
    if path == "/api/agent-bridge/tasks":
        return JsonRouteResult(
            {
                "tasks": deps.bridge.tasks(
                    project=project,
                    owner=owner,
                    status=deps.first(params, "status") or "",
                    limit=deps.parse_int(deps.first(params, "limit"), default=100),
                )
            }
        )
    if path == "/api/agent-bridge/memory":
        return JsonRouteResult(
            {
                "events": deps.bridge.memory(
                    project=project,
                    query=deps.first(params, "q") or "",
                    limit=deps.parse_int(deps.first(params, "limit"), default=50),
                )
            }
        )
    if path == "/api/agent-bridge/calendar/events":
        return JsonRouteResult(
            {
                "events": deps.bridge.calendar_events(
                    project=project,
                    calendar_id=deps.first(params, "calendar_id") or "",
                    limit=deps.parse_int(deps.first(params, "limit"), default=100),
                )
            }
        )
    if path == "/api/agent-bridge/documents":
        return JsonRouteResult(
            deps.bridge.documents_list(
                project=project,
                owner=owner,
                query=deps.first(params, "q") or "",
                limit=deps.parse_int(deps.first(params, "limit"), default=50),
            )
        )
    return None


def handle_post(path: str, deps: AgentBridgeRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/agent-bridge"):
        return None
    payload = deps.read_json()
    owner = deps.payload_owner(payload)
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/agent-bridge/tasks":
            task = deps.bridge.create_task(
                goal=str(payload.get("goal", "")).strip(),
                project=project,
                owner=owner,
                agent=str(payload.get("agent", "assistant")).strip() or "assistant",
                payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
            )
            return JsonRouteResult({"task": task}, HTTPStatus.CREATED)
        if path == "/api/agent-bridge/memory":
            event = deps.bridge.write_memory(
                title=str(payload.get("title", "")).strip(),
                content=str(payload.get("content", "")).strip(),
                project=project,
                memory_type=str(payload.get("type", "agent_memory")).strip() or "agent_memory",
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"event": event}, HTTPStatus.CREATED)
        if path == "/api/agent-bridge/calendar/events":
            event = deps.bridge.create_calendar_event(
                title=str(payload.get("title", "")).strip(),
                starts_at=str(payload.get("starts_at", "")).strip(),
                ends_at=str(payload.get("ends_at", "")).strip(),
                project=project,
                calendar_id=str(payload.get("calendar_id", "")).strip(),
                description=str(payload.get("description", "")),
            )
            return JsonRouteResult({"event": event}, HTTPStatus.CREATED)
        if path == "/api/agent-bridge/documents":
            document = deps.bridge.create_document(
                title=str(payload.get("title", "")).strip(),
                content=str(payload.get("content", "")),
                project=project,
                owner=owner,
                tags=[str(item) for item in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else [],
            )
            return JsonRouteResult({"document": document}, HTTPStatus.CREATED)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None
