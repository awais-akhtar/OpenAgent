"""Core capability, context, event, and diagnostic HTTP handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "core"
ROUTES = (
    ("GET", "/api/capabilities"),
    ("GET", "/api/core/runtime"),
    ("GET", "/api/mission/features"),
    ("GET", "/api/context"),
    ("GET", "/api/events"),
    ("GET", "/api/chat/diagnose"),
    ("POST", "/api/events"),
    ("POST", "/api/capabilities/hard-audit"),
    ("POST", "/api/mission/run-check"),
)


def contract() -> dict[str, object]:
    result = route_contract(ROUTE_GROUP, ROUTES)
    result["handling"] = "concrete"
    return result


@dataclass(frozen=True)
class CoreRouteDependencies:
    capabilities: Callable[[], dict[str, Any]]
    runtime_plan: Callable[..., dict[str, Any]]
    mission_features: Callable[[str], dict[str, Any]]
    context_snapshot: Callable[[str | None], dict[str, Any]]
    search_events: Callable[..., list[dict[str, Any]]]
    diagnose_chat: Callable[[str], dict[str, Any]]
    ingest_event: Callable[..., dict[str, Any]]
    hard_audit: Callable[[str], dict[str, Any]]
    mission_run_check: Callable[[str], dict[str, Any]]
    read_json: Callable[[], dict[str, object]]


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def _integer(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def handle_get(path: str, query: str, deps: CoreRouteDependencies) -> JsonRouteResult | None:
    params = parse_qs(query)
    if path == "/api/capabilities":
        return JsonRouteResult(deps.capabilities())
    if path == "/api/core/runtime":
        target_ram = _first(params, "target_ram_gb")
        try:
            target_ram_gb = float(target_ram) if target_ram else None
        except (TypeError, ValueError):
            return JsonRouteResult({"error": "target_ram_gb must be a number."}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(deps.runtime_plan(
            task=_first(params, "task") or "chat",
            target_ram_gb=target_ram_gb,
            quantization_mode=_first(params, "quantization_mode") or "auto",
        ))
    if path == "/api/mission/features":
        return JsonRouteResult(deps.mission_features(_first(params, "project") or "default"))
    if path == "/api/context":
        return JsonRouteResult(deps.context_snapshot(_first(params, "project") or None))
    if path == "/api/events":
        return JsonRouteResult({"events": deps.search_events(
            _first(params, "q"),
            project=_first(params, "project") or None,
            limit=_integer(_first(params, "limit"), 50),
        )})
    if path == "/api/chat/diagnose":
        return JsonRouteResult(deps.diagnose_chat(_first(params, "project") or "default"))
    return None


def handle_post(path: str, deps: CoreRouteDependencies) -> JsonRouteResult | None:
    if path not in {route_path for method, route_path in ROUTES if method == "POST"}:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    if path == "/api/events":
        title = str(payload.get("title", "")).strip()
        content = str(payload.get("content", "")).strip()
        if not title or not content:
            return JsonRouteResult({"error": "Both title and content are required."}, HTTPStatus.BAD_REQUEST)
        event = deps.ingest_event(
            title=title,
            content=content,
            type=str(payload.get("type", "note")),
            source=str(payload.get("source", "manual")),
            project=project,
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )
        return JsonRouteResult({"event": event}, HTTPStatus.CREATED)
    if path == "/api/capabilities/hard-audit":
        audit_project = str(payload.get("project", "hard-audit")).strip() or "hard-audit"
        return JsonRouteResult(deps.hard_audit(audit_project), HTTPStatus.CREATED)
    return JsonRouteResult(deps.mission_run_check(project), HTTPStatus.CREATED)
