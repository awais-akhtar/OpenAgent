"""Digital twin HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "twin"
ROUTES = (
    ("GET", "/api/twin/status"),
    ("GET", "/api/twin/profile"),
    ("GET", "/api/twin/patterns"),
    ("GET", "/api/twin/suggestions"),
    ("GET", "/api/twin/goals"),
    ("POST", "/api/twin/profile"),
    ("POST", "/api/twin/interaction"),
    ("POST", "/api/twin/predict"),
    ("POST", "/api/twin/goals"),
    ("POST", "/api/twin/consolidate"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class TwinRouteDependencies:
    twin: Any
    first: Callable[..., str | None]
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, query: str, deps: TwinRouteDependencies) -> JsonRouteResult | None:
    if path not in {route for method, route in ROUTES if method == "GET"}:
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    if path == "/api/twin/status":
        return JsonRouteResult(deps.twin.status(project=project))
    if path == "/api/twin/profile":
        return JsonRouteResult(deps.twin.get_profile(project=project))
    if path == "/api/twin/patterns":
        return JsonRouteResult(deps.twin.analyse_patterns(project=project))
    if path == "/api/twin/suggestions":
        return JsonRouteResult({"suggestions": deps.twin.generate_proactive_suggestions(project=project)})
    return JsonRouteResult({"goals": deps.twin.list_goals(project=project)})


def handle_post(path: str, deps: TwinRouteDependencies) -> JsonRouteResult | None:
    if path not in {route for method, route in ROUTES if method == "POST"}:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    if path == "/api/twin/profile":
        updates = {key: value for key, value in payload.items() if key != "project"}
        return JsonRouteResult(deps.twin.update_profile(updates, project=project))
    if path == "/api/twin/interaction":
        result = deps.twin.record_interaction(
            str(payload.get("kind", "chat")), str(payload.get("content", "")),
            project=project,
            context=payload.get("context") if isinstance(payload.get("context"), dict) else None,
        )
        return JsonRouteResult(result, HTTPStatus.CREATED)
    if path == "/api/twin/predict":
        return JsonRouteResult(deps.twin.predict_intent(
            str(payload.get("context", "")).strip(), project=project,
        ))
    if path == "/api/twin/goals":
        try:
            goal = deps.twin.create_goal(
                str(payload.get("title", "")).strip(), project=project,
                description=str(payload.get("description", "")),
                target_date=str(payload.get("target_date", "")),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"goal": goal}, HTTPStatus.CREATED)
    return JsonRouteResult(deps.twin.consolidate_memory(project=project))
