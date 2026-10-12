"""Workspace item HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "workspace"
ROUTES = (
    ("GET", "/api/workspace/items"),
    ("POST", "/api/workspace/items"),
    ("POST", "/api/workspace/items/update"),
    ("POST", "/api/workspace/templates/instantiate"),
    ("GET", "/api/workspace/dashboard"),
    ("GET", "/api/workspace/canvas"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class WorkspaceRouteDependencies:
    workspace: Any
    first: Callable[..., str | None]
    parse_int: Callable[..., int]
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, query: str, deps: WorkspaceRouteDependencies) -> JsonRouteResult | None:
    if path not in {"/api/workspace/items", "/api/workspace/dashboard", "/api/workspace/canvas"}:
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    if path == "/api/workspace/dashboard":
        return JsonRouteResult(deps.workspace.dashboard(project=project))
    if path == "/api/workspace/canvas":
        return JsonRouteResult(deps.workspace.canvas_model(project=project))
    search = deps.first(params, "q") or ""
    limit = deps.parse_int(deps.first(params, "limit"), default=100)
    items = (
        deps.workspace.search(search, project=project, limit=limit)
        if search
        else deps.workspace.list_items(
            project=project,
            kind=deps.first(params, "kind") or "",
            parent_id=deps.first(params, "parent_id"),
            limit=limit,
        )
    )
    return JsonRouteResult({"items": items})


def handle_post(path: str, deps: WorkspaceRouteDependencies) -> JsonRouteResult | None:
    if path not in {"/api/workspace/items", "/api/workspace/items/update", "/api/workspace/templates/instantiate"}:
        return None
    payload = deps.read_json()
    try:
        if path == "/api/workspace/items":
            item = deps.workspace.create_item(
                kind=str(payload.get("kind", "note")).strip() or "note",
                title=str(payload.get("title", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                parent_id=str(payload.get("parent_id", "")).strip(),
                content=str(payload.get("content", "")),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"item": item}, HTTPStatus.CREATED)
        if path == "/api/workspace/items/update":
            item = deps.workspace.update_item(
                str(payload.get("item_id", "")).strip(),
                title=str(payload.get("title", "")).strip() if "title" in payload else None,
                content=str(payload.get("content", "")) if "content" in payload else None,
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
            )
            return JsonRouteResult({"item": item})
        item = deps.workspace.instantiate_template(
            str(payload.get("template_id", "")).strip(),
            title=str(payload.get("title", "")).strip(),
            project=str(payload.get("project", "default")).strip() or "default",
            variables=payload.get("variables") if isinstance(payload.get("variables"), dict) else {},
        )
        return JsonRouteResult({"item": item}, HTTPStatus.CREATED)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
