"""Workspace item HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from . import JsonRouteResult
from ._contracts import route_contract
from ..workspace import WorkspaceItemNotFoundError

ROUTE_GROUP = "workspace"
ROUTES = (
    ("GET", "/api/workspace/items"),
    ("POST", "/api/workspace/items"),
    ("POST", "/api/workspace/items/update"),
    ("POST", "/api/workspace/items/{item_id}/archive"),
    ("POST", "/api/workspace/items/{item_id}/restore"),
    ("DELETE", "/api/workspace/items/{item_id}"),
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
    include_archived_raw = (deps.first(params, "include_archived") or "").lower()
    include_archived = include_archived_raw in {"1", "true", "yes"}
    items = (
        deps.workspace.search(search, project=project, limit=limit)
        if search
        else deps.workspace.list_items(
            project=project,
            kind=deps.first(params, "kind") or "",
            parent_id=deps.first(params, "parent_id"),
            limit=limit,
            include_archived=include_archived,
        )
    )
    return JsonRouteResult({"items": items})


def handle_post(path: str, deps: WorkspaceRouteDependencies) -> JsonRouteResult | None:
    # archive / restore lifecycle
    for action in ("archive", "restore"):
        prefix = f"/api/workspace/items/"
        suffix = f"/{action}"
        if path.startswith(prefix) and path.endswith(suffix):
            item_id = unquote(path.removeprefix(prefix).removesuffix(suffix)).strip()
            if not item_id or "/" in item_id:
                return JsonRouteResult({"error": "item_id is required."}, HTTPStatus.BAD_REQUEST)
            payload = deps.read_json()
            project = str(payload.get("project", "default")).strip() or "default"
            try:
                if action == "archive":
                    return JsonRouteResult(deps.workspace.archive_item(item_id, project=project))
                return JsonRouteResult(deps.workspace.restore_item(item_id, project=project))
            except KeyError:
                return JsonRouteResult({"error": "Workspace item not found."}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
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
                project=str(payload.get("project", "default")).strip() or "default",
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
    except WorkspaceItemNotFoundError:
        return JsonRouteResult({"error": "Workspace item not found."}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def handle_delete(path: str, query: str, deps: WorkspaceRouteDependencies) -> JsonRouteResult | None:
    prefix = "/api/workspace/items/"
    if not path.startswith(prefix):
        return None
    item_id = unquote(path.removeprefix(prefix)).strip()
    if not item_id or "/" in item_id:
        return None
    project = deps.first(parse_qs(query), "project") or "default"
    try:
        result = deps.workspace.delete_item(item_id, project=project)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.CONFLICT)
    if not result.get("deleted"):
        return JsonRouteResult({"error": "Workspace item not found."}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult(result)
