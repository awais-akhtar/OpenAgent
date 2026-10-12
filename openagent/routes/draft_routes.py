"""Editor draft route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "draft"
ROUTES = (
    ("GET", "/api/editor-drafts"),
    ("POST", "/api/editor-drafts"),
    ("GET", "/api/editor-drafts/{draft_id}"),
    ("PUT", "/api/editor-drafts/{draft_id}"),
    ("DELETE", "/api/editor-drafts/{draft_id}"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class DraftRouteDependencies:
    documents: Any
    read_json: Callable[[], dict[str, Any]]
    auth_scope_owner: Callable[[], str]
    document_write_owner: Callable[[dict[str, Any]], str]


def _draft_id(path: str) -> str | None:
    prefix = "/api/editor-drafts/"
    return unquote(path.removeprefix(prefix)).strip() if path.startswith(prefix) else None


def handle_get(path: str, query: str, deps: DraftRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/editor-drafts":
        params = parse_qs(query)
        project = params.get("project", [None])[0] or "default"
        doc_id = params.get("doc_id", [None])[0] or ""
        return JsonRouteResult({"drafts": deps.documents.list_drafts(
            project=project, doc_id=doc_id, owner=deps.auth_scope_owner(),
        )})
    draft_id = _draft_id(path)
    if draft_id is None:
        return None
    try:
        return JsonRouteResult({"draft": deps.documents.get_draft(draft_id, owner=deps.auth_scope_owner())})
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)


def handle_post(path: str, deps: DraftRouteDependencies) -> JsonRouteResult | None:
    if path != "/api/editor-drafts":
        return None
    payload = deps.read_json()
    doc_id = str(payload.get("doc_id", "")).strip()
    try:
        result = deps.documents.create_draft(
            doc_id,
            str(payload.get("content", "")),
            base_version=int(payload.get("base_version", 0) or 0),
            project=str(payload.get("project", "default")).strip() or "default",
            owner=deps.document_write_owner(payload),
            scope_owner=deps.auth_scope_owner(),
        )
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult(result, HTTPStatus.CREATED)


def handle_put(path: str, deps: DraftRouteDependencies) -> JsonRouteResult | None:
    draft_id = _draft_id(path)
    if draft_id is None:
        return None
    payload = deps.read_json()
    try:
        result = deps.documents.update_draft(
            draft_id,
            str(payload.get("content", "")),
            base_version=int(payload["base_version"]) if "base_version" in payload else None,
            owner=deps.auth_scope_owner(),
        )
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult({"draft": result})


def handle_delete(path: str, deps: DraftRouteDependencies) -> JsonRouteResult | None:
    draft_id = _draft_id(path)
    if draft_id is None:
        return None
    try:
        result = deps.documents.delete_draft(draft_id, owner=deps.auth_scope_owner())
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult(result)
