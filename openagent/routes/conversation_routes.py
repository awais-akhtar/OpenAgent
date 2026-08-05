"""Conversation organization lifecycle routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..conversation_organization import ConversationOrganization
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "conversation-organization"
ROUTES = (
    ("GET", "/api/conversations/organized"), ("GET", "/api/conversations/folders"),
    ("GET", "/api/conversations/{id}/branches"), ("GET", "/api/conversations/{id}/compactions"),
    ("POST", "/api/conversations/bulk"), ("POST", "/api/conversations/folders"),
    ("POST", "/api/conversations/{id}/compact"), ("PATCH", "/api/conversations/{id}"),
    ("DELETE", "/api/conversations/{id}"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ConversationRouteDependencies:
    organization: ConversationOrganization
    first: Any
    parse_bool: Any
    parse_int: Any
    query_owner: Any
    payload_owner: Any
    read_json: Any


def handle_get(path: str, query: str, deps: ConversationRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/conversations"):
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    owner = deps.query_owner(params)
    try:
        if path == "/api/conversations/organized":
            return JsonRouteResult({"conversations": deps.organization.list(
                project=project, include_archived=deps.parse_bool(deps.first(params, "include_archived"), default=False),
                folder=deps.first(params, "folder") or "", sort=deps.first(params, "sort") or "pinned",
                query=deps.first(params, "q") or "", limit=deps.parse_int(deps.first(params, "limit"), default=100))})
        if path == "/api/conversations/folders":
            return JsonRouteResult({"folders": deps.organization.folders(project=project, owner=owner)})
        suffix = unquote(path.removeprefix("/api/conversations/")).strip("/")
        if suffix.endswith("/branches"):
            return JsonRouteResult(deps.organization.branch_graph(suffix.removesuffix("/branches").strip("/")))
        if suffix.endswith("/compactions"):
            return JsonRouteResult({"compactions": deps.organization.compactions(suffix.removesuffix("/compactions").strip("/"))})
    except KeyError:
        return JsonRouteResult({"error": "Conversation not found."}, HTTPStatus.NOT_FOUND)
    return None


def handle_post(path: str, deps: ConversationRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/conversations"):
        return None
    payload = deps.read_json()
    owner = deps.payload_owner(payload)
    try:
        if path == "/api/conversations/bulk":
            ids = payload.get("conversation_ids")
            return JsonRouteResult(deps.organization.bulk([str(item) for item in ids] if isinstance(ids, list) else [],
                action=str(payload.get("action", "")), folder=str(payload.get("folder", "")), owner=owner))
        if path == "/api/conversations/folders":
            return JsonRouteResult({"folder": deps.organization.create_folder(str(payload.get("name", "")),
                project=str(payload.get("project", "default")), owner=owner, color=str(payload.get("color", "")))}, HTTPStatus.CREATED)
        suffix = unquote(path.removeprefix("/api/conversations/")).strip("/")
        if suffix.endswith("/compact"):
            return JsonRouteResult({"compaction": deps.organization.compact(suffix.removesuffix("/compact").strip("/"),
                preserve=int(payload.get("preserve", 8)), mode=str(payload.get("mode", "manual")))}, HTTPStatus.CREATED)
    except KeyError:
        return JsonRouteResult({"error": "Conversation not found."}, HTTPStatus.NOT_FOUND)
    except (ValueError, TypeError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None


def handle_patch(path: str, deps: ConversationRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/conversations/"):
        return None
    payload = deps.read_json()
    try:
        return JsonRouteResult({"conversation": deps.organization.update(
            unquote(path.removeprefix("/api/conversations/")).strip("/"), patch=payload, owner=deps.payload_owner(payload))})
    except KeyError:
        return JsonRouteResult({"error": "Conversation not found."}, HTTPStatus.NOT_FOUND)
    except (ValueError, TypeError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def handle_delete(path: str, deps: ConversationRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/conversations/"):
        return None
    conversation_id = unquote(path.removeprefix("/api/conversations/")).strip("/")
    try:
        return JsonRouteResult(deps.organization.bulk([conversation_id], action="delete"))
    except KeyError:
        return JsonRouteResult({"error": "Conversation not found."}, HTTPStatus.NOT_FOUND)
