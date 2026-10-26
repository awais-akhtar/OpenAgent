"""Chat, conversation, artifact, and streaming HTTP handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs

from ..chat import ChatService
from ..chat_artifacts import ChatArtifactService
from ..model_gateway import ProviderConfigurationError
from ..storage import OpenAgentStore
from . import JsonRouteResult, StreamRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "chat"
ROUTES = (
    ("GET", "/api/conversations"), ("GET", "/api/messages"),
    ("GET", "/api/conversation/messages"), ("GET", "/api/chat/search"),
    ("GET", "/api/chat/export"), ("GET", "/api/chat/replay"),
    ("POST", "/api/chat"), ("POST", "/api/chat/stream"),
    ("POST", "/api/chat/branch"), ("POST", "/api/chat/import"),
    ("POST", "/api/chat/artifact"), ("POST", "/api/chat/research-record"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ChatRouteDependencies:
    store: OpenAgentStore
    chat: ChatService
    artifacts: ChatArtifactService
    first: Any
    parse_bool: Any
    parse_int: Any
    read_json: Any
    cancelled: Any = None
    auth_scope_owner: Any = None


def handle_get(path: str, query: str, deps: ChatRouteDependencies) -> JsonRouteResult | None:
    params = parse_qs(query)
    if path == "/api/conversations":
        return JsonRouteResult({"conversations": deps.store.list_conversations(
            project=deps.first(params, "project"),
            limit=deps.parse_int(deps.first(params, "limit"), default=50),
        )})
    if path in {"/api/messages", "/api/conversation/messages"}:
        conversation_id = deps.first(params, "conversation_id")
        if not conversation_id:
            return JsonRouteResult({"error": "conversation_id is required."}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"messages": deps.store.list_chat_messages(conversation_id)})
    if path == "/api/chat/search":
        return JsonRouteResult({"results": deps.chat.search(
            deps.first(params, "q") or "",
            project=deps.first(params, "project") or "default",
            limit=deps.parse_int(deps.first(params, "limit"), default=50),
        )})
    if path in {"/api/chat/export", "/api/chat/replay"}:
        conversation_id = deps.first(params, "conversation_id")
        if not conversation_id:
            return JsonRouteResult({"error": "conversation_id is required."}, HTTPStatus.BAD_REQUEST)
        try:
            if path.endswith("/export"):
                value = deps.chat.export_conversation(
                    conversation_id,
                    write_file=deps.parse_bool(deps.first(params, "write_file"), default=True),
                )
                return JsonRouteResult({"export": value})
            return JsonRouteResult({"replay": deps.chat.replay(conversation_id)})
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None


def handle_post(path: str, deps: ChatRouteDependencies) -> JsonRouteResult | StreamRouteResult | None:
    if path not in {route_path for method, route_path in ROUTES if method == "POST"}:
        return None
    payload = deps.read_json()
    try:
        if path in {"/api/chat", "/api/chat/stream"}:
            content = str(payload.get("message", "")).strip()
            if not content:
                return JsonRouteResult({"error": "A message is required."}, HTTPStatus.BAD_REQUEST)
            arguments = {
                "conversation_id": str(payload.get("conversation_id") or "").strip() or None,
                "project": str(payload.get("project", "default")).strip() or "default",
                "provider": str(payload.get("provider", "local")).strip() or "local",
                "model": str(payload.get("model", "")).strip() or None,
                "allow_external": deps.parse_bool(payload.get("allow_external"), default=False),
                "attachments": payload.get("attachments") if isinstance(payload.get("attachments"), list) else [],
                "owner": deps.auth_scope_owner() if callable(deps.auth_scope_owner) else "",
            }
            if path.endswith("/stream"):
                return StreamRouteResult(deps.chat.stream(content, cancelled=deps.cancelled, **arguments))
            return JsonRouteResult(deps.chat.send(content, **arguments), HTTPStatus.CREATED)
        if path == "/api/chat/artifact":
            artifact = deps.artifacts.render(
                title=str(payload.get("title", "OpenAgent document")),
                content=str(payload.get("content", "")),
                file_format=str(payload.get("format", "pdf")),
                project=str(payload.get("project", "default")) or "default",
            )
            return JsonRouteResult({"artifact": artifact}, HTTPStatus.CREATED)
        if path == "/api/chat/research-record":
            result = deps.chat.record_research(
                str(payload.get("query", "")), str(payload.get("report", "")),
                conversation_id=str(payload.get("conversation_id", "")).strip() or None,
                project=str(payload.get("project", "default")).strip() or "default",
                sources=payload.get("sources") if isinstance(payload.get("sources"), list) else [],
                research_session_id=str(payload.get("research_session_id", "")).strip(),
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path == "/api/chat/branch":
            branch = deps.chat.branch(
                str(payload.get("conversation_id", "")).strip(),
                at_message_id=str(payload.get("at_message_id", "")).strip(),
                title=str(payload.get("title", "")).strip(),
            )
            return JsonRouteResult({"branch": branch}, HTTPStatus.CREATED)
        imported = deps.chat.import_conversation(
            payload.get("conversation_export") if isinstance(payload.get("conversation_export"), dict) else payload,
            project=str(payload.get("project", "default")).strip() or "default",
        )
        return JsonRouteResult({"imported": imported}, HTTPStatus.CREATED)
    except (ProviderConfigurationError, ValueError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
