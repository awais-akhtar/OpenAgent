"""Voice WebSocket management routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import unquote

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "voice_websocket"
ROUTES = (
    ("POST", "/api/voice/stream/continuous"),
    ("GET", "/api/voice/stream/continuous/{session_id}"),
    ("POST", "/api/voice/stream/continuous/{session_id}/frame"),
    ("POST", "/api/voice/stream/continuous/{session_id}/finalize"),
    ("POST", "/api/voice/stream/continuous/{session_id}/cancel"),
    ("GET", "/api/voice/stream/continuous/sessions"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class VoiceWebSocketRouteDependencies:
    voice_websocket: Any
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, deps: VoiceWebSocketRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/voice/stream/continuous/sessions":
        sessions = deps.voice_websocket.list_sessions(active_only=True)
        return JsonRouteResult({"sessions": sessions})

    if path.startswith("/api/voice/stream/continuous/"):
        session_id = path.removeprefix("/api/voice/stream/continuous/").strip("/")
        if not session_id or "/" in session_id:
            return JsonRouteResult({"error": "session_id required"}, HTTPStatus.BAD_REQUEST)
        try:
            status = deps.voice_websocket.get_session_status(unquote(session_id))
            return JsonRouteResult({"session": status})
        except ValueError:
            return JsonRouteResult({"error": "Session not found"}, HTTPStatus.NOT_FOUND)

    return None


def handle_post(path: str, deps: VoiceWebSocketRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/voice/stream/continuous":
        result = deps.voice_websocket.create_session()
        return JsonRouteResult(result, HTTPStatus.CREATED)

    if path.startswith("/api/voice/stream/continuous/"):
        session_id = unquote(path.removeprefix("/api/voice/stream/continuous/").removesuffix("/frame").removesuffix("/finalize").removesuffix("/cancel")).strip("/")

        if path.endswith("/frame"):
            payload = deps.read_json()
            audio_data = payload.get("audio", "")
            encoding = str(payload.get("encoding", "base64")).strip() or "base64"
            try:
                result = deps.voice_websocket.process_audio_frame(session_id, audio_data, encoding=encoding)
                return JsonRouteResult(result)
            except (ValueError, RuntimeError) as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        if path.endswith("/finalize"):
            try:
                result = deps.voice_websocket.finalize_session(session_id)
                return JsonRouteResult(result)
            except ValueError:
                return JsonRouteResult({"error": "Session not found"}, HTTPStatus.NOT_FOUND)

        if path.endswith("/cancel"):
            try:
                result = deps.voice_websocket.cancel_session(session_id)
                return JsonRouteResult(result)
            except ValueError:
                return JsonRouteResult({"error": "Session not found"}, HTTPStatus.NOT_FOUND)

    return None
