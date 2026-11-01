"""Routes for G05-G06: Full voice conversation."""

from __future__ import annotations
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from ..voice_conversation_manager import VoiceConversationManager
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "voice_conversation"
ROUTES = (
    ("POST", "/api/voice/conversations"),
    ("GET", "/api/voice/conversations"),
    ("GET", "/api/voice/conversations/{id}"),
    ("POST", "/api/voice/conversations/{id}/audio"),
    ("POST", "/api/voice/conversations/{id}/respond"),
    ("POST", "/api/voice/conversations/{id}/full-conversation"),
    ("POST", "/api/voice/conversations/{id}/close"),
    ("GET", "/api/voice/conversation-models"),
    ("GET", "/api/voice/conversation-voices"),
    ("POST", "/api/voice/conversation-models/{id}"),
    ("POST", "/api/voice/conversation-voices/{id}"),
)

def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)

@dataclass(frozen=True)
class VoiceConversationRouteDependencies:
    manager: VoiceConversationManager
    read_json: Any
    query_owner: Callable[[dict[str, list[str]]], str]
    auth_scope_owner: Callable[[dict[str, object]], str]


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""

def handle_get(path: str, query: str, deps: VoiceConversationRouteDependencies) -> JsonRouteResult | None:
    """Handle GET requests."""
    params = parse_qs(query)
    project = _first(params, "project") or "default"
    owner = deps.query_owner(params) or "local"

    if path == "/api/voice/conversations":
        return JsonRouteResult({"conversations": deps.manager.list_conversations(project=project, owner=owner)})

    if path == "/api/voice/conversation-models":
        return JsonRouteResult(deps.manager.get_transcription_models(project=project, owner=owner))

    if path == "/api/voice/conversation-voices":
        return JsonRouteResult(deps.manager.get_synthesis_voices(project=project, owner=owner))

    if path.startswith("/api/voice/conversations/"):
        conv_id = unquote(path.removeprefix("/api/voice/conversations/")).strip()
        conv = deps.manager.get_conversation(conv_id, project=project, owner=owner)
        if not conv:
            return JsonRouteResult({"error": f"Conversation {conv_id} not found"}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(conv)

    return None

def handle_post(path: str, deps: VoiceConversationRouteDependencies) -> JsonRouteResult | None:
    """Handle POST requests."""
    if not path.startswith("/api/voice/"):
        return None

    payload = deps.read_json()
    project = str(payload.get("project", "")).strip() or "default"
    owner = deps.auth_scope_owner(payload)

    if path == "/api/voice/conversations":
        return JsonRouteResult(deps.manager.create_conversation(project=project, owner=owner), HTTPStatus.CREATED)

    if path.startswith("/api/voice/conversation-models/"):
        model = unquote(path.removeprefix("/api/voice/conversation-models/")).strip()
        if not deps.manager.set_transcription_model(model, project=project, owner=owner):
            return JsonRouteResult({"error": f"Unknown transcription model: {model}"}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"model": model, "status": "selected"})

    if path.startswith("/api/voice/conversation-voices/"):
        voice = unquote(path.removeprefix("/api/voice/conversation-voices/")).strip()
        if not deps.manager.set_synthesis_voice(voice, project=project, owner=owner):
            return JsonRouteResult({"error": f"Unknown synthesis voice: {voice}"}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"voice": voice, "status": "selected"})

    if path.startswith("/api/voice/conversations/") and path.endswith("/audio"):
        conv_id = unquote(path.removeprefix("/api/voice/conversations/").removesuffix("/audio")).strip()
        audio_b64 = str(payload.get("audio", "")).strip()

        if not audio_b64:
            return JsonRouteResult({"error": "audio required"}, HTTPStatus.BAD_REQUEST)

        try:
            import base64
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return JsonRouteResult({"error": f"Invalid audio: {e}"}, HTTPStatus.BAD_REQUEST)

        result = deps.manager.process_audio_input(conv_id, audio_bytes, project=project, owner=owner)
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    if path.startswith("/api/voice/conversations/") and path.endswith("/respond"):
        conv_id = unquote(path.removeprefix("/api/voice/conversations/").removesuffix("/respond")).strip()
        response_text = str(payload.get("text", "")).strip()

        if not response_text:
            return JsonRouteResult({"error": "text required"}, HTTPStatus.BAD_REQUEST)

        result = deps.manager.generate_speech_response(conv_id, response_text, project=project, owner=owner)
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)

        import base64
        audio_b64 = base64.b64encode(result.get("audio_bytes", b"")).decode()
        return JsonRouteResult({"audio_base64": audio_b64, "duration_seconds": result.get("duration_seconds", 0)})

    if path.startswith("/api/voice/conversations/") and path.endswith("/full-conversation"):
        conv_id = unquote(path.removeprefix("/api/voice/conversations/").removesuffix("/full-conversation")).strip()
        audio_b64 = str(payload.get("audio", "")).strip()
        response_text = str(payload.get("response_text", "")).strip()

        if not audio_b64 or not response_text:
            return JsonRouteResult({"error": "audio and response_text required"}, HTTPStatus.BAD_REQUEST)

        try:
            import base64
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return JsonRouteResult({"error": f"Invalid audio: {e}"}, HTTPStatus.BAD_REQUEST)

        result = deps.manager.full_voice_conversation(conv_id, audio_bytes, response_text, project=project, owner=owner)
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)

        import base64
        response_b64 = base64.b64encode(result.get("response_audio_bytes", b"")).decode()
        return JsonRouteResult({
            "user_transcription": result.get("user_transcription", ""),
            "user_confidence": result.get("user_confidence", 0),
            "assistant_response": result.get("assistant_response", ""),
            "response_audio_base64": response_b64,
            "response_duration": result.get("response_duration", 0)
        })

    if path.startswith("/api/voice/conversations/") and path.endswith("/close"):
        conv_id = unquote(path.removeprefix("/api/voice/conversations/").removesuffix("/close")).strip()
        if not deps.manager.close_conversation(conv_id, project=project, owner=owner):
            return JsonRouteResult({"error": "Conversation not found"}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult({"status": "closed", "conversation_id": conv_id})

    return None
