"""Routes for G05-G06: Full voice conversation (listen, transcribe, respond, speak)."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
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
    ("GET", "/api/voice/models"),
    ("GET", "/api/voice/voices"),
    ("POST", "/api/voice/models/{id}"),
    ("POST", "/api/voice/voices/{id}"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class VoiceConversationRouteDependencies:
    manager: VoiceConversationManager
    read_json: Any


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def handle_get(path: str, query: str, deps: VoiceConversationRouteDependencies) -> JsonRouteResult | None:
    """Handle GET requests for voice conversation."""
    params = parse_qs(query)

    # List all conversations
    if path == "/api/voice/conversations":
        conversations = deps.manager.list_conversations()
        return JsonRouteResult({"conversations": conversations})

    # Get specific conversation
    if path.startswith("/api/voice/conversations/") and not path.count("/") > 4:
        conv_id = unquote(path.removeprefix("/api/voice/conversations/")).strip()
        conversation = deps.manager.get_conversation(conv_id)
        if not conversation:
            return JsonRouteResult({"error": f"Conversation {conv_id} not found"}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(conversation)

    # Get available transcription models
    if path == "/api/voice/models":
        models = deps.manager.get_transcription_models()
        return JsonRouteResult(models)

    # Get available synthesis voices
    if path == "/api/voice/voices":
        voices = deps.manager.get_synthesis_voices()
        return JsonRouteResult(voices)

    return None


def handle_post(path: str, deps: VoiceConversationRouteDependencies) -> JsonRouteResult | None:
    """Handle POST requests for voice conversation."""
    if not path.startswith("/api/voice/"):
        return None

    payload = deps.read_json()

    # Create new conversation
    if path == "/api/voice/conversations":
        conversation = deps.manager.create_conversation()
        return JsonRouteResult(conversation, HTTPStatus.CREATED)

    # Process audio input (transcribe speech to text)
    if path.startswith("/api/voice/conversations/") and path.endswith("/audio"):
        conv_id = unquote(path.removeprefix("/api/voice/conversations/").removesuffix("/audio")).strip()

        # Audio data should be base64 encoded in JSON
        audio_b64 = str(payload.get("audio", "")).strip()
        audio_format = str(payload.get("format", "wav")).strip()

        if not audio_b64:
            return JsonRouteResult({"error": "audio data required"}, HTTPStatus.BAD_REQUEST)

        try:
            import base64

            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return JsonRouteResult({"error": f"Invalid audio data: {e}"}, HTTPStatus.BAD_REQUEST)

        result = deps.manager.process_audio_input(conv_id, audio_bytes, audio_format)

        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)

        return JsonRouteResult(result)

    # Generate speech response (synthesize text to speech)
    if path.startswith("/api/voice/conversations/") and path.endswith("/respond"):
        conv_id = unquote(path.removeprefix("/api/voice/conversations/").removesuffix("/respond")).strip()

        response_text = str(payload.get("text", "")).strip()

        if not response_text:
            return JsonRouteResult({"error": "response text required"}, HTTPStatus.BAD_REQUEST)

        result = deps.manager.generate_speech_response(conv_id, response_text)

        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)

        # Return audio as base64
        import base64

        audio_b64 = base64.b64encode(result.get("audio_bytes", b"")).decode()

        return JsonRouteResult({
            "audio_base64": audio_b64,
            "duration_seconds": result.get("duration_seconds", 0),
            "engine": result.get("engine", "unknown"),
            "format": result.get("format", "wav"),
        })

    # Full voice conversation (listen + process + respond)
    if path.startswith("/api/voice/conversations/") and path.endswith("/full-conversation"):
        conv_id = unquote(path.removeprefix("/api/voice/conversations/").removesuffix("/full-conversation")).strip()

        # Input: audio bytes + backend response text
        audio_b64 = str(payload.get("audio", "")).strip()
        response_text = str(payload.get("response_text", "")).strip()

        if not audio_b64 or not response_text:
            return JsonRouteResult(
                {"error": "audio and response_text required"},
                HTTPStatus.BAD_REQUEST,
            )

        try:
            import base64

            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return JsonRouteResult({"error": f"Invalid audio data: {e}"}, HTTPStatus.BAD_REQUEST)

        result = deps.manager.full_voice_conversation(conv_id, audio_bytes, response_text)

        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)

        # Return response audio as base64
        import base64

        response_audio_b64 = base64.b64encode(result.get("response_audio_bytes", b"")).decode()

        return JsonRouteResult({
            "user_transcription": result.get("user_transcription", ""),
            "user_confidence": result.get("user_confidence", 0),
            "assistant_response": result.get("assistant_response", ""),
            "response_audio_base64": response_audio_b64,
            "response_duration": result.get("response_duration", 0),
        })

    # Close conversation
    if path.startswith("/api/voice/conversations/") and path.endswith("/close"):
        conv_id = unquote(path.removeprefix("/api/voice/conversations/").removesuffix("/close")).strip()
        success = deps.manager.close_conversation(conv_id)

        if not success:
            return JsonRouteResult({"error": "Conversation not found"}, HTTPStatus.NOT_FOUND)

        return JsonRouteResult({"status": "closed", "conversation_id": conv_id})

    # Set transcription model
    if path.startswith("/api/voice/models/"):
        model_id = unquote(path.removeprefix("/api/voice/models/")).strip()
        success = deps.manager.set_transcription_model(model_id)

        if not success:
            return JsonRouteResult({"error": f"Unknown model: {model_id}"}, HTTPStatus.BAD_REQUEST)

        return JsonRouteResult({"model": model_id, "status": "set"})

    # Set synthesis voice
    if path.startswith("/api/voice/voices/"):
        voice_id = unquote(path.removeprefix("/api/voice/voices/")).strip()
        success = deps.manager.set_synthesis_voice(voice_id)

        if not success:
            return JsonRouteResult({"error": f"Unknown voice: {voice_id}"}, HTTPStatus.BAD_REQUEST)

        return JsonRouteResult({"voice": voice_id, "status": "set"})

    return None
