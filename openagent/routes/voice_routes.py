"""Voice streaming and session HTTP routes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "voice"
ROUTES = (
    ("POST", "/api/voice/stream/continuous"),
    ("POST", "/api/voice/websocket/create"),
    ("POST", "/api/voice/websocket/chunk"),
    ("POST", "/api/voice/websocket/finish"),
    ("POST", "/api/voice/websocket/cancel"),
    ("GET", "/api/voice/websocket/status"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class VoiceRouteDependencies:
    voice: Any
    voice_websocket: Any
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_bool: Callable[[object, bool], bool]
    parse_int: Callable[[object, int], int]
    query_owner: Callable[[dict[str, list[str]]], str]
    payload_owner: Callable[[dict[str, object], str], str]
    read_json: Callable[[], dict[str, Any]]
    read_query_string: Callable[[], dict[str, list[str]]]


def handle_post(path: str, deps: VoiceRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/voice/"):
        return None

    payload = deps.read_json()
    voice = deps.voice
    voice_websocket = deps.voice_websocket

    try:
        if path == "/api/voice/stream/continuous":
            # Handle continuous audio streaming
            session_id = str(payload.get("session_id", "")).strip()
            if not session_id:
                return JsonRouteResult({"error": "session_id is required."}, HTTPStatus.BAD_REQUEST)

            # Audio data can be provided as base64
            audio_base64 = str(payload.get("audio_base64", "")).strip()
            if not audio_base64:
                return JsonRouteResult({"error": "audio_base64 is required."}, HTTPStatus.BAD_REQUEST)

            # Decode audio data
            try:
                import base64
                audio_data = base64.b64decode(audio_base64)
            except Exception as exc:
                return JsonRouteResult(
                    {"error": f"Invalid base64 audio data: {exc}"},
                    HTTPStatus.BAD_REQUEST
                )

            # Process the continuous audio stream
            project = str(payload.get("project", "default")).strip() or "default"
            owner = deps.payload_owner(payload, "")
            language = str(payload.get("language", "en")).strip() or "en"

            result = voice.process_continuous_stream(
                session_id,
                audio_data,
                project=project,
                owner=owner,
                language=language,
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"stream_result": result}, HTTPStatus.OK)

        elif path == "/api/voice/websocket/create":
            # Create a new WebSocket streaming session
            voice_session_id = str(payload.get("voice_session_id", "")).strip()
            if not voice_session_id:
                return JsonRouteResult({"error": "voice_session_id is required."}, HTTPStatus.BAD_REQUEST)

            project = str(payload.get("project", "default")).strip() or "default"
            owner = deps.payload_owner(payload, "")
            language = str(payload.get("language", "en")).strip() or "en"
            stt_engine = str(payload.get("stt_engine", "")).strip()
            allow_fallback_stt = deps.parse_bool(payload.get("allow_fallback_stt"), default=False)

            result = voice_websocket.create_session(
                voice_session_id,
                project=project,
                owner=owner,
                language=language,
                stt_engine=stt_engine,
                allow_fallback_stt=allow_fallback_stt,
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)

        elif path == "/api/voice/websocket/chunk":
            # Push audio chunk to WebSocket session
            stream_id = str(payload.get("stream_id", "")).strip()
            if not stream_id:
                return JsonRouteResult({"error": "stream_id is required."}, HTTPStatus.BAD_REQUEST)

            audio_base64 = str(payload.get("audio_base64", "")).strip()
            if not audio_base64:
                return JsonRouteResult({"error": "audio_base64 is required."}, HTTPStatus.BAD_REQUEST)

            try:
                import base64
                audio_data = base64.b64decode(audio_base64)
            except Exception as exc:
                return JsonRouteResult(
                    {"error": f"Invalid base64 audio data: {exc}"},
                    HTTPStatus.BAD_REQUEST
                )

            result = voice_websocket.push_chunk(stream_id, audio_data)
            return JsonRouteResult(result, HTTPStatus.OK)

        elif path == "/api/voice/websocket/cancel":
            # Interrupt a live stream and discard its buffered audio
            stream_id = str(payload.get("stream_id", "")).strip()
            if not stream_id:
                return JsonRouteResult({"error": "stream_id is required."}, HTTPStatus.BAD_REQUEST)

            result = voice_websocket.cancel(stream_id)
            return JsonRouteResult(result, HTTPStatus.OK)

        elif path == "/api/voice/websocket/finish":
            # Finish streaming and run STT
            stream_id = str(payload.get("stream_id", "")).strip()
            if not stream_id:
                return JsonRouteResult({"error": "stream_id is required."}, HTTPStatus.BAD_REQUEST)

            result = voice_websocket.finish(stream_id)
            return JsonRouteResult(result, HTTPStatus.OK)

    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # noqa: BLE001
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return None


def handle_get(path: str, deps: VoiceRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/voice/"):
        return None

    voice_websocket = deps.voice_websocket

    try:
        if path == "/api/voice/websocket/status":
            # Get status of a WebSocket streaming session
            query = deps.read_query_string()
            stream_id = deps.first(query, "stream_id") or ""
            if not stream_id:
                return JsonRouteResult({"error": "stream_id query param is required."}, HTTPStatus.BAD_REQUEST)

            result = voice_websocket.get_session_status(stream_id)
            return JsonRouteResult(result, HTTPStatus.OK)

    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # noqa: BLE001
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return None
