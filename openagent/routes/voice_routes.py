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
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class VoiceRouteDependencies:
    voice: Any
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_bool: Callable[[object, bool], bool]
    parse_int: Callable[[object, int], int]
    query_owner: Callable[[dict[str, list[str]]], str]
    payload_owner: Callable[[dict[str, object], str], str]
    read_json: Callable[[], dict[str, Any]]


def handle_post(path: str, deps: VoiceRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/voice/"):
        return None

    payload = deps.read_json()
    voice = deps.voice

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

    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # noqa: BLE001
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return None
