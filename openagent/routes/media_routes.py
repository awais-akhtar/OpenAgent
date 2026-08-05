"""Media and voice-adjacent route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "media"
ROUTES = (
    ("GET", "/api/media/status"),
    ("GET", "/api/media/assets"),
    ("GET", "/api/media/analyses"),
    ("POST", "/api/media/ingest"),
    ("POST", "/api/media/analyze"),
    ("GET", "/api/voice/status"),
    ("GET", "/api/voice/personas"),
    ("GET", "/api/voice/sessions"),
    ("GET", "/api/voice/turns"),
    ("POST", "/api/voice/sessions"),
    ("POST", "/api/voice/vosk/setup"),
    ("POST", "/api/voice/clone"),
    ("POST", "/api/voice/sessions/close"),
    ("POST", "/api/voice/transcript"),
    ("POST", "/api/voice/transcribe-upload"),
    ("POST", "/api/voice/upload"),
    ("POST", "/api/voice/stt"),
    ("POST", "/api/voice/tts"),
    ("POST", "/api/voice/translate"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
