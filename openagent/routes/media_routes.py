"""Media and voice route contracts and handlers."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from . import JsonRouteResult
from ._contracts import route_contract
from ..media import MediaAssetInUse

ROUTE_GROUP = "media"
ROUTES = (
    ("GET", "/api/media/status"),
    ("GET", "/api/media/assets"),
    ("DELETE", "/api/media/assets/{id}"),
    ("GET", "/api/voice/assets/{id}/content"),
    ("GET", "/api/media/analyses"),
    ("POST", "/api/media/ingest"),
    ("POST", "/api/media/analyze"),
    ("GET", "/api/voice/status"),
    ("GET", "/api/voice/personas"),
    ("GET", "/api/voice/sessions"),
    ("GET", "/api/voice/turns"),
    ("POST", "/api/voice/personas"),
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
    ("POST", "/api/voice/stream/start"),
    ("POST", "/api/voice/stream/chunk"),
    ("POST", "/api/voice/stream/finish"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class MediaRouteDependencies:
    media: Any
    voice: Any
    read_json: Callable[[], dict[str, Any]]
    handle_voice_upload: Callable[[], None]
    personal_remote: bool = False
    auth_scope_owner: Callable[[], str] = lambda: ""


@dataclass(frozen=True)
class ResponseAlreadySent:
    """The runtime upload callback has written the HTTP response."""


@dataclass(frozen=True)
class MediaContentRouteResult:
    """Verified stored audio bytes and a safe content type for playback."""

    body: bytes
    content_type: str


def _first(params: dict[str, list[str]], key: str) -> str | None:
    return params.get(key, [None])[0] or None


def _parse_int(value: str | None, *, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _parse_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _project(payload: dict[str, Any]) -> str:
    return str(payload.get("project", "default")).strip() or "default"


def handle_get(path: str, query: str, deps: MediaRouteDependencies) -> JsonRouteResult | MediaContentRouteResult | None:
    audio_content_route = path.startswith("/api/voice/assets/") and path.endswith("/content")
    if path not in {route for method, route in ROUTES if method == "GET"} and not audio_content_route:
        return None
    params = parse_qs(query)
    project = _first(params, "project") or "default"
    if audio_content_route:
        asset_id = unquote(path.removeprefix("/api/voice/assets/").removesuffix("/content").strip("/"))
        if not asset_id or "/" in asset_id:
            return JsonRouteResult({"error": "Audio asset not found."}, HTTPStatus.NOT_FOUND)
        result = deps.media.read_audio_asset(
            asset_id, project=project, owner=deps.auth_scope_owner(),
        )
        if result is None:
            return JsonRouteResult({"error": "Audio asset not found."}, HTTPStatus.NOT_FOUND)
        asset, body = result
        mime = str(asset.get("mime_type", "")).split(";", 1)[0].strip().lower()
        if not mime.startswith("audio/") or any(char in mime for char in "\r\n"):
            mime = "application/octet-stream"
        return MediaContentRouteResult(body, mime)
    if path == "/api/media/status":
        return JsonRouteResult(deps.media.status(project=project, owner=deps.auth_scope_owner()))
    if path == "/api/media/assets":
        return JsonRouteResult({"assets": deps.media.assets(
            project=project, media_type=_first(params, "media_type") or "",
            limit=_parse_int(_first(params, "limit"), default=50),
            owner=deps.auth_scope_owner(),
        )})
    if path == "/api/media/analyses":
        return JsonRouteResult({"analyses": deps.media.analyses(
            project=project, asset_id=_first(params, "asset_id") or "",
            limit=_parse_int(_first(params, "limit"), default=50),
            owner=deps.auth_scope_owner(),
        )})
    if path == "/api/voice/status":
        return JsonRouteResult(deps.voice.status(project=project))
    if path == "/api/voice/personas":
        return JsonRouteResult({"personas": deps.voice.personas(project=project)})
    if path == "/api/voice/sessions":
        return JsonRouteResult({"sessions": deps.voice.sessions(
            project=project, status=_first(params, "status") or "",
            limit=_parse_int(_first(params, "limit"), default=50),
        )})
    session_id = _first(params, "session_id") or ""
    if not session_id:
        return JsonRouteResult({"error": "session_id is required."}, HTTPStatus.BAD_REQUEST)
    return JsonRouteResult({"turns": deps.voice.turns(session_id)})


def handle_post(path: str, deps: MediaRouteDependencies) -> JsonRouteResult | ResponseAlreadySent | None:
    if path not in {route for method, route in ROUTES if method == "POST"}:
        return None


    if path in {"/api/voice/transcribe-upload", "/api/voice/upload"}:
        deps.handle_voice_upload()
        return ResponseAlreadySent()
    payload = deps.read_json()
    project = _project(payload)
    if path == "/api/media/ingest":
        owner = deps.auth_scope_owner()
        try:
            if payload.get("path"):
                if deps.personal_remote:
                    return JsonRouteResult(
                        {"error": "server-path media ingestion is disabled in personal_remote; upload the file contents instead."},
                        HTTPStatus.FORBIDDEN,
                    )
                asset = deps.media.ingest_file(
                    str(payload.get("path", "")).strip(), project=project,
                    title=str(payload.get("title", "")).strip(),
                    source=str(payload.get("source", "file")).strip() or "file",
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    copy_to_store=_parse_bool(payload.get("copy_to_store"), default=True),
                    owner=owner,
                )
            else:
                raw = str(payload.get("data_base64", "")).strip()
                if not raw:
                    return JsonRouteResult({"error": "path or data_base64 is required."}, HTTPStatus.BAD_REQUEST)
                asset = deps.media.ingest_bytes(
                    base64.b64decode(raw.encode("utf-8")),
                    filename=str(payload.get("filename", "upload.bin")).strip() or "upload.bin",
                    project=project, title=str(payload.get("title", "")).strip(),
                    source=str(payload.get("source", "upload")).strip() or "upload",
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    copy_to_store=_parse_bool(payload.get("copy_to_store"), default=True),
                    owner=owner,
                )
        except (ValueError, OSError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"asset": asset}, HTTPStatus.CREATED)
    if path == "/api/media/analyze":
        try:
            analysis = deps.media.analyze_asset(
                str(payload.get("asset_id", "")).strip(),
                project=project,
                owner=deps.auth_scope_owner(),
            )
        except KeyError:
            return JsonRouteResult({"error": "Media asset not found."}, HTTPStatus.NOT_FOUND)
        except (ValueError, OSError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"analysis": analysis}, HTTPStatus.CREATED)
    if path == "/api/voice/personas":
        persona = deps.voice.upsert_persona(
            name=str(payload.get("name", "")).strip(), project=project,
            language=str(payload.get("language", "en")).strip() or "en",
            style=str(payload.get("style", "clear")).strip() or "clear",
            voiceprint_asset_id=str(payload.get("voiceprint_asset_id", "")).strip(),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )
        return JsonRouteResult({"persona": persona}, HTTPStatus.CREATED)
    if path == "/api/voice/sessions":
        try:
            session = deps.voice.create_session(
                project=project, mode=str(payload.get("mode", "push_to_talk")).strip() or "push_to_talk",
                persona_id=str(payload.get("persona_id", "")).strip(),
                language=str(payload.get("language", "en")).strip() or "en",
                wake_word=str(payload.get("wake_word", "")).strip(),
                always_listening=_parse_bool(payload.get("always_listening"), default=False),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"session": session}, HTTPStatus.CREATED)
    if path == "/api/voice/vosk/setup":
        try:
            result = deps.voice.setup_vosk_model(
                allow_network=_parse_bool(payload.get("allow_network"), default=False), project=project,
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc), "status": deps.voice.status(project=project)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result, HTTPStatus.CREATED)
    if path == "/api/voice/clone":
        try:
            audio_data = base64.b64decode(str(payload.get("audio_base64", "")).encode("ascii"), validate=True)
            clone = deps.voice.clone_voice_from_audio(
                name=str(payload.get("name", "")).strip(), audio_data=audio_data,
                filename=str(payload.get("filename", "voiceprint.wav")).strip() or "voiceprint.wav",
                project=project, owner=deps.auth_scope_owner(),
                language=str(payload.get("language", "en")).strip() or "en",
                style=str(payload.get("style", "cloned-local")).strip() or "cloned-local",
                consent=str(payload.get("consent", "")).strip(),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
        except (ValueError, binascii.Error) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"clone": clone}, HTTPStatus.CREATED)
    if path == "/api/voice/sessions/close":
        try:
            session = deps.voice.close_session(str(payload.get("session_id", "")).strip())
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"session": session})
    if path == "/api/voice/transcript":
        try:
            turn = deps.voice.add_transcript(
                str(payload.get("session_id", "")).strip(), text=str(payload.get("text", "")).strip(),
                speaker=str(payload.get("speaker", "user")).strip() or "user",
                role=str(payload.get("role", "user")).strip() or "user",
                audio_asset_id=str(payload.get("audio_asset_id", "")).strip(),
                is_final=_parse_bool(payload.get("is_final"), default=True),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"turn": turn}, HTTPStatus.CREATED)
    if path == "/api/voice/stt":
        try:
            audio_data = base64.b64decode(str(payload.get("audio_base64", "")).encode("ascii"), validate=True)
            transcription = deps.voice.transcribe_audio(
                str(payload.get("session_id", "")).strip(), audio_data=audio_data,
                filename=str(payload.get("filename", "speech.wav")).strip() or "speech.wav",
                speaker=str(payload.get("speaker", "user")).strip() or "user",
                transcript_hint=str(payload.get("transcript_hint", "")).strip(),
                language=str(payload.get("language", "en")).strip() or "en",
                stt_engine=str(payload.get("stt_engine", "")).strip(),
                allow_fallback_stt=_parse_bool(payload.get("allow_fallback_stt"), default=False),
                owner=deps.auth_scope_owner(),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
        except (ValueError, binascii.Error) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"transcription": transcription}, HTTPStatus.CREATED)
    if path == "/api/voice/tts":
        try:
            script = deps.voice.synthesize_text(
                text=str(payload.get("text", "")).strip(),
                session_id=str(payload.get("session_id", "")).strip(), project=project,
                owner=deps.auth_scope_owner(),
                persona_id=str(payload.get("persona_id", "")).strip(),
                voice=str(payload.get("voice", "local-script")).strip() or "local-script",
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"script": script}, HTTPStatus.CREATED)
    if path == "/api/voice/translate":
        try:
            translation = deps.voice.translate_text(
                text=str(payload.get("text", "")).strip(), project=project,
                source_language=str(payload.get("source_language", "auto")).strip() or "auto",
                target_language=str(payload.get("target_language", "en")).strip() or "en",
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"translation": translation}, HTTPStatus.CREATED)
    if path == "/api/voice/stream/start":
        try:
            result = deps.voice.start_stream_session(
                project=project,
                owner=deps.auth_scope_owner(),
                voice_session_id=str(payload.get("voice_session_id", "")).strip(),
                audio_format=str(payload.get("audio_format", "webm")).strip() or "webm",
                language=str(payload.get("language", "en")).strip() or "en",
                stt_engine=str(payload.get("stt_engine", "")).strip(),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result, HTTPStatus.CREATED)
    if path == "/api/voice/stream/chunk":
        try:
            raw = str(payload.get("audio_base64", "")).strip()
            if not raw:
                return JsonRouteResult({"error": "audio_base64 is required."}, HTTPStatus.BAD_REQUEST)
            audio_data = base64.b64decode(raw.encode("ascii"), validate=True)
            result = deps.voice.push_stream_chunk(
                str(payload.get("stream_id", "")).strip(),
                audio_data=audio_data,
                sequence=int(payload.get("sequence", 0)),
                owner=deps.auth_scope_owner(),
            )
        except (ValueError, binascii.Error) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)
    if path == "/api/voice/stream/finish":
        try:
            result = deps.voice.finish_stream_session(
                str(payload.get("stream_id", "")).strip(),
                owner=deps.auth_scope_owner(),
                speaker=str(payload.get("speaker", "user")).strip() or "user",
                transcript_hint=str(payload.get("transcript_hint", "")).strip(),
                allow_fallback_stt=_parse_bool(payload.get("allow_fallback_stt"), default=False),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)
    return None


def handle_delete(path: str, query: str, deps: MediaRouteDependencies) -> JsonRouteResult | None:
    prefix = "/api/media/assets/"
    if not path.startswith(prefix):
        return None
    encoded_id = path.removeprefix(prefix).strip("/")
    asset_id = unquote(encoded_id)
    if not asset_id or "/" in asset_id:
        return JsonRouteResult({"error": "Media asset not found."}, HTTPStatus.NOT_FOUND)
    params = parse_qs(query, keep_blank_values=True)
    raw_project = params.get("project", [None])[0]
    project = str(raw_project or "").strip()
    if not project:
        return JsonRouteResult({"error": "project is required."}, HTTPStatus.BAD_REQUEST)
    try:
        result = deps.media.delete_asset(asset_id, project=project, owner=deps.auth_scope_owner())
    except MediaAssetInUse as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.CONFLICT)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    if result is None:
        return JsonRouteResult({"error": "Media asset not found."}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult({"deleted": True, **result})
