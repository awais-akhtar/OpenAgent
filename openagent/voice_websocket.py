"""WebSocket streaming for continuous voice capture and real-time STT."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from typing import Any, Callable
from uuid import uuid4

from .models import AuditEvent, utc_now
from .voice import VoiceRuntime

logger = logging.getLogger(__name__)


class VoiceWebSocketSession:
    """Manages a continuous voice WebSocket session with real-time STT."""

    def __init__(
        self,
        voice_runtime: VoiceRuntime,
        session_id: str,
        *,
        project: str = "default",
        owner: str = "",
        language: str = "en",
        stt_engine: str = "",
        allow_fallback_stt: bool = False,
    ):
        self.voice_runtime = voice_runtime
        self.session_id = session_id
        self.project = project
        self.owner = owner or "local"
        self.language = language
        self.stt_engine = stt_engine
        self.allow_fallback_stt = allow_fallback_stt
        self.stream_id = uuid4().hex
        self.chunks: list[bytes] = []
        self.total_bytes = 0
        self.last_chunk_at = utc_now()
        self.status = "open"
        self.transcription = ""
        self.is_final = False

    def add_audio_chunk(self, audio_data: bytes) -> dict[str, Any]:
        """Add an audio chunk to the stream buffer."""
        if self.status != "open":
            raise ValueError("Stream session is closed.")
        if not audio_data:
            return {"error": "audio_data is empty"}

        self.chunks.append(audio_data)
        self.total_bytes += len(audio_data)
        self.last_chunk_at = utc_now()

        self.voice_runtime.store.add_audit_event(
            AuditEvent(
                actor=self.owner,
                action="voice.websocket.chunk",
                resource=f"stream:{self.stream_id}",
                project=self.project,
                metadata={"chunk_size": len(audio_data), "total_bytes": self.total_bytes},
            )
        )

        return {
            "stream_id": self.stream_id,
            "chunk_received": True,
            "total_bytes": self.total_bytes,
            "chunk_count": len(self.chunks),
            "status": "buffering",
        }

    def finish_and_transcribe(self) -> dict[str, Any]:
        """Assemble all chunks and run STT to get final transcription."""
        if self.status == "closed":
            return {
                "stream_id": self.stream_id,
                "status": "closed",
                "transcription": self.transcription,
                "is_final": self.is_final,
            }

        if not self.chunks:
            self.status = "closed"
            return {
                "stream_id": self.stream_id,
                "status": "closed",
                "transcription": "",
                "is_final": False,
                "message": "No audio chunks were received.",
            }

        assembled = b"".join(self.chunks)
        result = {
            "stream_id": self.stream_id,
            "total_bytes": self.total_bytes,
            "chunk_count": len(self.chunks),
            "assembly_hash": hashlib.sha256(assembled).hexdigest()[:16],
        }

        try:
            transcription_result = self.voice_runtime.transcribe_audio(
                self.session_id,
                audio_data=assembled,
                filename=f"websocket-stream-{self.stream_id}.webm",
                speaker="user",
                transcript_hint="",
                language=self.language,
                stt_engine=self.stt_engine,
                allow_fallback_stt=self.allow_fallback_stt,
                owner=self.owner,
            )

            self.transcription = transcription_result.get("transcript", "")
            self.is_final = True
            self.status = "closed"

            result.update(
                {
                    "transcription": self.transcription,
                    "is_final": True,
                    "status": "transcribed",
                    "stt_engine": transcription_result.get("stt_engine", ""),
                    "confidence": transcription_result.get("confidence", "unknown"),
                    "transcript_status": transcription_result.get("transcript_status", ""),
                    "stt_message": transcription_result.get("stt_message", ""),
                    "audio_profile": transcription_result.get("audio_profile", {}),
                    "turn_id": transcription_result.get("turn", {}).get("id", ""),
                }
            )

            self.voice_runtime.store.add_audit_event(
                AuditEvent(
                    actor=self.owner,
                    action="voice.websocket.transcribed",
                    resource=f"stream:{self.stream_id}",
                    project=self.project,
                    metadata={
                        "total_bytes": self.total_bytes,
                        "stt_engine": result.get("stt_engine", ""),
                        "confidence": result.get("confidence", ""),
                        "transcript_length": len(self.transcription),
                    },
                )
            )

        except Exception as exc:
            self.status = "error"
            result.update(
                {
                    "status": "error",
                    "transcription": "",
                    "is_final": False,
                    "error": str(exc),
                }
            )
            self.voice_runtime.store.add_audit_event(
                AuditEvent(
                    actor=self.owner,
                    action="voice.websocket.error",
                    resource=f"stream:{self.stream_id}",
                    project=self.project,
                    metadata={"error": str(exc)},
                )
            )

        return result


class VoiceWebSocketStreamManager:
    """Manages active WebSocket streaming sessions."""

    def __init__(self, voice_runtime: VoiceRuntime):
        self.voice_runtime = voice_runtime
        self.sessions: dict[str, VoiceWebSocketSession] = {}

    def create_session(
        self,
        voice_session_id: str,
        *,
        project: str = "default",
        owner: str = "",
        language: str = "en",
        stt_engine: str = "",
        allow_fallback_stt: bool = False,
    ) -> dict[str, Any]:
        """Create a new WebSocket streaming session."""
        try:
            voice_session = self.voice_runtime.get_session(voice_session_id)
            if voice_session is None:
                raise ValueError(f"Voice session '{voice_session_id}' not found.")
        except Exception as exc:
            raise ValueError(f"Cannot create WebSocket session: {exc}") from exc

        stream_session = VoiceWebSocketSession(
            self.voice_runtime,
            voice_session_id,
            project=project or "default",
            owner=owner or "local",
            language=language or "en",
            stt_engine=stt_engine,
            allow_fallback_stt=allow_fallback_stt,
        )

        self.sessions[stream_session.stream_id] = stream_session

        self.voice_runtime.store.add_audit_event(
            AuditEvent(
                actor=stream_session.owner,
                action="voice.websocket.session.create",
                resource=f"stream:{stream_session.stream_id}",
                project=project,
                metadata={
                    "voice_session_id": voice_session_id,
                    "language": language,
                    "stt_engine": stt_engine,
                },
            )
        )

        return {
            "stream_id": stream_session.stream_id,
            "voice_session_id": voice_session_id,
            "project": project,
            "language": language,
            "stt_engine": stt_engine,
            "status": "open",
            "created_at": utc_now(),
            "stt_engines": self.voice_runtime.local_stt_engines() if hasattr(self.voice_runtime, 'local_stt_engines') else [],
        }

    def push_chunk(self, stream_id: str, audio_data: bytes) -> dict[str, Any]:
        """Push an audio chunk to a stream session."""
        stream_id = str(stream_id or "").strip()
        if not stream_id:
            raise ValueError("stream_id is required.")

        session = self.sessions.get(stream_id)
        if session is None:
            raise ValueError("Stream session not found.")

        return session.add_audio_chunk(audio_data)

    def finish(self, stream_id: str) -> dict[str, Any]:
        """Finish streaming, assemble chunks, and run STT."""
        stream_id = str(stream_id or "").strip()
        if not stream_id:
            raise ValueError("stream_id is required.")

        session = self.sessions.get(stream_id)
        if session is None:
            raise ValueError("Stream session not found.")

        result = session.finish_and_transcribe()

        if session.status == "closed" or session.status == "error":
            self.sessions.pop(stream_id, None)

        return result

    def get_session_status(self, stream_id: str) -> dict[str, Any]:
        """Get current status of a streaming session."""
        stream_id = str(stream_id or "").strip()
        if not stream_id:
            raise ValueError("stream_id is required.")

        session = self.sessions.get(stream_id)
        if session is None:
            raise ValueError("Stream session not found.")

        return {
            "stream_id": stream_id,
            "voice_session_id": session.session_id,
            "status": session.status,
            "chunk_count": len(session.chunks),
            "total_bytes": session.total_bytes,
            "transcription": session.transcription if session.is_final else "",
            "is_final": session.is_final,
            "last_chunk_at": session.last_chunk_at,
        }

    def cleanup_stale_sessions(self, timeout_seconds: int = 3600) -> int:
        """Remove sessions older than timeout_seconds."""
        from datetime import datetime, timedelta

        now_time = datetime.now() if not isinstance(utc_now(), str) else datetime.fromisoformat(utc_now())
        stale = []
        for stream_id, session in self.sessions.items():
            last_chunk_time = (
                session.last_chunk_at
                if isinstance(session.last_chunk_at, str)
                else (session.last_chunk_at.isoformat() if hasattr(session.last_chunk_at, "isoformat") else str(session.last_chunk_at))
            )
            try:
                if isinstance(last_chunk_time, str):
                    last_chunk_dt = datetime.fromisoformat(last_chunk_time)
                else:
                    last_chunk_dt = last_chunk_time
                age_seconds = (now_time - last_chunk_dt).total_seconds()
                if age_seconds > timeout_seconds:
                    stale.append(stream_id)
            except (ValueError, TypeError):
                # If we can't parse the timestamp, skip it
                continue

        for stream_id in stale:
            self.sessions.pop(stream_id, None)

        return len(stale)
