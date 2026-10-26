"""Voice WebSocket transport for continuous audio streaming."""

from __future__ import annotations

import base64
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class VoiceWebSocketSession:
    """Manages continuous WebSocket session for voice streaming."""

    def __init__(self, session_id: str = "", stt_provider: Any = None) -> None:
        self.session_id = session_id or f"voice-{uuid4().hex[:8]}"
        self.stt_provider = stt_provider
        self.audio_buffer: bytearray = bytearray()
        self.frame_count = 0
        self.is_active = False
        self.created_at = ""
        self.last_frame_at = ""
        self.status = "idle"
        self.transcript = ""
        self.confidence = 0.0
        self.error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "frame_count": self.frame_count,
            "transcript": self.transcript,
            "confidence": self.confidence,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_frame_at": self.last_frame_at,
            "error": self.error,
        }


class VoiceWebSocketStreamManager:
    """Manages continuous WebSocket connections for voice/audio streaming."""

    def __init__(self, stt_provider: Any = None) -> None:
        self.stt_provider = stt_provider
        self.sessions: dict[str, VoiceWebSocketSession] = {}
        self.max_sessions = 100

    def create_session(self) -> dict[str, Any]:
        """Create new WebSocket session for voice streaming."""
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError(f"Maximum concurrent sessions ({self.max_sessions}) reached.")

        session = VoiceWebSocketSession(stt_provider=self.stt_provider)
        session.status = "created"
        session.is_active = True

        from .models import utc_now
        session.created_at = utc_now()

        self.sessions[session.session_id] = session

        return {
            "session_id": session.session_id,
            "status": "created",
            "message": "WebSocket session ready for audio frames",
        }

    def get_session(self, session_id: str) -> VoiceWebSocketSession | None:
        """Get active session by ID."""
        return self.sessions.get(session_id)

    def process_audio_frame(
        self,
        session_id: str,
        audio_data: bytes,
        encoding: str = "base64",
    ) -> dict[str, Any]:
        """Process audio frame from WebSocket stream."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found.")

        if not session.is_active:
            raise RuntimeError(f"Session {session_id} is not active.")

        from .models import utc_now
        session.last_frame_at = utc_now()
        session.frame_count += 1
        session.status = "receiving"

        # Decode audio data
        try:
            if encoding == "base64":
                frame_bytes = base64.b64decode(audio_data)
            else:
                frame_bytes = audio_data if isinstance(audio_data, bytes) else audio_data.encode()

            session.audio_buffer.extend(frame_bytes)
        except Exception as exc:
            session.error = f"Failed to decode audio: {exc}"
            return {
                "session_id": session_id,
                "status": "error",
                "error": session.error,
                "frame": session.frame_count,
            }

        return {
            "session_id": session_id,
            "status": "receiving",
            "frame": session.frame_count,
            "buffered_bytes": len(session.audio_buffer),
        }

    def finalize_session(self, session_id: str) -> dict[str, Any]:
        """Finalize and close WebSocket session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found.")

        session.is_active = False
        session.status = "completed"

        return {
            "session_id": session_id,
            "status": "completed",
            "frames_processed": session.frame_count,
            "transcript": session.transcript,
        }

    def cancel_session(self, session_id: str) -> dict[str, Any]:
        """Cancel active session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found.")

        session.is_active = False
        session.status = "cancelled"

        return {
            "session_id": session_id,
            "status": "cancelled",
            "frames_processed": session.frame_count,
        }

    def list_sessions(self, active_only: bool = True) -> list[dict[str, Any]]:
        """List all sessions."""
        sessions = []
        for session in self.sessions.values():
            if active_only and not session.is_active:
                continue
            sessions.append(session.to_dict())
        return sessions

    def get_session_status(self, session_id: str) -> dict[str, Any]:
        """Get detailed session status."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found.")

        return session.to_dict()
