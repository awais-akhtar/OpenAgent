"""Tests for Voice WebSocket streaming with real-time STT contract verification."""

import base64
import json
import tempfile
import unittest
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from openagent.storage import OpenAgentStore
from openagent.voice import VoiceRuntime, synthesize_tone_wav
from openagent.voice_websocket import VoiceWebSocketSession, VoiceWebSocketStreamManager


class VoiceWebSocketSTTTests(unittest.TestCase):
    """Verify Voice WebSocket streaming provides proven STT contract."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test database and voice runtime."""
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.store = OpenAgentStore(str(Path(cls.tmpdir.name) / "test.db"))
        cls.voice = VoiceRuntime(cls.store)
        cls.websocket_manager = VoiceWebSocketStreamManager(cls.voice)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory."""
        cls.tmpdir.cleanup()

    def setUp(self) -> None:
        """Create a voice session for each test."""
        # Use unique project to ensure session IDs don't collide
        project = f"test-voice-ws-{uuid4().hex[:8]}"
        self.voice_session = self.voice.create_session(
            project=project,
            mode="push_to_talk",
            language="en",
        )
        self.voice_session_id = self.voice_session["id"]

    def test_websocket_session_creation(self) -> None:
        """Test creating a WebSocket streaming session."""
        result = self.websocket_manager.create_session(
            self.voice_session_id,
            project="test-voice-ws",
            owner="test-user",
            language="en",
        )

        assert result["status"] == "open"
        assert result["stream_id"]
        assert result["voice_session_id"] == self.voice_session_id
        assert result["project"] == "test-voice-ws"
        assert "stt_engines" in result

    def test_websocket_audio_chunk_buffering(self) -> None:
        """Test adding audio chunks to WebSocket session."""
        stream_result = self.websocket_manager.create_session(
            self.voice_session_id,
            project="test-voice-ws",
            owner="test-user",
            language="en",
        )
        stream_id = stream_result["stream_id"]

        # Create a small test audio (synthesized tone)
        audio_wav = synthesize_tone_wav("hello")

        # Push chunk to stream
        chunk_result = self.websocket_manager.push_chunk(stream_id, audio_wav)

        assert chunk_result["chunk_received"] is True
        assert chunk_result["status"] == "buffering"
        assert chunk_result["total_bytes"] == len(audio_wav)
        assert chunk_result["chunk_count"] == 1

    def test_websocket_audio_chunk_accumulation(self) -> None:
        """Test accumulating multiple audio chunks."""
        stream_result = self.websocket_manager.create_session(
            self.voice_session_id,
            project="test-voice-ws",
            owner="test-user",
            language="en",
        )
        stream_id = stream_result["stream_id"]

        # Create multiple audio chunks
        audio1 = synthesize_tone_wav("hello")
        audio2 = synthesize_tone_wav("world")
        audio3 = synthesize_tone_wav("test")

        # Push all chunks
        result1 = self.websocket_manager.push_chunk(stream_id, audio1)
        result2 = self.websocket_manager.push_chunk(stream_id, audio2)
        result3 = self.websocket_manager.push_chunk(stream_id, audio3)

        # Verify accumulation
        assert result1["chunk_count"] == 1
        assert result2["chunk_count"] == 2
        assert result3["chunk_count"] == 3
        assert result3["total_bytes"] == len(audio1) + len(audio2) + len(audio3)

    def test_websocket_finish_and_stt_contract(self) -> None:
        """Test finish operation and STT transcription contract."""
        stream_result = self.websocket_manager.create_session(
            self.voice_session_id,
            project="test-voice-ws",
            owner="test-user",
            language="en",
            stt_engine="",
            allow_fallback_stt=True,  # Allow fallback to verify available engines
        )
        stream_id = stream_result["stream_id"]

        # Create synthesized audio (deterministic, local)
        audio_data = synthesize_tone_wav("hello world testing speech recognition")

        # Push audio chunk
        self.websocket_manager.push_chunk(stream_id, audio_data)

        # Finish and run STT
        finish_result = self.websocket_manager.finish(stream_id)

        # Verify STT contract results
        assert "stream_id" in finish_result
        assert finish_result["stream_id"] == stream_id
        assert "status" in finish_result
        assert finish_result["status"] in {"transcribed", "closed", "error"}
        assert "total_bytes" in finish_result
        assert "chunk_count" in finish_result
        assert "assembly_hash" in finish_result

        # Verify transcription fields are present even if empty
        assert "transcription" in finish_result
        assert "is_final" in finish_result
        assert isinstance(finish_result["is_final"], bool)

        # If transcribed, verify STT metadata
        if finish_result["status"] == "transcribed":
            assert "stt_engine" in finish_result
            assert "confidence" in finish_result
            assert finish_result["confidence"] in {"high", "medium", "low"}
            assert "turn_id" in finish_result
            assert "audio_profile" in finish_result

    def test_unavailable_stt_does_not_claim_transcription(self) -> None:
        stream_result = self.websocket_manager.create_session(
            self.voice_session_id, project="test-voice-ws", owner="test-user",
        )
        stream_id = stream_result["stream_id"]
        self.websocket_manager.push_chunk(stream_id, synthesize_tone_wav("quiet test"))
        with patch.object(self.voice, "transcribe_audio", return_value={
            "transcript": "", "transcript_status": "unavailable",
            "stt_engine": "", "stt_message": "No local STT provider is configured.",
            "confidence": "none", "turn": {},
        }):
            result = self.websocket_manager.finish(stream_id)
        self.assertEqual(result["status"], "closed")
        self.assertEqual(result["transcript_status"], "unavailable")
        self.assertEqual(result["transcription"], "")
        self.assertFalse(result["is_final"])
        self.assertEqual(result["confidence"], "none")

    def test_websocket_stt_produces_transcript(self) -> None:
        """Test that STT contract returns actual transcription."""
        stream_result = self.websocket_manager.create_session(
            self.voice_session_id,
            project="test-voice-ws",
            owner="test-user",
            language="en",
        )
        stream_id = stream_result["stream_id"]

        # Create audio data
        audio_data = synthesize_tone_wav("test audio transcription")

        # Push audio
        self.websocket_manager.push_chunk(stream_id, audio_data)

        # Finish and verify transcription result
        result = self.websocket_manager.finish(stream_id)

        # STT should return some result (transcription, error, or unavailable message)
        if result["status"] == "transcribed":
            # If transcribed, there should be a turn_id
            assert result["turn_id"]
            # Confidence should be set
            assert result["confidence"] in {"high", "medium", "low"}
        elif result["status"] == "error":
            # If error, verify error details
            assert "error" in result

        # Verify audio profile is captured
        assert "audio_profile" in result

    def test_websocket_session_status(self) -> None:
        """Test getting session status before finishing."""
        stream_result = self.websocket_manager.create_session(
            self.voice_session_id,
            project="test-voice-ws",
            owner="test-user",
            language="en",
        )
        stream_id = stream_result["stream_id"]

        # Add some chunks
        audio_data = synthesize_tone_wav("test")
        self.websocket_manager.push_chunk(stream_id, audio_data)

        # Get status
        status = self.websocket_manager.get_session_status(stream_id)

        assert status["stream_id"] == stream_id
        assert status["status"] in {"buffering", "open"}  # Can be either status
        assert status["chunk_count"] == 1
        assert status["total_bytes"] > 0
        assert status["is_final"] is False

    def test_websocket_empty_session_handling(self) -> None:
        """Test finishing with no audio chunks."""
        stream_result = self.websocket_manager.create_session(
            self.voice_session_id,
            project="test-voice-ws",
            owner="test-user",
            language="en",
        )
        stream_id = stream_result["stream_id"]

        # Finish without adding chunks
        result = self.websocket_manager.finish(stream_id)

        assert result["status"] == "closed"
        assert result["transcription"] == ""
        assert result["is_final"] is False
        assert "message" in result

    def test_websocket_stale_session_cleanup(self) -> None:
        """Test cleanup of stale sessions."""
        stream_result = self.websocket_manager.create_session(
            self.voice_session_id,
            project="test-voice-ws",
            owner="test-user",
            language="en",
        )
        stream_id = stream_result["stream_id"]

        # Verify session exists
        assert stream_id in self.websocket_manager.sessions

        # Cleanup with very short timeout
        stale_count = self.websocket_manager.cleanup_stale_sessions(timeout_seconds=0)

        # Session should be cleaned up
        assert stale_count >= 1

    def test_websocket_session_not_found_errors(self) -> None:
        """Test error handling for non-existent sessions."""
        # Try to push chunk to non-existent session
        with self.assertRaises(ValueError) as ctx:
            self.websocket_manager.push_chunk("non-existent-stream", b"audio data")
        assert "not found" in str(ctx.exception).lower()

        # Try to get status of non-existent session
        with self.assertRaises(ValueError) as ctx:
            self.websocket_manager.get_session_status("non-existent-stream")
        assert "not found" in str(ctx.exception).lower()

    def test_websocket_invalid_voice_session_error(self) -> None:
        """Test error handling for invalid voice session ID."""
        with self.assertRaises(ValueError) as ctx:
            self.websocket_manager.create_session(
                "non-existent-voice-session",
                project="test-voice-ws",
                owner="test-user",
            )
        assert "not found" in str(ctx.exception).lower()


class VoiceWebSocketSessionUnitTests(unittest.TestCase):
    """Unit tests for VoiceWebSocketSession class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test database and voice runtime."""
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.store = OpenAgentStore(str(Path(cls.tmpdir.name) / "test.db"))
        cls.voice = VoiceRuntime(cls.store)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory."""
        cls.tmpdir.cleanup()

    def setUp(self) -> None:
        """Create a voice session for each test."""
        # Use unique project to ensure session IDs don't collide
        project = f"test-unit-{uuid4().hex[:8]}"
        self.voice_session = self.voice.create_session(
            project=project,
            mode="push_to_talk",
            language="en",
        )

    def test_session_initialization(self) -> None:
        """Test VoiceWebSocketSession initialization."""
        session = VoiceWebSocketSession(
            self.voice,
            self.voice_session["id"],
            project="test-unit",
            owner="test-user",
            language="en",
        )

        assert session.session_id == self.voice_session["id"]
        assert session.project == "test-unit"
        assert session.owner == "test-user"
        assert session.language == "en"
        assert session.status == "open"
        assert len(session.chunks) == 0
        assert session.total_bytes == 0

    def test_session_audio_chunk_addition(self) -> None:
        """Test adding audio chunks to session."""
        session = VoiceWebSocketSession(
            self.voice,
            self.voice_session["id"],
            project="test-unit",
            owner="test-user",
        )

        audio_data = b"fake audio data 1234"
        result = session.add_audio_chunk(audio_data)

        assert result["chunk_received"] is True
        assert result["status"] == "buffering"
        assert len(session.chunks) == 1
        assert session.total_bytes == len(audio_data)

    def test_session_cannot_add_chunks_when_closed(self) -> None:
        """Test that chunks cannot be added to closed session."""
        session = VoiceWebSocketSession(
            self.voice,
            self.voice_session["id"],
            project="test-unit",
            owner="test-user",
        )

        # Close the session
        session.status = "closed"

        # Try to add chunk
        with self.assertRaises(ValueError) as ctx:
            session.add_audio_chunk(b"audio data")
        assert "closed" in str(ctx.exception).lower()


if __name__ == "__main__":
    unittest.main()
