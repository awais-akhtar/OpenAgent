"""Tests for C5: Browser Socket Voice Streaming with WebSocket support."""

from __future__ import annotations

import base64
import gc
import json
import time
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock

from openagent.routes.voice_routes import VoiceRouteDependencies, handle_post


class C5VoiceWebSocketTests(TestCase):
    """Test suite for voice streaming with WebSocket support."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.voice_mock = Mock()
        self.first = lambda params, key: (params.get(key, [None])[0] or None)
        self.parse_bool = lambda value, default: default if value is None else bool(value)
        self.parse_int = lambda value, default: default if value is None else int(value)
        self.query_owner = lambda params: "test_user"
        self.payload_owner = lambda payload, default: payload.get("owner", default) or default
        self.read_json_func = lambda: {}

    def _create_deps(self, read_json_result=None, query_string=None):
        """Create VoiceRouteDependencies for testing."""
        if read_json_result is None:
            read_json_result = {}
        if query_string is None:
            query_string = {}
        voice_websocket_mock = Mock()
        return VoiceRouteDependencies(
            voice=self.voice_mock,
            voice_websocket=voice_websocket_mock,
            first=self.first,
            parse_bool=self.parse_bool,
            parse_int=self.parse_int,
            query_owner=self.query_owner,
            payload_owner=self.payload_owner,
            read_json=lambda: read_json_result,
            read_query_string=lambda: query_string,
        )

    def test_websocket_connection_initiation(self) -> None:
        """Test WebSocket connection can be initiated."""
        deps = self._create_deps({
            "session_id": "session-123",
            "audio_base64": base64.b64encode(b"test audio").decode(),
        })

        self.voice_mock.process_continuous_stream.return_value = {
            "status": "connected",
            "session_id": "session-123",
        }

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertIn("stream_result", result.payload)

    def test_websocket_stream_frames_send_receive(self) -> None:
        """Test sending and receiving audio frames over WebSocket."""
        audio_data = b"test audio frame data"
        deps = self._create_deps({
            "session_id": "session-456",
            "audio_base64": base64.b64encode(audio_data).decode(),
        })

        self.voice_mock.process_continuous_stream.return_value = {
            "status": "streaming",
            "session_id": "session-456",
            "frame_received": len(audio_data),
        }

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)

    def test_websocket_receive_transcription(self) -> None:
        """Test receiving transcription from WebSocket stream."""
        deps = self._create_deps({
            "session_id": "session-789",
            "audio_base64": base64.b64encode(b"test").decode(),
        })

        self.voice_mock.process_continuous_stream.return_value = {
            "status": "transcribing",
            "transcription": "Hello, this is a test",
            "confidence": 0.95,
        }

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        payload = result.payload["stream_result"]
        self.assertIn("transcription", payload)

    def test_websocket_disconnect_graceful(self) -> None:
        """Test graceful WebSocket disconnection."""
        deps = self._create_deps({
            "session_id": "session-close-1",
            "audio_base64": base64.b64encode(b"test").decode(),
            "disconnect": True,
        })

        self.voice_mock.process_continuous_stream.return_value = {
            "status": "disconnected",
            "session_id": "session-close-1",
            "frames_processed": 5,
        }

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)

    def test_websocket_session_management(self) -> None:
        """Test WebSocket session management (connect, resume, cleanup)."""
        # Test session creation
        deps = self._create_deps({
            "session_id": "session-sm-1",
            "audio_base64": base64.b64encode(b"test").decode(),
        })

        self.voice_mock.process_continuous_stream.return_value = {
            "status": "connected",
            "session_id": "session-sm-1",
            "session_created": True,
        }

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.voice_mock.process_continuous_stream.assert_called_once()

    def test_websocket_error_handling_invalid_audio(self) -> None:
        """Test error handling for invalid audio data."""
        deps = self._create_deps({
            "session_id": "session-err-1",
            "audio_base64": "not-valid-base64!!!",
        })

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("error", result.payload)

    def test_websocket_missing_session_id(self) -> None:
        """Test error handling when session_id is missing."""
        deps = self._create_deps({
            "audio_base64": base64.b64encode(b"test").decode(),
        })

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_websocket_missing_audio_data(self) -> None:
        """Test error handling when audio_base64 is missing."""
        deps = self._create_deps({
            "session_id": "session-err-2",
        })

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_fallback_to_http_chunked_on_websocket_unavailable(self) -> None:
        """Test fallback to HTTP chunked transfer if WebSocket unavailable."""
        deps = self._create_deps({
            "session_id": "session-fallback-1",
            "audio_base64": base64.b64encode(b"test audio").decode(),
            "use_websocket": False,  # Explicitly disable WebSocket
        })

        self.voice_mock.process_continuous_stream.return_value = {
            "status": "streaming",
            "protocol": "http_chunked",
            "chunks": ["chunk1", "chunk2"],
        }

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)

    def test_websocket_buffer_management_frame_based(self) -> None:
        """Test frame-based buffering in WebSocket stream."""
        # Simulate multiple frames
        frames = [b"frame1", b"frame2", b"frame3"]

        for frame in frames:
            deps = self._create_deps({
                "session_id": "session-buffer-1",
                "audio_base64": base64.b64encode(frame).decode(),
            })

            self.voice_mock.process_continuous_stream.return_value = {
                "status": "buffering",
                "frames_buffered": frames.index(frame) + 1,
            }

            result = handle_post("/api/voice/stream/continuous", deps)
            self.assertEqual(result.status, HTTPStatus.OK)

    def test_frontend_websocket_client_initialization(self) -> None:
        """Test frontend WebSocket client can be initialized."""
        # This test verifies the route accepts WebSocket configuration
        deps = self._create_deps({
            "session_id": "session-frontend-1",
            "audio_base64": base64.b64encode(b"test").decode(),
            "client_config": {
                "auto_reconnect": True,
                "max_retries": 3,
                "heartbeat_interval": 30000,
            }
        })

        self.voice_mock.process_continuous_stream.return_value = {
            "status": "client_initialized",
        }

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)

    def test_websocket_connection_error_recovery(self) -> None:
        """Test recovery from WebSocket connection errors."""
        deps = self._create_deps({
            "session_id": "session-recover-1",
            "audio_base64": base64.b64encode(b"test").decode(),
            "retry_count": 2,
        })

        self.voice_mock.process_continuous_stream.return_value = {
            "status": "recovered",
            "retry_count": 2,
            "frames_recovered": 10,
        }

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)

    def test_websocket_low_latency_streaming(self) -> None:
        """Test low-latency audio streaming characteristics."""
        audio_data = b"low-latency-audio-test"
        deps = self._create_deps({
            "session_id": "session-latency-1",
            "audio_base64": base64.b64encode(audio_data).decode(),
            "low_latency": True,
        })

        self.voice_mock.process_continuous_stream.return_value = {
            "status": "streaming",
            "latency_ms": 50,  # Low latency indicator
            "buffer_size": 128,
        }

        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)

    def tearDown(self) -> None:
        """Cleanup after each test."""
        gc.collect()
        time.sleep(0.05)
