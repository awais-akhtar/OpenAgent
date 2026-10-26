from __future__ import annotations

import base64
from http import HTTPStatus
from unittest import TestCase
from unittest.mock import Mock

from openagent.routes.voice_routes import (
    ROUTES, VoiceRouteDependencies, contract, handle_post,
)


def _deps(payload=None, *, owner=""):
    voice = Mock()
    read_json = Mock(return_value=payload if payload is not None else {})
    first = lambda params, key: (params.get(key, [None])[0] or None)
    query_owner = lambda params: owner
    payload_owner = lambda p, default: owner or default
    parse_bool = lambda value, default: default if value is None else bool(value)
    parse_int = lambda value, default: default if value is None else int(value)
    return voice, VoiceRouteDependencies(
        voice=voice,
        read_json=read_json,
        first=first,
        parse_bool=parse_bool,
        parse_int=parse_int,
        query_owner=query_owner,
        payload_owner=payload_owner,
    )


class VoiceRouteContractTests(TestCase):
    def test_contract_has_stream_route(self):
        self.assertIn(("POST", "/api/voice/stream/continuous"), ROUTES)
        self.assertEqual(contract()["group"], "voice")


class VoiceContinuousStreamTests(TestCase):
    def test_stream_continuous_success(self):
        audio_data = b"fake audio data"
        audio_b64 = base64.b64encode(audio_data).decode("ascii")
        payload = {
            "session_id": "session-1",
            "audio_base64": audio_b64,
            "project": "default",
            "language": "en",
        }
        voice, deps = _deps(payload=payload, owner="user-1")
        voice.process_continuous_stream.return_value = {
            "chunk_id": "chunk-1",
            "session_id": "session-1",
            "chunk_size": len(audio_data),
            "buffering": True,
        }
        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertIn("stream_result", result.payload)
        voice.process_continuous_stream.assert_called_once()
        args, kwargs = voice.process_continuous_stream.call_args
        self.assertEqual(args[0], "session-1")
        self.assertEqual(args[1], audio_data)
        self.assertEqual(kwargs.get("language"), "en")

    def test_stream_missing_session_id(self):
        audio_data = b"fake audio data"
        audio_b64 = base64.b64encode(audio_data).decode("ascii")
        payload = {
            "audio_base64": audio_b64,
            "project": "default",
        }
        voice, deps = _deps(payload=payload)
        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("error", result.payload)

    def test_stream_missing_audio_data(self):
        payload = {
            "session_id": "session-1",
            "project": "default",
        }
        voice, deps = _deps(payload=payload)
        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("error", result.payload)

    def test_stream_invalid_base64(self):
        payload = {
            "session_id": "session-1",
            "audio_base64": "not-valid-base64!!!",
            "project": "default",
        }
        voice, deps = _deps(payload=payload)
        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_stream_session_not_found(self):
        audio_data = b"fake audio data"
        audio_b64 = base64.b64encode(audio_data).decode("ascii")
        payload = {
            "session_id": "missing-session",
            "audio_base64": audio_b64,
            "project": "default",
        }
        voice, deps = _deps(payload=payload)
        voice.process_continuous_stream.side_effect = ValueError("Voice session 'missing-session' not found.")
        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_stream_with_metadata(self):
        audio_data = b"fake audio data"
        audio_b64 = base64.b64encode(audio_data).decode("ascii")
        payload = {
            "session_id": "session-1",
            "audio_base64": audio_b64,
            "project": "default",
            "metadata": {"speaker": "user-1", "confidence": 0.95},
        }
        voice, deps = _deps(payload=payload)
        voice.process_continuous_stream.return_value = {
            "chunk_id": "chunk-1",
            "session_id": "session-1",
            "chunk_size": len(audio_data),
            "buffering": True,
        }
        result = handle_post("/api/voice/stream/continuous", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        _, kwargs = voice.process_continuous_stream.call_args
        self.assertEqual(kwargs.get("metadata"), {"speaker": "user-1", "confidence": 0.95})

    def test_stream_with_owner(self):
        audio_data = b"fake audio data"
        audio_b64 = base64.b64encode(audio_data).decode("ascii")
        payload = {
            "session_id": "session-1",
            "audio_base64": audio_b64,
            "project": "default",
        }
        voice, deps = _deps(payload=payload, owner="user-2")
        voice.process_continuous_stream.return_value = {
            "chunk_id": "chunk-1",
            "session_id": "session-1",
            "chunk_size": len(audio_data),
            "buffering": True,
        }
        handle_post("/api/voice/stream/continuous", deps)
        _, kwargs = voice.process_continuous_stream.call_args
        self.assertEqual(kwargs.get("owner"), "user-2")
