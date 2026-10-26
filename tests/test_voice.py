from __future__ import annotations

import base64
from http import HTTPStatus
from unittest import TestCase
from unittest.mock import Mock

from openagent.routes.media_routes import (
    ROUTES, MediaRouteDependencies, handle_post,
)


def _deps(payload=None, *, owner=""):
    media = Mock()
    voice = Mock()
    read_json = Mock(return_value=payload if payload is not None else {})
    return media, voice, MediaRouteDependencies(
        media=media,
        voice=voice,
        read_json=read_json,
        handle_voice_upload=Mock(),
        auth_scope_owner=lambda: owner,
    )


class VoiceStreamRouteContractTests(TestCase):
    def test_stream_routes_in_contract(self):
        paths = [p for _, p in ROUTES]
        self.assertIn("/api/voice/stream/start", paths)
        self.assertIn("/api/voice/stream/chunk", paths)
        self.assertIn("/api/voice/stream/finish", paths)


class VoiceMediaOwnerRouteTests(TestCase):
    def test_tts_persists_audio_for_authenticated_owner(self):
        _, voice, deps = _deps({"project": "default", "text": "playback"}, owner="usr-tts")
        voice.synthesize_text.return_value = {"audio_asset_id": "asset-tts"}
        result = handle_post("/api/voice/tts", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        self.assertEqual(voice.synthesize_text.call_args.kwargs["owner"], "usr-tts")

    def test_stt_persists_audio_for_authenticated_owner(self):
        audio = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEdata").decode("ascii")
        _, voice, deps = _deps({"project": "default", "session_id": "session-1", "audio_base64": audio}, owner="usr-stt")
        voice.transcribe_audio.return_value = {"audio_asset": {"id": "asset-stt"}}
        result = handle_post("/api/voice/stt", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        self.assertEqual(voice.transcribe_audio.call_args.kwargs["owner"], "usr-stt")

    def test_voice_clone_persists_audio_for_authenticated_owner(self):
        audio = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEdata").decode("ascii")
        _, voice, deps = _deps({"project": "default", "name": "Local voice", "audio_base64": audio, "consent": "local test consent"}, owner="usr-clone")
        voice.clone_voice_from_audio.return_value = {"asset": {"id": "asset-clone"}}
        result = handle_post("/api/voice/clone", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        self.assertEqual(voice.clone_voice_from_audio.call_args.kwargs["owner"], "usr-clone")


class VoiceStreamStartTests(TestCase):
    def test_start_stream_success(self):
        _, voice, deps = _deps({"project": "default", "audio_format": "webm", "language": "en"})
        voice.start_stream_session.return_value = {"stream_id": "s-1", "status": "open"}
        result = handle_post("/api/voice/stream/start", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        self.assertEqual(result.payload["stream_id"], "s-1")
        voice.start_stream_session.assert_called_once()

    def test_start_stream_bad_request_on_value_error(self):
        _, voice, deps = _deps({})
        voice.start_stream_session.side_effect = ValueError("invalid format")
        result = handle_post("/api/voice/stream/start", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_start_stream_passes_owner(self):
        _, voice, deps = _deps({}, owner="usr-1")
        voice.start_stream_session.return_value = {"stream_id": "s-2"}
        handle_post("/api/voice/stream/start", deps)
        _, kwargs = voice.start_stream_session.call_args
        self.assertEqual(kwargs.get("owner"), "usr-1")

    def test_start_stream_defaults_audio_format_to_webm(self):
        _, voice, deps = _deps({})
        voice.start_stream_session.return_value = {"stream_id": "s-3"}
        handle_post("/api/voice/stream/start", deps)
        _, kwargs = voice.start_stream_session.call_args
        self.assertEqual(kwargs.get("audio_format"), "webm")


class VoiceStreamChunkTests(TestCase):
    def test_push_chunk_success(self):
        audio_b64 = base64.b64encode(b"\x00\x01\x02").decode("ascii")
        _, voice, deps = _deps({"stream_id": "s-1", "audio_base64": audio_b64, "sequence": 1})
        voice.push_stream_chunk.return_value = {"accepted": True, "sequence": 1}
        result = handle_post("/api/voice/stream/chunk", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertTrue(result.payload["accepted"])

    def test_push_chunk_missing_audio_returns_bad_request(self):
        _, voice, deps = _deps({"stream_id": "s-1"})
        result = handle_post("/api/voice/stream/chunk", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_push_chunk_invalid_base64_returns_bad_request(self):
        _, voice, deps = _deps({"stream_id": "s-1", "audio_base64": "not-valid-base64!!"})
        result = handle_post("/api/voice/stream/chunk", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)


class VoiceStreamFinishTests(TestCase):
    def test_finish_stream_success(self):
        _, voice, deps = _deps({"stream_id": "s-1", "speaker": "user"})
        voice.finish_stream_session.return_value = {"session_id": "s-1", "transcript": "hello world", "status": "complete"}
        result = handle_post("/api/voice/stream/finish", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertIn("transcript", result.payload)

    def test_finish_stream_bad_request_on_value_error(self):
        _, voice, deps = _deps({"stream_id": "s-bad"})
        voice.finish_stream_session.side_effect = ValueError("stream not found")
        result = handle_post("/api/voice/stream/finish", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_finish_stream_passes_owner(self):
        _, voice, deps = _deps({"stream_id": "s-2"}, owner="usr-2")
        voice.finish_stream_session.return_value = {"session_id": "s-2", "transcript": ""}
        handle_post("/api/voice/stream/finish", deps)
        _, kwargs = voice.finish_stream_session.call_args
        self.assertEqual(kwargs.get("owner"), "usr-2")

    def test_finish_stream_defaults_speaker_to_user(self):
        _, voice, deps = _deps({"stream_id": "s-3"})
        voice.finish_stream_session.return_value = {"session_id": "s-3"}
        handle_post("/api/voice/stream/finish", deps)
        _, kwargs = voice.finish_stream_session.call_args
        self.assertEqual(kwargs.get("speaker"), "user")
