from __future__ import annotations

import base64
from http import HTTPStatus
from unittest import TestCase
from unittest.mock import Mock

from openagent.routes.media_routes import (
    MediaContentRouteResult, MediaRouteDependencies, ResponseAlreadySent, ROUTES, contract, handle_delete, handle_get, handle_post,
)


class MediaRouteTests(TestCase):
    def dependencies(self, payload=None, *, owner=""):
        media, voice = Mock(), Mock()
        read_json = Mock(return_value=payload if payload is not None else {})
        upload = Mock()
        return media, voice, read_json, upload, MediaRouteDependencies(
            media, voice, read_json, upload, auth_scope_owner=lambda: owner,
        )

    def test_contract_includes_all_runtime_paths(self):
        self.assertEqual(len(ROUTES), 25)
        self.assertIn(("POST", "/api/voice/personas"), ROUTES)
        self.assertIn(("DELETE", "/api/media/assets/{id}"), ROUTES)
        self.assertEqual(contract()["group"], "media")

    def test_delete_media_asset_requires_explicit_project_and_hides_other_projects(self):
        media, _, _, _, deps = self.dependencies()
        media.delete_asset.return_value = {
            "id": "asset-1", "title": "notes.txt", "project": "default",
            "stored_bytes_removed": True, "cleanup_pending": False,
        }
        result = handle_delete("/api/media/assets/asset-1", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertTrue(result.payload["deleted"])
        media.delete_asset.assert_called_once_with("asset-1", project="default", owner="")
        self.assertEqual(handle_delete("/api/media/assets/asset-1", "project=", deps).status, HTTPStatus.BAD_REQUEST)
        media.delete_asset.return_value = None
        self.assertEqual(handle_delete("/api/media/assets/asset-1", "project=other", deps).status,
                         HTTPStatus.NOT_FOUND)
        media.delete_asset.side_effect = ValueError("unsafe stored path")
        self.assertEqual(handle_delete("/api/media/assets/asset-1", "project=default", deps).status,
                         HTTPStatus.BAD_REQUEST)

    def test_get_media_paths(self):
        media, _, _, _, deps = self.dependencies()
        media.status.return_value = {"ready": True}
        media.assets.return_value = [{"id": "a"}]
        media.analyses.return_value = [{"id": "b"}]
        self.assertEqual(handle_get("/api/media/status", "project=p", deps).payload, {"ready": True})
        media.status.assert_called_once_with(project="p", owner="")
        self.assertEqual(handle_get("/api/media/assets", "project=p&media_type=image&limit=bad", deps).payload,
                         {"assets": [{"id": "a"}]})
        media.assets.assert_called_once_with(project="p", media_type="image", limit=50, owner="")
        self.assertEqual(handle_get("/api/media/analyses", "asset_id=a&limit=4", deps).payload,
                         {"analyses": [{"id": "b"}]})
        media.analyses.assert_called_once_with(project="default", asset_id="a", limit=4, owner="")

    def test_voice_audio_content_is_project_scoped_and_decodes_asset_ids(self):
        media, _, _, _, deps = self.dependencies()
        media.read_audio_asset.return_value = ({"id": "asset:one", "mime_type": "audio/wav"}, b"RIFFfixture")
        result = handle_get("/api/voice/assets/asset%3Aone/content", "project=default", deps)
        self.assertEqual(result, MediaContentRouteResult(b"RIFFfixture", "audio/wav"))
        media.read_audio_asset.assert_called_once_with("asset:one", project="default", owner="")
        media.read_audio_asset.return_value = None
        missing = handle_get("/api/voice/assets/asset%3Aone/content", "project=other", deps)
        self.assertEqual((missing.payload, missing.status), ({"error": "Audio asset not found."}, HTTPStatus.NOT_FOUND))

    def test_get_voice_paths(self):
        _, voice, _, _, deps = self.dependencies()
        voice.status.return_value = {"engine": "local"}
        voice.personas.return_value = [{"id": "p"}]
        voice.sessions.return_value = [{"id": "s"}]
        voice.turns.return_value = [{"id": "t"}]
        self.assertEqual(handle_get("/api/voice/status", "", deps).payload, {"engine": "local"})
        voice.status.assert_called_once_with(project="default")
        self.assertEqual(handle_get("/api/voice/personas", "project=p", deps).payload, {"personas": [{"id": "p"}]})
        voice.personas.assert_called_once_with(project="p")
        self.assertEqual(handle_get("/api/voice/sessions", "project=p&status=active&limit=no", deps).payload,
                         {"sessions": [{"id": "s"}]})
        voice.sessions.assert_called_once_with(project="p", status="active", limit=50)
        missing = handle_get("/api/voice/turns", "", deps)
        self.assertEqual((missing.payload, missing.status), ({"error": "session_id is required."}, HTTPStatus.BAD_REQUEST))
        self.assertEqual(handle_get("/api/voice/turns", "session_id=s", deps).payload, {"turns": [{"id": "t"}]})
        voice.turns.assert_called_once_with("s")

    def test_media_ingest_and_analyze_paths(self):
        media, _, _, _, deps = self.dependencies({"path": " file.txt ", "project": " ", "metadata": []})
        media.ingest_file.return_value = {"id": "a"}
        result = handle_post("/api/media/ingest", deps)
        self.assertEqual((result.payload, result.status), ({"asset": {"id": "a"}}, HTTPStatus.CREATED))
        media.ingest_file.assert_called_once_with("file.txt", project="default", title="", source="file",
                                                  metadata={}, copy_to_store=True, owner="")
        _, _, _, _, deps = self.dependencies({"data_base64": base64.b64encode(b"audio").decode(),
                                              "filename": "clip.wav", "copy_to_store": "false"})
        result = handle_post("/api/media/ingest", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        deps.media.ingest_bytes.assert_called_once_with(b"audio", filename="clip.wav", project="default",
                                                       title="", source="upload", metadata={}, copy_to_store=False,
                                                       owner="")
        _, _, _, _, authenticated_deps = self.dependencies(
            {"data_base64": base64.b64encode(b"owner-scoped").decode(), "filename": "private.txt"},
            owner="installation-owner",
        )
        handle_post("/api/media/ingest", authenticated_deps)
        authenticated_deps.media.ingest_bytes.assert_called_once_with(
            b"owner-scoped", filename="private.txt", project="default", title="", source="upload",
            metadata={}, copy_to_store=True, owner="installation-owner",
        )
        _, _, _, _, deps = self.dependencies({})
        self.assertEqual(handle_post("/api/media/ingest", deps).status, HTTPStatus.BAD_REQUEST)
        deps.media.analyze_asset.return_value = {"id": "analysis"}
        result = handle_post("/api/media/analyze", deps)
        self.assertEqual(result.payload, {"analysis": {"id": "analysis"}})
        deps.media.analyze_asset.assert_called_once_with("", project="default", owner="")
        deps.media.analyze_asset.side_effect = ValueError("missing")
        self.assertEqual(handle_post("/api/media/analyze", deps).payload, {"error": "missing"})
        deps.media.analyze_asset.side_effect = KeyError("private")
        self.assertEqual(handle_post("/api/media/analyze", deps).status, HTTPStatus.NOT_FOUND)

    def test_voice_persona_session_and_setup(self):
        _, voice, _, _, deps = self.dependencies({"name": "  Alice ", "project": "core", "always_listening": "yes"})
        voice.upsert_persona.return_value = {"id": "p"}
        voice.create_session.return_value = {"id": "s"}
        voice.setup_vosk_model.return_value = {"ready": True}
        self.assertEqual(handle_post("/api/voice/personas", deps).payload, {"persona": {"id": "p"}})
        voice.upsert_persona.assert_called_once_with(name="Alice", project="core", language="en", style="clear",
                                                    voiceprint_asset_id="", metadata={})
        self.assertEqual(handle_post("/api/voice/sessions", deps).status, HTTPStatus.CREATED)
        self.assertTrue(voice.create_session.call_args.kwargs["always_listening"])
        self.assertEqual(handle_post("/api/voice/vosk/setup", deps).payload, {"ready": True})
        voice.setup_vosk_model.assert_called_once_with(allow_network=False, project="core")
        voice.setup_vosk_model.side_effect = ValueError("network blocked")
        voice.status.return_value = {"ready": False}
        result = handle_post("/api/voice/vosk/setup", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "network blocked", "status": {"ready": False}}, HTTPStatus.BAD_REQUEST))

    def test_voice_clone_close_transcript_and_stt(self):
        payload = {"session_id": "s", "audio_base64": base64.b64encode(b"wav").decode(),
                   "consent": "yes", "text": "hello", "is_final": "false"}
        _, voice, _, _, deps = self.dependencies(payload)
        voice.clone_voice_from_audio.return_value = {"id": "clone"}
        voice.close_session.return_value = {"id": "s"}
        voice.add_transcript.return_value = {"id": "turn"}
        voice.transcribe_audio.return_value = {"text": "hi"}
        self.assertEqual(handle_post("/api/voice/clone", deps).payload, {"clone": {"id": "clone"}})
        self.assertEqual(voice.clone_voice_from_audio.call_args.kwargs["audio_data"], b"wav")
        self.assertEqual(handle_post("/api/voice/sessions/close", deps).payload, {"session": {"id": "s"}})
        voice.close_session.assert_called_once_with("s")
        self.assertEqual(handle_post("/api/voice/transcript", deps).payload, {"turn": {"id": "turn"}})
        self.assertFalse(voice.add_transcript.call_args.kwargs["is_final"])
        self.assertEqual(handle_post("/api/voice/stt", deps).payload, {"transcription": {"text": "hi"}})
        self.assertEqual(voice.transcribe_audio.call_args.kwargs["audio_data"], b"wav")
        _, _, _, _, bad = self.dependencies({"audio_base64": "!"})
        self.assertEqual(handle_post("/api/voice/clone", bad).status, HTTPStatus.BAD_REQUEST)
        self.assertEqual(handle_post("/api/voice/stt", bad).status, HTTPStatus.BAD_REQUEST)

    def test_voice_tts_and_translate(self):
        _, voice, _, _, deps = self.dependencies({"text": "hello", "project": "core"})
        voice.synthesize_text.return_value = {"id": "script"}
        voice.translate_text.return_value = {"text": "hola"}
        self.assertEqual(handle_post("/api/voice/tts", deps).payload, {"script": {"id": "script"}})
        voice.synthesize_text.assert_called_once_with(text="hello", session_id="", project="core",
                                                      owner="", persona_id="", voice="local-script")
        self.assertEqual(handle_post("/api/voice/translate", deps).payload, {"translation": {"text": "hola"}})
        voice.translate_text.assert_called_once_with(text="hello", project="core", source_language="auto",
                                                     target_language="en")
        voice.translate_text.side_effect = ValueError("unsupported")
        self.assertEqual(handle_post("/api/voice/translate", deps).status, HTTPStatus.BAD_REQUEST)

    def test_upload_aliases_delegate_and_signal_response_sent(self):
        _, _, read_json, upload, deps = self.dependencies()
        for path in ("/api/voice/transcribe-upload", "/api/voice/upload"):
            self.assertIsInstance(handle_post(path, deps), ResponseAlreadySent)
        self.assertEqual(upload.call_count, 2)
        read_json.assert_not_called()

    def test_unknown_paths_are_not_handled(self):
        _, _, read_json, upload, deps = self.dependencies()
        self.assertIsNone(handle_get("/api/nope", "", deps))
        self.assertIsNone(handle_post("/api/nope", deps))
        read_json.assert_not_called()
        upload.assert_not_called()
