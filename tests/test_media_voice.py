from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import struct
import unittest
from unittest.mock import patch
import zipfile

from openagent.media import MediaManager
from openagent.storage import OpenAgentStore
from openagent.voice import (
    VOSK_SMALL_EN_MODEL,
    VoiceRuntime,
    analyze_wav_audio,
    local_stt_engines,
    prepare_audio_for_stt,
    synthesize_tone_wav,
    transcribe_with_local_engine,
)


class MediaAndVoiceTests(unittest.TestCase):
    def test_media_ingestion_understanding_and_audit(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            media = MediaManager(store)

            pdf = media.ingest_bytes(
                b"%PDF-1.4\nBT (Local PDF text for OpenAgent) Tj ET\n%%EOF",
                filename="brief.pdf",
                project="core",
                title="Brief",
            )
            self.assertEqual(pdf["media_type"], "document")
            self.assertIn("Local PDF text", pdf["analysis"]["extracted_text"])
            self.assertEqual(pdf["analysis"]["analyzer"], "local-pdf-text-scan")

            png_bytes = (
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\r"
                + b"IHDR"
                + struct.pack(">II", 3, 2)
                + b"\x08\x02\x00\x00\x00"
            )
            image = media.ingest_bytes(
                png_bytes,
                filename="canvas.png",
                project="core",
                title="Canvas",
            )
            self.assertEqual(image["media_type"], "image")
            self.assertEqual(image["analysis"]["findings"]["width"], 3)
            self.assertEqual(image["analysis"]["findings"]["height"], 2)

            status = media.status(project="core")
            self.assertEqual(status["assets_by_type"]["document"], 1)
            self.assertEqual(status["assets_by_type"]["image"], 1)
            self.assertIn("local", status["privacy_boundary"].lower())

            audit_actions = {event["action"] for event in store.list_audit_events(project="core", limit=20)}
            self.assertIn("media.ingest", audit_actions)
            self.assertIn("media.analyze", audit_actions)

    def test_voice_sessions_transcripts_tts_translation_and_audit(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            voice = VoiceRuntime(store)

            persona = voice.upsert_persona(
                name="Research Voice",
                project="core",
                language="en",
                style="calm briefing",
                metadata={"voice_cloning": "local-voiceprint-record"},
            )
            clone = voice.clone_voice_from_audio(
                name="Awais Local Clone",
                project="core",
                audio_data=synthesize_tone_wav("voice consent sample"),
                consent="Owner submitted this local voice sample for OpenAgent testing.",
            )
            self.assertEqual(clone["persona"]["voiceprint_asset_id"], clone["voiceprint_asset"]["id"])
            self.assertEqual(clone["audio_profile"]["format"], "wav")
            session = voice.create_session(
                project="core",
                mode="wake_word",
                persona_id=persona["id"],
                wake_word="open agent",
            )
            self.assertEqual(session["mode"], "wake_word")
            self.assertFalse(session["always_listening"])

            turn = voice.add_transcript(
                session["id"],
                text="Summarize the private roadmap.",
                speaker="awais",
            )
            self.assertEqual(turn["text"], "Summarize the private roadmap.")

            stt = voice.transcribe_audio(
                session["id"],
                audio_data=synthesize_tone_wav("private roadmap"),
                filename="speech.wav",
                speaker="awais",
                transcript_hint="Private roadmap audio transcribed.",
            )
            self.assertEqual(stt["transcript"], "Private roadmap audio transcribed.")
            self.assertEqual(stt["audio_profile"]["format"], "wav")
            self.assertEqual(stt["turn"]["audio_asset_id"], stt["audio_asset"]["id"])

            webm_stt = voice.transcribe_audio(
                session["id"],
                audio_data=b"\x1a\x45\xdf\xa3openagent-browser-audio",
                filename="voice.webm",
                speaker="awais",
                metadata={"mime_type": "audio/webm"},
            )
            self.assertEqual(webm_stt["audio_profile"]["format"], "webm")
            self.assertEqual(webm_stt["audio_asset"]["media_type"], "audio")
            self.assertIn(
                webm_stt["transcript_status"],
                {"transcribed", "no_engine", "unavailable", "unsupported_format", "engine_error", "empty_audio", "low_audio"},
            )
            if webm_stt["transcript_status"] == "transcribed":
                self.assertTrue(webm_stt["transcript"])
                self.assertIn(webm_stt["confidence"], {"high", "medium", "low"})
            else:
                self.assertEqual(webm_stt["transcript"], "")
                self.assertIn("speech-to-text", webm_stt["display_text"].lower())
                self.assertTrue(webm_stt["stt_message"])

            script = voice.synthesize_text(
                session_id=session["id"],
                text="Roadmap summary ready. No external TTS was called.",
                voice="local-script",
            )
            self.assertEqual(script["audio_status"], "generated_wav")
            self.assertTrue(script["audio_asset_id"])
            self.assertGreaterEqual(len(voice.turns(session["id"])), 2)
            media = MediaManager(store)
            audio_asset = media.get_asset(script["audio_asset_id"])
            self.assertIsNotNone(audio_asset)
            self.assertEqual(audio_asset["media_type"], "audio")

            translation = voice.translate_text(
                text="hello thanks",
                project="core",
                target_language="es",
            )
            self.assertEqual(translation["translated_text"], "hola gracias")

            closed = voice.close_session(session["id"])
            self.assertEqual(closed["status"], "closed")
            status = voice.status(project="core")
            self.assertEqual(status["sessions"], 1)
            self.assertIn("local sqlite", status["privacy_boundary"].lower())

            audit_actions = {event["action"] for event in store.list_audit_events(project="core", limit=50)}
            self.assertIn("voice.clone.create", audit_actions)
            self.assertIn("voice.session.create", audit_actions)
            self.assertIn("voice.transcript.add", audit_actions)
            self.assertIn("voice.stt.transcribe", audit_actions)
            self.assertIn("voice.tts.audio", audit_actions)
            self.assertIn("voice.translate", audit_actions)

    def test_vosk_setup_requires_explicit_network_approval(self) -> None:
        with TemporaryDirectory() as tmp:
            voice = VoiceRuntime(OpenAgentStore(tmp))
            with patch("openagent.voice.first_existing_vosk_archive", return_value=None), self.assertRaises(ValueError):
                voice.setup_vosk_model(allow_network=False, project="core")

    def test_vosk_setup_uses_local_archive_without_network(self) -> None:
        with TemporaryDirectory() as tmp:
            voice = VoiceRuntime(OpenAgentStore(tmp))
            archive_path = voice.vosk_model_path().parent / f"{VOSK_SMALL_EN_MODEL}.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"{VOSK_SMALL_EN_MODEL}/README", "local test model")

            result = voice.setup_vosk_model(allow_network=False, project="core")

            self.assertTrue(Path(result["model_path"]).exists())
            self.assertFalse(result["downloaded"])
            self.assertTrue(result["local_archive"])

    def test_voice_audio_preprocessing_resamples_to_stt_wav(self) -> None:
        with TemporaryDirectory() as tmp:
            source = synthesize_tone_wav("clear local voice sample", sample_rate=8000)
            original = Path(tmp) / "speech.wav"
            original.write_bytes(source)

            original_profile = analyze_wav_audio(source)
            self.assertEqual(original_profile["sample_rate"], 8000)

            prepared = prepare_audio_for_stt(original, profile=original_profile, workdir=Path(tmp))
            prepared_profile = prepared["profile"]
            self.assertEqual(prepared_profile["format"], "wav")
            self.assertEqual(prepared_profile["sample_rate"], 16000)
            self.assertEqual(prepared_profile["channels"], 1)
            self.assertIn("resampled", " ".join(prepared["warnings"]).lower())

    def test_custom_local_stt_command_is_cleaned_and_scored(self) -> None:
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_stt.py"
            script.write_text("print('hello hello hello from openagent')\n", encoding="utf-8")
            command = f'"{sys.executable}" "{script}" "{{input}}" "{{language}}"'
            with patch.dict(os.environ, {"OPENAGENT_STT_COMMAND": command}, clear=False):
                result = transcribe_with_local_engine(
                    synthesize_tone_wav("hello from openagent"),
                    filename="speech.wav",
                    language="en",
                )
            self.assertEqual(result["status"], "transcribed")
            self.assertEqual(result["engine"], "custom-command")
            self.assertEqual(result["transcript"], "Hello hello hello from openagent.")
            self.assertIn(result["confidence"], {"high", "medium"})
            self.assertEqual(result["prepared_audio_profile"]["sample_rate"], 16000)

    def test_windows_speech_is_explicit_fallback_not_default(self) -> None:
        with patch.dict(os.environ, {"OPENAGENT_ENABLE_WINDOWS_SPEECH": ""}, clear=False):
            engines = local_stt_engines()
        windows = next(engine for engine in engines if engine["id"] == "windows-speech")
        self.assertEqual(windows["quality"], "fallback")
        if windows["available"]:
            self.assertFalse(windows["enabled"])


if __name__ == "__main__":
    unittest.main()
