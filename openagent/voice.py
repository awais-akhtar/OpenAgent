"""Stage 11 local voice session runtime."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any
import wave
import zipfile

from .media import MediaManager
from .egress import EgressClient
from .models import AuditEvent, ContextEvent, utc_now
from .process_policy import DEFAULT_ENV_KEYS, ProcessPolicy, parse_argv_template, run_process
from .storage import OpenAgentStore


VOSK_SMALL_EN_MODEL = "vosk-model-small-en-us-0.15"
VOSK_SMALL_EN_URL = f"https://alphacephei.com/vosk/models/{VOSK_SMALL_EN_MODEL}.zip"
VOSK_SMALL_EN_URLS = [VOSK_SMALL_EN_URL]


class VoiceRuntime:
    """Local voice session records, transcripts, personas, and TTS scripts."""

    def __init__(self, store: OpenAgentStore):
        self.store = store
        self.initialize()
        self.auto_configure_vosk_model()
        self.seed_personas()

    def initialize(self) -> None:
        self.store.initialize()

    def seed_personas(self) -> None:
        self.upsert_persona(
            name="Default Local Voice",
            project="default",
            language="en",
            style="clear, concise, privacy-first",
            metadata={"seeded": True, "voice_cloning": "local-voiceprint-record"},
        )

    def upsert_persona(
        self,
        *,
        name: str,
        project: str = "default",
        language: str = "en",
        style: str = "clear",
        voiceprint_asset_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = name.strip() or "Local Voice"
        project = project.strip() or "default"
        persona_id = stable_id("voice-persona", project, name)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO voice_personas (
                    id, project, name, language, style, voiceprint_asset_id,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM voice_personas WHERE id = ?), ?), ?)
                """,
                (
                    persona_id,
                    project,
                    name,
                    language.strip() or "en",
                    style.strip() or "clear",
                    voiceprint_asset_id.strip(),
                    json.dumps(metadata or {}, sort_keys=True),
                    persona_id,
                    now,
                    now,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="voice",
                action="voice.persona.upsert",
                resource=f"persona:{persona_id}",
                project=project,
                metadata={"language": language, "voiceprint_asset_id": bool(voiceprint_asset_id)},
            )
        )
        return self.get_persona(persona_id) or {}

    def clone_voice_from_audio(
        self,
        *,
        name: str,
        audio_data: bytes,
        filename: str = "voiceprint.wav",
        project: str = "default",
        language: str = "en",
        style: str = "cloned-local",
        consent: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not audio_data:
            raise ValueError("Voice clone audio data is required.")
        if not consent.strip():
            raise ValueError("Voice cloning requires an explicit local consent note.")
        media = MediaManager(self.store)
        audio_profile = analyze_wav_audio(audio_data)
        asset = media.ingest_bytes(
            audio_data,
            filename=filename,
            project=project,
            title=f"Voiceprint: {name}",
            source="voice:clone",
            metadata={
                "consent": consent,
                "audio_profile": audio_profile,
                "voice_clone": True,
                **(metadata or {}),
            },
        )
        persona = self.upsert_persona(
            name=name,
            project=project,
            language=language,
            style=style,
            voiceprint_asset_id=asset["id"],
            metadata={
                "voice_cloning": "local-voiceprint-record",
                "consent": consent,
                "audio_profile": audio_profile,
                **(metadata or {}),
            },
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="voice",
                action="voice.clone.create",
                resource=f"persona:{persona['id']}",
                project=project,
                metadata={"voiceprint_asset_id": asset["id"], "language": language},
            )
        )
        return {
            "persona": persona,
            "voiceprint_asset": asset,
            "audio_profile": audio_profile,
            "privacy_boundary": "Voice clone records store submitted voiceprint audio locally and require explicit consent metadata.",
        }

    def get_persona(self, persona_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM voice_personas WHERE id = ?", (persona_id,)).fetchone()
        return persona_row(row) if row else None

    def personas(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM voice_personas WHERE project IN (?, 'default') ORDER BY project, name",
                (project,),
            ).fetchall()
        return [persona_row(row) for row in rows]

    def create_session(
        self,
        *,
        project: str = "default",
        mode: str = "push_to_talk",
        persona_id: str = "",
        language: str = "en",
        wake_word: str = "",
        always_listening: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if always_listening and mode != "always_listening":
            mode = "always_listening"
        if mode not in {"push_to_talk", "wake_word", "always_listening", "transcription"}:
            raise ValueError("Unsupported voice session mode.")
        project = project.strip() or "default"
        now = utc_now()
        session_id = stable_id("voice-session", project, mode, language, now)
        boundary = {
            "privacy_boundary": "Voice runtime stores transcripts and submitted audio locally. Microphone capture enters only through explicit client audio upload.",
            "explicit_client_audio_input": True,
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO voice_sessions (
                    id, project, mode, persona_id, language, wake_word, always_listening,
                    status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    project,
                    mode,
                    persona_id,
                    language.strip() or "en",
                    wake_word.strip(),
                    1 if always_listening else 0,
                    "active",
                    json.dumps({**boundary, **(metadata or {})}, sort_keys=True),
                    now,
                    now,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="voice",
                action="voice.session.create",
                resource=f"voice-session:{session_id}",
                project=project,
                metadata={"mode": mode, "always_listening": always_listening},
            )
        )
        return self.get_session(session_id) or {}

    def close_session(self, session_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "UPDATE voice_sessions SET status = ?, updated_at = ? WHERE id = ?",
                ("closed", now, session_id),
            )
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown voice session '{session_id}'.")
        self.store.add_audit_event(
            AuditEvent(
                actor="voice",
                action="voice.session.close",
                resource=f"voice-session:{session_id}",
                project=session["project"],
            )
        )
        return session

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM voice_sessions WHERE id = ?", (session_id,)).fetchone()
        return session_row(row) if row else None

    def sessions(self, *, project: str = "default", status: str = "", limit: int = 50) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if status:
            clauses.append("status = ?")
            params.append(status)
        params.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM voice_sessions WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [session_row(row) for row in rows]

    def add_transcript(
        self,
        session_id: str,
        *,
        text: str,
        speaker: str = "user",
        role: str = "user",
        audio_asset_id: str = "",
        is_final: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown voice session '{session_id}'.")
        text = text.strip()
        if not text:
            raise ValueError("Transcript text is required.")
        now = utc_now()
        turn_id = stable_id("voice-turn", session_id, speaker, role, text, now)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO voice_turns (
                    id, session_id, project, speaker, role, text, audio_asset_id,
                    is_final, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    session["project"],
                    speaker.strip() or "user",
                    role.strip() or "user",
                    text,
                    audio_asset_id.strip(),
                    1 if is_final else 0,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                ),
            )
            connection.execute(
                "UPDATE voice_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        self.store.add_event(
            ContextEvent(
                title=f"Voice transcript: {speaker}",
                content=text,
                type="voice_transcript",
                source="voice",
                project=session["project"],
                metadata={"session_id": session_id, "turn_id": turn_id, "audio_asset_id": audio_asset_id},
            )
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="voice",
                action="voice.transcript.add",
                resource=f"voice-turn:{turn_id}",
                project=session["project"],
                metadata={"session_id": session_id, "is_final": is_final},
            )
        )
        return self.get_turn(turn_id) or {}

    def transcribe_audio(
        self,
        session_id: str,
        *,
        audio_data: bytes,
        filename: str = "speech.wav",
        speaker: str = "user",
        transcript_hint: str = "",
        language: str = "en",
        stt_engine: str = "",
        allow_fallback_stt: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown voice session '{session_id}'.")
        if not audio_data:
            raise ValueError("Audio data is required for speech-to-text.")
        media = MediaManager(self.store)
        audio_profile = analyze_wav_audio(audio_data)
        stt_result = (
            {
                "status": "provided_hint",
                "engine": "transcript_hint",
                "transcript": clean_transcript_text(transcript_hint),
                "message": "",
                "confidence": "provided",
            }
            if transcript_hint.strip()
            else transcribe_with_local_engine(
                audio_data,
                filename=filename,
                language=language,
                preferred_engine=stt_engine,
                allow_fallback=allow_fallback_stt,
            )
        )
        asset = media.ingest_bytes(
            audio_data,
            filename=filename,
            project=session["project"],
            title="Local speech-to-text audio",
            source="voice:stt",
            metadata={
                "language": language,
                "speaker": speaker,
                "transcript_hint": transcript_hint,
                "audio_profile": audio_profile,
                "stt_result": stt_result,
                **(metadata or {}),
            },
        )
        transcript = str(stt_result.get("transcript", "")).strip()
        transcript_status = str(stt_result.get("status", "unavailable"))
        display_text = transcript or stt_result.get("message") or local_audio_capture_notice(audio_profile, asset["content_hash"])
        turn = self.add_transcript(
            session_id,
            text=display_text,
            speaker=speaker,
            role="user",
            audio_asset_id=asset["id"],
            metadata={
                "stt": {
                    "provider": stt_result.get("engine", "unavailable"),
                    "status": transcript_status,
                    "language": language,
                    "requested_engine": stt_engine,
                    "allow_fallback_stt": allow_fallback_stt,
                    "audio_asset_id": asset["id"],
                    "audio_profile": audio_profile,
                    "prepared_audio_profile": stt_result.get("prepared_audio_profile", {}),
                    "confidence": stt_result.get("confidence", "unknown"),
                    "warnings": stt_result.get("warnings", []),
                    "message": stt_result.get("message", ""),
                    "engines": stt_result.get("engines", []),
                    "privacy_boundary": "Audio bytes were supplied by the client and stored locally before transcript creation.",
                }
            },
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="voice",
                action="voice.stt.transcribe",
                resource=f"voice-turn:{turn['id']}",
                project=session["project"],
                metadata={"audio_asset_id": asset["id"], "language": language},
            )
        )
        return {
            "project": session["project"],
            "session_id": session_id,
            "turn": turn,
            "audio_asset": asset,
            "transcript": transcript,
            "display_text": display_text,
            "transcript_status": transcript_status,
            "stt_engine": stt_result.get("engine", ""),
            "stt_message": stt_result.get("message", ""),
            "confidence": stt_result.get("confidence", "unknown"),
            "warnings": stt_result.get("warnings", []),
            "prepared_audio_profile": stt_result.get("prepared_audio_profile", {}),
            "available_stt_engines": stt_result.get("engines", local_stt_engines()),
            "audio_profile": audio_profile,
            "privacy_boundary": "Speech-to-text stores submitted audio and transcript locally; microphone capture is explicit client input.",
        }

    def synthesize_text(
        self,
        *,
        text: str,
        session_id: str = "",
        project: str = "default",
        persona_id: str = "",
        voice: str = "local-script",
    ) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Text is required for speech synthesis.")
        if session_id:
            session = self.get_session(session_id)
            if session is None:
                raise ValueError(f"Unknown voice session '{session_id}'.")
            project = session["project"]
        segments = [segment.strip() for segment in text.replace("\n", " ").split(".") if segment.strip()]
        media = MediaManager(self.store)
        wav_bytes = synthesize_tone_wav(text)
        audio_asset = media.ingest_bytes(
            wav_bytes,
            filename=f"tts-{stable_id(project, session_id, text)}.wav",
            project=project,
            title="Local generated speech audio",
            source="voice:local-synth",
            metadata={"source_text": text, "voice": voice, "persona_id": persona_id},
        )
        result = {
            "project": project,
            "session_id": session_id,
            "persona_id": persona_id,
            "voice": voice,
            "audio_status": "generated_wav",
            "audio_asset_id": audio_asset["id"],
            "audio_mime_type": audio_asset["mime_type"],
            "segments": segments or [text],
            "privacy_boundary": "No external TTS provider is called; a local WAV artifact is generated and stored inside the OpenAgent data directory.",
            "created_at": utc_now(),
        }
        if session_id:
            self.add_transcript(
                session_id,
                text=text,
                speaker=voice,
                role="assistant",
                metadata={"tts": result},
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="voice",
                action="voice.tts.audio",
                resource=f"voice-session:{session_id or project}",
                project=project,
                metadata={"voice": voice, "segments": len(result["segments"]), "audio_asset_id": audio_asset["id"]},
            )
        )
        return result

    def translate_text(
        self,
        *,
        text: str,
        project: str = "default",
        source_language: str = "auto",
        target_language: str = "en",
    ) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Text is required for translation.")
        translated = local_translate(text, target_language=target_language)
        translation_id = stable_id("voice-translation", project, source_language, target_language, text, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO voice_translations (
                    id, project, source_language, target_language, source_text,
                    translated_text, provider, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    translation_id,
                    project,
                    source_language,
                    target_language,
                    text,
                    translated,
                    "local-rule",
                    json.dumps({"lossless": translated == text}, sort_keys=True),
                    created_at,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="voice",
                action="voice.translate",
                resource=f"translation:{translation_id}",
                project=project,
                metadata={"source_language": source_language, "target_language": target_language},
            )
        )
        return {
            "id": translation_id,
            "project": project,
            "source_language": source_language,
            "target_language": target_language,
            "source_text": text,
            "translated_text": translated,
            "provider": "local-rule",
            "created_at": created_at,
        }

    def get_turn(self, turn_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM voice_turns WHERE id = ?", (turn_id,)).fetchone()
        return turn_row(row) if row else None

    def turns(self, session_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM voice_turns WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [turn_row(row) for row in rows]

    def status(self, *, project: str = "default") -> dict[str, Any]:
        self.auto_configure_vosk_model()
        with self.store.session() as connection:
            sessions = connection.execute(
                "SELECT count(*) FROM voice_sessions WHERE project = ?",
                (project,),
            ).fetchone()[0]
            turns = connection.execute(
                "SELECT count(*) FROM voice_turns WHERE project = ?",
                (project,),
            ).fetchone()[0]
            translations = connection.execute(
                "SELECT count(*) FROM voice_translations WHERE project = ?",
                (project,),
            ).fetchone()[0]
        return {
            "project": project,
            "sessions": sessions,
            "turns": turns,
            "translations": translations,
            "runtime": "local-session-runtime",
            "microphone_control": "explicit-client-audio-input",
            "stt": {
                "engines": local_stt_engines(),
                "privacy_boundary": "Only local STT engines are used by default. Browser or cloud speech services are not called by the backend.",
            },
            "privacy_boundary": "Voice text, submitted audio, and generated WAV artifacts stay in local SQLite/media storage.",
        }

    def vosk_model_path(self) -> Path:
        return self.store.data_dir / "models" / "vosk" / VOSK_SMALL_EN_MODEL

    def auto_configure_vosk_model(self) -> None:
        if os.environ.get("OPENAGENT_VOSK_MODEL", "").strip():
            return
        candidate = self.vosk_model_path()
        if candidate.exists() and candidate.is_dir():
            os.environ["OPENAGENT_VOSK_MODEL"] = str(candidate)

    def setup_vosk_model(self, *, allow_network: bool = False, project: str = "default") -> dict[str, Any]:
        model_dir = self.vosk_model_path()
        root = model_dir.parent
        root.mkdir(parents=True, exist_ok=True)
        downloaded = False
        extracted_local_archive = False
        if not model_dir.exists():
            archive_path = root / f"{VOSK_SMALL_EN_MODEL}.zip"
            local_archive = first_existing_vosk_archive(root)
            if local_archive:
                with zipfile.ZipFile(local_archive) as archive:
                    safe_extract_zip(archive, root)
                extracted_local_archive = True
            else:
                if not allow_network:
                    raise ValueError(
                        "Vosk model setup requires allow_network=true or a local Vosk zip archive "
                        f"at {archive_path} or .tmp-vosk.zip in the OpenAgent repo root."
                    )
                errors = []
                for url in VOSK_SMALL_EN_URLS:
                    try:
                        download_file(url, archive_path, approved=allow_network)
                        with zipfile.ZipFile(archive_path) as archive:
                            safe_extract_zip(archive, root)
                        downloaded = True
                        break
                    except Exception as exc:
                        archive_path.unlink(missing_ok=True)
                        errors.append(f"{url}: {exc}")
            if not model_dir.exists():
                if local_archive:
                    raise ValueError(f"Local Vosk archive did not contain {VOSK_SMALL_EN_MODEL}.")
                raise ValueError("Vosk model download failed. " + " | ".join(errors[-2:]))
        os.environ["OPENAGENT_VOSK_MODEL"] = str(model_dir)
        package_available = module_available("vosk")
        self.store.add_audit_event(
            AuditEvent(
                actor="voice",
                action="voice.vosk.setup",
                resource=f"vosk:{VOSK_SMALL_EN_MODEL}",
                project=project,
                metadata={
                    "allow_network": allow_network,
                    "downloaded": downloaded,
                    "local_archive": extracted_local_archive,
                    "package_available": package_available,
                },
            )
        )
        return {
            "engine": "vosk",
            "model": VOSK_SMALL_EN_MODEL,
            "model_path": str(model_dir),
            "downloaded": downloaded,
            "local_archive": extracted_local_archive,
            "package_available": package_available,
            "enabled": package_available and model_dir.exists(),
            "next_step": (
                "Vosk is ready for local STT."
                if package_available
                else "Install the optional Python package with: python -m pip install \"openagent[voice]\""
            ),
            "privacy_boundary": "The Vosk acoustic model is downloaded into the local OpenAgent data directory and used only by local speech-to-text.",
            "stt": {"engines": local_stt_engines()},
        }


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def safe_extract_zip(archive: zipfile.ZipFile, target_dir: Path) -> None:
    root = target_dir.resolve()
    for member in archive.infolist():
        destination = (root / member.filename).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe archive path: {member.filename}") from exc
    archive.extractall(root)


def first_existing_vosk_archive(model_root: Path) -> Path | None:
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        model_root / f"{VOSK_SMALL_EN_MODEL}.zip",
        repo_root / f"{VOSK_SMALL_EN_MODEL}.zip",
        repo_root / ".tmp-vosk.zip",
        Path.cwd() / f"{VOSK_SMALL_EN_MODEL}.zip",
        Path.cwd() / ".tmp-vosk.zip",
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def download_file(url: str, target: Path, *, approved: bool = False) -> None:
    EgressClient().download(
        url,
        target,
        approved=approved,
        purpose="Vosk acoustic model setup",
        timeout_seconds=600,
        max_total_bytes=300 * 1024 * 1024,
        resume=False,
    )
    with target.open("rb") as handle:
        signature = handle.read(4)
    if signature != b"PK\x03\x04":
        target.unlink(missing_ok=True)
        raise ValueError("Downloaded payload was not a ZIP archive.")


def local_translate(text: str, *, target_language: str) -> str:
    phrasebook = {
        "es": {"hello": "hola", "thanks": "gracias", "meeting": "reunion"},
        "fr": {"hello": "bonjour", "thanks": "merci", "meeting": "reunion"},
        "ur": {"hello": "salam", "thanks": "shukriya", "meeting": "meeting"},
    }
    mapping = phrasebook.get(target_language.lower())
    if not mapping:
        return text
    words = text.split()
    translated = [mapping.get(word.lower().strip(".,!?"), word) for word in words]
    return " ".join(translated)


def synthesize_tone_wav(text: str, *, sample_rate: int = 8000) -> bytes:
    """Create a tiny deterministic local WAV cue for text.

    This dependency-free offline synthesizer gives the voice runtime a real
    local audio artifact to pass through storage, audit, playback, and sync paths.
    """

    frames = bytearray()
    cleaned = text.strip() or "OpenAgent"
    amplitude = 8000
    for char in cleaned[:240]:
        if char.isspace():
            frames.extend(struct.pack("<h", 0) * int(sample_rate * 0.035))
            continue
        frequency = 360 + (ord(char.lower()) % 32) * 18
        duration = 0.045
        samples = int(sample_rate * duration)
        for index in range(samples):
            envelope = 1.0 - (index / samples) * 0.35
            value = int(amplitude * envelope * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        frames.extend(struct.pack("<h", 0) * int(sample_rate * 0.01))
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return buffer.getvalue()


def decode_audio_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid audio base64: {exc}") from exc


def analyze_wav_audio(audio_data: bytes) -> dict[str, Any]:
    profile: dict[str, Any] = {"format": "binary", "size_bytes": len(audio_data)}
    if audio_data.startswith(b"\x1a\x45\xdf\xa3"):
        profile.update(
            {
                "format": "webm",
                "container": "matroska/webm",
                "duration_seconds": 0,
                "browser_recording": True,
            }
        )
        return profile
    if audio_data.startswith(b"OggS"):
        profile.update(
            {
                "format": "ogg",
                "container": "ogg",
                "duration_seconds": 0,
                "browser_recording": True,
            }
        )
        return profile
    if len(audio_data) > 12 and audio_data[4:8] == b"ftyp":
        profile.update(
            {
                "format": "mp4",
                "container": "iso-bmff",
                "duration_seconds": 0,
                "browser_recording": True,
            }
        )
        return profile
    try:
        with wave.open(io.BytesIO(audio_data), "rb") as wav:
            frame_count = wav.getnframes()
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            raw = wav.readframes(min(frame_count, sample_rate * 5))
        values = []
        if width == 2:
            values = [sample[0] for sample in struct.iter_unpack("<h", raw)]
        energy = sum(abs(value) for value in values) / max(1, len(values)) if values else 0
        peak = max((abs(value) for value in values), default=0)
        clipped = sum(1 for value in values if abs(value) >= 32700)
        rms = math.sqrt(sum(value * value for value in values) / max(1, len(values))) if values else 0
        profile.update(
            {
                "format": "wav",
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": width,
                "frames": frame_count,
                "duration_seconds": round(frame_count / sample_rate, 3) if sample_rate else 0,
                "average_energy": round(energy, 3),
                "rms": round(rms, 3),
                "peak": peak,
                "clipped_percent": round((clipped / max(1, len(values))) * 100, 4),
                "silence_likely": bool(peak < 120 or energy < 25),
            }
        )
    except Exception:
        profile["format"] = "binary-audio"
        profile["duration_seconds"] = 0
    return profile


def local_audio_capture_notice(profile: dict[str, Any], content_hash: str) -> str:
    if profile.get("format") == "wav":
        return (
            "Audio was captured locally, but no local speech-to-text engine returned text. "
            f"Duration {profile.get('duration_seconds', 0)}s, "
            f"sample rate {profile.get('sample_rate', 0)} Hz, "
            f"fingerprint {content_hash[:12]}."
        )
    if profile.get("browser_recording"):
        return (
            "Browser microphone recording was captured locally, but no local speech-to-text engine returned text. "
            f"Container {profile.get('format')}, "
            f"size {profile.get('size_bytes', 0)} bytes, "
            f"fingerprint {content_hash[:12]}."
        )
    return f"Audio was captured locally, but no local speech-to-text engine returned text. Fingerprint {content_hash[:12]}."


def local_audio_transcript(profile: dict[str, Any], content_hash: str) -> str:
    return local_audio_capture_notice(profile, content_hash)


def local_stt_engines() -> list[dict[str, Any]]:
    command = os.environ.get("OPENAGENT_STT_COMMAND", "").strip()
    faster_model = faster_whisper_model_name()
    faster_auto_model = not os.environ.get("OPENAGENT_FASTER_WHISPER_MODEL", "").strip() and module_available("faster_whisper")
    whisper_model = os.environ.get("OPENAGENT_WHISPER_MODEL", "").strip()
    whisper_cpp_exe = os.environ.get("OPENAGENT_WHISPER_CPP_EXE", "").strip()
    whisper_cpp_model = os.environ.get("OPENAGENT_WHISPER_CPP_MODEL", "").strip()
    vosk_model = os.environ.get("OPENAGENT_VOSK_MODEL", "").strip()
    vosk_package = module_available("vosk")
    if vosk_package and vosk_model:
        vosk_reason = ""
    elif vosk_model:
        vosk_reason = "Vosk model is configured, but the Python vosk package is not installed. Run: python -m pip install \"openagent[voice]\""
    elif vosk_package:
        vosk_reason = "Vosk package is installed; run Setup Vosk to extract or download a local model."
    else:
        vosk_reason = "Install Vosk extras and run Setup Vosk, or set OPENAGENT_VOSK_MODEL to a local model directory."
    windows_speech_allowed = os.environ.get("OPENAGENT_ENABLE_WINDOWS_SPEECH", "").strip().lower() in {"1", "true", "yes", "on"}
    ffmpeg_path = shutil.which("ffmpeg")
    return [
        {
            "id": "custom-command",
            "name": "Custom local command",
            "available": bool(command),
            "enabled": bool(command),
            "configured": bool(command),
            "quality": "user-provided",
            "reason": "" if command else "Set OPENAGENT_STT_COMMAND to a local command template.",
            "privacy_boundary": "Runs the user-provided local command template from OPENAGENT_STT_COMMAND.",
        },
        {
            "id": "faster-whisper",
            "name": "faster-whisper",
            "available": module_available("faster_whisper") and bool(faster_model),
            "enabled": module_available("faster_whisper") and bool(faster_model),
            "configured": bool(faster_model),
            "model": faster_model,
            "auto_model": faster_auto_model,
            "quality": "high",
            "reason": "" if module_available("faster_whisper") and faster_model else "Install faster-whisper or set OPENAGENT_FASTER_WHISPER_MODEL to a local model path/name.",
            "privacy_boundary": "Runs faster-whisper locally with the configured model.",
        },
        {
            "id": "openai-whisper",
            "name": "Whisper",
            "available": module_available("whisper") and bool(whisper_model),
            "enabled": module_available("whisper") and bool(whisper_model),
            "configured": bool(whisper_model),
            "model": whisper_model,
            "quality": "high",
            "reason": "" if module_available("whisper") and whisper_model else "Install open-source Whisper and set OPENAGENT_WHISPER_MODEL.",
            "privacy_boundary": "Runs the open-source Whisper package locally with the configured model.",
        },
        {
            "id": "whisper-cpp",
            "name": "whisper.cpp",
            "available": bool(whisper_cpp_exe and Path(whisper_cpp_exe).exists() and whisper_cpp_model and Path(whisper_cpp_model).exists()),
            "enabled": bool(whisper_cpp_exe and Path(whisper_cpp_exe).exists() and whisper_cpp_model and Path(whisper_cpp_model).exists()),
            "configured": bool(whisper_cpp_exe or whisper_cpp_model),
            "model": whisper_cpp_model,
            "quality": "high",
            "reason": "" if whisper_cpp_exe and whisper_cpp_model else "Set OPENAGENT_WHISPER_CPP_EXE and OPENAGENT_WHISPER_CPP_MODEL.",
            "privacy_boundary": "Runs the configured whisper.cpp executable and model locally.",
        },
        {
            "id": "windows-speech",
            "name": "Windows Speech fallback",
            "available": bool(os.name == "nt" and shutil.which("powershell")),
            "enabled": bool(os.name == "nt" and shutil.which("powershell") and windows_speech_allowed),
            "configured": os.name == "nt",
            "quality": "fallback",
            "reason": (
                "Disabled by default because accuracy is often weak. Set OPENAGENT_ENABLE_WINDOWS_SPEECH=1 "
                "or enable fallback in the voice UI to use it."
            ),
            "privacy_boundary": "Uses the local Windows speech recognition engine for WAV input when installed.",
        },
        {
            "id": "vosk",
            "name": "Vosk",
            "available": vosk_package and bool(vosk_model),
            "enabled": vosk_package and bool(vosk_model),
            "configured": bool(vosk_model),
            "model": vosk_model,
            "quality": "medium",
            "reason": vosk_reason,
            "privacy_boundary": "Runs Vosk locally with the configured model directory for WAV input.",
        },
        {
            "id": "ffmpeg",
            "name": "FFmpeg audio converter",
            "available": bool(ffmpeg_path),
            "enabled": bool(ffmpeg_path),
            "configured": bool(ffmpeg_path),
            "quality": "converter",
            "path": ffmpeg_path or "",
            "reason": "" if ffmpeg_path else "Install ffmpeg to convert WebM/Ogg/MP4 recordings before local STT.",
            "privacy_boundary": "Runs local ffmpeg only for audio format conversion.",
        },
    ]


def transcribe_with_local_engine(
    audio_data: bytes,
    *,
    filename: str,
    language: str = "en",
    preferred_engine: str = "",
    allow_fallback: bool = False,
) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower() or ".bin"
    engines = local_stt_engines()
    profile = analyze_wav_audio(audio_data)
    with tempfile.TemporaryDirectory(prefix="openagent-stt-") as tmp:
        audio_path = Path(tmp) / f"input{suffix}"
        audio_path.write_bytes(audio_data)
        prepared = prepare_audio_for_stt(audio_path, profile=profile, workdir=Path(tmp))
        warnings = list(prepared.get("warnings", []))
        prepared_profile = prepared.get("profile", {})
        if prepared.get("silence_likely"):
            return {
                "status": "low_audio",
                "engine": "none",
                "transcript": "",
                "message": "Audio was too quiet or too short for reliable transcription. Move closer to the microphone and record again.",
                "confidence": "none",
                "warnings": warnings,
                "prepared_audio_profile": prepared_profile,
                "engines": engines,
            }
        attempt_map = {
            "custom-command": _transcribe_custom_command,
            "faster-whisper": _transcribe_faster_whisper,
            "openai-whisper": _transcribe_openai_whisper,
            "whisper-cpp": _transcribe_whisper_cpp,
            "vosk": _transcribe_vosk,
            "windows-speech": _transcribe_windows_speech,
        }
        order = ["custom-command", "faster-whisper", "whisper-cpp", "openai-whisper", "vosk"]
        fallback_enabled = allow_fallback or os.environ.get("OPENAGENT_ENABLE_WINDOWS_SPEECH", "").strip().lower() in {"1", "true", "yes", "on"}
        if fallback_enabled:
            order.append("windows-speech")
        preferred_engine = preferred_engine.strip()
        if preferred_engine in attempt_map:
            order = [preferred_engine] + [engine_id for engine_id in order if engine_id != preferred_engine]
        errors = []
        candidates = []
        for engine_id in order:
            attempt = attempt_map[engine_id]
            input_path = choose_stt_input_path(engine_id, original=audio_path, prepared=prepared)
            if input_path is None:
                errors.append({"engine": engine_id, "status": "unsupported_format", "message": "No WAV conversion available for this engine."})
                continue
            result = attempt(input_path, language=language)
            result["transcript"] = clean_transcript_text(str(result.get("transcript", "")))
            status = result.get("status")
            if status == "transcribed" and str(result.get("transcript", "")).strip():
                scored = score_stt_result(result, prepared_profile)
                candidates.append({**result, **scored})
                if scored["confidence"] == "high" and engine_id != "windows-speech":
                    break
            if status not in {"not_configured", "unsupported_format"}:
                errors.append({"engine": result.get("engine", ""), "status": status, "message": result.get("message", "")})
        if candidates:
            best = sorted(candidates, key=lambda item: item.get("score", 0), reverse=True)[0]
            return {
                **best,
                "engines": engines,
                "warnings": warnings + list(best.get("warnings", [])),
                "prepared_audio_profile": prepared_profile,
                "attempts": [{"engine": item.get("engine"), "confidence": item.get("confidence"), "score": item.get("score")} for item in candidates],
            }
        return {
            "status": "unavailable",
            "engine": "none",
            "transcript": "",
            "message": (
                "No high-quality local speech-to-text engine returned text. Configure one of: "
                "OPENAGENT_STT_COMMAND, OPENAGENT_FASTER_WHISPER_MODEL, "
                "OPENAGENT_WHISPER_MODEL, OPENAGENT_WHISPER_CPP_EXE plus OPENAGENT_WHISPER_CPP_MODEL, "
                "or OPENAGENT_VOSK_MODEL. Windows Speech is available only as an explicit fallback because it can produce wrong transcripts."
            ),
            "engines": engines,
            "errors": errors,
            "warnings": warnings,
            "prepared_audio_profile": prepared_profile,
            "confidence": "none",
        }


def faster_whisper_model_name() -> str:
    configured = os.environ.get("OPENAGENT_FASTER_WHISPER_MODEL", "").strip()
    if configured:
        return configured
    if module_available("faster_whisper"):
        return os.environ.get("OPENAGENT_DEFAULT_STT_MODEL", "base.en").strip() or "base.en"
    return ""


def prepare_audio_for_stt(audio_path: Path, *, profile: dict[str, Any], workdir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    if profile.get("format") == "wav":
        try:
            samples, sample_rate = read_wav_as_mono_float(audio_path)
        except Exception as exc:
            return {
                "path": audio_path,
                "profile": profile,
                "warnings": [f"Could not normalize WAV audio: {exc}"],
                "silence_likely": False,
            }
        prepared_samples, prep_warnings = normalize_voice_samples(samples, sample_rate)
        warnings.extend(prep_warnings)
        if sample_rate != 16000:
            prepared_samples = resample_linear(prepared_samples, sample_rate, 16000)
            sample_rate = 16000
            warnings.append("Audio was resampled to 16 kHz mono PCM for local STT.")
        prepared_path = workdir / "prepared.wav"
        write_mono_wav(prepared_path, prepared_samples, sample_rate)
        prepared_bytes = prepared_path.read_bytes()
        prepared_profile = analyze_wav_audio(prepared_bytes)
        return {
            "path": prepared_path,
            "profile": prepared_profile,
            "warnings": warnings,
            "silence_likely": bool(prepared_profile.get("duration_seconds", 0) < 0.25 or prepared_profile.get("silence_likely")),
        }

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {
            "path": audio_path,
            "profile": profile,
            "warnings": warnings + ["Input is not WAV and ffmpeg is not installed, so some local STT engines cannot read it."],
            "silence_likely": False,
        }
    prepared_path = workdir / "prepared.wav"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(prepared_path),
    ]
    try:
        completed = run_process(
            command,
            cwd=workdir,
            policy=ProcessPolicy(
                timeout_seconds=90,
                max_output_bytes=32_000,
                allowed_cwd_roots=(workdir,),
                allowed_executables=(ffmpeg,),
            ),
        )
    except Exception as exc:
        return {
            "path": audio_path,
            "profile": profile,
            "warnings": warnings + [f"Local ffmpeg conversion failed: {exc}"],
            "silence_likely": False,
        }
    if completed.returncode != 0 or not prepared_path.exists():
        return {
            "path": audio_path,
            "profile": profile,
            "warnings": warnings + [completed.stderr.strip() or "Local ffmpeg conversion failed."],
            "silence_likely": False,
        }
    prepared_profile = analyze_wav_audio(prepared_path.read_bytes())
    return {
        "path": prepared_path,
        "profile": prepared_profile,
        "warnings": warnings + ["Audio was converted to 16 kHz mono WAV before local STT."],
        "silence_likely": bool(prepared_profile.get("duration_seconds", 0) < 0.25 or prepared_profile.get("silence_likely")),
    }


def choose_stt_input_path(engine_id: str, *, original: Path, prepared: dict[str, Any]) -> Path | None:
    prepared_path = prepared.get("path")
    prepared_is_wav = isinstance(prepared_path, Path) and prepared_path.suffix.lower() == ".wav" and prepared_path.exists()
    if engine_id in {"windows-speech", "vosk", "whisper-cpp"}:
        return prepared_path if prepared_is_wav else None
    if engine_id in {"custom-command", "faster-whisper", "openai-whisper"}:
        return prepared_path if prepared_is_wav else original
    return original


def read_wav_as_mono_float(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = max(1, wav.getnchannels())
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width not in {1, 2, 4}:
        raise ValueError(f"Unsupported WAV sample width {width}.")
    samples: list[float] = []
    frame: list[float] = []
    if width == 1:
        values = ((byte - 128) / 128.0 for byte in raw)
    elif width == 2:
        values = (value[0] / 32768.0 for value in struct.iter_unpack("<h", raw))
    else:
        values = (value[0] / 2147483648.0 for value in struct.iter_unpack("<i", raw))
    for value in values:
        frame.append(float(value))
        if len(frame) == channels:
            samples.append(sum(frame) / channels)
            frame = []
    return samples, sample_rate


def normalize_voice_samples(samples: list[float], sample_rate: int) -> tuple[list[float], list[str]]:
    warnings: list[str] = []
    if not samples:
        return [], ["No audio samples were present in the WAV file."]
    peak = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    if peak < 0.004 or rms < 0.001:
        return samples, ["Audio level is extremely low; transcription may be wrong."]

    threshold = max(0.006, min(0.04, peak * 0.025), rms * 0.55)
    first = 0
    while first < len(samples) and abs(samples[first]) < threshold:
        first += 1
    last = len(samples) - 1
    while last > first and abs(samples[last]) < threshold:
        last -= 1
    pad = int(sample_rate * 0.18)
    if first > 0 or last < len(samples) - 1:
        start = max(0, first - pad)
        end = min(len(samples), last + pad)
        samples = samples[start:end]
        warnings.append("Leading/trailing silence was trimmed before transcription.")

    peak = max((abs(sample) for sample in samples), default=0.0)
    if peak > 0:
        gain = min(8.0, 0.88 / peak)
        if gain > 1.15:
            warnings.append("Audio gain was normalized before transcription.")
        samples = [max(-0.98, min(0.98, sample * gain)) for sample in samples]

    fade = min(int(sample_rate * 0.006), len(samples) // 2)
    if fade > 0:
        for index in range(fade):
            factor = index / fade
            samples[index] *= factor
            samples[-index - 1] *= factor
    return samples, warnings


def resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if not samples or source_rate == target_rate:
        return samples
    target_count = max(1, int(len(samples) * target_rate / source_rate))
    ratio = source_rate / target_rate
    output: list[float] = []
    last_index = len(samples) - 1
    for index in range(target_count):
        source_pos = index * ratio
        left = min(int(source_pos), last_index)
        right = min(left + 1, last_index)
        fraction = source_pos - left
        output.append(samples[left] * (1.0 - fraction) + samples[right] * fraction)
    return output


def write_mono_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    frames = bytearray()
    for sample in samples:
        clamped = max(-1.0, min(1.0, sample))
        frames.extend(struct.pack("<h", int(clamped * 32767)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))


def clean_transcript_text(text: str) -> str:
    cleaned = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
    cleaned = cleaned.strip(" \t\n\r\"'")
    if not cleaned:
        return ""
    repeated = []
    last = ""
    repeats = 0
    for word in cleaned.split():
        key = word.lower().strip(".,!?")
        if key == last:
            repeats += 1
            if repeats > 2:
                continue
        else:
            repeats = 0
        repeated.append(word)
        last = key
    cleaned = " ".join(repeated)
    if cleaned and cleaned[-1] not in ".?!":
        cleaned += "."
    return cleaned[0].upper() + cleaned[1:] if cleaned else cleaned


def score_stt_result(result: dict[str, Any], audio_profile: dict[str, Any]) -> dict[str, Any]:
    transcript = str(result.get("transcript", "")).strip()
    words = [word for word in transcript.split() if word.strip(".,!?")]
    engine = str(result.get("engine", ""))
    engine_weight = {
        "custom-command": 0.9,
        "faster-whisper": 1.0,
        "openai-whisper": 0.96,
        "whisper-cpp": 0.94,
        "vosk": 0.78,
        "windows-speech": 0.46,
    }.get(engine, 0.5)
    duration = float(audio_profile.get("duration_seconds", 0) or 0)
    score = engine_weight * 100
    if len(words) < 2 and duration > 1.0:
        score -= 25
    if len(words) > 2:
        score += min(10, len(words) * 0.5)
    if audio_profile.get("clipped_percent", 0) > 1:
        score -= 10
    if audio_profile.get("silence_likely"):
        score -= 20
    lower = transcript.lower()
    suspicious = {"thank you for watching", "subscribe", "music", "applause"}
    warnings = []
    if any(phrase in lower for phrase in suspicious):
        score -= 20
        warnings.append("Transcript contains common hallucination/filler text; verify before sending.")
    if engine == "windows-speech":
        warnings.append("Windows Speech is a fallback engine and may be inaccurate; use Whisper/faster-whisper for final transcripts.")
    confidence = "high" if score >= 82 else "medium" if score >= 62 else "low"
    return {"score": round(score, 3), "confidence": confidence, "warnings": warnings}


def _transcribe_custom_command(audio_path: Path, *, language: str) -> dict[str, Any]:
    template = os.environ.get("OPENAGENT_STT_COMMAND", "").strip()
    if not template:
        return {"status": "not_configured", "engine": "custom-command"}
    try:
        command = parse_argv_template(template, input=str(audio_path), language=language)
        completed = run_process(
            command,
            cwd=audio_path.parent,
            policy=ProcessPolicy(
                timeout_seconds=120,
                max_output_bytes=256_000,
                allowed_cwd_roots=(audio_path.parent,),
                allowed_executables=(command[0],),
            ),
        )
    except Exception as exc:
        return {"status": "error", "engine": "custom-command", "message": str(exc)}
    text = completed.stdout.strip()
    if completed.returncode != 0:
        return {"status": "error", "engine": "custom-command", "message": completed.stderr.strip() or f"exit {completed.returncode}"}
    return {
        "status": "transcribed" if text else "no_speech",
        "engine": "custom-command",
        "transcript": text,
        "message": completed.stderr.strip(),
        "output_truncated": completed.output_truncated,
    }


def _transcribe_faster_whisper(audio_path: Path, *, language: str) -> dict[str, Any]:
    model_name = faster_whisper_model_name()
    if not model_name:
        return {"status": "not_configured", "engine": "faster-whisper"}
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel(
            model_name,
            device=os.environ.get("OPENAGENT_FASTER_WHISPER_DEVICE", "cpu"),
            compute_type=os.environ.get("OPENAGENT_FASTER_WHISPER_COMPUTE", "int8"),
            local_files_only=os.environ.get("OPENAGENT_STT_OFFLINE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"},
        )
        segments, _info = model.transcribe(str(audio_path), language=language or None)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    except Exception as exc:
        return {"status": "error", "engine": "faster-whisper", "message": str(exc)}
    return {"status": "transcribed" if text else "no_speech", "engine": "faster-whisper", "transcript": text}


def _transcribe_openai_whisper(audio_path: Path, *, language: str) -> dict[str, Any]:
    model_name = os.environ.get("OPENAGENT_WHISPER_MODEL", "").strip()
    if not model_name:
        return {"status": "not_configured", "engine": "openai-whisper"}
    try:
        import whisper  # type: ignore
        model = whisper.load_model(model_name)
        result = model.transcribe(str(audio_path), language=language or None)
        text = str(result.get("text", "")).strip()
    except Exception as exc:
        return {"status": "error", "engine": "openai-whisper", "message": str(exc)}
    return {"status": "transcribed" if text else "no_speech", "engine": "openai-whisper", "transcript": text}


def _transcribe_whisper_cpp(audio_path: Path, *, language: str) -> dict[str, Any]:
    exe = os.environ.get("OPENAGENT_WHISPER_CPP_EXE", "").strip()
    model = os.environ.get("OPENAGENT_WHISPER_CPP_MODEL", "").strip()
    if not exe or not model:
        return {"status": "not_configured", "engine": "whisper-cpp"}
    if not Path(exe).exists() or not Path(model).exists():
        return {"status": "error", "engine": "whisper-cpp", "message": "Configured whisper.cpp executable or model does not exist."}
    output_base = audio_path.with_suffix("")
    command = [exe, "-m", model, "-f", str(audio_path), "-otxt", "-of", str(output_base)]
    if language:
        command.extend(["-l", language])
    try:
        completed = run_process(
            command,
            cwd=audio_path.parent,
            policy=ProcessPolicy(
                timeout_seconds=180,
                max_output_bytes=100_000,
                allowed_cwd_roots=(audio_path.parent,),
                allowed_executables=(exe,),
            ),
        )
    except Exception as exc:
        return {"status": "error", "engine": "whisper-cpp", "message": str(exc)}
    output_file = output_base.with_suffix(".txt")
    text = output_file.read_text(encoding="utf-8", errors="replace").strip() if output_file.exists() else completed.stdout.strip()
    if completed.returncode != 0:
        return {"status": "error", "engine": "whisper-cpp", "message": completed.stderr.strip() or f"exit {completed.returncode}"}
    return {"status": "transcribed" if text else "no_speech", "engine": "whisper-cpp", "transcript": text}


def _transcribe_windows_speech(audio_path: Path, *, language: str) -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "not_configured", "engine": "windows-speech"}
    if audio_path.suffix.lower() != ".wav":
        return {"status": "unsupported_format", "engine": "windows-speech", "message": "Windows Speech requires WAV input."}
    powershell = shutil.which("powershell")
    if not powershell:
        return {"status": "not_configured", "engine": "windows-speech"}
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$p=$env:OPENAGENT_STT_INPUT; "
        "$r=New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
        "$r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
        "$r.SetInputToWaveFile($p); "
        "$o=$r.Recognize(); "
        "if ($o -ne $null) { [Console]::OutputEncoding=[Text.Encoding]::UTF8; Write-Output $o.Text }; "
        "$r.Dispose();"
    )
    env = {**os.environ, "OPENAGENT_STT_INPUT": str(audio_path)}
    try:
        completed = run_process(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            cwd=audio_path.parent,
            policy=ProcessPolicy(
                timeout_seconds=90,
                max_output_bytes=100_000,
                allowed_cwd_roots=(audio_path.parent,),
                allowed_executables=(powershell,),
                env_keys=DEFAULT_ENV_KEYS | {"OPENAGENT_STT_INPUT"},
            ),
            env=env,
        )
    except Exception as exc:
        return {"status": "error", "engine": "windows-speech", "message": str(exc)}
    text = completed.stdout.strip()
    if completed.returncode != 0:
        return {"status": "error", "engine": "windows-speech", "message": completed.stderr.strip() or f"exit {completed.returncode}"}
    return {"status": "transcribed" if text else "no_speech", "engine": "windows-speech", "transcript": text, "message": completed.stderr.strip()}


def _transcribe_vosk(audio_path: Path, *, language: str) -> dict[str, Any]:
    model_path = os.environ.get("OPENAGENT_VOSK_MODEL", "").strip()
    if not model_path:
        return {"status": "not_configured", "engine": "vosk"}
    if audio_path.suffix.lower() != ".wav":
        return {"status": "unsupported_format", "engine": "vosk", "message": "Vosk requires WAV input."}
    try:
        import vosk  # type: ignore
        model = vosk.Model(model_path)
        with wave.open(str(audio_path), "rb") as wav:
            recognizer = vosk.KaldiRecognizer(model, wav.getframerate())
            pieces = []
            while True:
                data = wav.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    pieces.append(json.loads(recognizer.Result()).get("text", ""))
            pieces.append(json.loads(recognizer.FinalResult()).get("text", ""))
        text = " ".join(piece for piece in pieces if piece).strip()
    except Exception as exc:
        return {"status": "error", "engine": "vosk", "message": str(exc)}
    return {"status": "transcribed" if text else "no_speech", "engine": "vosk", "transcript": text}


def module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def persona_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "name": row["name"],
        "language": row["language"],
        "style": row["style"],
        "voiceprint_asset_id": row["voiceprint_asset_id"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def session_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "mode": row["mode"],
        "persona_id": row["persona_id"],
        "language": row["language"],
        "wake_word": row["wake_word"],
        "always_listening": bool(row["always_listening"]),
        "status": row["status"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def turn_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "project": row["project"],
        "speaker": row["speaker"],
        "role": row["role"],
        "text": row["text"],
        "audio_asset_id": row["audio_asset_id"],
        "is_final": bool(row["is_final"]),
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }
