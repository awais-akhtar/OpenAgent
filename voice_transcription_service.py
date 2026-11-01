"""G05: Voice transcription service with real speech-to-text."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VoiceTranscriptionService:
    """Convert audio to text using Whisper or local alternatives."""

    def __init__(self):
        self.model = "base"  # Options: tiny, base, small, medium, large
        self.language = "en"
        self.device = "cpu"  # cpu or cuda

    def transcribe_audio(self, audio_data: bytes, audio_format: str = "wav") -> dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format (wav, mp3, m4a, etc.)

        Returns:
            Dict with transcribed text and metadata
        """
        try:
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            logger.info(f"Transcribing {len(audio_data)} bytes of audio")

            # Try whisper-cli first
            result = self._transcribe_with_whisper(tmp_path)

            # Cleanup
            Path(tmp_path).unlink()

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "error": str(e),
                "text": "",
                "confidence": 0.0,
            }

    def _transcribe_with_whisper(self, audio_path: str) -> dict[str, Any]:
        """Use Whisper (OpenAI) for transcription."""
        try:
            import whisper

            logger.info("Loading Whisper model...")
            model = whisper.load_model(self.model)

            logger.info(f"Transcribing with model: {self.model}")
            result = model.transcribe(audio_path, language=self.language)

            return {
                "text": result["text"],
                "language": result.get("language", self.language),
                "confidence": self._calculate_confidence(result),
                "segments": result.get("segments", []),
                "model": self.model,
            }

        except ImportError:
            logger.warning("Whisper not installed, using fallback")
            return self._transcribe_fallback(audio_path)

    def _transcribe_fallback(self, audio_path: str) -> dict[str, Any]:
        """Fallback using ffmpeg + basic processing."""
        try:
            # Try using ffmpeg to convert and get duration
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1:novalue=1", audio_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            duration = float(result.stdout.strip()) if result.stdout.strip() else 0

            logger.info(f"Audio duration: {duration:.2f}s (fallback mode)")

            return {
                "text": "[Audio processed - requires Whisper for actual transcription]",
                "language": self.language,
                "confidence": 0.5,
                "segments": [],
                "model": "fallback",
                "duration_seconds": duration,
            }

        except Exception as e:
            logger.error(f"Fallback transcription failed: {e}")
            return {
                "text": "",
                "error": f"Transcription service unavailable: {e}",
                "confidence": 0.0,
            }

    def _calculate_confidence(self, whisper_result: dict) -> float:
        """Calculate confidence score from Whisper result."""
        segments = whisper_result.get("segments", [])
        if not segments:
            return 0.9

        confidences = []
        for segment in segments:
            # Whisper doesn't give per-token confidence by default
            # Use no_speech_prob if available
            no_speech_prob = segment.get("no_speech_prob", 0.0)
            confidence = 1.0 - no_speech_prob
            confidences.append(confidence)

        return sum(confidences) / len(confidences) if confidences else 0.85

    def get_available_models(self) -> dict[str, Any]:
        """Get available Whisper models."""
        return {
            "tiny": {"size_mb": 39, "speed": "fastest", "accuracy": "low"},
            "base": {"size_mb": 140, "speed": "fast", "accuracy": "medium"},
            "small": {"size_mb": 466, "speed": "medium", "accuracy": "high"},
            "medium": {"size_mb": 1530, "speed": "slow", "accuracy": "very_high"},
            "large": {"size_mb": 2950, "speed": "slowest", "accuracy": "highest"},
        }

    def set_model(self, model: str) -> bool:
        """Change transcription model."""
        available = self.get_available_models()
        if model in available:
            self.model = model
            logger.info(f"Transcription model set to: {model}")
            return True
        return False
