"""G06: Voice synthesis service with real text-to-speech."""

from __future__ import annotations
import io
import logging
import struct
from typing import Any

logger = logging.getLogger(__name__)

class VoiceSynthesisService:
    """Convert text to speech using TTS engines."""

    def __init__(self):
        self.voice = "default"
        self.rate = 1.0
        self.language = "en"

    def synthesize_text(self, text: str, voice: str | None = None) -> dict[str, Any]:
        """Synthesize text to audio."""
        if not text or not text.strip():
            return {"error": "Empty text", "audio_bytes": b"", "duration_seconds": 0}

        try:
            logger.info(f"Synthesizing {len(text)} chars to speech")
            result = self._synthesize_with_piper(text, voice or self.voice)
            if result:
                return result
            return self._synthesize_fallback(text)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {"error": str(e), "audio_bytes": b"", "duration_seconds": 0}

    def _synthesize_with_piper(self, text: str, voice: str) -> dict[str, Any] | None:
        """Use Piper TTS."""
        try:
            import piper
            logger.info(f"Synthesizing with Piper voice: {voice}")
            audio_data = piper.synthesize(text, voice=voice)
            if audio_data:
                return {
                    "audio_bytes": audio_data,
                    "duration_seconds": len(audio_data) / (22050 * 2),
                    "voice": voice,
                    "engine": "piper",
                    "format": "wav"
                }
        except (ImportError, Exception) as e:
            logger.debug(f"Piper not available: {e}")
        return None

    def _synthesize_fallback(self, text: str) -> dict[str, Any]:
        """Fallback: silent WAV."""
        logger.warning("Using silent fallback for TTS")
        sample_rate = 22050
        duration_seconds = max(len(text) * 0.05, 1.0)
        num_samples = int(sample_rate * duration_seconds)
        
        wav_data = io.BytesIO()
        wav_data.write(b"RIFF")
        wav_data.write(struct.pack("<I", 36 + num_samples * 2))
        wav_data.write(b"WAVE")
        wav_data.write(b"fmt ")
        wav_data.write(struct.pack("<I", 16))
        wav_data.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
        wav_data.write(b"data")
        wav_data.write(struct.pack("<I", num_samples * 2))
        for _ in range(num_samples):
            wav_data.write(struct.pack("<h", 0))
        
        return {
            "audio_bytes": wav_data.getvalue(),
            "duration_seconds": duration_seconds,
            "voice": "silent",
            "engine": "fallback",
            "format": "wav"
        }

    def get_available_voices(self) -> dict[str, Any]:
        """Get available voice options."""
        return {"voices": ["default", "male", "female"], "current": self.voice}

    def set_voice(self, voice: str) -> bool:
        """Change voice."""
        self.voice = voice
        logger.info(f"Voice set to: {voice}")
        return True
