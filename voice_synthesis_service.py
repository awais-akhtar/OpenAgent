"""G06: Voice synthesis service with real text-to-speech."""

from __future__ import annotations

import io
import logging
from typing import Any

logger = logging.getLogger(__name__)


class VoiceSynthesisService:
    """Convert text to speech using TTS engines."""

    def __init__(self):
        self.voice = "default"
        self.rate = 1.0  # Speed multiplier
        self.pitch = 1.0  # Pitch multiplier
        self.language = "en"

    def synthesize_text(self, text: str, voice: str | None = None) -> dict[str, Any]:
        """
        Synthesize text to audio.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (optional)

        Returns:
            Dict with audio data and metadata
        """
        if not text or not text.strip():
            return {
                "error": "Empty text",
                "audio_bytes": b"",
                "duration_seconds": 0,
            }

        try:
            logger.info(f"Synthesizing {len(text)} chars to speech")

            # Try piper TTS first (local, no API key needed)
            result = self._synthesize_with_piper(text, voice or self.voice)

            if result:
                return result

            # Fallback to gTTS (Google Text-to-Speech)
            result = self._synthesize_with_gtts(text)

            if result:
                return result

            # Last fallback
            return self._synthesize_fallback(text)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "error": str(e),
                "audio_bytes": b"",
                "duration_seconds": 0,
            }

    def _synthesize_with_piper(self, text: str, voice: str) -> dict[str, Any] | None:
        """Use Piper TTS (local, lightweight)."""
        try:
            import piper

            logger.info(f"Synthesizing with Piper voice: {voice}")

            # Generate audio
            audio_data = piper.synthesize(text, voice=voice, rate=self.rate)

            if audio_data:
                return {
                    "audio_bytes": audio_data,
                    "duration_seconds": len(audio_data) / (22050 * 2),  # Estimate
                    "voice": voice,
                    "engine": "piper",
                    "format": "wav",
                }

        except ImportError:
            logger.debug("Piper not installed")
        except Exception as e:
            logger.debug(f"Piper synthesis failed: {e}")

        return None

    def _synthesize_with_gtts(self, text: str) -> dict[str, Any] | None:
        """Use Google Text-to-Speech (requires internet)."""
        try:
            from gtts import gTTS

            logger.info("Synthesizing with Google TTS")

            tts = gTTS(text, lang=self.language, slow=False)

            # Convert to bytes
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            data = audio_bytes.getvalue()

            if data:
                return {
                    "audio_bytes": data,
                    "duration_seconds": len(text) * 0.1,  # Rough estimate
                    "voice": "google-default",
                    "engine": "gtts",
                    "format": "mp3",
                }

        except ImportError:
            logger.debug("gTTS not installed")
        except Exception as e:
            logger.debug(f"gTTS synthesis failed: {e}")

        return None

    def _synthesize_fallback(self, text: str) -> dict[str, Any]:
        """Fallback: return silent audio with metadata."""
        logger.warning("Using silent fallback for TTS")

        # Create minimal silent WAV (22.05kHz, 1 second)
        import struct

        sample_rate = 22050
        duration_seconds = max(len(text) * 0.05, 1.0)  # Roughly 50ms per char, min 1s
        num_samples = int(sample_rate * duration_seconds)

        # WAV header
        channels = 1
        sample_width = 2

        wav_data = io.BytesIO()

        # RIFF header
        wav_data.write(b"RIFF")
        wav_data.write(struct.pack("<I", 36 + num_samples * sample_width))
        wav_data.write(b"WAVE")

        # fmt subchunk
        wav_data.write(b"fmt ")
        wav_data.write(struct.pack("<I", 16))
        wav_data.write(struct.pack("<H", 1))  # PCM
        wav_data.write(struct.pack("<H", channels))
        wav_data.write(struct.pack("<I", sample_rate))
        wav_data.write(struct.pack("<I", sample_rate * channels * sample_width))
        wav_data.write(struct.pack("<H", channels * sample_width))
        wav_data.write(struct.pack("<H", 16))

        # data subchunk
        wav_data.write(b"data")
        wav_data.write(struct.pack("<I", num_samples * sample_width))

        # Silent samples
        for _ in range(num_samples):
            wav_data.write(struct.pack("<h", 0))

        return {
            "audio_bytes": wav_data.getvalue(),
            "duration_seconds": duration_seconds,
            "voice": "silent-fallback",
            "engine": "fallback",
            "format": "wav",
            "note": "Text-to-speech not available, returning silent audio",
        }

    def get_available_voices(self) -> dict[str, Any]:
        """Get available voice options."""
        return {
            "piper_voices": {
                "en-us-amy": "Amy (Female, US)",
                "en-us-libritts": "LibriTTS (Female, US)",
                "en-us-lessac": "LeSsac (Male, US)",
            },
            "google_voice": "default (requires internet)",
            "note": "Piper is local and fast, Google requires internet",
        }

    def set_voice(self, voice: str) -> bool:
        """Change voice."""
        self.voice = voice
        logger.info(f"Voice set to: {voice}")
        return True

    def set_rate(self, rate: float) -> bool:
        """Set speech rate (0.5 - 2.0)."""
        if 0.5 <= rate <= 2.0:
            self.rate = rate
            logger.info(f"Speech rate set to: {rate}")
            return True
        return False
