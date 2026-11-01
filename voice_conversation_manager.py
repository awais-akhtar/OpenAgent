"""G05-G06: Full voice conversation orchestration (listen→transcribe→process→respond→speak)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .voice_transcription_service import VoiceTranscriptionService
from .voice_synthesis_service import VoiceSynthesisService

logger = logging.getLogger(__name__)


class VoiceConversation:
    """Single voice conversation session."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.created_at = time.time()
        self.messages: list[dict[str, Any]] = []
        self.status = "active"

    def add_user_message(self, text: str, audio_duration: float = 0) -> None:
        """Record user's spoken message."""
        self.messages.append({
            "role": "user",
            "text": text,
            "timestamp": time.time(),
            "audio_duration": audio_duration,
        })

    def add_assistant_message(self, text: str, audio_bytes: bytes) -> None:
        """Record assistant's spoken response."""
        self.messages.append({
            "role": "assistant",
            "text": text,
            "timestamp": time.time(),
            "audio_length": len(audio_bytes),
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "status": self.status,
            "message_count": len(self.messages),
            "messages": self.messages,
            "duration_seconds": time.time() - self.created_at,
        }


class VoiceConversationManager:
    """Manage full voice conversations with transcription and synthesis."""

    def __init__(self):
        self.transcriber = VoiceTranscriptionService()
        self.synthesizer = VoiceSynthesisService()
        self.conversations: dict[str, VoiceConversation] = {}
        self.state_file = Path("voice_conversations_state.json")
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted conversation state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    for conv_id, conv_data in data.get("conversations", {}).items():
                        conv = VoiceConversation(conv_id)
                        conv.messages = conv_data.get("messages", [])
                        self.conversations[conv_id] = conv
            except Exception as e:
                logger.error(f"Failed to load conversation state: {e}")

    def _save_state(self) -> None:
        """Save conversation state to disk."""
        state = {
            "conversations": {
                cid: conv.to_dict()
                for cid, conv in self.conversations.items()
            }
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def create_conversation(self) -> dict[str, Any]:
        """Create a new voice conversation."""
        conv_id = str(uuid4())
        conv = VoiceConversation(conv_id)
        self.conversations[conv_id] = conv

        logger.info(f"Created voice conversation: {conv_id}")
        self._save_state()

        return {
            "conversation_id": conv_id,
            "status": "ready",
            "created_at": conv.created_at,
        }

    def process_audio_input(
        self,
        conversation_id: str,
        audio_data: bytes,
        audio_format: str = "wav",
    ) -> dict[str, Any]:
        """
        Process audio input: transcribe and return text.

        Full flow:
        1. Receive audio bytes
        2. Transcribe to text (G05)
        3. Return transcribed text
        """
        if conversation_id not in self.conversations:
            return {"error": f"Conversation {conversation_id} not found"}

        conv = self.conversations[conversation_id]

        logger.info(f"Processing audio input for {conversation_id}")

        try:
            # Step 1: Transcribe audio to text
            transcription_result = self.transcriber.transcribe_audio(audio_data, audio_format)

            if "error" in transcription_result:
                return {
                    "error": transcription_result["error"],
                    "transcription": "",
                }

            text = transcription_result.get("text", "")

            # Step 2: Store in conversation
            audio_duration = transcription_result.get("duration_seconds", 0)
            conv.add_user_message(text, audio_duration)

            logger.info(f"Transcribed: {text}")

            return {
                "transcription": text,
                "confidence": transcription_result.get("confidence", 0.8),
                "language": transcription_result.get("language", "en"),
                "segments": transcription_result.get("segments", []),
            }

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {"error": str(e)}

    def generate_speech_response(
        self,
        conversation_id: str,
        response_text: str,
    ) -> dict[str, Any]:
        """
        Generate speech from text response.

        Full flow:
        1. Take response text
        2. Synthesize to audio (G06)
        3. Return audio bytes
        """
        if conversation_id not in self.conversations:
            return {"error": f"Conversation {conversation_id} not found"}

        conv = self.conversations[conversation_id]

        logger.info(f"Generating speech response for {conversation_id}")

        try:
            # Step 1: Synthesize text to speech
            synthesis_result = self.synthesizer.synthesize_text(response_text)

            if "error" in synthesis_result:
                return {
                    "error": synthesis_result["error"],
                    "audio_bytes": b"",
                }

            audio_bytes = synthesis_result.get("audio_bytes", b"")

            # Step 2: Store in conversation
            conv.add_assistant_message(response_text, audio_bytes)

            logger.info(f"Generated {len(audio_bytes)} bytes of speech")

            return {
                "audio_bytes": audio_bytes,
                "duration_seconds": synthesis_result.get("duration_seconds", 0),
                "engine": synthesis_result.get("engine", "unknown"),
                "format": synthesis_result.get("format", "wav"),
            }

        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            return {"error": str(e)}

    def full_voice_conversation(
        self,
        conversation_id: str,
        audio_input: bytes,
        backend_response_text: str,
    ) -> dict[str, Any]:
        """
        Complete full-duplex voice conversation.

        Flow:
        1. Listen: Receive audio → transcribe (G05)
        2. Process: Backend processes transcribed text
        3. Respond: Generate speech from response (G06)

        Returns both transcription and audio response in one call.
        """
        if conversation_id not in self.conversations:
            return {"error": f"Conversation {conversation_id} not found"}

        try:
            logger.info(f"Full voice conversation for {conversation_id}")

            # Step 1: Process audio input (Listen + Transcribe)
            transcription = self.process_audio_input(
                conversation_id,
                audio_input,
                audio_format="wav",
            )

            if "error" in transcription:
                return {"error": f"Transcription failed: {transcription['error']}"}

            user_text = transcription.get("transcription", "")
            logger.info(f"User said: {user_text}")

            # Step 2: Backend processes (handled by caller)
            # Step 3: Generate speech response (Respond + Speak)
            synthesis = self.generate_speech_response(
                conversation_id,
                backend_response_text,
            )

            if "error" in synthesis:
                return {"error": f"Synthesis failed: {synthesis['error']}"}

            return {
                "user_transcription": user_text,
                "user_confidence": transcription.get("confidence", 0),
                "assistant_response": backend_response_text,
                "response_audio_bytes": synthesis.get("audio_bytes", b""),
                "response_duration": synthesis.get("duration_seconds", 0),
            }

        except Exception as e:
            logger.error(f"Full voice conversation failed: {e}")
            return {"error": str(e)}

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get conversation details."""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id].to_dict()
        return None

    def list_conversations(self) -> list[dict[str, Any]]:
        """List all conversations."""
        return [conv.to_dict() for conv in self.conversations.values()]

    def close_conversation(self, conversation_id: str) -> bool:
        """Close and finalize a conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].status = "closed"
            self._save_state()
            logger.info(f"Closed conversation: {conversation_id}")
            return True
        return False

    def get_transcription_models(self) -> dict[str, Any]:
        """Get available transcription models."""
        return {
            "models": self.transcriber.get_available_models(),
            "current": self.transcriber.model,
        }

    def get_synthesis_voices(self) -> dict[str, Any]:
        """Get available synthesis voices."""
        return {
            "voices": self.synthesizer.get_available_voices(),
            "current": self.synthesizer.voice,
        }

    def set_transcription_model(self, model: str) -> bool:
        """Change transcription model."""
        return self.transcriber.set_model(model)

    def set_synthesis_voice(self, voice: str) -> bool:
        """Change synthesis voice."""
        return self.synthesizer.set_voice(voice)
