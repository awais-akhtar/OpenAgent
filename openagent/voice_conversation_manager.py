"""G05-G06: Full voice conversation orchestration."""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

class VoiceConversation:
    """Single voice conversation session."""
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.created_at = time.time()
        self.messages: list[dict[str, Any]] = []
        self.status = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "status": self.status,
            "message_count": len(self.messages),
            "messages": self.messages,
        }

class VoiceConversationManager:
    """Manage full voice conversations (listen→transcribe→process→respond→speak)."""

    def __init__(self):
        from .voice_transcription_service import VoiceTranscriptionService
        from .voice_synthesis_service import VoiceSynthesisService
        
        self.transcriber = VoiceTranscriptionService()
        self.synthesizer = VoiceSynthesisService()
        self.conversations: dict[str, VoiceConversation] = {}
        self.state_file = Path("voice_conversations_state.json")

    def create_conversation(self) -> dict[str, Any]:
        """Create new voice conversation."""
        conv_id = str(uuid4())
        conv = VoiceConversation(conv_id)
        self.conversations[conv_id] = conv
        logger.info(f"Created voice conversation: {conv_id}")
        return {"conversation_id": conv_id, "status": "ready", "created_at": conv.created_at}

    def process_audio_input(self, conv_id: str, audio_data: bytes) -> dict[str, Any]:
        """Process audio: transcribe to text."""
        if conv_id not in self.conversations:
            return {"error": f"Conversation {conv_id} not found"}
        
        logger.info(f"Processing audio input for {conv_id}")
        result = self.transcriber.transcribe_audio(audio_data)
        
        if "error" in result:
            return {"error": result["error"], "transcription": ""}
        
        text = result.get("text", "")
        self.conversations[conv_id].messages.append({
            "role": "user",
            "text": text,
            "timestamp": time.time()
        })
        
        return {
            "transcription": text,
            "confidence": result.get("confidence", 0.8),
            "language": "en"
        }

    def generate_speech_response(self, conv_id: str, response_text: str) -> dict[str, Any]:
        """Generate speech from text response."""
        if conv_id not in self.conversations:
            return {"error": f"Conversation {conv_id} not found"}
        
        logger.info(f"Generating speech response for {conv_id}")
        result = self.synthesizer.synthesize_text(response_text)
        
        if "error" in result:
            return {"error": result["error"], "audio_bytes": b""}
        
        audio_bytes = result.get("audio_bytes", b"")
        self.conversations[conv_id].messages.append({
            "role": "assistant",
            "text": response_text,
            "timestamp": time.time()
        })
        
        return {
            "audio_bytes": audio_bytes,
            "duration_seconds": result.get("duration_seconds", 0),
            "engine": result.get("engine", "unknown")
        }

    def full_voice_conversation(self, conv_id: str, audio_input: bytes, response_text: str) -> dict[str, Any]:
        """Complete full-duplex: listen + transcribe + respond + speak."""
        if conv_id not in self.conversations:
            return {"error": f"Conversation {conv_id} not found"}
        
        logger.info(f"Full voice conversation for {conv_id}")
        
        transcription = self.process_audio_input(conv_id, audio_input)
        if "error" in transcription:
            return {"error": f"Transcription failed: {transcription['error']}"}
        
        synthesis = self.generate_speech_response(conv_id, response_text)
        if "error" in synthesis:
            return {"error": f"Synthesis failed: {synthesis['error']}"}
        
        return {
            "user_transcription": transcription.get("transcription", ""),
            "user_confidence": transcription.get("confidence", 0),
            "assistant_response": response_text,
            "response_audio_bytes": synthesis.get("audio_bytes", b""),
            "response_duration": synthesis.get("duration_seconds", 0)
        }

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        """Get conversation details."""
        return self.conversations.get(conv_id).to_dict() if conv_id in self.conversations else None

    def list_conversations(self) -> list[dict[str, Any]]:
        """List all conversations."""
        return [conv.to_dict() for conv in self.conversations.values()]

    def close_conversation(self, conv_id: str) -> bool:
        """Close a conversation."""
        if conv_id in self.conversations:
            self.conversations[conv_id].status = "closed"
            logger.info(f"Closed conversation: {conv_id}")
            return True
        return False
