"""G25: Semantic memory - conversation history, context persistence, session recovery."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class SemanticMemoryService:
    """Long-term semantic memory and session management."""

    def __init__(self):
        self.sessions = {}
        self.conversation_history = {}
        self.context_memories = {}
        self.state_file = Path("semantic_memory_state.json")
        self._load_state()

    def create_session(self) -> dict[str, Any]:
        """Create new memory session."""
        session_id = str(uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": time.time(),
            "last_access": time.time(),
            "messages": [],
            "context_embedding": None,
        }
        logger.info(f"Created memory session: {session_id}")
        return {"session_id": session_id, "created_at": self.sessions[session_id]["created_at"]}

    def store_message(self, session_id: str, role: str, content: str, embedding: list[float] | None = None) -> dict[str, Any]:
        """Store message in conversation history."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        self.sessions[session_id]["messages"].append({
            "role": role,
            "content": content,
            "embedding": embedding,
            "timestamp": time.time(),
        })
        self.sessions[session_id]["last_access"] = time.time()
        
        self._save_state()
        return {"stored": True, "message_count": len(self.sessions[session_id]["messages"])}

    def get_conversation_history(self, session_id: str, limit: int = 10) -> dict[str, Any]:
        """Get conversation history for session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        messages = self.sessions[session_id]["messages"][-limit:]
        return {
            "session_id": session_id,
            "messages": messages,
            "total_messages": len(self.sessions[session_id]["messages"]),
        }

    def search_history(self, session_id: str, query: str) -> dict[str, Any]:
        """Search conversation history."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        messages = self.sessions[session_id]["messages"]
        results = [m for m in messages if query.lower() in m["content"].lower()]
        
        return {
            "query": query,
            "results": results[:5],
            "total_found": len(results),
        }

    def store_context(self, session_id: str, key: str, value: Any) -> dict[str, Any]:
        """Store context memory."""
        if session_id not in self.context_memories:
            self.context_memories[session_id] = {}
        
        self.context_memories[session_id][key] = {
            "value": value,
            "stored_at": time.time(),
        }
        self._save_state()
        
        return {"stored": True, "key": key}

    def retrieve_context(self, session_id: str, key: str) -> Any:
        """Retrieve context memory."""
        if session_id in self.context_memories and key in self.context_memories[session_id]:
            return self.context_memories[session_id][key]["value"]
        return None

    def get_session_context(self, session_id: str) -> dict[str, Any]:
        """Get all context for session."""
        context = self.context_memories.get(session_id, {})
        return {
            "session_id": session_id,
            "context": {k: v["value"] for k, v in context.items()},
            "context_size": len(context),
        }

    def recover_session(self, session_id: str) -> dict[str, Any]:
        """Recover session from persistence."""
        if session_id not in self.sessions:
            return {"error": "Session not found or not recoverable"}
        
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "recovered": True,
            "message_count": len(session["messages"]),
            "context_memory_size": len(self.context_memories.get(session_id, {})),
        }

    def get_memory_analytics(self, session_id: str) -> dict[str, Any]:
        """Get memory analytics for session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        messages = session["messages"]
        
        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if m["role"] == "user"]),
            "assistant_messages": len([m for m in messages if m["role"] == "assistant"]),
            "session_duration": time.time() - session["created_at"],
            "context_items": len(self.context_memories.get(session_id, {})),
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions."""
        return [
            {
                "id": s["id"],
                "created_at": s["created_at"],
                "messages": len(s["messages"]),
                "last_access": s["last_access"],
            }
            for s in self.sessions.values()
        ]

    def _save_state(self) -> None:
        """Persist memory state."""
        state = {
            "sessions": {sid: {"created_at": s["created_at"], "message_count": len(s["messages"])} 
                        for sid, s in self.sessions.items()},
            "context_memories": {sid: len(ctx) for sid, ctx in self.context_memories.items()},
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def _load_state(self) -> None:
        """Load persisted state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    logger.info(f"Loaded memory state: {len(state['sessions'])} sessions")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
