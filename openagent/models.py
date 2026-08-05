"""Shared data models for OpenAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ContextEvent:
    title: str
    content: str
    type: str = "note"
    source: str = "manual"
    project: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "project": self.project,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AgentResponse:
    agent: str
    goal: str
    response: str
    next_actions: list[str]
    confidence: float
    context_used: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_record(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "goal": self.goal,
            "response": self.response,
            "next_actions": self.next_actions,
            "confidence": self.confidence,
            "context_used": self.context_used,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ChatMessage:
    conversation_id: str
    role: str
    content: str
    provider: str = "local"
    model: str = "openagent-local"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AuditEvent:
    actor: str
    action: str
    resource: str
    project: str = "default"
    status: str = "allowed"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "project": self.project,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
