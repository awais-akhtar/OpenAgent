"""Conversation service backed by local memory and the model gateway."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import secrets
from typing import Iterator

from .context import ContextEngine
from .model_gateway import ModelGateway
from .models import AuditEvent, ChatMessage, utc_now
from .storage import OpenAgentStore


class ChatService:
    def __init__(self, store: OpenAgentStore, gateway: ModelGateway | None = None):
        self.store = store
        self.context = ContextEngine(store)
        self.gateway = gateway or ModelGateway()
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_branches (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    source_conversation_id TEXT NOT NULL,
                    branch_conversation_id TEXT NOT NULL,
                    at_message_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_branches_project
                    ON conversation_branches(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS conversation_replays (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    replay_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def send(
        self,
        content: str,
        *,
        conversation_id: str | None = None,
        project: str = "default",
        provider: str = "local",
        model: str | None = None,
        allow_external: bool = False,
    ) -> dict[str, object]:
        conversation, user_message = self._record_user_message(
            content,
            conversation_id=conversation_id,
            project=project,
        )
        messages = self._build_model_messages(conversation["id"], project)
        result = self.gateway.complete(
            messages,
            provider_id=provider,
            model=model,
            allow_external=allow_external,
        )
        assistant_message = self._record_assistant_message(
            conversation,
            result.content,
            provider=result.provider,
            model=result.model,
            metadata=result.metadata,
            project=project,
        )
        return {
            "conversation": conversation,
            "conversation_id": conversation["id"],
            "messages": [user_message, assistant_message],
            "provider": result.provider,
            "model": result.model,
            "runtime_profile": result.metadata.get("smooth_runtime", {}),
            "routing": {
                "fallback_from": result.metadata.get("fallback_from", ""),
                "smooth_options": result.metadata.get("smooth_options", {}),
            },
        }

    def stream(
        self,
        content: str,
        *,
        conversation_id: str | None = None,
        project: str = "default",
        provider: str = "local",
        model: str | None = None,
        allow_external: bool = False,
    ) -> Iterator[dict[str, object]]:
        conversation, user_message = self._record_user_message(
            content,
            conversation_id=conversation_id,
            project=project,
        )
        yield {"type": "conversation", "conversation": conversation, "message": user_message}
        yield {
            "type": "status",
            "status": "started",
            "provider": provider,
            "model": model or "",
            "conversation_id": conversation["id"],
        }

        messages = self._build_model_messages(conversation["id"], project)
        parts: list[str] = []
        chunk_provider = provider
        chunk_model = model or ""
        chunk_metadata = {}
        for chunk in self.gateway.stream_complete(
            messages,
            provider_id=provider,
            model=model,
            allow_external=allow_external,
        ):
            if chunk.content:
                parts.append(chunk.content)
            chunk_provider = chunk.provider
            chunk_model = chunk.model
            chunk_metadata = chunk.metadata
            yield {
                "type": "chunk",
                "provider": chunk.provider,
                "model": chunk.model,
                "content": chunk.content,
                "done": chunk.done,
                "metadata": chunk.metadata,
            }

        assistant_message = self._record_assistant_message(
            conversation,
            "".join(parts),
            provider=chunk_provider,
            model=chunk_model,
            metadata=chunk_metadata,
            project=project,
        )
        yield {"type": "message", "message": assistant_message, "done": True}

    def _record_user_message(
        self,
        content: str,
        *,
        conversation_id: str | None,
        project: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        content = content.strip()
        if not content:
            raise ValueError("Message content is required.")
        conversation = self.store.ensure_conversation(
            conversation_id,
            title=content[:80],
            project=project,
        )
        user_message = self.store.add_chat_message(
            ChatMessage(
                conversation_id=str(conversation["id"]),
                role="user",
                content=content,
                provider="user",
                model="human",
            )
        )
        self.context.ingest(
            title=f"User message: {conversation['title']}",
            content=content,
            type="conversation",
            source="chat",
            project=project,
            metadata={"conversation_id": conversation["id"], "role": "user"},
        )
        return conversation, user_message

    def _record_assistant_message(
        self,
        conversation: dict[str, object],
        content: str,
        *,
        provider: str,
        model: str,
        metadata: dict[str, object],
        project: str,
    ) -> dict[str, object]:
        assistant_message = self.store.add_chat_message(
            ChatMessage(
                conversation_id=str(conversation["id"]),
                role="assistant",
                content=content,
                provider=provider,
                model=model,
                metadata=metadata,
            )
        )
        self.context.ingest(
            title=f"Assistant response: {conversation['title']}",
            content=content,
            type="conversation",
            source=f"model:{provider}",
            project=project,
            metadata={"conversation_id": conversation["id"], "role": "assistant"},
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="chat",
                action="model.complete",
                resource=f"conversation:{conversation['id']}",
                project=project,
                status="allowed",
                metadata={"provider": provider, "model": model, **metadata},
            )
        )
        return assistant_message

    def _build_model_messages(self, conversation_id: str, project: str) -> list[dict[str, str]]:
        snapshot = self.context.snapshot(project=project, limit=40)
        terms = ", ".join(item["term"] for item in snapshot["top_terms"][:8]) or "none yet"
        system = (
            "You are OpenAgent, a self-hosted privacy-first AI operating system. "
            "Respect local-first boundaries. "
            f"Project: {project}. Memory events: {snapshot['event_count']}. Top terms: {terms}."
        )
        history = self.store.list_chat_messages(conversation_id, limit=20)
        messages = [{"role": "system", "content": system}]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item["role"] in {"user", "assistant", "system"}
        )
        return messages

    def branch(
        self,
        conversation_id: str,
        *,
        at_message_id: str = "",
        title: str = "",
    ) -> dict[str, object]:
        source = self.store.get_conversation(conversation_id)
        if source is None:
            raise ValueError(f"Unknown conversation '{conversation_id}'.")
        messages = self.store.list_chat_messages(conversation_id, limit=500)
        selected = []
        for message in messages:
            selected.append(message)
            if at_message_id and message["id"] == at_message_id:
                break
        if at_message_id and (not selected or selected[-1]["id"] != at_message_id):
            raise ValueError(f"Unknown message '{at_message_id}' in conversation '{conversation_id}'.")
        branch_title = title.strip() or f"Branch: {source['title']}"
        branch_conversation = self.store.create_conversation(
            branch_title,
            project=str(source["project"]),
            conversation_id=unique_id("conversation-branch", conversation_id),
        )
        for message in selected:
            metadata = dict(message.get("metadata", {}))
            metadata.update(
                {
                    "branched_from_conversation_id": conversation_id,
                    "branched_from_message_id": message["id"],
                }
            )
            self.store.add_chat_message(
                ChatMessage(
                    conversation_id=str(branch_conversation["id"]),
                    role=message["role"],
                    content=message["content"],
                    provider=message["provider"],
                    model=message["model"],
                    metadata=metadata,
                )
            )
        branch_id = stable_id("branch", conversation_id, str(branch_conversation["id"]))
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO conversation_branches (
                    id, project, source_conversation_id, branch_conversation_id,
                    at_message_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    branch_id,
                    source["project"],
                    conversation_id,
                    branch_conversation["id"],
                    at_message_id,
                    created_at,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="chat",
                action="conversation.branch",
                resource=f"conversation:{conversation_id}",
                project=str(source["project"]),
                metadata={"branch_conversation_id": branch_conversation["id"], "message_count": len(selected)},
            )
        )
        return {
            "id": branch_id,
            "source": source,
            "conversation": branch_conversation,
            "messages": self.store.list_chat_messages(str(branch_conversation["id"]), limit=500),
            "created_at": created_at,
        }

    def search(self, query: str, *, project: str = "default", limit: int = 50) -> list[dict[str, object]]:
        query = query.strip()
        if not query:
            return []
        pattern = f"%{query.lower()}%"
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT m.*, c.title AS conversation_title, c.project
                FROM chat_messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.project = ?
                  AND (lower(m.content) LIKE ? OR lower(c.title) LIKE ?)
                ORDER BY m.created_at DESC
                LIMIT ?
                """,
                (project, pattern, pattern, max(1, min(limit, 200))),
            ).fetchall()
        return [
            {
                **self.store._chat_message_row_to_dict(row),  # noqa: SLF001 - shared row conversion.
                "conversation_title": row["conversation_title"],
                "project": row["project"],
            }
            for row in rows
        ]

    def export_conversation(self, conversation_id: str, *, write_file: bool = True) -> dict[str, object]:
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Unknown conversation '{conversation_id}'.")
        payload = {
            "format": "openagent.conversation.v1",
            "exported_at": utc_now(),
            "conversation": conversation,
            "messages": self.store.list_chat_messages(conversation_id, limit=500),
        }
        path = ""
        if write_file:
            export_dir = self.store.data_dir / "conversation_exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            path = str((export_dir / f"{conversation_id}.json").resolve())
            Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.store.add_audit_event(
            AuditEvent(
                actor="chat",
                action="conversation.export",
                resource=f"conversation:{conversation_id}",
                project=str(conversation["project"]),
                metadata={"path": path, "message_count": len(payload["messages"])},
            )
        )
        return {**payload, "path": path}

    def import_conversation(self, payload: dict[str, object], *, project: str = "default") -> dict[str, object]:
        conversation_payload = payload.get("conversation") if isinstance(payload.get("conversation"), dict) else {}
        messages_payload = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        title = str(conversation_payload.get("title", "Imported conversation")).strip() or "Imported conversation"
        conversation = self.store.create_conversation(
            f"Imported: {title}",
            project=project,
            conversation_id=unique_id("conversation-import", title),
        )
        imported_messages = []
        for message in messages_payload:
            if not isinstance(message, dict):
                continue
            imported_messages.append(
                self.store.add_chat_message(
                    ChatMessage(
                        conversation_id=str(conversation["id"]),
                        role=str(message.get("role", "user")),
                        content=str(message.get("content", "")),
                        provider=str(message.get("provider", "import")),
                        model=str(message.get("model", "imported")),
                        metadata={
                            "imported": True,
                            **(message.get("metadata") if isinstance(message.get("metadata"), dict) else {}),
                        },
                    )
                )
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="chat",
                action="conversation.import",
                resource=f"conversation:{conversation['id']}",
                project=project,
                metadata={"message_count": len(imported_messages)},
            )
        )
        return {"conversation": conversation, "messages": imported_messages}

    def replay(self, conversation_id: str) -> dict[str, object]:
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Unknown conversation '{conversation_id}'.")
        messages = self.store.list_chat_messages(conversation_id, limit=500)
        context_events = self.store.search_events(str(conversation["title"]), project=str(conversation["project"]), limit=20)
        replay = {
            "conversation": conversation,
            "timeline": [
                {
                    "type": "message",
                    "role": message["role"],
                    "content": message["content"],
                    "created_at": message["created_at"],
                    "provider": message["provider"],
                }
                for message in messages
            ],
            "context_events": context_events,
            "message_count": len(messages),
        }
        replay_id = stable_id("conversation-replay", conversation_id, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO conversation_replays (
                    id, project, conversation_id, replay_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (replay_id, conversation["project"], conversation_id, json.dumps(replay, sort_keys=True), created_at),
            )
        return {"id": replay_id, "project": conversation["project"], "replay": replay, "created_at": created_at}


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:32]


def unique_id(*parts: str) -> str:
    return stable_id(*parts, utc_now(), secrets.token_hex(8))
