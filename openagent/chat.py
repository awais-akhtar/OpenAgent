"""Conversation service backed by local memory and the model gateway."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Iterator

from .context import ContextEngine
from .media import MediaManager
from .model_gateway import ModelGateway, ModelResult, ProviderUnavailableError
from .models import AuditEvent, ChatMessage, utc_now
from .storage import OpenAgentStore


class ChatService:
    def __init__(self, store: OpenAgentStore, gateway: ModelGateway | None = None):
        self.store = store
        self.context = ContextEngine(store)
        self.gateway = gateway or ModelGateway()
        self.media = MediaManager(store)
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
        attachments: list[dict[str, Any]] | None = None,
        owner: str = "",
    ) -> dict[str, object]:
        # Provider names and the local/external boundary are request
        # validation, not generation failures. Resolve them before writing a
        # conversation turn so a rejected provider cannot leave a dangling
        # user message behind.
        self.gateway.select_provider(provider, allow_external=allow_external)
        attachments = self._resolve_attachments(attachments, project=project, owner=owner)
        conversation, user_message = self._record_user_message(
            content,
            conversation_id=conversation_id,
            project=project,
            attachments=attachments,
        )
        messages = self._build_model_messages(
            conversation["id"], project, query=content, attachments=attachments, owner=owner,
        )
        effective_attachments = attachments or self._historical_attachments(
            self.store.list_chat_messages(str(conversation["id"]), limit=5000), None,
            project=project, owner=owner,
        )
        result = self._complete_with_attachments(
            messages, attachments=effective_attachments, provider=provider, model=model,
            allow_external=allow_external, project=project, owner=owner,
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
        attachments: list[dict[str, Any]] | None = None,
        cancelled=None,
        owner: str = "",
    ) -> Iterator[dict[str, object]]:
        from .adaptive_engine.execution_service import ExecutionCancelled

        if cancelled is not None and not callable(cancelled):
            raise ValueError("stream cancellation must be callable")

        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise ExecutionCancelled("chat stream request cancelled")

        # This generator is lazy: callers can obtain it before any body runs.
        # Keep provider validation ahead of the first event and, critically,
        # ahead of persisting the user's message.
        check_cancelled()
        self.gateway.select_provider(provider, allow_external=allow_external)
        attachments = self._resolve_attachments(attachments, project=project, owner=owner)
        check_cancelled()
        conversation, user_message = self._record_user_message(
            content,
            conversation_id=conversation_id,
            project=project,
            attachments=attachments,
        )
        yield {"type": "conversation", "conversation": conversation, "message": user_message}
        yield {
            "type": "status",
            "status": "started",
            "provider": provider,
            "model": model or "",
            "conversation_id": conversation["id"],
        }

        check_cancelled()
        messages = self._build_model_messages(
            conversation["id"], project, query=content, attachments=attachments, owner=owner,
        )
        effective_attachments = attachments or self._historical_attachments(
            self.store.list_chat_messages(str(conversation["id"]), limit=5000), None,
            project=project, owner=owner,
        )
        attachment_context = self._attachment_context(effective_attachments, project=project, owner=owner)
        vision = self._vision_attachment(effective_attachments, project=project, owner=owner)
        if attachment_context and vision is None:
            check_cancelled()
            result = self._complete_with_attachments(
                messages, attachments=effective_attachments, provider=provider, model=model,
                allow_external=allow_external, cancelled=cancelled, project=project, owner=owner,
            )
            check_cancelled()
            yield {
                "type": "chunk", "provider": result.provider, "model": result.model,
                "content": result.content, "done": True, "metadata": result.metadata,
            }
            check_cancelled()
            assistant_message = self._record_assistant_message(
                conversation, result.content, provider=result.provider, model=result.model,
                metadata=result.metadata, project=project,
            )
            yield {"type": "message", "message": assistant_message, "done": True}
            return

        parts: list[str] = []
        chunk_provider = provider
        chunk_model = model or ""
        chunk_metadata = {}
        if vision is not None:
            image, media_type = vision
            response = self.gateway.stream_complete_vision(messages, image,
                media_type=media_type, provider_id=provider, model=model,
                allow_external=allow_external, cancelled=cancelled)
        else:
            response = self.gateway.stream_complete(messages, provider_id=provider,
                model=model, allow_external=allow_external, cancelled=cancelled)
        try:
            for chunk in response:
                check_cancelled()
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
        finally:
            close = getattr(response, 'close', None)
            if callable(close):
                # A provider iterator's close hook is best-effort cleanup when
                # iteration already failed or the caller closed this generator.
                # Do not replace that primary failure with a secondary teardown
                # error; when the stream otherwise completed, surface cleanup
                # failure so the caller does not mistake it for a clean finish.
                primary_error = sys.exc_info()[1]
                try:
                    close()
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    try:
                        primary_error.add_note(
                            "chat provider stream cleanup failed: " + type(cleanup_error).__name__
                        )
                    except BaseException:
                        pass

        check_cancelled()
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
        attachments: list[dict[str, Any]] | None = None,
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
                metadata={"attachments": self._attachment_records(attachments)},
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

    def record_research(
        self,
        query: str,
        report: str,
        *,
        conversation_id: str | None = None,
        project: str = "default",
        sources: list[dict[str, object]] | None = None,
        research_session_id: str = "",
    ) -> dict[str, object]:
        """Persist a completed research exchange in ordinary chat history."""

        conversation, user_message = self._record_user_message(
            query, conversation_id=conversation_id, project=project,
        )
        source_records = sources or []
        assistant_message = self._record_assistant_message(
            conversation,
            report,
            provider="research",
            model="source-backed-research",
            metadata={
                "research_session_id": research_session_id,
                "source_count": len(source_records),
                "sources": source_records[:20],
            },
            project=project,
        )
        return {
            "conversation": conversation,
            "conversation_id": conversation["id"],
            "messages": [user_message, assistant_message],
        }

    def _build_model_messages(
        self,
        conversation_id: str,
        project: str,
        *,
        query: str = "",
        attachments: list[dict[str, Any]] | None = None,
        owner: str = "",
    ) -> list[dict[str, str]]:
        snapshot = self.context.snapshot(project=project, limit=40)
        terms = ", ".join(item["term"] for item in snapshot["top_terms"][:8]) or "none yet"
        history = self.store.list_chat_messages(conversation_id, limit=5000)
        attachment_text = self._attachment_context(attachments, project=project, owner=owner)
        historical_attachments = self._historical_attachments(
            history, attachments, project=project, owner=owner,
        )
        historical_attachment_text = self._attachment_context(
            historical_attachments, project=project, owner=owner,
        )
        has_file_evidence = bool(attachment_text or historical_attachment_text)
        evidence = self.context.search(query, project=project, limit=6) if query.strip() and not has_file_evidence else []
        evidence_text = "\n".join(
            f"- {item.get('title') or item.get('type') or 'Local source'}: {str(item.get('content', ''))[:700]}"
            for item in evidence
            if str(item.get("content", "")).strip()
        )
        system = (
            "You are OpenAgent, a self-hosted privacy-first AI operating system. "
            "Respect local-first boundaries. "
            f"Project: {project}. Memory events: {snapshot['event_count']}. Top terms: {terms}. "
            "Use retrieved local evidence when it is relevant, but never pretend it is current web research. "
            "Infer the user's intent when spelling or grammar is imperfect and ask only when ambiguity changes the answer."
        )
        if evidence_text:
            system += f"\n\nRetrieved local knowledge for this request:\n{evidence_text}"
        if attachment_text:
            system += (
                "\n\nFiles attached to the current message were processed locally. Their evidence is included in the current user turn. "
                "Read it and answer from it; never claim that attachments are unavailable. State a limitation only when "
                "the supplied extraction is genuinely insufficient."
            )
        elif historical_attachment_text:
            system += (
                "\n\nFiles from earlier turns in this conversation remain available locally. Use the relevant file "
                "when the user refers to 'this file', its filename, or asks a follow-up about its contents. The selected "
                "file evidence is included in the current user turn."
            )
        project_memory = self._project_conversation_memory(
            project=project,
            current_conversation_id=conversation_id,
            current_query=query,
        )
        if project_memory:
            system += (
                "\n\nRecent memory from this project and other conversations follows. "
                "Use it to resolve follow-up references and stable user preferences; ignore unrelated items.\n"
                + project_memory
            )
        recent_history = history[-100:]
        earlier_memory = self._select_conversation_memory(history[:-100], query=query, limit=48)
        if earlier_memory:
            system += (
                "\n\nDurable memory selected from earlier turns in this conversation. These turns remain stored even "
                "after the app is closed; use them when they answer a follow-up.\n" + earlier_memory
            )
        messages = [{"role": "system", "content": system}]
        turn_attachment_text = attachment_text or historical_attachment_text
        for item in recent_history:
            if item["role"] not in {"user", "assistant", "system"}:
                continue
            content = str(item["content"])
            if turn_attachment_text and item is recent_history[-1] and item["role"] == "user":
                content += "\n\n<attached_evidence>\n" + turn_attachment_text + "\n</attached_evidence>"
            messages.append({"role": item["role"], "content": content})
        return messages

    def _complete_with_attachments(
        self,
        messages: list[dict[str, str]],
        *,
        attachments: list[dict[str, Any]] | None,
        provider: str,
        model: str | None,
        allow_external: bool,
        cancelled=None,
        project: str = "default",
        owner: str = "",
    ) -> ModelResult:
        vision = self._vision_attachment(attachments, project=project, owner=owner)
        attachment_context = self._attachment_context(attachments, project=project, owner=owner)
        if vision is not None:
            image, media_type = vision
            try:
                result = self.gateway.complete_vision(
                    messages, image, media_type=media_type, provider_id=provider,
                    model=model, allow_external=allow_external,
                )
                if attachment_context and _denies_attachment_access(result.content):
                    return self.gateway.complete_vision(
                        self._attachment_correction(messages), image, media_type=media_type,
                        provider_id=provider, model=model, allow_external=allow_external,
                    )
                return result
            except ProviderUnavailableError as exc:
                messages = [*messages, {
                    "role": "system",
                    "content": "Native image understanding was unavailable for this turn. Use the locally extracted "
                    f"image metadata already supplied and explain the exact limitation briefly. Runtime detail: {exc}",
                }]
        completion_options = {
            "provider_id": provider,
            "model": model,
            "allow_external": allow_external,
        }
        if cancelled is not None:
            completion_options["cancelled"] = cancelled
        result = self.gateway.complete(messages, **completion_options)
        if attachment_context and _denies_attachment_access(result.content):
            if cancelled is not None and cancelled():
                from .adaptive_engine.execution_service import ExecutionCancelled
                raise ExecutionCancelled("chat stream request cancelled")
            return self.gateway.complete(
                self._attachment_correction(messages), provider_id=provider,
                model=model, allow_external=allow_external, **({"cancelled": cancelled} if cancelled is not None else {}),
            )
        return result

    @staticmethod
    def _attachment_correction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [*messages, {
            "role": "system",
            "content": "Correction: the attachment was already processed and its readable contents are inside the "
            "current user message under <attached_evidence>. Answer directly from that evidence. Do not say that you "
            "cannot access or process the attachment, and do not ask the user to paste text that is already present.",
        }]

    def _vision_attachment(
        self, attachments: list[dict[str, Any]] | None, *, project: str = "default", owner: str = "",
    ) -> tuple[bytes, str] | None:
        for item in attachments or []:
            if not isinstance(item, dict) or str(item.get("media_type", "")) != "image":
                continue
            asset_id = str(item.get("id") or item.get("asset_id") or "").strip()
            asset = self.media.get_asset(asset_id, project=project, owner=owner) if asset_id else None
            stored_path = str((asset or {}).get("stored_path", ""))
            path = Path(stored_path) if stored_path else None
            if path and path.is_file():
                maximum = 256*1024**2
                with path.open('rb') as source:
                    before = os.fstat(source.fileno())
                    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
                        raise ValueError('Image attachment must be a non-empty regular file within 256 MiB.')
                    data = source.read(before.st_size+1)
                    after = os.fstat(source.fileno())
                current = path.stat()
                def stamp(info):
                    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                def content_identity(info):
                    # Windows can report a small ctime discrepancy between a
                    # still-open handle and a path lookup even when the file
                    # identity/content is unchanged. The asset's stored hash
                    # below is the content authority; retain identity, size,
                    # and mtime checks here without treating that metadata-only
                    # discrepancy as an attachment mutation.
                    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                expected_hash = (asset or {}).get('content_hash')
                if (len(data) != before.st_size or stamp(before) != stamp(after) or
                        content_identity(before) != content_identity(current) or
                        type(expected_hash) is not str or
                        not re.fullmatch(r'[0-9a-f]{64}', expected_hash) or
                        hashlib.sha256(data).hexdigest() != expected_hash):
                    raise ValueError('Image attachment changed while loading.')
                return data, str((asset or {}).get("mime_type") or item.get("mime_type") or "image/png")
        return None

    @staticmethod
    def _select_conversation_memory(
        messages: list[dict[str, Any]], *, query: str, limit: int = 48,
    ) -> str:
        if not messages:
            return ""
        query_terms = {term for term in _memory_terms(query) if len(term) > 2}
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for index, message in enumerate(messages):
            content = " ".join(str(message.get("content", "")).split())
            if not content:
                continue
            terms = set(_memory_terms(content))
            score = len(query_terms & terms) * 20
            if message.get("role") == "user":
                score += 3
            if any(marker in content.lower() for marker in ("remember", "prefer", "my name", "always", "never", "important")):
                score += 12
            ranked.append((score, index, message))
        selected = sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)[:max(1, min(limit, 80))]
        selected.sort(key=lambda item: item[1])
        return "\n".join(
            f"- {str(item.get('role', 'message'))}: {' '.join(str(item.get('content', '')).split())[:700]}"
            for _, _, item in selected
        )

    @staticmethod
    def _attachment_records(attachments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        records = []
        for item in attachments or []:
            if not isinstance(item, dict):
                continue
            records.append({
                "id": str(item.get("id") or item.get("asset_id") or ""),
                "name": str(item.get("name") or item.get("filename") or "attachment"),
                "media_type": str(item.get("media_type") or "document"),
                "mime_type": str(item.get("mime_type") or ""),
            })
        return records[:12]

    def _resolve_attachments(
        self, attachments: list[dict[str, Any]] | None, *, project: str = "default", owner: str = "",
    ) -> list[dict[str, Any]]:
        image_count = sum(
            1 for item in attachments or []
            if isinstance(item, dict) and item.get("media_type") == "image"
        )
        if image_count > 1:
            # The gateway's vision contract accepts exactly one image. Do not
            # silently process only the first image while recording all of
            # them as part of the user's turn.
            raise ValueError(
                "Chat currently supports one image attachment per turn; "
                "multiple images cannot be sent to the vision provider."
            )
        resolved = []
        for item in attachments or []:
            if not isinstance(item, dict):
                continue
            record = dict(item)
            asset_id = str(record.get("id") or record.get("asset_id") or "").strip()
            extracted = str(record.get("extracted_text") or record.get("content") or "").strip()
            if asset_id and not _usable_attachment_text(extracted):
                asset = self.media.get_asset(asset_id, project=project, owner=owner) or {}
                analyses = self.media.analyses(
                    project=str(asset.get("project") or project), asset_id=asset_id, limit=1, owner=owner,
                )
                analysis = analyses[0] if analyses else {}
                candidate = str(analysis.get("extracted_text", "")).strip()
                stored_suffix = Path(str(asset.get("stored_path", ""))).suffix.lower()
                if str(asset.get("media_type", record.get("media_type", ""))) == "document" and not _usable_attachment_text(candidate) and stored_suffix == ".pdf":
                    try:
                        analysis = self.media.analyze_asset(asset_id, project=project, owner=owner)
                        candidate = str(analysis.get("extracted_text", "")).strip()
                    except (OSError, ValueError):
                        candidate = ""
                if _usable_attachment_text(candidate):
                    record["extracted_text"] = candidate[:50_000]
                    record["summary"] = str(analysis.get("summary", record.get("summary", "")))
            resolved.append(record)
        return resolved[:12]

    def _historical_attachments(
        self, history: list[dict[str, Any]], current: list[dict[str, Any]] | None,
        *, project: str = "default", owner: str = "",
    ) -> list[dict[str, Any]]:
        current_ids = {str(item.get("id") or item.get("asset_id") or "") for item in current or [] if isinstance(item, dict)}
        selected = []
        seen = set(current_ids)
        for message in reversed(history):
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            for item in reversed(metadata.get("attachments", []) if isinstance(metadata.get("attachments"), list) else []):
                if not isinstance(item, dict):
                    continue
                asset_id = str(item.get("id") or item.get("asset_id") or "").strip()
                if not asset_id or asset_id in seen:
                    continue
                seen.add(asset_id)
                selected.append(item)
                if len(selected) >= 3:
                    return self._resolve_attachments(selected, project=project, owner=owner)
        return self._resolve_attachments(selected, project=project, owner=owner)

    def _attachment_context(
        self, attachments: list[dict[str, Any]] | None, *, project: str = "default", owner: str = "",
    ) -> str:
        blocks = []
        for index, item in enumerate(
            self._resolve_attachments(attachments, project=project, owner=owner), start=1,
        ):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("filename") or f"attachment-{index}")
            summary = " ".join(str(item.get("summary") or "").split())
            extracted = str(item.get("extracted_text") or item.get("content") or "").strip()
            findings = item.get("findings") if isinstance(item.get("findings"), dict) else {}
            details = json.dumps(findings, sort_keys=True)[:1800] if findings else ""
            evidence = extracted[:14_000] if _usable_attachment_text(extracted) else summary or details or "No readable text was extracted."
            blocks.append(f"[Attachment {index}: {name}]\n{evidence}")
        return "\n\n".join(blocks[:12])

    def _project_conversation_memory(
        self,
        *,
        project: str,
        current_conversation_id: str,
        current_query: str,
        limit: int = 12,
    ) -> str:
        events = self.store.list_events(project=project, limit=160)
        relevant = self.context.search(current_query, project=project, limit=12) if current_query.strip() else []
        ordered = [*relevant, *events]
        selected: list[str] = []
        seen: set[str] = set()
        for item in ordered:
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if str(item.get("type", "")) != "conversation" or metadata.get("role") != "user":
                continue
            content = " ".join(str(item.get("content", "")).split())
            if not content or content.casefold() == current_query.strip().casefold():
                continue
            key = content.casefold()
            if key in seen:
                continue
            seen.add(key)
            origin = "this conversation" if metadata.get("conversation_id") == current_conversation_id else "another conversation"
            selected.append(f"- ({origin}) {content[:500]}")
            if len(selected) >= max(1, min(limit, 30)):
                break
        return "\n".join(selected)

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
        messages = self.store.list_chat_messages(conversation_id, limit=5000)
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


def _memory_terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_'-]{1,}", str(value).lower())


def _denies_attachment_access(value: str) -> bool:
    text = " ".join(str(value).lower().split())
    return bool(re.search(
        r"\b(?:unable|cannot|can't|can not)\b.{0,80}\b(?:access|process|read|open)\b.{0,50}\battachments?\b",
        text,
    ))


def _usable_attachment_text(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    printable = sum(character.isprintable() or character in "\n\r\t" for character in text)
    controls = sum(ord(character) < 32 and character not in "\n\r\t" for character in text)
    letters = sum(character.isalpha() for character in text)
    return printable / len(text) >= 0.92 and letters / len(text) >= 0.20 and controls == 0
