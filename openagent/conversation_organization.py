"""Durable local organization and context controls for conversations."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any
from uuid import uuid4

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class ConversationOrganization:
    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_preferences (
                    conversation_id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    folder TEXT NOT NULL DEFAULT '',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    compaction_mode TEXT NOT NULL DEFAULT 'adaptive',
                    compact_after INTEGER NOT NULL DEFAULT 20,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_preferences_project
                    ON conversation_preferences(project, archived, pinned, updated_at DESC);
                CREATE TABLE IF NOT EXISTS conversation_folders (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL,
                    color TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(project, owner, name)
                );
                CREATE TABLE IF NOT EXISTS conversation_compactions (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    source_message_count INTEGER NOT NULL,
                    preserved_message_count INTEGER NOT NULL,
                    digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_compactions_conversation
                    ON conversation_compactions(conversation_id, created_at DESC);
                """
            )

    def _conversation(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        return conversation

    def _ensure_preferences(self, conversation: dict[str, Any], *, owner: str = "local") -> None:
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversation_preferences
                    (conversation_id, project, owner, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation["id"], conversation["project"], owner or "local", utc_now()),
            )

    def update(self, conversation_id: str, *, patch: dict[str, Any], owner: str = "local") -> dict[str, Any]:
        conversation = self._conversation(conversation_id)
        self._ensure_preferences(conversation, owner=owner)
        allowed = {"folder", "pinned", "archived", "compaction_mode", "compact_after", "title"}
        unknown = set(patch) - allowed - {"owner", "project"}
        if unknown:
            raise ValueError(f"Unsupported conversation fields: {', '.join(sorted(unknown))}")
        mode = str(patch.get("compaction_mode", "adaptive"))
        if mode not in {"off", "adaptive", "aggressive", "manual"}:
            raise ValueError("compaction_mode must be off, adaptive, aggressive, or manual")
        compact_after = max(4, min(int(patch.get("compact_after", 20)), 500))
        now = utc_now()
        with self.store.session() as connection:
            if "title" in patch:
                title = str(patch["title"]).strip()
                if not title:
                    raise ValueError("title cannot be empty")
                connection.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, now, conversation_id))
            current = connection.execute("SELECT * FROM conversation_preferences WHERE conversation_id = ?", (conversation_id,)).fetchone()
            connection.execute(
                """
                UPDATE conversation_preferences SET folder = ?, pinned = ?, archived = ?,
                    compaction_mode = ?, compact_after = ?, owner = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    str(patch.get("folder", current["folder"])).strip(),
                    int(bool(patch.get("pinned", current["pinned"]))),
                    int(bool(patch.get("archived", current["archived"]))),
                    mode if "compaction_mode" in patch else current["compaction_mode"],
                    compact_after if "compact_after" in patch else current["compact_after"],
                    owner or current["owner"], now, conversation_id,
                ),
            )
        self._audit("conversation.organization.update", conversation, {key: patch[key] for key in patch if key in allowed})
        return self.get(conversation_id)

    def get(self, conversation_id: str) -> dict[str, Any]:
        conversation = self._conversation(conversation_id)
        self._ensure_preferences(conversation)
        with self.store.session() as connection:
            pref = connection.execute("SELECT * FROM conversation_preferences WHERE conversation_id = ?", (conversation_id,)).fetchone()
            count = connection.execute("SELECT COUNT(*) AS n FROM chat_messages WHERE conversation_id = ?", (conversation_id,)).fetchone()["n"]
        return {**conversation, "folder": pref["folder"], "pinned": bool(pref["pinned"]), "archived": bool(pref["archived"]),
                "compaction_mode": pref["compaction_mode"], "compact_after": pref["compact_after"], "message_count": count}

    def list(self, *, project: str = "default", include_archived: bool = False, archived_only: bool = False, folder: str = "",
             sort: str = "pinned", query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        conversations = self.store.list_conversations(project=project, limit=max(1, min(limit, 200)))
        records = [self.get(str(item["id"])) for item in conversations]
        records = [
            item for item in records
            if ((item["archived"] if archived_only else (include_archived or not item["archived"])))
            and (not folder or item["folder"] == folder)
        ]
        if sort == "semantic" and query.strip():
            for item in records:
                messages = self.store.list_chat_messages(str(item["id"]), limit=80)
                corpus = " ".join([str(item["title"]), *(str(message["content"]) for message in messages)])
                item["semantic_score"] = self._semantic_score(query, corpus)
            records.sort(key=lambda item: (item.get("semantic_score", 0), item["updated_at"]), reverse=True)
        elif sort == "oldest":
            records.sort(key=lambda item: item["updated_at"])
        elif sort == "recent":
            records.sort(key=lambda item: item["updated_at"], reverse=True)
        else:
            records.sort(key=lambda item: (item["pinned"], item["updated_at"]), reverse=True)
        return records

    def bulk(self, conversation_ids: list[str], *, action: str, folder: str = "", owner: str = "local") -> dict[str, Any]:
        ids = list(dict.fromkeys(item.strip() for item in conversation_ids if item.strip()))
        if not ids:
            raise ValueError("conversation_ids is required")
        if action not in {"pin", "unpin", "archive", "unarchive", "move", "delete"}:
            raise ValueError("Unsupported bulk action")
        changed = []
        for conversation_id in ids:
            if action == "delete":
                conversation = self._conversation(conversation_id)
                with self.store.session() as connection:
                    connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
                self._audit("conversation.delete", conversation, {})
                changed.append({"id": conversation_id, "deleted": True})
                continue
            patch: dict[str, Any] = {}
            if action in {"pin", "unpin"}:
                patch["pinned"] = action == "pin"
            elif action in {"archive", "unarchive"}:
                patch["archived"] = action == "archive"
            else:
                patch["folder"] = folder
            changed.append(self.update(conversation_id, patch=patch, owner=owner))
        return {"action": action, "changed": changed, "count": len(changed)}

    def create_folder(self, name: str, *, project: str = "default", owner: str = "local", color: str = "") -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Folder name is required")
        folder = {"id": uuid4().hex, "project": project, "owner": owner or "local", "name": name, "color": color.strip(), "created_at": utc_now()}
        with self.store.session() as connection:
            connection.execute("INSERT INTO conversation_folders (id, project, owner, name, color, created_at) VALUES (?, ?, ?, ?, ?, ?)", tuple(folder.values()))
        return folder

    def folders(self, *, project: str = "default", owner: str = "local") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM conversation_folders WHERE project = ? AND owner = ? ORDER BY name", (project, owner or "local")).fetchall()
        return [dict(row) for row in rows]

    def compact(self, conversation_id: str, *, preserve: int = 8, mode: str = "manual") -> dict[str, Any]:
        conversation = self._conversation(conversation_id)
        messages = self.store.list_chat_messages(conversation_id, limit=500)
        preserve = max(2, min(int(preserve), 100))
        source = messages[:-preserve] if len(messages) > preserve else []
        digest = "\n".join(f"{item['role']}: {self._compact_text(str(item['content']))}" for item in source)
        record = {"id": uuid4().hex, "conversation_id": conversation_id, "project": conversation["project"], "mode": mode,
                  "source_message_count": len(source), "preserved_message_count": min(len(messages), preserve), "digest": digest, "created_at": utc_now()}
        with self.store.session() as connection:
            connection.execute("INSERT INTO conversation_compactions (id, conversation_id, project, mode, source_message_count, preserved_message_count, digest, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(record.values()))
        self._audit("conversation.compact", conversation, {"source_message_count": len(source), "preserve": preserve, "mode": mode})
        return record

    def compactions(self, conversation_id: str) -> list[dict[str, Any]]:
        self._conversation(conversation_id)
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM conversation_compactions WHERE conversation_id = ? ORDER BY created_at DESC", (conversation_id,)).fetchall()
        return [dict(row) for row in rows]

    def branch_graph(self, conversation_id: str) -> dict[str, Any]:
        root = self._conversation(conversation_id)
        with self.store.session() as connection:
            rows = connection.execute(
                """SELECT * FROM conversation_branches WHERE project = ? AND
                   (source_conversation_id = ? OR branch_conversation_id = ?) ORDER BY created_at""",
                (root["project"], conversation_id, conversation_id),
            ).fetchall()
        edges = [dict(row) for row in rows]
        node_ids = {conversation_id, *(edge["source_conversation_id"] for edge in edges), *(edge["branch_conversation_id"] for edge in edges)}
        return {"root": conversation_id, "nodes": [self.get(node_id) for node_id in node_ids],
                "edges": [{"id": edge["id"], "source": edge["source_conversation_id"], "target": edge["branch_conversation_id"], "at_message_id": edge["at_message_id"], "created_at": edge["created_at"]} for edge in edges]}

    @staticmethod
    def _compact_text(value: str, limit: int = 240) -> str:
        value = re.sub(r"\s+", " ", value).strip()
        return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."

    @staticmethod
    def _semantic_score(query: str, corpus: str) -> float:
        query_terms = re.findall(r"[a-z0-9_]{2,}", query.lower())
        frequencies = Counter(re.findall(r"[a-z0-9_]{2,}", corpus.lower()))
        return sum(1.0 + min(frequencies[term], 8) / 8 for term in set(query_terms) if frequencies[term])

    def _audit(self, action: str, conversation: dict[str, Any], metadata: dict[str, Any]) -> None:
        self.store.add_audit_event(AuditEvent(actor="conversation-organization", action=action,
            resource=f"conversation:{conversation['id']}", project=str(conversation["project"]), metadata=metadata))
