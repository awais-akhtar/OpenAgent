"""Editor document lifecycle: versioned documents and autosave drafts.

Independent implementation of a document/draft subsystem for OpenAgent.
Documents keep an immutable version history so every edit can be inspected
or restored; editor drafts are lightweight autosave snapshots that get
reconciled against the live document before being promoted into a new
version. This is separate from the knowledge-base ingestion documents in
knowledge.py — these are user-authored, versioned editor documents.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from .models import utc_now
from .storage import OpenAgentStore


class DocumentError(RuntimeError):
    """Raised when a document or draft operation is invalid."""


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class DocumentManager:
    """Owns editor documents, their version history, and autosave drafts."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS editor_documents (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    title TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    doc_format TEXT NOT NULL DEFAULT 'markdown',
                    status TEXT NOT NULL DEFAULT 'active',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    folder TEXT NOT NULL DEFAULT '',
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_editor_documents_project
                    ON editor_documents(project, status, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_editor_documents_owner_project
                    ON editor_documents(owner, project, status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS editor_document_versions (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(doc_id, version),
                    FOREIGN KEY(doc_id) REFERENCES editor_documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS editor_document_events (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES editor_documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS editor_drafts (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    base_version INTEGER NOT NULL DEFAULT 0,
                    content TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES editor_documents(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_editor_drafts_doc
                    ON editor_drafts(doc_id, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_editor_drafts_owner_project
                    ON editor_drafts(owner, project, updated_at DESC);

                CREATE TABLE IF NOT EXISTS editor_draft_versions (
                    id TEXT PRIMARY KEY,
                    draft_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    base_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(draft_id) REFERENCES editor_drafts(id) ON DELETE CASCADE
                );
                """
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_event(self, connection: Any, doc_id: str, event_type: str, detail: str = "") -> None:
        connection.execute(
            "INSERT INTO editor_document_events (id, doc_id, event_type, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid4().hex, doc_id, event_type, detail, utc_now()),
        )

    def _row_to_document(self, row: Any, *, include_content: bool = True) -> dict[str, Any]:
        payload = {
            "doc_id": row["id"],
            "project": row["project"],
            "owner": row["owner"],
            "title": row["title"],
            "format": row["doc_format"],
            "status": row["status"],
            "tags": json.loads(row["tags_json"] or "[]"),
            "folder": row["folder"],
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if include_content:
            payload["content"] = row["content"]
        return payload

    def _get_document_row(self, connection: Any, doc_id: str, *, owner: str = "") -> Any:
        owner = (owner or "").strip()
        if owner:
            row = connection.execute(
                "SELECT * FROM editor_documents WHERE id = ? AND owner = ?",
                (doc_id, owner),
            ).fetchone()
        else:
            row = connection.execute("SELECT * FROM editor_documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None or row["status"] == "deleted":
            raise KeyError("document not found")
        return row

    def _row_to_draft(self, connection: Any, row: Any) -> dict[str, Any]:
        doc = connection.execute("SELECT version FROM editor_documents WHERE id = ?", (row["doc_id"],)).fetchone()
        current_version = doc["version"] if doc is not None else None
        return {
            "draft_id": row["id"],
            "doc_id": row["doc_id"],
            "project": row["project"],
            "owner": row["owner"],
            "content": row["content"],
            "base_version": row["base_version"],
            "current_doc_version": current_version,
            "conflict": current_version is not None and current_version != row["base_version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # Document lifecycle
    # ------------------------------------------------------------------

    def create_document(
        self,
        title: str,
        content: str = "",
        *,
        project: str = "default",
        owner: str = "local",
        doc_format: str = "markdown",
        tags: list[str] | None = None,
        folder: str = "",
    ) -> dict[str, Any]:
        title = (title or "").strip()
        if not title:
            raise ValueError("title is required")
        doc_id = uuid4().hex
        now = utc_now()
        tags_json = json.dumps(list(tags or []))
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO editor_documents
                    (id, project, owner, title, content, doc_format, tags_json, folder, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    doc_id,
                    project or "default",
                    owner or "local",
                    title,
                    content,
                    doc_format or "markdown",
                    tags_json,
                    folder or "",
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO editor_document_versions (id, doc_id, version, title, content, content_hash, created_at)
                VALUES (?, ?, 1, ?, ?, ?, ?)
                """,
                (uuid4().hex, doc_id, title, content, _content_hash(content), now),
            )
            self._record_event(connection, doc_id, "created")
            row = connection.execute("SELECT * FROM editor_documents WHERE id = ?", (doc_id,)).fetchone()
            return self._row_to_document(row)

    def get_document(self, doc_id: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._get_document_row(connection, doc_id, owner=owner)
            return self._row_to_document(row)

    def update_document(
        self,
        doc_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        if_version: int | None = None,
        owner: str = "",
    ) -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._get_document_row(connection, doc_id, owner=owner)
            if if_version is not None and int(if_version) != row["version"]:
                raise DocumentError(f"version conflict: expected {row['version']}, got {if_version}")
            new_title = title if title is not None else row["title"]
            new_content = content if content is not None else row["content"]
            new_version = row["version"] + 1
            now = utc_now()
            connection.execute(
                "UPDATE editor_documents SET title = ?, content = ?, version = ?, updated_at = ? WHERE id = ?",
                (new_title, new_content, new_version, now, doc_id),
            )
            connection.execute(
                """
                INSERT INTO editor_document_versions (id, doc_id, version, title, content, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (uuid4().hex, doc_id, new_version, new_title, new_content, _content_hash(new_content), now),
            )
            self._record_event(connection, doc_id, "updated", f"version={new_version}")
            row = connection.execute("SELECT * FROM editor_documents WHERE id = ?", (doc_id,)).fetchone()
            return self._row_to_document(row)

    def patch_document(
        self,
        doc_id: str,
        *,
        tags: list[str] | None = None,
        folder: str | None = None,
        title: str | None = None,
        owner: str = "",
    ) -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._get_document_row(connection, doc_id, owner=owner)
            new_tags = json.dumps(list(tags)) if tags is not None else row["tags_json"]
            new_folder = folder if folder is not None else row["folder"]
            new_title = title if title is not None else row["title"]
            now = utc_now()
            connection.execute(
                "UPDATE editor_documents SET tags_json = ?, folder = ?, title = ?, updated_at = ? WHERE id = ?",
                (new_tags, new_folder, new_title, now, doc_id),
            )
            self._record_event(connection, doc_id, "metadata_patched")
            row = connection.execute("SELECT * FROM editor_documents WHERE id = ?", (doc_id,)).fetchone()
            return self._row_to_document(row)

    def archive_document(self, doc_id: str, archived: bool = True, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            self._get_document_row(connection, doc_id, owner=owner)
            status = "archived" if archived else "active"
            connection.execute(
                "UPDATE editor_documents SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), doc_id),
            )
            self._record_event(connection, doc_id, "archived" if archived else "unarchived")
            return {"doc_id": doc_id, "status": status}

    def delete_document(self, doc_id: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            self._get_document_row(connection, doc_id, owner=owner)
            connection.execute(
                "UPDATE editor_documents SET status = 'deleted', updated_at = ? WHERE id = ?",
                (utc_now(), doc_id),
            )
            self._record_event(connection, doc_id, "deleted")
            return {"doc_id": doc_id, "status": "deleted"}

    def list_library(
        self,
        *,
        project: str = "default",
        query: str = "",
        tag: str = "",
        folder: str = "",
        status: str = "",
        owner: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        clauses = ["project = ?"]
        args: list[Any] = [project or "default"]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        if status:
            clauses.append("status = ?")
            args.append(status)
        else:
            clauses.append("status != 'deleted'")
        if folder:
            clauses.append("folder = ?")
            args.append(folder)
        if query:
            clauses.append("(title LIKE ? OR content LIKE ?)")
            like = f"%{query}%"
            args.extend([like, like])
        sql = "SELECT * FROM editor_documents WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC, rowid DESC LIMIT ?"
        args.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
            items = []
            for row in rows:
                doc = self._row_to_document(row, include_content=False)
                if tag and tag not in doc["tags"]:
                    continue
                items.append(doc)
            return {"items": items, "count": len(items)}

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------

    def list_versions(self, doc_id: str, *, owner: str = "") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            self._get_document_row(connection, doc_id, owner=owner)
            rows = connection.execute(
                "SELECT version, content_hash, created_at FROM editor_document_versions "
                "WHERE doc_id = ? ORDER BY version DESC",
                (doc_id,),
            ).fetchall()
            return [
                {"version": r["version"], "hash": r["content_hash"], "created_at": r["created_at"]}
                for r in rows
            ]

    def get_version(self, doc_id: str, version: int, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            self._get_document_row(connection, doc_id, owner=owner)
            row = connection.execute(
                "SELECT * FROM editor_document_versions WHERE doc_id = ? AND version = ?",
                (doc_id, version),
            ).fetchone()
            if row is None:
                raise KeyError("version not found")
            return {
                "version": row["version"],
                "title": row["title"],
                "content": row["content"],
                "hash": row["content_hash"],
                "created_at": row["created_at"],
            }

    def restore_version(self, doc_id: str, version: int, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._get_document_row(connection, doc_id, owner=owner)
            target = connection.execute(
                "SELECT * FROM editor_document_versions WHERE doc_id = ? AND version = ?",
                (doc_id, version),
            ).fetchone()
            if target is None:
                raise KeyError("version not found")
            new_version = row["version"] + 1
            now = utc_now()
            connection.execute(
                "UPDATE editor_documents SET title = ?, content = ?, version = ?, updated_at = ? WHERE id = ?",
                (target["title"], target["content"], new_version, now, doc_id),
            )
            connection.execute(
                """
                INSERT INTO editor_document_versions (id, doc_id, version, title, content, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (uuid4().hex, doc_id, new_version, target["title"], target["content"], target["content_hash"], now),
            )
            self._record_event(connection, doc_id, "restored", f"from_version={version} new_version={new_version}")
            return {"doc_id": doc_id, "restored_from": version, "new_version": new_version}

    # ------------------------------------------------------------------
    # Editor drafts (autosave)
    # ------------------------------------------------------------------

    def create_draft(
        self,
        doc_id: str,
        content: str,
        *,
        base_version: int = 0,
        project: str = "default",
        owner: str = "local",
        scope_owner: str = "",
    ) -> dict[str, Any]:
        with self.store.session() as connection:
            self._get_document_row(connection, doc_id, owner=scope_owner)
            draft_id = uuid4().hex
            now = utc_now()
            connection.execute(
                """
                INSERT INTO editor_drafts (id, doc_id, project, owner, base_version, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (draft_id, doc_id, project or "default", owner or "local", base_version, content, now, now),
            )
            connection.execute(
                "INSERT INTO editor_draft_versions (id, draft_id, content, base_version, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (uuid4().hex, draft_id, content, base_version, now),
            )
            return {"draft_id": draft_id, "doc_id": doc_id, "owner": owner or "local", "updated_at": now}

    def get_draft(self, draft_id: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            owner = (owner or "").strip()
            if owner:
                row = connection.execute(
                    "SELECT * FROM editor_drafts WHERE id = ? AND owner = ?",
                    (draft_id, owner),
                ).fetchone()
            else:
                row = connection.execute("SELECT * FROM editor_drafts WHERE id = ?", (draft_id,)).fetchone()
            if row is None:
                raise KeyError("draft not found")
            return self._row_to_draft(connection, row)

    def list_drafts(self, *, project: str = "default", doc_id: str = "", owner: str = "") -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project or "default"]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        if doc_id:
            clauses.append("doc_id = ?")
            args.append(doc_id)
        sql = "SELECT * FROM editor_drafts WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC, rowid DESC"
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
            return [self._row_to_draft(connection, row) for row in rows]

    def update_draft(
        self,
        draft_id: str,
        content: str,
        *,
        base_version: int | None = None,
        owner: str = "",
    ) -> dict[str, Any]:
        with self.store.session() as connection:
            owner = (owner or "").strip()
            if owner:
                row = connection.execute(
                    "SELECT * FROM editor_drafts WHERE id = ? AND owner = ?",
                    (draft_id, owner),
                ).fetchone()
            else:
                row = connection.execute("SELECT * FROM editor_drafts WHERE id = ?", (draft_id,)).fetchone()
            if row is None:
                raise KeyError("draft not found")
            new_base_version = base_version if base_version is not None else row["base_version"]
            now = utc_now()
            connection.execute(
                "UPDATE editor_drafts SET content = ?, base_version = ?, updated_at = ? WHERE id = ?",
                (content, new_base_version, now, draft_id),
            )
            connection.execute(
                "INSERT INTO editor_draft_versions (id, draft_id, content, base_version, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (uuid4().hex, draft_id, content, new_base_version, now),
            )
            row = connection.execute("SELECT * FROM editor_drafts WHERE id = ?", (draft_id,)).fetchone()
            return self._row_to_draft(connection, row)

    def list_draft_versions(self, draft_id: str, *, owner: str = "") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            self.get_draft(draft_id, owner=owner)
            rows = connection.execute(
                "SELECT * FROM editor_draft_versions WHERE draft_id = ? ORDER BY created_at ASC, rowid ASC",
                (draft_id,),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "draft_id": row["draft_id"],
                    "content": row["content"],
                    "base_version": row["base_version"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def delete_draft(self, draft_id: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            owner = (owner or "").strip()
            if owner:
                row = connection.execute(
                    "SELECT id FROM editor_drafts WHERE id = ? AND owner = ?",
                    (draft_id, owner),
                ).fetchone()
            else:
                row = connection.execute("SELECT id FROM editor_drafts WHERE id = ?", (draft_id,)).fetchone()
            if row is None:
                raise KeyError("draft not found")
            connection.execute("DELETE FROM editor_drafts WHERE id = ?", (draft_id,))
            return {"deleted": True, "draft_id": draft_id}
