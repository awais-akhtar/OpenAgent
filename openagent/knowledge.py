"""Stage 2 local knowledge and memory engine."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from .context import STOPWORDS, TOKEN_RE
from .models import AuditEvent, ContextEvent, utc_now
from .storage import OpenAgentStore
from .vector_store import VectorStore

ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|https?://\S+|\d{4}-\d{2}-\d{2})\b"
)


class KnowledgeEngine:
    """Owns local document memory, retrieval, graph, compression, and replay."""

    def __init__(self, store: OpenAgentStore, vector_store: VectorStore | None = None):
        self.store = store
        self.vector_store = vector_store or VectorStore(store)
        self.fts_available = False
        self._vector_sync_projects: set[str] = set()
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_path TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_project_created
                    ON documents(project, created_at DESC);
                """)
            try:
                connection.execute("ALTER TABLE documents ADD COLUMN owner TEXT NOT NULL DEFAULT ''")
            except Exception:
                pass
            connection.executescript("""

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_document_chunks_project
                    ON document_chunks(project, document_id, chunk_index);

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_project
                    ON chunk_embeddings(project);

                CREATE TABLE IF NOT EXISTS knowledge_entities (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_entities_project_name
                    ON knowledge_entities(project, name);

                CREATE TABLE IF NOT EXISTS knowledge_relationships (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_relationships_project_subject
                    ON knowledge_relationships(project, subject);

                CREATE TABLE IF NOT EXISTS memory_replays (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    label TEXT NOT NULL,
                    query TEXT NOT NULL DEFAULT '',
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_replays_project_created
                    ON memory_replays(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS document_tags (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_document_tags_project_tag
                    ON document_tags(project, tag);
                """
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts
                    USING fts5(chunk_id UNINDEXED, project UNINDEXED, title, content)
                    """
                )
                self.fts_available = True
            except Exception:
                self.fts_available = False

    def ingest_text(
        self,
        title: str,
        content: str,
        *,
        project: str = "default",
        source: str = "manual",
        source_path: str = "",
        metadata: dict[str, Any] | None = None,
        embedding_provider: str = "",
        owner: str = "",
        _document_id: str = "",
    ) -> dict[str, Any]:
        title = title.strip() or "Untitled document"
        content = content.strip()
        if not content:
            raise ValueError("Document content is required.")
        project = project.strip() or "default"
        embedding_provider = (embedding_provider or os.getenv("OPENAGENT_EMBEDDING_PROVIDER", "local-hash")).strip() or "local-hash"
        document_id = _document_id.strip() or stable_id("document", project, title, content)
        created_at = utc_now()
        chunks = semantic_chunks(content)
        tags = extract_tags(f"{title}\n{content}")
        vector_chunks: list[dict[str, Any]] = []
        replacing_existing = False
        with self.store.session() as connection:
            existing = connection.execute(
                "SELECT id, created_at FROM documents WHERE id = ? AND project = ?",
                (document_id, project),
            ).fetchone()
            if existing is not None:
                replacing_existing = True
                # Editing a source must not rewrite its original creation date.
                created_at = str(existing["created_at"] or created_at)
                old_chunks = connection.execute(
                    "SELECT id FROM document_chunks WHERE document_id = ? AND project = ?",
                    (document_id, project),
                ).fetchall()
                for old_chunk in old_chunks:
                    old_chunk_id = str(old_chunk["id"])
                    if self.fts_available:
                        connection.execute(
                            "DELETE FROM document_chunks_fts WHERE project = ? AND chunk_id = ?",
                            (project, old_chunk_id),
                        )
                    connection.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (old_chunk_id,))
                    connection.execute("DELETE FROM knowledge_entities WHERE project = ? AND source_id = ?", (project, old_chunk_id))
                    connection.execute("DELETE FROM knowledge_relationships WHERE project = ? AND source_id = ?", (project, old_chunk_id))
                connection.execute("DELETE FROM document_chunks WHERE document_id = ? AND project = ?", (document_id, project))
                connection.execute("DELETE FROM document_tags WHERE project = ? AND target_id = ?", (project, document_id))
            connection.execute(
                """
                INSERT INTO documents (
                    id, project, title, source, source_path, content_hash, metadata_json, created_at, owner
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    project = excluded.project,
                    title = excluded.title,
                    source = excluded.source,
                    source_path = excluded.source_path,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at,
                    owner = excluded.owner
                """,
                (
                    document_id,
                    project,
                    title,
                    source,
                    source_path,
                    sha256(content),
                    json.dumps(metadata or {}, sort_keys=True),
                    created_at,
                    owner,
                ),
            )
            for tag, confidence in tags:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO document_tags (
                        id, project, target_type, target_id, tag, source, confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("tag", project, document_id, tag),
                        project,
                        "document",
                        document_id,
                        tag,
                        "auto",
                        confidence,
                        created_at,
                    ),
                )
            for index, chunk in enumerate(chunks):
                chunk_id = stable_id("chunk", document_id, str(index), chunk)
                connection.execute(
                    """
                    INSERT INTO document_chunks (
                        id, document_id, project, chunk_index, title, content, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        project,
                        index,
                        title,
                        chunk,
                        json.dumps({"source": source, "source_path": source_path}, sort_keys=True),
                        created_at,
                    ),
                )
                if self.fts_available:
                    connection.execute(
                        "INSERT INTO document_chunks_fts (chunk_id, project, title, content) VALUES (?, ?, ?, ?)",
                        (chunk_id, project, title, chunk),
                    )
                embedding = self.vector_store.embeddings.embed(
                    chunk,
                    provider=embedding_provider,
                    dimensions=128,
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO chunk_embeddings (
                        chunk_id, project, provider, model, dimensions, vector_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        project,
                        embedding.provider,
                        embedding.model,
                        embedding.dimensions,
                        json.dumps(embedding.vector),
                        created_at,
                    ),
                )
                self._extract_graph(connection, project, chunk_id, chunk, created_at)
                vector_chunks.append(
                    {
                        "source_id": document_id,
                        "chunk_id": chunk_id,
                        "content": chunk,
                        "embedding": embedding,
                        "embedding_provider": embedding.provider,
                        "metadata": {
                            "document_id": document_id,
                            "title": title,
                            "source": source,
                            "source_path": source_path,
                            "chunk_index": index,
                        },
                    }
                )

        if replacing_existing:
            self.vector_store.delete_source(collection="knowledge", source_id=document_id, project=project)
        for vector_chunk in vector_chunks:
            self.vector_store.upsert_text(
                collection="knowledge",
                project=project,
                dimensions=128,
                **vector_chunk,
            )
        self._vector_sync_projects.add(project)

        self.store.add_event(
            ContextEvent(
                title=f"Document {'updated' if replacing_existing else 'ingested'}: {title}",
                content=f"{len(chunks)} chunks stored from {source}.",
                type="document",
                source="knowledge",
                project=project,
                metadata={"document_id": document_id, "chunk_count": len(chunks)},
            )
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="knowledge",
                action="document.update" if replacing_existing else "document.ingest",
                resource=f"document:{document_id}",
                project=project,
                metadata={"chunk_count": len(chunks), "source": source, **({"updated": True} if replacing_existing else {})},
            )
        )
        return {
            "id": document_id,
            "project": project,
            "title": title,
            "source": source,
            "source_path": source_path,
            "chunk_count": len(chunks),
            "tags": [tag for tag, _ in tags],
            "created_at": created_at,
        }

    def update_document(
        self,
        document_id: str,
        title: str,
        content: str,
        *,
        project: str = "default",
        embedding_provider: str = "",
        owner: str = "",
    ) -> dict[str, Any]:
        """Replace an indexed source and rebuild its derived index records in place."""

        document_id = document_id.strip()
        project = project.strip() or "default"
        if not document_id:
            raise KeyError("knowledge document not found")
        with self.store.session() as connection:
            existing = connection.execute(
                "SELECT source, source_path, metadata_json FROM documents WHERE id = ? AND project = ?" + (" AND owner = ?" if owner else ""),
                (document_id, project, owner) if owner else (document_id, project),
            ).fetchone()
        if existing is None:
            raise KeyError("knowledge document not found")
        try:
            metadata = json.loads(existing["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return self.ingest_text(
            title,
            content,
            project=project,
            source=str(existing["source"] or "manual"),
            source_path=str(existing["source_path"] or ""),
            metadata=metadata if isinstance(metadata, dict) else {},
            embedding_provider=embedding_provider,
            owner=owner,
            _document_id=document_id,
        )

    def ingest_file(self, path: str | Path, *, project: str = "default") -> dict[str, Any]:
        file_path = Path(path).expanduser().resolve()
        if file_path.suffix.lower() not in {".txt", ".md", ".csv", ".json", ".log"}:
            raise ValueError("Only text-like files are supported by the local bootstrap ingester.")
        content = file_path.read_text(encoding="utf-8")
        return self.ingest_text(
            file_path.name,
            content,
            project=project,
            source="file",
            source_path=str(file_path),
        )

    def delete_document(self, document_id: str, *, project: str = "default", owner: str = "") -> dict[str, Any]:
        """Delete a knowledge document, graph lineage, embeddings, and vector mirrors."""

        project = project.strip() or "default"
        document_id = document_id.strip()
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT id, title FROM documents WHERE id = ? AND project = ?" + (" AND owner = ?" if owner else ""),
                (document_id, project, owner) if owner else (document_id, project),
            ).fetchone()
            if row is None:
                return {"deleted": False, "document_id": document_id, "vector_chunks": 0}
            chunk_rows = connection.execute(
                "SELECT id FROM document_chunks WHERE document_id = ? AND project = ?",
                (document_id, project),
            ).fetchall()
            chunk_ids = [str(item["id"]) for item in chunk_rows]
            for chunk_id in chunk_ids:
                if self.fts_available:
                    connection.execute(
                        "DELETE FROM document_chunks_fts WHERE project = ? AND chunk_id = ?",
                        (project, chunk_id),
                    )
                connection.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
                connection.execute("DELETE FROM knowledge_entities WHERE project = ? AND source_id = ?", (project, chunk_id))
                connection.execute("DELETE FROM knowledge_relationships WHERE project = ? AND source_id = ?", (project, chunk_id))
            connection.execute("DELETE FROM document_chunks WHERE document_id = ? AND project = ?", (document_id, project))
            connection.execute("DELETE FROM document_tags WHERE project = ? AND target_id = ?", (project, document_id))
            connection.execute("DELETE FROM documents WHERE id = ? AND project = ?", (document_id, project))
        vector_result = self.vector_store.delete_source(
            collection="knowledge",
            source_id=document_id,
            project=project,
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="knowledge",
                action="document.delete",
                resource=f"document:{document_id}",
                project=project,
                metadata={"chunks": len(chunk_ids), "vector_chunks": vector_result["deleted"]},
            )
        )
        return {"deleted": True, "document_id": document_id, "chunks": len(chunk_ids), "vector_chunks": vector_result["deleted"]}

    def list_documents(self, *, project: str = "default", limit: int = 100, owner: str = "") -> list[dict[str, Any]]:
        """Return the durable source library without exposing local file paths."""

        project = project.strip() or "default"
        where_extra = " AND documents.owner = ?" if owner else ""
        params: list[Any] = [project]
        if owner:
            params.append(owner)
        params.append(max(1, min(limit, 500)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    documents.id,
                    documents.project,
                    documents.title,
                    documents.source,
                    documents.metadata_json,
                    documents.created_at,
                    COUNT(document_chunks.id) AS chunk_count,
                    GROUP_CONCAT(DISTINCT document_tags.tag) AS tags
                FROM documents
                LEFT JOIN document_chunks
                    ON document_chunks.document_id = documents.id
                    AND document_chunks.project = documents.project
                LEFT JOIN document_tags
                    ON document_tags.target_id = documents.id
                    AND document_tags.project = documents.project
                    AND document_tags.target_type = 'document'
                WHERE documents.project = ?{where_extra}
                GROUP BY documents.id
                ORDER BY documents.created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [
            {
                "id": str(row["id"]),
                "project": str(row["project"]),
                "title": str(row["title"]),
                "source": str(row["source"]),
                "chunk_count": int(row["chunk_count"] or 0),
                "tags": [tag for tag in str(row["tags"] or "").split(",") if tag],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def get_document(self, document_id: str, *, project: str = "default", owner: str = "") -> dict[str, Any]:
        """Return one indexed source and its stored chunks within a project."""

        project = project.strip() or "default"
        document_id = document_id.strip()
        if not document_id:
            raise KeyError("knowledge document not found")
        sql = "SELECT id, project, title, source, metadata_json, created_at FROM documents WHERE id = ? AND project = ?"
        params: list[Any] = [document_id, project]
        if owner:
            sql += " AND owner = ?"
            params.append(owner)
        with self.store.session() as connection:
            row = connection.execute(sql, tuple(params)).fetchone()
            if row is None:
                raise KeyError("knowledge document not found")
            chunks = connection.execute(
                """SELECT id, chunk_index, content, metadata_json, created_at
                   FROM document_chunks WHERE document_id = ? AND project = ?
                   ORDER BY chunk_index""",
                (document_id, project),
            ).fetchall()
            tags = connection.execute(
                """SELECT tag FROM document_tags
                   WHERE target_id = ? AND project = ? AND target_type = 'document'
                   ORDER BY tag""",
                (document_id, project),
            ).fetchall()
        return {
            "id": str(row["id"]),
            "project": str(row["project"]),
            "title": str(row["title"]),
            "source": str(row["source"]),
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "created_at": str(row["created_at"]),
            "tags": [str(tag["tag"]) for tag in tags],
            "chunks": [
                {
                    "id": str(chunk["id"]),
                    "index": int(chunk["chunk_index"]),
                    "content": str(chunk["content"]),
                    "metadata": json.loads(chunk["metadata_json"] or "{}"),
                    "created_at": str(chunk["created_at"]),
                }
                for chunk in chunks
            ],
            "content": "\n\n".join(str(chunk["content"]) for chunk in chunks),
        }

    def _owner_document_ids(self, project: str, owner: str) -> set[str] | None:
        """Return the set of document ids owned by `owner`, or None if no filter needed."""
        if not owner:
            return None
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT id FROM documents WHERE project = ? AND owner = ?",
                (project, owner),
            ).fetchall()
        return {str(row["id"]) for row in rows}

    def search(self, query: str, *, project: str = "default", limit: int = 10, owner: str = "") -> list[dict[str, Any]]:
        query = query.strip()
        allowed = self._owner_document_ids(project, owner)
        if not query:
            results = self.list_chunks(project=project, limit=limit)
            return results if allowed is None else [r for r in results if r.get("document_id") in allowed]
        limit = max(1, min(limit, 50))
        if self.fts_available:
            with self.store.session() as connection:
                rows = connection.execute(
                    """
                    SELECT c.*, bm25(document_chunks_fts) AS rank
                    FROM document_chunks_fts
                    JOIN document_chunks c ON c.id = document_chunks_fts.chunk_id
                    WHERE document_chunks_fts MATCH ? AND c.project = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query(query), project, limit),
                ).fetchall()
            if rows:
                results = [chunk_row(row, score=-float(row["rank"])) for row in rows]
                return results if allowed is None else [r for r in results if r.get("document_id") in allowed]
        results = self._like_search(query, project=project, limit=limit)
        return results if allowed is None else [r for r in results if r.get("document_id") in allowed]

    def semantic_search(
        self,
        query: str,
        *,
        project: str = "default",
        limit: int = 10,
        embedding_provider: str = "",
        owner: str = "",
    ) -> list[dict[str, Any]]:
        allowed = self._owner_document_ids(project, owner)
        limit = max(1, min(limit, 50))
        embedding_provider = (embedding_provider or os.getenv("OPENAGENT_EMBEDDING_PROVIDER", "local-hash")).strip() or "local-hash"
        self.sync_vector_index(project=project, embedding_provider=embedding_provider)
        accelerated = self.vector_store.search(
            query,
            collection="knowledge",
            project=project,
            limit=limit,
            embedding_provider=embedding_provider,
        )
        if accelerated["results"]:
            results = [
                {
                    "id": str(item["chunk_id"]),
                    "document_id": str(item["source_id"]),
                    "project": project,
                    "chunk_index": int(item.get("metadata", {}).get("chunk_index", 0)),
                    "title": str(item.get("metadata", {}).get("title", "Knowledge source")),
                    "content": str(item["content"]),
                    "metadata": {
                        **item.get("metadata", {}),
                        "vector_acceleration": accelerated["acceleration"],
                    },
                    "created_at": item["created_at"],
                    "score": item["score"],
                    "retrieval": item["retrieval"],
                    "semantic_score": item.get("semantic_score", item["score"]),
                }
                for item in accelerated["results"]
            ]
            return results if allowed is None else [r for r in results if r.get("document_id") in allowed]

        query_vector = self.vector_store.embeddings.embed(
            query,
            provider=embedding_provider,
            dimensions=128,
        ).vector
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT c.*, e.vector_json
                FROM chunk_embeddings e
                JOIN document_chunks c ON c.id = e.chunk_id
                WHERE e.project = ?
                """,
                (project,),
            ).fetchall()
        scored = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            scored.append((cosine(query_vector, vector), chunk_row(row)))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [{**item, "score": round(score, 6)} for score, item in scored[:limit]]
        return results if allowed is None else [r for r in results if r.get("document_id") in allowed]

    def sync_vector_index(
        self,
        *,
        project: str = "default",
        force: bool = False,
        embedding_provider: str = "",
    ) -> dict[str, Any]:
        """Reconcile durable knowledge chunks into the accelerated vector collection."""

        project = project.strip() or "default"
        embedding_provider = (embedding_provider or os.getenv("OPENAGENT_EMBEDDING_PROVIDER", "local-hash")).strip() or "local-hash"
        collection = self.vector_store.ensure_collection("knowledge", project=project, dimensions=128)
        with self.store.session() as connection:
            source_count = int(
                connection.execute(
                    "SELECT count(*) FROM document_chunks WHERE project = ?",
                    (project,),
                ).fetchone()[0]
            )
            indexed_count = int(
                connection.execute(
                    "SELECT count(*) FROM vector_store_chunks WHERE project = ? AND collection_id = ?",
                    (project, collection["id"]),
                ).fetchone()[0]
            )
            indexed_with_provider = int(
                connection.execute(
                    "SELECT count(*) FROM vector_store_chunks WHERE project = ? AND collection_id = ? AND embedding_provider = ?",
                    (project, collection["id"], embedding_provider),
                ).fetchone()[0]
            )
            if not force and indexed_with_provider == source_count and indexed_count == source_count:
                self._vector_sync_projects.add(project)
                return {"project": project, "source_count": source_count, "indexed_count": indexed_count, "synced": 0}
            rows = connection.execute(
                "SELECT * FROM document_chunks WHERE project = ? ORDER BY document_id, chunk_index",
                (project,),
            ).fetchall()
        synced = 0
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            self.vector_store.upsert_text(
                collection="knowledge",
                project=project,
                source_id=str(row["document_id"]),
                chunk_id=str(row["id"]),
                content=str(row["content"]),
                dimensions=128,
                embedding_provider=embedding_provider,
                metadata={
                    **metadata,
                    "document_id": str(row["document_id"]),
                    "title": str(row["title"]),
                    "chunk_index": int(row["chunk_index"]),
                },
            )
            synced += 1
        self._vector_sync_projects.add(project)
        return {"project": project, "source_count": source_count, "indexed_count": source_count, "synced": synced}

    def hybrid_search(
        self,
        query: str,
        *,
        project: str = "default",
        limit: int = 10,
        embedding_provider: str = "",
        owner: str = "",
    ) -> list[dict[str, Any]]:
        lexical = self.search(query, project=project, limit=limit, owner=owner)
        semantic = self.semantic_search(
            query,
            project=project,
            limit=limit,
            embedding_provider=embedding_provider,
            owner=owner,
        )
        merged: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(lexical, start=1):
            score = 1.0 / rank
            merged[item["id"]] = {**item, "score": score, "retrieval": ["lexical"]}
        for item in semantic:
            existing = merged.get(item["id"])
            semantic_score = float(item.get("score", 0.0))
            semantic_retrieval = list(item.get("retrieval") or ["semantic"])
            if existing:
                existing["score"] = round(float(existing["score"]) + semantic_score, 6)
                existing["retrieval"] = list(
                    dict.fromkeys([*existing["retrieval"], "semantic", *semantic_retrieval])
                )
            else:
                merged[item["id"]] = {
                    **item,
                    "score": semantic_score,
                    "retrieval": list(dict.fromkeys(["semantic", *semantic_retrieval])),
                }
        results = sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True)
        return results[: max(1, min(limit, 50))]

    def tags(self, *, project: str = "default", limit: int = 100) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM document_tags
                WHERE project = ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
                """,
                (project, max(1, min(limit, 500))),
            ).fetchall()
        return [tag_row(row) for row in rows]

    def timeline(self, *, project: str = "default", limit: int = 100) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with self.store.session() as connection:
            document_rows = connection.execute(
                "SELECT * FROM documents WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, max(1, min(limit, 500))),
            ).fetchall()
            date_entities = connection.execute(
                """
                SELECT * FROM knowledge_entities
                WHERE project = ? AND entity_type = 'date'
                ORDER BY name DESC
                LIMIT ?
                """,
                (project, max(1, min(limit, 500))),
            ).fetchall()
        for row in document_rows:
            items.append(
                {
                    "type": "document",
                    "timestamp": row["created_at"],
                    "title": row["title"],
                    "source_id": row["id"],
                    "metadata": json.loads(row["metadata_json"]),
                }
            )
        for row in date_entities:
            items.append(
                {
                    "type": "mentioned_date",
                    "timestamp": row["name"],
                    "title": f"Mentioned date: {row['name']}",
                    "source_id": row["source_id"],
                    "metadata": json.loads(row["metadata_json"]),
                }
            )
        items.extend(
            {
                "type": "context_event",
                "timestamp": event["created_at"],
                "title": event["title"],
                "source_id": event["id"],
                "metadata": event["metadata"],
            }
            for event in self.store.list_events(project=project, limit=limit)
        )
        return sorted(items, key=lambda item: str(item["timestamp"]), reverse=True)[:limit]

    def synthesize(
        self,
        query: str,
        *,
        project: str = "default",
        limit: int = 8,
        embedding_provider: str = "",
    ) -> dict[str, Any]:
        results = self.hybrid_search(
            query,
            project=project,
            limit=limit,
            embedding_provider=embedding_provider,
        )
        graph = self.graph(project=project, limit=50)
        tags = self.tags(project=project, limit=25)
        source_titles = []
        key_terms = Counter()
        for result in results:
            source_titles.append(result["title"])
            for token in TOKEN_RE.findall(result["content"].lower()):
                if token not in STOPWORDS and len(token) > 3:
                    key_terms[token] += 1
        synthesis = {
            "query": query,
            "project": project,
            "summary": (
                f"Found {len(results)} local sources across "
                f"{len(set(source_titles))} document(s). "
                f"Top terms: {', '.join(term for term, _ in key_terms.most_common(8)) or 'none'}."
            ),
            "sources": [
                {
                    "chunk_id": result["id"],
                    "document_id": result["document_id"],
                    "title": result["title"],
                    "score": result.get("score", 0),
                    "retrieval": result.get("retrieval", []),
                }
                for result in results
            ],
            "related_entities": graph["entities"][:10],
            "related_relationships": graph["relationships"][:10],
            "tags": tags[:10],
            "reasoning_steps": [
                "Search local lexical index.",
                "Search local vector adapter.",
                "Merge and rank sources.",
                "Extract entities, tags, and relationship hints.",
                "Return source-attributed synthesis without external calls.",
            ],
        }
        self.store.add_audit_event(
            AuditEvent(
                actor="knowledge",
                action="knowledge.synthesize",
                resource=f"query:{query[:80]}",
                project=project,
                metadata={"source_count": len(results), "embedding_provider": embedding_provider or "configured-default"},
            )
        )
        return synthesis

    def list_chunks(self, *, project: str = "default", limit: int = 20) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM document_chunks WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, max(1, min(limit, 100))),
            ).fetchall()
        return [chunk_row(row) for row in rows]

    def graph(self, *, project: str = "default", limit: int = 100) -> dict[str, Any]:
        bounded_limit = max(1, min(int(limit), 500))
        with self.store.session() as connection:
            entity_rows = connection.execute(
                "SELECT * FROM knowledge_entities WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, bounded_limit),
            ).fetchall()
            relationship_rows = connection.execute(
                "SELECT * FROM knowledge_relationships WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, bounded_limit),
            ).fetchall()
        entities = [entity_row(row) for row in entity_rows]
        relationships = [relationship_row(row) for row in relationship_rows]
        nodes_by_name: dict[str, dict[str, Any]] = {}

        def ensure_node(name: str, entity_type: str = "entity", source_id: str = "") -> dict[str, Any]:
            key = name.casefold().strip()
            node = nodes_by_name.get(key)
            if node is None:
                node = {
                    "id": stable_id("graph-node", project, key),
                    "name": name.strip(),
                    "entity_type": entity_type or "entity",
                    "degree": 0,
                    "source_ids": [],
                }
                nodes_by_name[key] = node
            if source_id and source_id not in node["source_ids"]:
                node["source_ids"].append(source_id)
            return node

        for entity in entities:
            ensure_node(entity["name"], entity["entity_type"], entity["source_id"])

        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for relationship in relationships:
            source = ensure_node(relationship["subject"], source_id=relationship["source_id"])
            target = ensure_node(relationship["object"], source_id=relationship["source_id"])
            edge_key = (source["id"], relationship["predicate"], target["id"])
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            source["degree"] += 1
            target["degree"] += 1
            edges.append({
                "id": stable_id("graph-edge", project, *edge_key),
                "source": source["id"],
                "target": target["id"],
                "label": relationship["predicate"],
                "weight": float(relationship["weight"]),
                "source_id": relationship["source_id"],
            })

        nodes = sorted(nodes_by_name.values(), key=lambda item: (-int(item["degree"]), item["name"].casefold()))
        return {
            "project": project,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "connected_nodes": sum(1 for node in nodes if node["degree"] > 0),
                "isolated_nodes": sum(1 for node in nodes if node["degree"] == 0),
            },
            "entities": entities,
            "relationships": relationships,
        }

    def compress_memory(self, *, project: str = "default", limit: int = 100) -> dict[str, Any]:
        chunks = self.list_chunks(project=project, limit=limit)
        terms = Counter()
        titles = Counter()
        for chunk in chunks:
            titles[chunk["title"]] += 1
            for token in TOKEN_RE.findall(chunk["content"].lower()):
                if len(token) > 3:
                    terms[token] += 1
        summary = {
            "project": project,
            "chunk_count": len(chunks),
            "top_documents": titles.most_common(8),
            "top_terms": terms.most_common(20),
        }
        event = self.store.add_event(
            ContextEvent(
                title=f"Memory compression: {project}",
                content=json.dumps(summary, sort_keys=True),
                type="memory_compression",
                source="knowledge",
                project=project,
                metadata=summary,
            )
        )
        return {"summary": summary, "event": event}

    def create_replay(
        self,
        *,
        project: str = "default",
        label: str = "Replay",
        query: str = "",
    ) -> dict[str, Any]:
        chunks = self.search(query, project=project, limit=20) if query else self.list_chunks(project=project, limit=20)
        events = self.store.search_events(query, project=project, limit=20) if query else self.store.list_events(project=project, limit=20)
        snapshot = {"events": events, "chunks": chunks}
        replay_id = stable_id("replay", project, label, json.dumps(snapshot, sort_keys=True))
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_replays (id, project, label, query, snapshot_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (replay_id, project, label, query, json.dumps(snapshot, sort_keys=True), created_at),
            )
        return {
            "id": replay_id,
            "project": project,
            "label": label,
            "query": query,
            "snapshot": snapshot,
            "created_at": created_at,
        }

    def list_collections(self, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        return self.vector_store.collections(project=project, owner=owner)

    def delete_collection(self, collection_name: str, *, project: str = "default", owner: str = "") -> dict[str, Any]:
        collection_name = collection_name.strip()
        owner = str(owner).strip()
        if not collection_name:
            raise ValueError("collection name is required.")
        with self.store.session() as connection:
            if owner:
                row = connection.execute(
                    "SELECT id, name FROM vector_store_collections WHERE project = ? AND name = ? AND owner = ?",
                    (project, collection_name, owner),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT id, name FROM vector_store_collections WHERE project = ? AND name = ?",
                    (project, collection_name),
                ).fetchone()
            if row is None:
                raise KeyError(f"Collection '{collection_name}' not found.")
            collection_id = str(row["id"])
            chunk_count = connection.execute(
                "SELECT count(*) FROM vector_store_chunks WHERE collection_id = ?",
                (collection_id,),
            ).fetchone()[0]
            connection.execute("DELETE FROM vector_store_chunks WHERE collection_id = ?", (collection_id,))
            connection.execute(
                "DELETE FROM vector_store_collections WHERE id = ?", (collection_id,)
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="knowledge",
                action="knowledge.collection.delete",
                resource=f"vector-collection:{collection_name}",
                project=project,
                metadata={"chunks_removed": int(chunk_count)},
            )
        )
        return {"deleted": True, "collection": collection_name, "chunks_removed": int(chunk_count)}

    def update_collection(self, collection_name: str, *, project: str = "default", owner: str = "", new_name: str = "") -> dict[str, Any]:
        collection_name = collection_name.strip()
        owner = str(owner).strip()
        if not collection_name:
            raise ValueError("collection name is required.")
        new_name = new_name.strip()
        if not new_name or new_name == collection_name:
            raise ValueError("new_name must differ from the current name.")
        from .models import utc_now as _utc_now
        now = _utc_now()
        with self.store.session() as connection:
            if owner:
                row = connection.execute(
                    "SELECT id FROM vector_store_collections WHERE project = ? AND name = ? AND owner = ?",
                    (project, collection_name, owner),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT id FROM vector_store_collections WHERE project = ? AND name = ?",
                    (project, collection_name),
                ).fetchone()
            if row is None:
                raise KeyError(f"Collection '{collection_name}' not found.")
            conflict = connection.execute(
                "SELECT 1 FROM vector_store_collections WHERE project = ? AND name = ?",
                (project, new_name),
            ).fetchone()
            if conflict is not None:
                raise ValueError(f"Collection '{new_name}' already exists.")
            connection.execute(
                "UPDATE vector_store_collections SET name = ?, updated_at = ? WHERE id = ?",
                (new_name, now, str(row["id"])),
            )
        return {"updated": True, "old_name": collection_name, "new_name": new_name}

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as connection:
            documents = connection.execute("SELECT count(*) FROM documents WHERE project = ?", (project,)).fetchone()[0]
            chunks = connection.execute("SELECT count(*) FROM document_chunks WHERE project = ?", (project,)).fetchone()[0]
            entities = connection.execute("SELECT count(*) FROM knowledge_entities WHERE project = ?", (project,)).fetchone()[0]
            relationships = connection.execute("SELECT count(*) FROM knowledge_relationships WHERE project = ?", (project,)).fetchone()[0]
            replays = connection.execute("SELECT count(*) FROM memory_replays WHERE project = ?", (project,)).fetchone()[0]
            tags = connection.execute("SELECT count(*) FROM document_tags WHERE project = ?", (project,)).fetchone()[0]
        return {
            "project": project,
            "documents": documents,
            "chunks": chunks,
            "entities": entities,
            "relationships": relationships,
            "replays": replays,
            "tags": tags,
            "fts_available": self.fts_available,
            "vector_adapter": "sqlite-faiss-chroma-fusion",
            "vector_store": self.vector_store.status(project=project),
        }

    def _like_search(self, query: str, *, project: str, limit: int) -> list[dict[str, Any]]:
        pattern = f"%{query.lower()}%"
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM document_chunks
                WHERE project = ? AND (lower(title) LIKE ? OR lower(content) LIKE ?)
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project, pattern, pattern, limit),
            ).fetchall()
        return [chunk_row(row, score=0.1) for row in rows]

    def _extract_graph(self, connection: Any, project: str, chunk_id: str, content: str, created_at: str) -> None:
        seen = []
        for match in ENTITY_RE.findall(content):
            name = match.strip(".,;:()[]{}")
            if len(name) < 3 or name in seen:
                continue
            seen.append(name)
            entity_type = classify_entity(name)
            connection.execute(
                """
                INSERT OR IGNORE INTO knowledge_entities (
                    id, project, name, entity_type, source_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id("entity", project, name, entity_type),
                    project,
                    name,
                    entity_type,
                    chunk_id,
                    "{}",
                    created_at,
                ),
            )
        for left, right in zip(seen, seen[1:]):
            connection.execute(
                """
                INSERT OR IGNORE INTO knowledge_relationships (
                    id, project, subject, predicate, object, source_id, weight, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id("rel", project, left, "related_to", right, chunk_id),
                    project,
                    left,
                    "related_to",
                    right,
                    chunk_id,
                    1.0,
                    "{}",
                    created_at,
                ),
            )


def semantic_chunks(content: str, *, max_chars: int = 900) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [content]:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 > max_chars and current:
                    chunks.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def fts_query(query: str) -> str:
    terms = [term for term in re.findall(r"[A-Za-z0-9_]+", query) if len(term) > 1]
    return " OR ".join(terms) or query


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_mag = sum(a * a for a in left) ** 0.5
    right_mag = sum(b * b for b in right) ** 0.5
    if not left_mag or not right_mag:
        return 0.0
    return dot / (left_mag * right_mag)


def classify_entity(name: str) -> str:
    if "@" in name:
        return "email"
    if name.startswith("http"):
        return "url"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", name):
        return "date"
    return "proper_noun"


def chunk_row(row: Any, *, score: float | None = None) -> dict[str, Any]:
    result = {
        "id": row["id"],
        "document_id": row["document_id"],
        "project": row["project"],
        "chunk_index": row["chunk_index"],
        "title": row["title"],
        "content": row["content"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }
    if score is not None:
        result["score"] = score
    return result


def entity_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "name": row["name"],
        "entity_type": row["entity_type"],
        "source_id": row["source_id"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def relationship_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "subject": row["subject"],
        "predicate": row["predicate"],
        "object": row["object"],
        "source_id": row["source_id"],
        "weight": row["weight"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def tag_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "target_type": row["target_type"],
        "target_id": row["target_id"],
        "tag": row["tag"],
        "source": row["source"],
        "confidence": row["confidence"],
        "created_at": row["created_at"],
    }


def extract_tags(text: str, *, limit: int = 12) -> list[tuple[str, float]]:
    counts = Counter(
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS and len(token) > 3
    )
    total = sum(counts.values()) or 1
    tags = []
    for token, count in counts.most_common(limit):
        tags.append((token, round(min(1.0, 0.4 + count / total), 4)))
    return tags
