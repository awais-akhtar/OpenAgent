"""Stage 2 local knowledge and memory engine."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .context import STOPWORDS, TOKEN_RE
from .model_gateway import local_hash_embedding
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
    ) -> dict[str, Any]:
        title = title.strip() or "Untitled document"
        content = content.strip()
        if not content:
            raise ValueError("Document content is required.")
        project = project.strip() or "default"
        document_id = stable_id("document", project, title, content)
        created_at = utc_now()
        chunks = semantic_chunks(content)
        tags = extract_tags(f"{title}\n{content}")
        vector_chunks: list[dict[str, Any]] = []
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents (
                    id, project, title, source, source_path, content_hash, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            connection.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM document_tags WHERE target_id = ?", (document_id,))
            if self.fts_available:
                connection.execute("DELETE FROM document_chunks_fts WHERE project = ? AND title = ?", (project, title))
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
                vector = local_hash_embedding(chunk, dimensions=64)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO chunk_embeddings (
                        chunk_id, project, provider, model, dimensions, vector_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        project,
                        "local",
                        "openagent-local-hash-embedding",
                        64,
                        json.dumps(vector),
                        created_at,
                    ),
                )
                self._extract_graph(connection, project, chunk_id, chunk, created_at)
                vector_chunks.append(
                    {
                        "source_id": document_id,
                        "chunk_id": chunk_id,
                        "content": chunk,
                        "metadata": {
                            "document_id": document_id,
                            "title": title,
                            "source": source,
                            "source_path": source_path,
                            "chunk_index": index,
                        },
                    }
                )

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
                title=f"Document ingested: {title}",
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
                action="document.ingest",
                resource=f"document:{document_id}",
                project=project,
                metadata={"chunk_count": len(chunks), "source": source},
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

    def delete_document(self, document_id: str, *, project: str = "default") -> dict[str, Any]:
        """Delete a knowledge document, graph lineage, embeddings, and vector mirrors."""

        project = project.strip() or "default"
        document_id = document_id.strip()
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT id, title FROM documents WHERE id = ? AND project = ?",
                (document_id, project),
            ).fetchone()
            if row is None:
                return {"deleted": False, "document_id": document_id, "vector_chunks": 0}
            chunk_rows = connection.execute(
                "SELECT id FROM document_chunks WHERE document_id = ? AND project = ?",
                (document_id, project),
            ).fetchall()
            chunk_ids = [str(item["id"]) for item in chunk_rows]
            if self.fts_available:
                connection.execute(
                    "DELETE FROM document_chunks_fts WHERE project = ? AND title = ?",
                    (project, row["title"]),
                )
            for chunk_id in chunk_ids:
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

    def search(self, query: str, *, project: str = "default", limit: int = 10) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return self.list_chunks(project=project, limit=limit)
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
                return [chunk_row(row, score=-float(row["rank"])) for row in rows]
        return self._like_search(query, project=project, limit=limit)

    def semantic_search(
        self,
        query: str,
        *,
        project: str = "default",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        self.sync_vector_index(project=project)
        accelerated = self.vector_store.search(
            query,
            collection="knowledge",
            project=project,
            limit=limit,
        )
        if accelerated["results"]:
            return [
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

        query_vector = local_hash_embedding(query, dimensions=64)
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
        return [{**item, "score": round(score, 6)} for score, item in scored[:limit]]

    def sync_vector_index(self, *, project: str = "default", force: bool = False) -> dict[str, Any]:
        """Reconcile durable knowledge chunks into the accelerated vector collection."""

        project = project.strip() or "default"
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
            if not force and project in self._vector_sync_projects and indexed_count == source_count:
                return {"project": project, "source_count": source_count, "indexed_count": indexed_count, "synced": 0}
            if not force and indexed_count == source_count:
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

    def hybrid_search(self, query: str, *, project: str = "default", limit: int = 10) -> list[dict[str, Any]]:
        lexical = self.search(query, project=project, limit=limit)
        semantic = self.semantic_search(query, project=project, limit=limit)
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

    def synthesize(self, query: str, *, project: str = "default", limit: int = 8) -> dict[str, Any]:
        results = self.hybrid_search(query, project=project, limit=limit)
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
                metadata={"source_count": len(results)},
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
        with self.store.session() as connection:
            entity_rows = connection.execute(
                "SELECT * FROM knowledge_entities WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, limit),
            ).fetchall()
            relationship_rows = connection.execute(
                "SELECT * FROM knowledge_relationships WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        return {
            "entities": [entity_row(row) for row in entity_rows],
            "relationships": [relationship_row(row) for row in relationship_rows],
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
