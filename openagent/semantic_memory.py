"""Stage 11 semantic memory, vector, rerank, and graph analytics runtime."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import sqlite3
from typing import Any
from uuid import uuid4

from .context import STOPWORDS, TOKEN_RE
from .model_gateway import local_hash_embedding
from .models import AuditEvent, ContextEvent, utc_now
from .storage import OpenAgentStore


class SemanticMemoryManager:
    """Local semantic memory and vector retrieval without external services."""

    def __init__(self, store: OpenAgentStore):
        self.store = store
        self.initialize()

    def initialize(self) -> None:
        self.store.initialize()

    def add_memory(
        self,
        *,
        title: str,
        content: str,
        project: str = "default",
        memory_type: str = "semantic",
        profile_id: str = "",
        tags: list[str] | None = None,
        source: str = "manual",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        title = title.strip() or "Untitled memory"
        content = content.strip()
        if not content:
            raise ValueError("Memory content is required.")
        project = project.strip() or "default"
        normalized_tags = sorted({tag.strip().lower() for tag in tags or [] if tag.strip()})
        now = utc_now()
        memory_id = uuid4().hex
        vector = local_hash_embedding(f"{title}\n{content}", dimensions=64)
        record = {
            "id": memory_id,
            "project": project,
            "memory_type": memory_type.strip() or "semantic",
            "title": title,
            "content": content,
            "profile_id": profile_id.strip(),
            "tags": normalized_tags,
            "source": source.strip() or "manual",
            "confidence": max(0.0, min(float(confidence), 1.0)),
            "metadata": metadata or {},
            "embedding": vector,
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO semantic_memories (
                    id, project, memory_type, title, content, profile_id, tags_json, source,
                    confidence, metadata_json, embedding_json, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["project"],
                    record["memory_type"],
                    record["title"],
                    record["content"],
                    record["profile_id"],
                    json.dumps(record["tags"], sort_keys=True),
                    record["source"],
                    record["confidence"],
                    json.dumps(record["metadata"], sort_keys=True),
                    json.dumps(record["embedding"]),
                    record["version"],
                    record["created_at"],
                    record["updated_at"],
                ),
            )
            self._save_version(connection, record)
        self.store.add_event(
            ContextEvent(
                title=f"Semantic memory: {title}",
                content=content,
                type=f"memory:{record['memory_type']}",
                source=f"semantic:{record['source']}",
                project=project,
                metadata={"memory_id": memory_id, "tags": normalized_tags, "profile_id": record["profile_id"]},
            )
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="semantic-memory",
                action="memory.add",
                resource=f"memory:{memory_id}",
                project=project,
                metadata={"memory_type": record["memory_type"], "source": record["source"]},
            )
        )
        return without_embedding(record)

    def list_memories(
        self,
        *,
        project: str = "default",
        memory_type: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        params.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM semantic_memories
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [without_embedding(memory_row(row)) for row in rows]

    def search_memories(self, query: str, *, project: str = "default", limit: int = 10) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return self.list_memories(project=project, limit=limit)
        query_vector = local_hash_embedding(query, dimensions=64)
        query_terms = useful_terms(query)
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM semantic_memories WHERE project = ?",
                (project,),
            ).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            item = memory_row(row)
            lexical = lexical_score(query_terms, f"{item['title']} {item['content']} {' '.join(item['tags'])}")
            semantic = cosine(query_vector, item["embedding"])
            score = (0.55 * semantic) + (0.35 * lexical) + (0.10 * float(item["confidence"]))
            scored.append((score, item))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {**without_embedding(item), "score": round(score, 6), "retrieval": ["semantic", "lexical"]}
            for score, item in scored[: max(1, min(limit, 50))]
        ]

    def consolidate(self, *, project: str = "default", limit: int = 100) -> dict[str, Any]:
        memories = self.list_memories(project=project, limit=limit)
        events = self.store.list_events(project=project, limit=limit)
        terms = Counter()
        by_type = Counter(memory["memory_type"] for memory in memories)
        for memory in memories:
            terms.update(useful_terms(f"{memory['title']} {memory['content']}"))
        for event in events:
            terms.update(useful_terms(f"{event['title']} {event['content']}"))
        summary = {
            "project": project,
            "memory_count": len(memories),
            "event_count": len(events),
            "memory_types": dict(by_type),
            "top_terms": terms.most_common(20),
        }
        consolidated = self.add_memory(
            title=f"Consolidated memory: {project}",
            content=json.dumps(summary, sort_keys=True),
            project=project,
            memory_type="long_term",
            source="consolidation",
            tags=["consolidated", "long-term"],
            metadata=summary,
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="semantic-memory",
                action="memory.consolidate",
                resource=f"project:{project}",
                project=project,
                metadata={"memory_id": consolidated["id"], "memory_count": len(memories)},
            )
        )
        return {"summary": summary, "memory": consolidated}

    def create_collection(
        self,
        name: str,
        *,
        project: str = "default",
        dimensions: int = 64,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = name.strip() or "default"
        project = project.strip() or "default"
        dimensions = max(1, min(int(dimensions), 4096))
        now = utc_now()
        collection_id = stable_id("vector-collection", project, name)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO vector_collections (
                    id, project, name, dimensions, distance, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM vector_collections WHERE id = ?), ?), ?)
                """,
                (
                    collection_id,
                    project,
                    name,
                    dimensions,
                    "cosine",
                    json.dumps(metadata or {}, sort_keys=True),
                    collection_id,
                    now,
                    now,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="semantic-memory",
                action="vector.collection.upsert",
                resource=f"collection:{collection_id}",
                project=project,
                metadata={"dimensions": dimensions},
            )
        )
        return self.get_collection(collection_id) or {}

    def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM vector_collections WHERE id = ?", (collection_id,)).fetchone()
        return collection_row(row) if row else None

    def collections(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM vector_collections WHERE project = ? ORDER BY updated_at DESC",
                (project,),
            ).fetchall()
        return [collection_row(row) for row in rows]

    def upsert_vector_record(
        self,
        collection_id: str,
        *,
        content: str,
        project: str = "default",
        external_id: str = "",
        vector: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        collection = self.get_collection(collection_id)
        if collection is None:
            raise ValueError(f"Unknown vector collection '{collection_id}'.")
        content = content.strip()
        if not content:
            raise ValueError("Vector record content is required.")
        project = project.strip() or collection["project"]
        dimensions = int(collection["dimensions"])
        vector = vector or local_hash_embedding(content, dimensions=dimensions)
        if len(vector) != dimensions:
            raise ValueError(f"Vector must have {dimensions} dimensions.")
        now = utc_now()
        record_id = stable_id("vector-record", collection_id, external_id or content)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO vector_records (
                    id, collection_id, project, external_id, content, content_hash,
                    metadata_json, vector_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM vector_records WHERE id = ?), ?), ?)
                """,
                (
                    record_id,
                    collection_id,
                    project,
                    external_id,
                    content,
                    sha256(content),
                    json.dumps(metadata or {}, sort_keys=True),
                    json.dumps([float(value) for value in vector]),
                    record_id,
                    now,
                    now,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="semantic-memory",
                action="vector.record.upsert",
                resource=f"vector-record:{record_id}",
                project=project,
                metadata={"collection_id": collection_id, "external_id": external_id},
            )
        )
        return self.get_vector_record(record_id) or {}

    def get_vector_record(self, record_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM vector_records WHERE id = ?", (record_id,)).fetchone()
        return vector_record_row(row) if row else None

    def vector_search(self, collection_id: str, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        collection = self.get_collection(collection_id)
        if collection is None:
            raise ValueError(f"Unknown vector collection '{collection_id}'.")
        query_vector = local_hash_embedding(query, dimensions=int(collection["dimensions"]))
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM vector_records WHERE collection_id = ?",
                (collection_id,),
            ).fetchall()
        scored = []
        for row in rows:
            item = vector_record_row(row)
            scored.append((cosine(query_vector, item["vector"]), item))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {**without_vector(item), "score": round(score, 6)}
            for score, item in scored[: max(1, min(limit, 50))]
        ]

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        project: str = "default",
        limit: int = 10,
    ) -> dict[str, Any]:
        query = query.strip()
        query_vector = local_hash_embedding(query, dimensions=64)
        query_terms = useful_terms(query)
        ranked = []
        for index, candidate in enumerate(candidates):
            content = str(candidate.get("content") or candidate.get("body") or candidate.get("text") or "")
            title = str(candidate.get("title") or candidate.get("id") or f"candidate-{index}")
            vector = local_hash_embedding(f"{title}\n{content}", dimensions=64)
            score = (0.6 * cosine(query_vector, vector)) + (0.4 * lexical_score(query_terms, f"{title} {content}"))
            ranked.append(
                {
                    "id": str(candidate.get("id") or stable_id("candidate", str(index), title, content)),
                    "title": title,
                    "content": content,
                    "metadata": candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {},
                    "score": round(score, 6),
                    "ranker": "local-hybrid-reranker",
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        result = ranked[: max(1, min(limit, 50))]
        run_id = uuid4().hex
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO semantic_rerank_runs (id, project, query, input_count, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, project, query, len(candidates), json.dumps(result, sort_keys=True), created_at),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="semantic-memory",
                action="retrieval.rerank",
                resource=f"rerank:{run_id}",
                project=project,
                metadata={"input_count": len(candidates), "returned": len(result)},
            )
        )
        return {"id": run_id, "project": project, "query": query, "results": result, "created_at": created_at}

    def graph_analytics(self, *, project: str = "default") -> dict[str, Any]:
        entity_counts: Counter[str] = Counter()
        relationship_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        try:
            with self.store.session() as connection:
                memory_rows = connection.execute(
                    "SELECT tags_json FROM semantic_memories WHERE project = ?",
                    (project,),
                ).fetchall()
                for row in memory_rows:
                    tag_counts.update(json.loads(row["tags_json"]))
                entity_rows = connection.execute(
                    "SELECT name FROM knowledge_entities WHERE project = ?",
                    (project,),
                ).fetchall()
                relationship_rows = connection.execute(
                    "SELECT subject, object FROM knowledge_relationships WHERE project = ?",
                    (project,),
                ).fetchall()
        except sqlite3.OperationalError:
            entity_rows = []
            relationship_rows = []
        for row in entity_rows:
            entity_counts[row["name"]] += 1
        for row in relationship_rows:
            relationship_counts[row["subject"]] += 1
            relationship_counts[row["object"]] += 1
        central_entities = [
            {"name": name, "score": count}
            for name, count in (relationship_counts or entity_counts).most_common(20)
        ]
        summary = {
            "project": project,
            "entities": sum(entity_counts.values()),
            "relationships": len(relationship_rows),
            "central_entities": central_entities,
            "top_tags": [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(20)],
            "analytics": ["degree_centrality", "tag_frequency", "entity_frequency"],
        }
        snapshot_id = uuid4().hex
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO graph_analytics_snapshots (id, project, summary_json, created_at) VALUES (?, ?, ?, ?)",
                (snapshot_id, project, json.dumps(summary, sort_keys=True), created_at),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="semantic-memory",
                action="graph.analytics",
                resource=f"graph-analytics:{snapshot_id}",
                project=project,
                metadata={"entities": summary["entities"], "relationships": summary["relationships"]},
            )
        )
        return {"id": snapshot_id, "created_at": created_at, **summary}

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as connection:
            memories = connection.execute(
                "SELECT count(*) FROM semantic_memories WHERE project = ?",
                (project,),
            ).fetchone()[0]
            collections = connection.execute(
                "SELECT count(*) FROM vector_collections WHERE project = ?",
                (project,),
            ).fetchone()[0]
            vectors = connection.execute(
                "SELECT count(*) FROM vector_records WHERE project = ?",
                (project,),
            ).fetchone()[0]
            reranks = connection.execute(
                "SELECT count(*) FROM semantic_rerank_runs WHERE project = ?",
                (project,),
            ).fetchone()[0]
        return {
            "project": project,
            "semantic_memories": memories,
            "vector_collections": collections,
            "vector_records": vectors,
            "rerank_runs": reranks,
            "embedding_provider": "local-hash",
            "privacy_boundary": "All embeddings, vectors, reranking, and analytics are computed locally in SQLite.",
        }

    def _save_version(self, connection: Any, record: dict[str, Any]) -> None:
        snapshot = without_embedding(record)
        connection.execute(
            """
            INSERT INTO semantic_memory_versions (
                id, memory_id, project, version, content_hash, snapshot_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("semantic-memory-version", record["id"], str(record["version"])),
                record["id"],
                record["project"],
                record["version"],
                sha256(record["content"]),
                json.dumps(snapshot, sort_keys=True),
                record["created_at"],
            ),
        )


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def useful_terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def lexical_score(query_terms: list[str], text: str) -> float:
    if not query_terms:
        return 0.0
    counts = Counter(useful_terms(text))
    hits = sum(counts.get(term, 0) for term in query_terms)
    return min(1.0, hits / max(1, len(query_terms)))


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_mag = sum(a * a for a in left) ** 0.5
    right_mag = sum(b * b for b in right) ** 0.5
    if not left_mag or not right_mag:
        return 0.0
    return dot / (left_mag * right_mag)


def without_embedding(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "embedding"}


def without_vector(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "vector"}


def memory_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "memory_type": row["memory_type"],
        "title": row["title"],
        "content": row["content"],
        "profile_id": row["profile_id"],
        "tags": json.loads(row["tags_json"]),
        "source": row["source"],
        "confidence": row["confidence"],
        "metadata": json.loads(row["metadata_json"]),
        "embedding": json.loads(row["embedding_json"]),
        "version": row["version"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def collection_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "name": row["name"],
        "dimensions": row["dimensions"],
        "distance": row["distance"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def vector_record_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "collection_id": row["collection_id"],
        "project": row["project"],
        "external_id": row["external_id"],
        "content": row["content"],
        "content_hash": row["content_hash"],
        "metadata": json.loads(row["metadata_json"]),
        "vector": json.loads(row["vector_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
