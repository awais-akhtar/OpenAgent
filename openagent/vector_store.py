"""Production-shaped local vector store with an offline SQLite adapter."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from typing import Any

from .embeddings import EmbeddingRouter
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore
from .vector_acceleration import LocalVectorAccelerator


class VectorStore:
    """Durable vector index with chunk lineage and deterministic fallback embeddings."""

    def __init__(self, store: OpenAgentStore, embeddings: EmbeddingRouter | None = None) -> None:
        self.store = store
        self.embeddings = embeddings or EmbeddingRouter()
        self.accelerator = LocalVectorAccelerator(store.data_dir)
        self._cache_lock = threading.RLock()
        self._item_cache: dict[str, tuple[int, list[dict[str, Any]]]] = {}
        self.initialize()

    def initialize(self) -> None:
        self.store.initialize()

    def close(self) -> None:
        with self._cache_lock:
            self._item_cache.clear()
        self.accelerator.close()

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as connection:
            collection_count = connection.execute(
                "SELECT count(*) FROM vector_store_collections WHERE project = ?",
                (project,),
            ).fetchone()[0]
            chunk_count = connection.execute(
                "SELECT count(*) FROM vector_store_chunks WHERE project = ?",
                (project,),
            ).fetchone()[0]
        acceleration = self.accelerator.status()
        return {
            "status": "ok",
            "project": project,
            "adapter": "sqlite",
            "source_of_truth": "sqlite",
            "collection_count": collection_count,
            "chunk_count": chunk_count,
            "embedding_providers": self.embeddings.providers(),
            "active_accelerators": acceleration["active"],
            "accelerators": acceleration["backends"],
            "optional_adapters": [
                {"id": "qdrant", "available": False, "privacy_boundary": "Optional local/HTTP adapter hook."},
                {"id": "lancedb", "available": False, "privacy_boundary": "Optional local file adapter hook."},
            ],
            "privacy_boundary": (
                "SQLite is the authoritative local vector store. Enabled FAISS and Chroma accelerators "
                "operate only inside the configured OpenAgent data directory."
            ),
        }

    def ensure_collection(
        self,
        name: str,
        *,
        project: str = "default",
        dimensions: int = 128,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        project = project.strip() or "default"
        name = name.strip() or "default"
        dimensions = max(8, min(int(dimensions), 1024))
        now = utc_now()
        collection_id = stable_id("vector-collection", project, name)
        with self.store.session() as connection:
            existing = connection.execute(
                "SELECT * FROM vector_store_collections WHERE project = ? AND name = ?",
                (project, name),
            ).fetchone()
            if existing is not None:
                dimensions = int(existing["dimensions"])
                metadata_payload = (
                    metadata if metadata is not None else json.loads(existing["metadata_json"] or "{}")
                )
            else:
                metadata_payload = metadata or {}
            connection.execute(
                """
                INSERT INTO vector_store_collections (
                    id, project, name, adapter, dimensions, distance, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'sqlite', ?, 'cosine', ?, ?, ?)
                ON CONFLICT(project, name) DO UPDATE SET
                    dimensions = excluded.dimensions,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    collection_id,
                    project,
                    name,
                    dimensions,
                    json.dumps(metadata_payload, sort_keys=True),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM vector_store_collections WHERE project = ? AND name = ?",
                (project, name),
            ).fetchone()
        return collection_row(row)

    def collections(self, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        owner = str(owner).strip()
        with self.store.session() as connection:
            if owner:
                rows = connection.execute(
                    "SELECT * FROM vector_store_collections WHERE project = ? AND owner = ? ORDER BY updated_at DESC",
                    (project, owner),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM vector_store_collections WHERE project = ? ORDER BY updated_at DESC",
                    (project,),
                ).fetchall()
        return [collection_row(row) for row in rows]

    def upsert_text(
        self,
        *,
        collection: str,
        content: str,
        project: str = "default",
        source_id: str = "",
        chunk_id: str = "",
        metadata: dict[str, Any] | None = None,
        dimensions: int = 128,
        embedding_provider: str = "local-hash",
        embedding: Any | None = None,
    ) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("content is required.")
        collection_record = self.ensure_collection(collection, project=project, dimensions=dimensions)
        source_id = source_id.strip() or stable_id("vector-source", project, collection, content)
        chunk_id = chunk_id.strip() or "0"
        embedding = embedding or self.embeddings.embed(
            content,
            provider=embedding_provider,
            dimensions=collection_record["dimensions"],
        )
        now = utc_now()
        record_id = stable_id("vector-chunk", collection_record["id"], source_id, chunk_id)
        digest = sha256(content)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO vector_store_chunks (
                    id, project, collection_id, source_id, chunk_id, content_hash,
                    content, metadata_json, embedding_provider, embedding_model,
                    dimensions, vector_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection_id, source_id, chunk_id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    content = excluded.content,
                    metadata_json = excluded.metadata_json,
                    embedding_provider = excluded.embedding_provider,
                    embedding_model = excluded.embedding_model,
                    dimensions = excluded.dimensions,
                    vector_json = excluded.vector_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record_id,
                    project,
                    collection_record["id"],
                    source_id,
                    chunk_id,
                    digest,
                    content,
                    json.dumps(metadata or {}, sort_keys=True),
                    embedding.provider,
                    embedding.model,
                    embedding.dimensions,
                    json.dumps(embedding.vector),
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE vector_store_collections SET revision = revision + 1, updated_at = ? WHERE id = ?",
                (now, collection_record["id"]),
            )
            collection_row_value = connection.execute(
                "SELECT * FROM vector_store_collections WHERE id = ?", (collection_record["id"],)
            ).fetchone()
            row = connection.execute("SELECT * FROM vector_store_chunks WHERE id = ?", (record_id,)).fetchone()
        collection_record = collection_row(collection_row_value)
        with self._cache_lock:
            self._item_cache.pop(collection_record["id"], None)
        item = chunk_row(row)
        mirror = self.accelerator.mirror_upsert(collection_record, item)
        self.store.add_audit_event(
            AuditEvent(
                actor="vector-store",
                action="vector.upsert",
                resource=f"vector:{record_id}",
                project=project,
                metadata={
                    "collection": collection,
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "accelerators": mirror["mirrored"],
                    "accelerator_errors": mirror["errors"],
                },
            )
        )
        return {**without_vector(item), "acceleration": mirror}

    def search(
        self,
        query: str,
        *,
        collection: str = "default",
        project: str = "default",
        limit: int = 10,
        embedding_provider: str = "local-hash",
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query is required.")
        collection_record = self.ensure_collection(collection, project=project)
        embedding = self.embeddings.embed(query, provider=embedding_provider, dimensions=collection_record["dimensions"])
        started = time.perf_counter()
        items = self._collection_items(collection_record)
        accelerated = self.accelerator.search(
            collection_record,
            embedding.vector,
            items,
            limit=max(max(1, min(limit, 100)) * 4, 24),
        )
        backend_scores: dict[str, list[float]] = {}
        backend_ranks: dict[str, list[tuple[str, int]]] = {}
        for backend, ranking in accelerated["rankings"].items():
            for rank, result in enumerate(ranking, start=1):
                record_id = str(result["id"])
                backend_scores.setdefault(record_id, []).append(float(result["score"]))
                backend_ranks.setdefault(record_id, []).append((backend, rank))

        query_terms = useful_terms(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        active_backend_count = max(1, len(accelerated["rankings"]))
        for item in items:
            scores = backend_scores.get(str(item["id"]), [])
            semantic = sum(scores) / len(scores) if scores else (
                0.0 if accelerated["rankings"] else cosine(embedding.vector, item["vector"])
            )
            lexical = lexical_score(query_terms, item["content"])
            rank_entries = backend_ranks.get(str(item["id"]), [])
            agreement = len(rank_entries) / active_backend_count
            reciprocal_rank = sum(1.0 / (60.0 + rank) for _, rank in rank_entries)
            score = (0.70 * semantic) + (0.20 * lexical) + (0.08 * agreement) + (0.02 * reciprocal_rank * 60)
            scored.append(
                (
                    score,
                    {
                        **item,
                        "semantic_score": round(semantic, 6),
                        "lexical_score": round(lexical, 6),
                        "backend_agreement": round(agreement, 6),
                        "backend_ranks": {backend: rank for backend, rank in rank_entries},
                    },
                )
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [
            {
                **without_vector(item),
                "score": round(score, 6),
                "retrieval": [
                    "sqlite-source-of-truth",
                    *item.get("backend_ranks", {}).keys(),
                    "local-embedding",
                    "lexical-boost",
                    "reciprocal-rank-fusion",
                ],
            }
            for score, item in scored[: max(1, min(limit, 100))]
        ]
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return {
            "project": project,
            "collection": collection,
            "query": query,
            "results": results,
            "latency_ms": latency_ms,
            "acceleration": {
                "active": list(accelerated["rankings"]),
                "search_latency_ms": accelerated["latency_ms"],
                "errors": accelerated["errors"],
            },
            "embedding": {
                "provider": embedding.provider,
                "model": embedding.model,
                "dimensions": embedding.dimensions,
            },
            "privacy_boundary": embedding.privacy_boundary,
        }

    def delete_source(
        self,
        *,
        collection: str,
        source_id: str,
        project: str = "default",
    ) -> dict[str, Any]:
        """Delete every vector chunk for one source and invalidate local mirrors."""

        project = project.strip() or "default"
        source_id = source_id.strip()
        if not source_id:
            raise ValueError("source_id is required.")
        with self.store.session() as connection:
            collection_row_value = connection.execute(
                "SELECT * FROM vector_store_collections WHERE project = ? AND name = ?",
                (project, collection),
            ).fetchone()
            if collection_row_value is None:
                return {"deleted": 0, "source_id": source_id, "mirrors": {"removed": [], "errors": {}}}
            collection_record = collection_row(collection_row_value)
            rows = connection.execute(
                "SELECT id FROM vector_store_chunks WHERE collection_id = ? AND source_id = ?",
                (collection_record["id"], source_id),
            ).fetchall()
            record_ids = [str(row["id"]) for row in rows]
            connection.execute(
                "DELETE FROM vector_store_chunks WHERE collection_id = ? AND source_id = ?",
                (collection_record["id"], source_id),
            )
            if record_ids:
                connection.execute(
                    "UPDATE vector_store_collections SET revision = revision + 1, updated_at = ? WHERE id = ?",
                    (utc_now(), collection_record["id"]),
                )
                refreshed = connection.execute(
                    "SELECT * FROM vector_store_collections WHERE id = ?",
                    (collection_record["id"],),
                ).fetchone()
                collection_record = collection_row(refreshed)
        with self._cache_lock:
            self._item_cache.pop(collection_record["id"], None)
        mirrors = self.accelerator.mirror_delete(collection_record, record_ids)
        self.store.add_audit_event(
            AuditEvent(
                actor="vector-store",
                action="vector.source.delete",
                resource=f"vector-source:{source_id}",
                project=project,
                metadata={"collection": collection, "deleted": len(record_ids), "mirror_errors": mirrors["errors"]},
            )
        )
        return {"deleted": len(record_ids), "source_id": source_id, "mirrors": mirrors}

    def _collection_items(self, collection: dict[str, Any]) -> list[dict[str, Any]]:
        collection_id = str(collection["id"])
        revision = int(collection.get("revision", 0))
        with self._cache_lock:
            cached = self._item_cache.get(collection_id)
            if cached is not None and cached[0] == revision:
                return cached[1]
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM vector_store_chunks WHERE project = ? AND collection_id = ?",
                (collection["project"], collection_id),
            ).fetchall()
        items = [chunk_row(row) for row in rows]
        with self._cache_lock:
            self._item_cache[collection_id] = (revision, items)
        return items

    def benchmark(self, *, project: str = "default", collection: str = "benchmark") -> dict[str, Any]:
        samples = [
            ("local-first", "OpenAgent keeps user data inside local infrastructure by default."),
            ("worker", "Durable background jobs use leases, retries, and dead letters."),
            ("voice", "Voice transcription runs with local speech-to-text providers."),
        ]
        started = time.perf_counter()
        for chunk_id, content in samples:
            self.upsert_text(
                collection=collection,
                content=content,
                project=project,
                source_id="vector-benchmark",
                chunk_id=chunk_id,
                metadata={"benchmark": True},
            )
        index_ms = round((time.perf_counter() - started) * 1000, 3)
        search_result = self.search("local durable jobs", collection=collection, project=project, limit=2)
        return {
            "status": "ok",
            "project": project,
            "collection": collection,
            "indexed": len(samples),
            "index_ms": index_ms,
            "query_latency_ms": search_result["latency_ms"],
            "accelerators": search_result["acceleration"],
            "top_result": search_result["results"][0] if search_result["results"] else None,
        }


def collection_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "name": row["name"],
        "adapter": row["adapter"],
        "dimensions": row["dimensions"],
        "distance": row["distance"],
        "revision": int(row["revision"]) if "revision" in row.keys() else 0,
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def chunk_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "collection_id": row["collection_id"],
        "source_id": row["source_id"],
        "chunk_id": row["chunk_id"],
        "content_hash": row["content_hash"],
        "content": row["content"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "embedding_provider": row["embedding_provider"],
        "embedding_model": row["embedding_model"],
        "dimensions": row["dimensions"],
        "vector": json.loads(row["vector_json"] or "[]"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def without_vector(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "vector"}


def useful_terms(text: str) -> set[str]:
    return {term.lower() for term in text.split() if len(term.strip(".,:;!?()[]{}").strip()) >= 3}


def lexical_score(query_terms: set[str], content: str) -> float:
    if not query_terms:
        return 0.0
    terms = useful_terms(content)
    if not terms:
        return 0.0
    return len(query_terms & terms) / len(query_terms)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]
