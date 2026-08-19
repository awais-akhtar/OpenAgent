"""Optional local vector accelerators backed by FAISS and ChromaDB.

SQLite remains the durable source of truth. FAISS is rebuilt lazily from those
rows for low-latency in-memory search, while Chroma stores a local persistent
mirror that can survive process restarts. Both adapters receive caller-supplied
embeddings so neither library downloads a model or sends text over the network.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


@dataclass
class _FaissSnapshot:
    signature: str
    index: Any
    record_ids: list[str]


class LocalVectorAccelerator:
    """Coordinate FAISS and Chroma without changing the source-of-truth store."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.vector_dir = self.data_dir / "vectors"
        self.chroma_dir = self.vector_dir / "chroma"
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._faiss: dict[str, _FaissSnapshot] = {}
        self._chroma_client: Any | None = None
        default_keep_open = "0" if os.name == "nt" else "1"
        self._keep_chroma_open = os.environ.get(
            "OPENAGENT_CHROMA_KEEP_OPEN", default_keep_open
        ).strip().lower() in {"1", "true", "yes", "on"}
        requested = os.environ.get("OPENAGENT_VECTOR_ACCELERATORS", "faiss,chroma")
        self.requested = {item.strip().lower() for item in requested.split(",") if item.strip()}

    def status(self) -> dict[str, Any]:
        faiss_available = _module_available("faiss") and _module_available("numpy")
        chroma_available = _module_available("chromadb")
        return {
            "source_of_truth": "sqlite",
            "active": [
                name
                for name, enabled in (
                    ("faiss", "faiss" in self.requested and faiss_available),
                    ("chroma", "chroma" in self.requested and chroma_available),
                )
                if enabled
            ],
            "backends": [
                {
                    "id": "faiss",
                    "available": faiss_available,
                    "enabled": "faiss" in self.requested and faiss_available,
                    "persistence": "rebuilt-lazily-from-sqlite",
                    "distance": "cosine-via-normalized-inner-product",
                },
                {
                    "id": "chroma",
                    "available": chroma_available,
                    "enabled": "chroma" in self.requested and chroma_available,
                    "persistence": str(self.chroma_dir),
                    "distance": "cosine",
                    "handle_mode": "persistent" if self._keep_chroma_open else "close-after-operation",
                },
            ],
            "privacy_boundary": (
                "FAISS and Chroma run in-process against the configured local data directory. "
                "Embeddings are supplied by OpenAgent; no backend embedding download or network call is permitted."
            ),
        }

    def invalidate(self, collection_id: str) -> None:
        with self._lock:
            self._faiss.pop(collection_id, None)

    def close(self) -> None:
        """Release native index handles, especially Chroma HNSW files on Windows."""

        with self._lock:
            client = self._chroma_client
            self._chroma_client = None
            self._faiss.clear()
        if client is not None:
            self._close_chroma_client(client)

    def mirror_upsert(self, collection: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        """Mirror one committed SQLite row into enabled secondary indexes."""

        self.invalidate(str(collection["id"]))
        mirrored: list[str] = []
        errors: dict[str, str] = {}
        if self._chroma_enabled():
            try:
                with self._chroma_session() as client:
                    chroma_collection = self._chroma_collection(collection, client)
                    chroma_collection.upsert(
                        ids=[str(record["id"])],
                        embeddings=[[float(value) for value in record["vector"]]],
                        documents=[str(record["content"])],
                        metadatas=[_chroma_metadata(record)],
                    )
                mirrored.append("chroma")
            except Exception as exc:
                errors["chroma"] = _safe_error(exc)
        return {"mirrored": mirrored, "errors": errors}

    def mirror_delete(self, collection: dict[str, Any], record_ids: list[str]) -> dict[str, Any]:
        """Remove committed rows from secondary indexes after SQLite deletion."""

        self.invalidate(str(collection["id"]))
        removed: list[str] = []
        errors: dict[str, str] = {}
        ids = [str(value) for value in record_ids if str(value)]
        if ids and self._chroma_enabled():
            try:
                with self._chroma_session() as client:
                    self._chroma_collection(collection, client).delete(ids=ids)
                removed.append("chroma")
            except Exception as exc:
                errors["chroma"] = _safe_error(exc)
        return {"removed": removed, "errors": errors}

    def search(
        self,
        collection: dict[str, Any],
        query_vector: list[float],
        rows: list[dict[str, Any]],
        *,
        limit: int,
    ) -> dict[str, Any]:
        """Search every enabled local accelerator and return backend rankings."""

        started = time.perf_counter()
        rankings: dict[str, list[dict[str, Any]]] = {}
        errors: dict[str, str] = {}
        candidate_limit = max(1, min(int(limit), max(1, len(rows))))
        if rows and self._faiss_enabled():
            try:
                rankings["faiss"] = self._search_faiss(
                    str(collection["id"]),
                    str(collection.get("revision", 0)),
                    query_vector,
                    rows,
                    candidate_limit,
                )
            except Exception as exc:
                errors["faiss"] = _safe_error(exc)
        if rows and self._chroma_enabled():
            try:
                rankings["chroma"] = self._search_chroma(
                    collection, query_vector, rows, candidate_limit
                )
            except Exception as exc:
                errors["chroma"] = _safe_error(exc)
        return {
            "rankings": rankings,
            "errors": errors,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def _search_faiss(
        self,
        collection_id: str,
        signature: str,
        query_vector: list[float],
        rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        import faiss  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        dimensions = len(query_vector)
        with self._lock:
            snapshot = self._faiss.get(collection_id)
            if snapshot is None or snapshot.signature != signature:
                matrix = np.asarray([row["vector"] for row in rows], dtype="float32")
                if matrix.ndim != 2 or matrix.shape[1] != dimensions:
                    raise ValueError("FAISS vector dimensions do not match the query.")
                faiss.normalize_L2(matrix)
                index = faiss.IndexFlatIP(dimensions)
                index.add(matrix)
                snapshot = _FaissSnapshot(
                    signature=signature,
                    index=index,
                    record_ids=[str(row["id"]) for row in rows],
                )
                self._faiss[collection_id] = snapshot
            query = np.asarray([query_vector], dtype="float32")
            faiss.normalize_L2(query)
            scores, indexes = snapshot.index.search(query, limit)
        results: list[dict[str, Any]] = []
        for score, position in zip(scores[0].tolist(), indexes[0].tolist()):
            if position < 0 or position >= len(snapshot.record_ids):
                continue
            results.append({"id": snapshot.record_ids[position], "score": float(score)})
        return results

    def _search_chroma(
        self,
        collection: dict[str, Any],
        query_vector: list[float],
        rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        with self._chroma_session() as client:
            chroma_collection = self._chroma_collection(collection, client)
            if int(chroma_collection.count()) != len(rows):
                self._sync_chroma(chroma_collection, rows)
            payload = chroma_collection.query(
                query_embeddings=[[float(value) for value in query_vector]],
                n_results=limit,
                include=["distances"],
            )
        ids = (payload.get("ids") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        return [
            {"id": str(record_id), "score": max(-1.0, min(1.0, 1.0 - float(distance)))}
            for record_id, distance in zip(ids, distances)
        ]

    def _sync_chroma(self, chroma_collection: Any, rows: list[dict[str, Any]]) -> None:
        batch_size = 256
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            chroma_collection.upsert(
                ids=[str(row["id"]) for row in batch],
                embeddings=[[float(value) for value in row["vector"]] for row in batch],
                documents=[str(row["content"]) for row in batch],
                metadatas=[_chroma_metadata(row) for row in batch],
            )

    def _chroma_collection(self, collection: dict[str, Any], client: Any) -> Any:
        name = f"oa_{hashlib.sha256(str(collection['id']).encode('utf-8')).hexdigest()[:24]}"
        return client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "openagent_collection": str(collection.get("name", "default")),
                "openagent_project": str(collection.get("project", "default")),
            },
            embedding_function=None,
        )

    def _new_chroma_client(self) -> Any:
        import chromadb  # type: ignore[import-not-found]
        from chromadb.config import Settings  # type: ignore[import-not-found]

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def _get_chroma_client(self) -> Any:
        with self._lock:
            if self._chroma_client is None:
                self._chroma_client = self._new_chroma_client()
            return self._chroma_client

    @contextmanager
    def _chroma_session(self) -> Any:
        client = self._get_chroma_client() if self._keep_chroma_open else self._new_chroma_client()
        try:
            yield client
        finally:
            if not self._keep_chroma_open:
                self._close_chroma_client(client)

    @staticmethod
    def _close_chroma_client(client: Any) -> None:
        """Release Chroma's SQLite handle before a Windows workspace is removed."""

        system = getattr(client, "_system", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()
        if os.name == "nt" and system is not None:
            stop = getattr(system, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
        gc.collect()

    def _faiss_enabled(self) -> bool:
        return "faiss" in self.requested and _module_available("faiss") and _module_available("numpy")

    def _chroma_enabled(self) -> bool:
        return "chroma" in self.requested and _module_available("chromadb")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _chroma_metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "project": str(record.get("project", "default")),
        "source_id": str(record.get("source_id", "")),
        "chunk_id": str(record.get("chunk_id", "")),
        "content_hash": str(record.get("content_hash", "")),
    }
    for key, value in (record.get("metadata") or {}).items():
        clean_key = f"meta_{str(key)[:48]}"
        if isinstance(value, (str, int, float, bool)):
            metadata[clean_key] = value
        elif value is not None:
            metadata[clean_key] = json.dumps(value, sort_keys=True)[:1000]
    return metadata


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:240]}"
