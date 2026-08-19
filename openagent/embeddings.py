"""Local-first embedding providers for OpenAgent."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import threading
from typing import Any

from .model_gateway import local_hash_embedding


@dataclass(frozen=True)
class EmbeddingResult:
    provider: str
    model: str
    dimensions: int
    vector: list[float]
    privacy_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "vector": self.vector,
            "privacy_boundary": self.privacy_boundary,
        }


class LocalHashEmbeddingProvider:
    """Deterministic CPU-only embedding fallback for offline indexing."""

    provider_id = "local-hash"
    model = "openagent-local-hash-embedding"
    privacy_boundary = (
        "Embedding is computed in-process from local text using deterministic hashing. "
        "No model server or external network call is used."
    )

    def embed(self, text: str, *, dimensions: int = 128) -> EmbeddingResult:
        dimensions = max(8, min(int(dimensions), 1024))
        return EmbeddingResult(
            provider=self.provider_id,
            model=self.model,
            dimensions=dimensions,
            vector=local_hash_embedding(text, dimensions=dimensions),
            privacy_boundary=self.privacy_boundary,
        )


class FastEmbedProvider:
    """Lazy local ONNX embeddings for users who want semantic quality over fallback speed.

    The model is loaded only when this lane is selected. That keeps a fresh
    installation lightweight and makes the network boundary explicit: the
    first model download, if needed, is a user-controlled setup action.
    """

    provider_id = "fastembed"
    default_model = "BAAI/bge-small-en-v1.5"
    privacy_boundary = (
        "FastEmbed runs the selected ONNX embedding model locally. Model setup may "
        "download model files only when the user enables this lane."
    )

    def __init__(self) -> None:
        self.model_name = os.getenv("OPENAGENT_FASTEMBED_MODEL", self.default_model).strip() or self.default_model
        cache_value = os.getenv("OPENAGENT_FASTEMBED_CACHE_DIR", "").strip()
        self.cache_dir = Path(cache_value).expanduser() if cache_value else None
        self._model: Any | None = None
        self._lock = threading.RLock()

    @staticmethod
    def available() -> bool:
        try:
            return importlib.util.find_spec("fastembed") is not None
        except (ImportError, ValueError):
            return False

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.available():
                raise ValueError(
                    "FastEmbed is not installed. Install the optional RAG dependencies with "
                    "`pip install -e .[rag]`."
                )
            from fastembed import TextEmbedding

            kwargs: dict[str, Any] = {"model_name": self.model_name}
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                kwargs["cache_dir"] = str(self.cache_dir)
            try:
                self._model = TextEmbedding(**kwargs)
            except TypeError:
                # Older FastEmbed releases do not expose cache_dir.
                kwargs.pop("cache_dir", None)
                self._model = TextEmbedding(**kwargs)
            return self._model

    def embed(self, text: str, *, dimensions: int = 128) -> EmbeddingResult:
        text = str(text or "").strip()
        if not text:
            raise ValueError("text is required for embedding")
        model = self._load()
        raw = next(iter(model.embed([text])))
        vector = _fit_dimensions([float(value) for value in raw], dimensions)
        return EmbeddingResult(
            provider=self.provider_id,
            model=self.model_name,
            dimensions=len(vector),
            vector=vector,
            privacy_boundary=self.privacy_boundary,
        )


class EmbeddingRouter:
    """Select embedding providers while preserving explicit privacy boundaries."""

    def __init__(self) -> None:
        self.local_hash = LocalHashEmbeddingProvider()
        self.fastembed = FastEmbedProvider()

    def providers(self) -> list[dict[str, Any]]:
        fastembed_available = self.fastembed.available()
        return [
            {
                "id": self.local_hash.provider_id,
                "name": "Local deterministic hash embeddings",
                "available": True,
                "default": True,
                "quality": "fallback",
                "privacy_boundary": self.local_hash.privacy_boundary,
            },
            {
                "id": "fastembed",
                "name": "FastEmbed / ONNX",
                "available": fastembed_available,
                "default": False,
                "quality": "high",
                "model": self.fastembed.model_name,
                "reason": "Install the optional RAG dependencies to enable this lane." if not fastembed_available else "Ready for local ONNX embeddings.",
                "privacy_boundary": "Runs local embedding models when installed and configured.",
            },
            {
                "id": "ollama",
                "name": "Ollama embeddings",
                "available": False,
                "default": False,
                "quality": "medium",
                "reason": "Provider hook; requires a local Ollama embedding model.",
                "privacy_boundary": "Calls the configured local Ollama server only.",
            },
            {
                "id": "openai-compatible",
                "name": "OpenAI-compatible embeddings",
                "available": False,
                "default": False,
                "quality": "external",
                "reason": "External provider hook; requires explicit approval and configuration.",
                "privacy_boundary": "Crosses the local boundary only when explicitly configured and approved.",
            },
        ]

    def embed(self, text: str, *, provider: str = "local-hash", dimensions: int = 128) -> EmbeddingResult:
        provider = (provider or os.getenv("OPENAGENT_EMBEDDING_PROVIDER", "local-hash")).strip().lower()
        if provider in {"", "local", "local-hash", "auto"}:
            return self.local_hash.embed(text, dimensions=dimensions)
        if provider in {"fastembed", "fastembed-local", "onnx"}:
            return self.fastembed.embed(text, dimensions=dimensions)
        raise ValueError(f"Embedding provider '{provider}' is not configured for local use.")


def _fit_dimensions(values: list[float], dimensions: int) -> list[float]:
    """Keep all local lanes compatible with a collection's fixed dimension."""

    target = max(8, min(int(dimensions), 1024))
    if len(values) > target:
        values = values[:target]
    elif len(values) < target:
        values = [*values, *([0.0] * (target - len(values)))]
    norm = sum(value * value for value in values) ** 0.5
    if norm:
        return [value / norm for value in values]
    return values
