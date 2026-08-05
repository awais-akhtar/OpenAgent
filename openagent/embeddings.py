"""Local-first embedding providers for OpenAgent."""

from __future__ import annotations

from dataclasses import dataclass
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


class EmbeddingRouter:
    """Select embedding providers while preserving explicit privacy boundaries."""

    def __init__(self) -> None:
        self.local_hash = LocalHashEmbeddingProvider()

    def providers(self) -> list[dict[str, Any]]:
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
                "available": False,
                "default": False,
                "quality": "high",
                "reason": "Optional dependency/provider hook; not required for offline fallback.",
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
        provider = (provider or "local-hash").strip()
        if provider in {"", "local", "local-hash", "auto"}:
            return self.local_hash.embed(text, dimensions=dimensions)
        raise ValueError(f"Embedding provider '{provider}' is not configured for local use.")
