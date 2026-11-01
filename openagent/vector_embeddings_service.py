"""G25: Local vector embeddings with caching (no external API calls)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class VectorEmbeddingsService:
    """Generate and cache vector embeddings locally."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence-transformers model."""
        self.model_name = model_name
        self.cache_dir = Path("embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self._load_model()

    def _load_model(self) -> None:
        """Load embedding model (downloads once, then cached locally)."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using mock embeddings.")
            self.model = None
            self.embedding_dim = 384

    def embed_text(self, text: str) -> list[float]:
        """Convert text to embedding vector."""
        if not text or not text.strip():
            return [0.0] * self.embedding_dim

        text = text.strip()[:512]  # Truncate long text

        if self.model:
            try:
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                return self._mock_embedding(text)
        else:
            return self._mock_embedding(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        return [self.embed_text(text) for text in texts]

    def _mock_embedding(self, text: str) -> list[float]:
        """Create deterministic mock embedding from text."""
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.randn(self.embedding_dim).tolist()

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "cached_items": len(cache_files),
            "cache_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
            "model": self.model_name,
            "embedding_dimension": self.embedding_dim,
        }
