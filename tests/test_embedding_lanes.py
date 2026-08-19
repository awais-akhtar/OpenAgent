from __future__ import annotations

from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openagent.embeddings import EmbeddingRouter, FastEmbedProvider
from openagent.knowledge import KnowledgeEngine
from openagent.storage import OpenAgentStore


class EmbeddingLaneTests(unittest.TestCase):
    def test_fastembed_lane_is_lazy_and_normalizes_to_collection_dimensions(self) -> None:
        class FakeTextEmbedding:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def embed(self, texts):
                return iter([[3.0, 4.0]])

        fake_module = SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        with patch.object(FastEmbedProvider, "available", return_value=True), patch.dict(
            "sys.modules", {"fastembed": fake_module}
        ):
            result = EmbeddingRouter().embed("local document", provider="fastembed", dimensions=8)

        self.assertEqual(result.provider, "fastembed")
        self.assertEqual(result.dimensions, 8)
        self.assertAlmostEqual(result.vector[0], 0.6)
        self.assertAlmostEqual(result.vector[1], 0.8)
        self.assertAlmostEqual(sum(value * value for value in result.vector), 1.0)

    def test_ingestion_records_the_selected_embedding_provider(self) -> None:
        with TemporaryDirectory() as temp:
            knowledge = KnowledgeEngine(OpenAgentStore(temp))
            with patch.object(knowledge.vector_store.embeddings, "embed", wraps=knowledge.vector_store.embeddings.embed) as embed:
                knowledge.ingest_text("Source", "A local source for a retrieval test.", embedding_provider="local-hash")

        self.assertEqual(embed.call_args.kwargs["provider"], "local-hash")


if __name__ == "__main__":
    unittest.main()
