from __future__ import annotations

import importlib.util
from tempfile import TemporaryDirectory
import unittest

from openagent.storage import OpenAgentStore
from openagent.vector_store import VectorStore


class VectorAccelerationTests(unittest.TestCase):
    def test_status_preserves_sqlite_authority_and_reports_accelerators(self) -> None:
        with TemporaryDirectory() as tmp:
            vectors = VectorStore(OpenAgentStore(tmp))
            status = vectors.status(project="core")

        self.assertEqual(status["adapter"], "sqlite")
        self.assertEqual(status["source_of_truth"], "sqlite")
        self.assertSetEqual({item["id"] for item in status["accelerators"]}, {"faiss", "chroma"})

    @unittest.skipUnless(
        importlib.util.find_spec("faiss") and importlib.util.find_spec("chromadb"),
        "FAISS and ChromaDB optional dependencies are not installed.",
    )
    def test_real_faiss_and_chroma_backends_are_used_together(self) -> None:
        with TemporaryDirectory() as tmp:
            vectors = VectorStore(OpenAgentStore(tmp))
            vectors.upsert_text(
                collection="knowledge",
                project="core",
                source_id="privacy",
                chunk_id="one",
                content="Private local vector retrieval uses FAISS and ChromaDB.",
            )
            vectors.upsert_text(
                collection="knowledge",
                project="core",
                source_id="calendar",
                chunk_id="two",
                content="Calendar scheduling handles local meetings.",
            )
            result = vectors.search(
                "private local vector retrieval",
                collection="knowledge",
                project="core",
                limit=2,
            )
            vectors.close()

        self.assertEqual(result["results"][0]["source_id"], "privacy")
        self.assertIn("faiss", result["acceleration"]["active"])
        self.assertIn("chroma", result["acceleration"]["active"])
        self.assertEqual(result["acceleration"]["errors"], {})
        self.assertIn("faiss", result["results"][0]["retrieval"])
        self.assertIn("chroma", result["results"][0]["retrieval"])


if __name__ == "__main__":
    unittest.main()
