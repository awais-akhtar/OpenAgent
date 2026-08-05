from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.advanced_rag import AdvancedRAGEngine
from openagent.knowledge import KnowledgeEngine
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.vector_store import VectorStore


class AdvancedRAGTests(unittest.TestCase):
    def test_grounded_query_uses_sources_and_generation_aware_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            vectors = VectorStore(store)
            knowledge = KnowledgeEngine(store, vector_store=vectors)
            rag = AdvancedRAGEngine(store, knowledge=knowledge, vector_store=vectors)
            knowledge.ingest_text(
                "Private retrieval",
                "FAISS performs fast local similarity search. ChromaDB keeps a persistent local vector mirror.",
                project="core",
            )

            first = rag.query("How is local vector retrieval accelerated?", project="core")
            second = rag.query("How is local vector retrieval accelerated?", project="core")
            knowledge.ingest_text(
                "Grounding",
                "Every RAG answer should preserve source citations and abstain when evidence is weak.",
                project="core",
            )
            after_change = rag.query("How is local vector retrieval accelerated?", project="core")
            rag.close()

        self.assertGreaterEqual(len(first["sources"]), 1)
        self.assertIn("[S1]", first["answer"])
        self.assertFalse(first["cache"]["hit"])
        self.assertTrue(second["cache"]["hit"])
        self.assertNotEqual(second["cache"]["knowledge_generation"], after_change["cache"]["knowledge_generation"])
        self.assertFalse(after_change["cache"]["hit"])
        self.assertIn("faiss-chroma-consensus", first["retrieval"]["techniques"])

    def test_http_rag_status_and_query_contracts(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                status = json.loads(urlopen(f"{base}/api/rag/status?project=core", timeout=10).read())
                request = Request(
                    f"{base}/api/rag/query",
                    data=json.dumps({"project": "core", "query": "missing evidence"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                result = json.loads(urlopen(request, timeout=10).read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(status["algorithm"], "openagent-lattice-rag-v1")
        self.assertEqual(result["confidence"]["level"], "insufficient")


if __name__ == "__main__":
    unittest.main()
