from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_upsert_search_and_status_are_local_and_deterministic(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            vectors = VectorStore(store)

            inserted = vectors.upsert_text(
                collection="knowledge",
                project="core",
                source_id="doc-1",
                chunk_id="0",
                content="Durable worker queues use leases, retries, and dead letters.",
            )
            self.assertEqual(inserted["source_id"], "doc-1")
            self.assertEqual(vectors.status(project="core")["chunk_count"], 1)

            result = vectors.search("worker retries", collection="knowledge", project="core", limit=3)
            self.assertEqual(len(result["results"]), 1)
            self.assertEqual(result["results"][0]["source_id"], "doc-1")
            self.assertIn("local-embedding", result["results"][0]["retrieval"])
            self.assertEqual(result["embedding"]["provider"], "local-hash")

    def test_benchmark_indexes_and_queries(self) -> None:
        with TemporaryDirectory() as tmp:
            vectors = VectorStore(OpenAgentStore(tmp))
            benchmark = vectors.benchmark(project="core", collection="bench")
            self.assertEqual(benchmark["status"], "ok")
            self.assertEqual(benchmark["indexed"], 3)
            self.assertIsNotNone(benchmark["top_result"])


class VectorRouteTests(unittest.TestCase):
    def test_vector_routes_upsert_search_and_benchmark(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                upsert = Request(
                    f"{base_url}/api/vector-store/upsert",
                    data=json.dumps(
                        {
                            "project": "core",
                            "collection": "knowledge",
                            "source_id": "note-1",
                            "content": "OpenAgent vector storage remains local.",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                created = json.loads(urlopen(upsert, timeout=5).read().decode("utf-8"))
                self.assertEqual(created["record"]["source_id"], "note-1")

                search = Request(
                    f"{base_url}/api/vector-store/search",
                    data=json.dumps(
                        {
                            "project": "core",
                            "collection": "knowledge",
                            "query": "local vector storage",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                result = json.loads(urlopen(search, timeout=5).read().decode("utf-8"))
                self.assertEqual(result["results"][0]["source_id"], "note-1")

                status = json.loads(urlopen(f"{base_url}/api/vector-store/status?project=core", timeout=5).read().decode("utf-8"))
                self.assertEqual(status["adapter"], "sqlite")
                self.assertEqual(status["chunk_count"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
