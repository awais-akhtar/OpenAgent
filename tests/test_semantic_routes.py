from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import threading
from unittest import TestCase
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import required_scope_for_path
from openagent.routes.semantic_routes import SemanticRouteDependencies, contract, handle_delete, handle_get, handle_post
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class SemanticRouteTests(TestCase):
    def dependencies(self, payload: dict | None = None) -> tuple[Mock, SemanticRouteDependencies]:
        memory = Mock()
        return memory, SemanticRouteDependencies(memory, lambda: payload if payload is not None else {})

    def test_get_routes_preserve_payloads_and_query_defaults(self) -> None:
        memory, deps = self.dependencies()
        memory.status.return_value = {"project": "core", "semantic_memories": 1}
        memory.list_memories.return_value = [{"id": "m1"}]
        memory.search_memories.return_value = [{"id": "m1", "score": 0.8}]
        memory.collections.return_value = [{"id": "c1"}]
        memory.graph_analytics.return_value = {"project": "core", "entities": 2}

        result = handle_get("/api/semantic/status", "project=core", deps)
        self.assertEqual((result.payload, result.status), (memory.status.return_value, HTTPStatus.OK))
        memory.status.assert_called_once_with(project="core")

        result = handle_get("/api/semantic/memories", "project=core&memory_type=preference&limit=bad", deps)
        self.assertEqual((result.payload, result.status), ({"memories": [{"id": "m1"}]}, HTTPStatus.OK))
        memory.list_memories.assert_called_once_with(project="core", memory_type="preference", limit=50)

        result = handle_get("/api/semantic/memories", "q=local&memory_type=ignored&limit=3", deps)
        self.assertEqual(result.payload, {"memories": memory.search_memories.return_value})
        memory.search_memories.assert_called_once_with("local", project="default", limit=3)

        result = handle_get("/api/semantic/collections", "", deps)
        self.assertEqual(result.payload, {"collections": memory.collections.return_value})
        memory.collections.assert_called_once_with(project="default", include_archived=False)

        memory.reset_mock()
        result = handle_get("/api/semantic/collections", "project=core&include_archived=true", deps)
        self.assertEqual(result.payload, {"collections": memory.collections.return_value})
        memory.collections.assert_called_once_with(project="core", include_archived=True)

        result = handle_get("/api/semantic/graph-analytics", "project=core", deps)
        self.assertEqual(result.payload, memory.graph_analytics.return_value)
        memory.graph_analytics.assert_called_once_with(project="core")

    def test_memory_post_normalization_and_validation(self) -> None:
        payload = {
            "title": "  Title  ", "content": "  Body  ", "project": "  ",
            "memory_type": " ", "profile_id": "  user  ", "tags": ["A", 2],
            "source": "  import  ", "confidence": 0, "metadata": ["ignored"],
        }
        memory, deps = self.dependencies(payload)
        memory.add_memory.return_value = {"id": "m1"}
        result = handle_post("/api/semantic/memories", deps)
        self.assertEqual((result.payload, result.status), ({"memory": {"id": "m1"}}, HTTPStatus.CREATED))
        memory.add_memory.assert_called_once_with(
            title="Title", content="Body", project="default", memory_type="semantic",
            profile_id="user", tags=["A", "2"], source="import", confidence=1.0, metadata={},
        )

        memory, deps = self.dependencies({"content": "Body", "confidence": "invalid"})
        result = handle_post("/api/semantic/memories", deps)
        self.assertEqual(
            (result.payload, result.status),
            ({"error": "could not convert string to float: 'invalid'"}, HTTPStatus.BAD_REQUEST),
        )
        memory.add_memory.assert_not_called()

        memory, deps = self.dependencies({"content": " "})
        memory.add_memory.side_effect = ValueError("Memory content is required.")
        result = handle_post("/api/semantic/memories", deps)
        self.assertEqual((result.payload, result.status), ({"error": "Memory content is required."}, HTTPStatus.BAD_REQUEST))

    def test_consolidate_and_collection_posts(self) -> None:
        memory, deps = self.dependencies({"project": " core ", "limit": "bad"})
        memory.consolidate.return_value = {"summary": {}, "memory": {"id": "m1"}}
        result = handle_post("/api/semantic/consolidate", deps)
        self.assertEqual((result.payload, result.status), (memory.consolidate.return_value, HTTPStatus.CREATED))
        memory.consolidate.assert_called_once_with(project="core", limit=100)

        memory, deps = self.dependencies({"name": " research ", "project": " core ", "dimensions": "bad", "metadata": []})
        memory.create_collection.return_value = {"id": "c1"}
        result = handle_post("/api/semantic/collections", deps)
        self.assertEqual((result.payload, result.status), ({"collection": {"id": "c1"}}, HTTPStatus.CREATED))
        memory.create_collection.assert_called_once_with("research", project="core", dimensions=64, metadata={})

    def test_semantic_collection_rename_and_delete_are_project_scoped(self) -> None:
        self.assertEqual(required_scope_for_path("/api/semantic/collections/research/rename", "POST"), "semantic:write")
        self.assertEqual(required_scope_for_path("/api/semantic/collections/research/archive", "POST"), "semantic:write")
        self.assertEqual(required_scope_for_path("/api/semantic/collections/research", "DELETE"), "semantic:write")
        memory, deps = self.dependencies({"new_name": " Renamed ", "project": " core "})
        memory.rename_collection.return_value = {"id": "c1", "name": "Renamed", "project": "core"}
        result = handle_post("/api/semantic/collections/original/rename", deps)
        self.assertEqual((result.payload, result.status), ({"collection": memory.rename_collection.return_value}, HTTPStatus.OK))
        memory.rename_collection.assert_called_once_with("original", new_name="Renamed", project="core")

        memory.rename_collection.side_effect = ValueError("A collection with that name already exists in this project.")
        result = handle_post("/api/semantic/collections/original/rename", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

        memory, deps = self.dependencies({"archived": True, "project": "core"})
        memory.set_collection_archived.return_value = {"id": "c1", "name": "original", "archived": True}
        result = handle_post("/api/semantic/collections/original/archive", deps)
        self.assertEqual((result.payload, result.status), ({"collection": memory.set_collection_archived.return_value}, HTTPStatus.OK))
        memory.set_collection_archived.assert_called_once_with("original", archived=True, project="core")
        memory, deps = self.dependencies({"archived": "true"})
        self.assertEqual(handle_post("/api/semantic/collections/original/archive", deps).status, HTTPStatus.BAD_REQUEST)

        memory, deps = self.dependencies()
        memory.delete_collection.return_value = {"deleted": True, "name": "Renamed", "records_removed": 3}
        result = handle_delete("/api/semantic/collections/Renamed", "project=core", deps)
        self.assertEqual((result.payload, result.status), (memory.delete_collection.return_value, HTTPStatus.OK))
        memory.delete_collection.assert_called_once_with("Renamed", project="core")
        memory.delete_collection.side_effect = KeyError("missing")
        result = handle_delete("/api/semantic/collections/missing", "project=core", deps)
        self.assertEqual((result.payload, result.status), ({"error": "Collection not found."}, HTTPStatus.NOT_FOUND))

    def test_vectors_post_converts_values_and_reports_errors(self) -> None:
        memory, deps = self.dependencies({
            "collection_id": " c1 ", "content": " body ", "project": " core ",
            "external_id": " doc ", "vector": [1, "2.5"], "metadata": {"source": "test"},
        })
        memory.upsert_vector_record.return_value = {"id": "v1"}
        result = handle_post("/api/semantic/vectors", deps)
        self.assertEqual((result.payload, result.status), ({"record": {"id": "v1"}}, HTTPStatus.CREATED))
        memory.upsert_vector_record.assert_called_once_with(
            "c1", content="body", project="core", external_id="doc",
            vector=[1.0, 2.5], metadata={"source": "test"},
        )

        memory, deps = self.dependencies({"vector": ["invalid"]})
        result = handle_post("/api/semantic/vectors", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("could not convert string to float", result.payload["error"])
        memory.upsert_vector_record.assert_not_called()

        memory, deps = self.dependencies({"collection_id": "missing"})
        memory.upsert_vector_record.side_effect = ValueError("Unknown vector collection 'missing'.")
        result = handle_post("/api/semantic/vectors", deps)
        self.assertEqual((result.payload, result.status), ({"error": "Unknown vector collection 'missing'."}, HTTPStatus.BAD_REQUEST))

    def test_vector_search_and_rerank_status_and_validation(self) -> None:
        memory, deps = self.dependencies({"collection_id": " c1 ", "query": " local ", "limit": "bad"})
        memory.vector_search.return_value = [{"id": "v1", "score": 0.9}]
        result = handle_post("/api/semantic/vector-search", deps)
        self.assertEqual((result.payload, result.status), ({"results": memory.vector_search.return_value}, HTTPStatus.OK))
        memory.vector_search.assert_called_once_with("c1", "local", limit=10)

        memory.vector_search.side_effect = ValueError("Unknown vector collection 'c1'.")
        result = handle_post("/api/semantic/vector-search", deps)
        self.assertEqual((result.payload, result.status), ({"error": "Unknown vector collection 'c1'."}, HTTPStatus.BAD_REQUEST))

        memory, deps = self.dependencies({"candidates": "invalid"})
        result = handle_post("/api/semantic/rerank", deps)
        self.assertEqual((result.payload, result.status), ({"error": "candidates must be a list."}, HTTPStatus.BAD_REQUEST))
        memory.rerank.assert_not_called()

        memory, deps = self.dependencies({"query": " local ", "candidates": [{"id": "a"}, 1, "bad"], "project": " core ", "limit": "4"})
        memory.rerank.return_value = {"id": "r1", "results": []}
        result = handle_post("/api/semantic/rerank", deps)
        self.assertEqual((result.payload, result.status), (memory.rerank.return_value, HTTPStatus.CREATED))
        memory.rerank.assert_called_once_with("local", [{"id": "a"}], project="core", limit=4)

    def test_contract_paths_are_handled_and_other_paths_are_not(self) -> None:
        memory, deps = self.dependencies()
        for route in contract()["routes"]:
            with self.subTest(route=route):
                result = handle_get(route["path"], "", deps) if route["method"] == "GET" else handle_delete(route["path"], "", deps) if route["method"] == "DELETE" else handle_post(route["path"], deps)
                self.assertIsNotNone(result)
        self.assertIsNone(handle_get("/api/other", "", deps))
        self.assertIsNone(handle_post("/api/other", deps))

    def test_http_contract_preserves_memory_creation_and_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = Request(
                    f"{base}/api/semantic/memories",
                    data=json.dumps({"project": "core", "title": "Source", "content": "Local notes"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.CREATED)
                    created = json.load(response)
                with urlopen(f"{base}/api/semantic/memories?project=core", timeout=10) as response:
                    listing = json.load(response)
                self.assertEqual(listing["memories"][0]["id"], created["memory"]["id"])
                invalid = Request(
                    f"{base}/api/semantic/memories",
                    data=json.dumps({"content": ""}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with self.assertRaises(HTTPError) as error:
                    urlopen(invalid, timeout=10)
                with error.exception as response:
                    self.assertEqual(response.code, HTTPStatus.BAD_REQUEST)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_runtime_has_no_inline_semantic_dispatch(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/semantic/', runtime)
