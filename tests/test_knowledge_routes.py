from __future__ import annotations

import json
import threading
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen

from openagent.model_gateway import ProviderConfigurationError
from openagent.routes.knowledge_routes import KnowledgeRouteDependencies, contract, handle_delete, handle_get, handle_post, handle_put
from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


class KnowledgeRouteTests(TestCase):
    def dependencies(self, payload: dict | None = None, *, body: bytes = b"", headers: dict | None = None, owner: str = ""):
        knowledge, gateway, media = Mock(), Mock(), Mock()
        read_json = Mock(return_value=payload if payload is not None else {})
        read_body = Mock(return_value=body)
        deps = KnowledgeRouteDependencies(
            knowledge, gateway, media, read_json, headers or {}, read_body, 100_000,
            auth_scope_owner=lambda: owner,
        )
        return knowledge, gateway, media, deps

    def test_get_routes_preserve_payloads_defaults_and_limit_fallback(self):
        knowledge, _, _, deps = self.dependencies()
        knowledge.status.return_value = {"documents": 2}
        knowledge.list_documents.return_value = [{"id": "d1"}]
        knowledge.search.return_value = [{"id": "lexical"}]
        knowledge.semantic_search.return_value = [{"id": "semantic"}]
        knowledge.graph.return_value = {"nodes": []}
        knowledge.tags.return_value = [{"tag": "local"}]
        knowledge.timeline.return_value = [{"type": "document"}]

        result = handle_get("/api/knowledge/status", "project=core", deps)
        self.assertEqual((result.payload, result.status), ({"documents": 2}, HTTPStatus.OK))
        knowledge.status.assert_called_once_with(project="core")

        result = handle_get("/api/knowledge/documents", "project=core&limit=bad", deps)
        self.assertEqual(result.payload, {"documents": [{"id": "d1"}]})
        knowledge.list_documents.assert_called_once_with(project="core", limit=100, owner="")

        result = handle_get("/api/knowledge/search", "q=local&mode=fts", deps)
        self.assertEqual(result.payload, {"results": [{"id": "lexical"}]})
        knowledge.search.assert_called_once_with("local", project="default", owner="")
        result = handle_get("/api/knowledge/search", "mode=semantic", deps)
        self.assertEqual(result.payload, {"results": [{"id": "semantic"}]})
        knowledge.semantic_search.assert_called_once_with("", project="default", owner="")

        result = handle_get("/api/knowledge/graph", "project=core", deps)
        self.assertEqual(result.payload, {"nodes": []})
        knowledge.graph.assert_called_once_with(project="core")
        result = handle_get("/api/knowledge/tags", "limit=4", deps)
        self.assertEqual(result.payload, {"tags": [{"tag": "local"}]})
        knowledge.tags.assert_called_once_with(project="default", limit=4)
        result = handle_get("/api/knowledge/timeline", "project=core&limit=bad", deps)
        self.assertEqual(result.payload, {"timeline": [{"type": "document"}]})
        knowledge.timeline.assert_called_once_with(project="core", limit=100)

    def test_document_detail_is_project_scoped_and_missing_document_is_404(self):
        knowledge, _, _, deps = self.dependencies()
        knowledge.get_document.return_value = {"id": "doc-1", "project": "core", "content": "Source text"}
        result = handle_get("/api/knowledge/documents/doc-1", "project=core", deps)
        self.assertEqual(result.payload["document"]["content"], "Source text")
        knowledge.get_document.assert_called_once_with("doc-1", project="core", owner="")
        knowledge.get_document.side_effect = KeyError("not found")
        result = handle_get("/api/knowledge/documents/missing", "project=core", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_document_delete_requires_matching_project_record(self):
        knowledge, _, _, deps = self.dependencies(owner="installation-owner")
        knowledge.delete_document.return_value = {"deleted": True, "id": "doc-1"}
        result = handle_delete("/api/knowledge/documents/doc-1", "project=core", deps)
        self.assertEqual(result.payload, {"deleted": True, "id": "doc-1"})
        knowledge.delete_document.assert_called_once_with("doc-1", project="core", owner="installation-owner")
        knowledge.delete_document.return_value = {"deleted": False}
        result = handle_delete("/api/knowledge/documents/doc-1", "project=other", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)
        self.assertEqual(result.payload, {"error": "Knowledge document not found."})
        self.assertIsNone(handle_delete("/api/knowledge/documents/", "", deps))

    def test_document_update_is_project_scoped_and_validates_missing_records(self):
        knowledge, _, _, deps = self.dependencies({
            "project": " core ", "title": " Revised title ", "content": " Revised body ",
            "embedding_provider": " local-hash ",
        }, owner="installation-owner")
        knowledge.update_document.return_value = {"id": "doc-1", "title": "Revised title"}
        result = handle_put("/api/knowledge/documents/doc-1", deps)
        self.assertEqual((result.status, result.payload), (HTTPStatus.OK, {"document": {"id": "doc-1", "title": "Revised title"}}))
        knowledge.update_document.assert_called_once_with(
            "doc-1", "Revised title", " Revised body ", project="core", embedding_provider="local-hash",
            owner="installation-owner",
        )
        knowledge.update_document.side_effect = KeyError("not found")
        result = handle_put("/api/knowledge/documents/missing", deps)
        self.assertEqual((result.status, result.payload), (HTTPStatus.NOT_FOUND, {"error": "Knowledge document not found."}))
        knowledge.update_document.side_effect = ValueError("Document content is required.")
        result = handle_put("/api/knowledge/documents/doc-1", deps)
        self.assertEqual((result.status, result.payload), (HTTPStatus.BAD_REQUEST, {"error": "Document content is required."}))
        self.assertIsNone(handle_put("/api/knowledge/documents/", deps))

    def test_ingest_normalization_and_validation(self):
        knowledge, _, _, deps = self.dependencies({
            "title": "  Title ", "content": " Body ", "project": "  ",
            "source": " ", "source_path": " p ", "metadata": ["ignored"],
            "embedding_provider": " local-hash ",
        })
        knowledge.ingest_text.return_value = {"id": "d1"}
        result = handle_post("/api/knowledge/ingest", deps)
        self.assertEqual((result.payload, result.status), ({"document": {"id": "d1"}}, HTTPStatus.CREATED))
        knowledge.ingest_text.assert_called_once_with(
            "Title", "Body", project="default", source="manual", source_path="p",
            metadata={}, embedding_provider="local-hash", owner="",
        )
        scoped_knowledge, _, _, scoped_deps = self.dependencies(
            {"title": "Scoped", "content": "Owner data", "project": "core"},
            owner="installation-owner",
        )
        scoped_knowledge.ingest_text.return_value = {"id": "scoped-doc"}
        handle_post("/api/knowledge/ingest", scoped_deps)
        scoped_knowledge.ingest_text.assert_called_once_with(
            "Scoped", "Owner data", project="core", source="manual", source_path="",
            metadata={}, embedding_provider="", owner="installation-owner",
        )
        knowledge.ingest_text.side_effect = ValueError("Document content is required.")
        result = handle_post("/api/knowledge/ingest", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "Document content is required."}, HTTPStatus.BAD_REQUEST))

    def test_search_synthesis_compress_and_replay_posts(self):
        knowledge, _, _, deps = self.dependencies({
            "query": "  local  ", "project": " core ", "limit": "bad",
            "embedding_provider": " hash ", "label": "  ",
        })
        knowledge.semantic_search.return_value = [{"id": "s"}]
        knowledge.hybrid_search.return_value = [{"id": "h"}]
        knowledge.synthesize.return_value = {"summary": "found"}
        knowledge.compress_memory.return_value = {"summary": {}}
        knowledge.create_replay.return_value = {"id": "r1"}
        for path, expected, status in (
            ("semantic-search", {"results": [{"id": "s"}]}, HTTPStatus.OK),
            ("hybrid-search", {"results": [{"id": "h"}]}, HTTPStatus.OK),
            ("synthesize", {"summary": "found"}, HTTPStatus.CREATED),
            ("compress", {"summary": {}}, HTTPStatus.CREATED),
            ("replay", {"replay": {"id": "r1"}}, HTTPStatus.CREATED),
        ):
            with self.subTest(path=path):
                result = handle_post(f"/api/knowledge/{path}", deps)
                self.assertEqual((result.payload, result.status), (expected, status))
        knowledge.semantic_search.assert_called_once_with(
            "local", project="core", limit=10, embedding_provider="hash", owner=""
        )
        knowledge.hybrid_search.assert_called_once_with(
            "local", project="core", limit=10, embedding_provider="hash", owner=""
        )
        knowledge.synthesize.assert_called_once_with(
            "local", project="core", limit=8, embedding_provider="hash"
        )
        knowledge.compress_memory.assert_called_once_with(project="core")
        knowledge.create_replay.assert_called_once_with(project="core", label="Replay", query="local")

    def test_embeddings_normalization_payload_and_errors(self):
        _, gateway, _, deps = self.dependencies({
            "texts": ["one", 2], "provider": " remote ", "model": " model ",
            "allow_external": "yes",
        })
        gateway.embed.return_value = SimpleNamespace(
            provider="remote", model="model", dimensions=2,
            embeddings=[[0.1, 0.2]], metadata={"source": "test"},
        )
        result = handle_post("/api/embeddings", deps)
        self.assertEqual((result.payload, result.status), ({
            "provider": "remote", "model": "model", "dimensions": 2,
            "embeddings": [[0.1, 0.2]], "metadata": {"source": "test"},
        }, HTTPStatus.CREATED))
        gateway.embed.assert_called_once_with(
            ["one", "2"], provider_id="remote", model="model", allow_external=True
        )
        gateway.embed.side_effect = ProviderConfigurationError("missing provider")
        result = handle_post("/api/embeddings", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "missing provider"}, HTTPStatus.BAD_REQUEST))
        _, gateway, _, deps = self.dependencies({"texts": "solo", "provider": " ", "allow_external": "false"})
        handle_post("/api/embeddings", deps)
        gateway.embed.assert_called_once_with(["solo"], provider_id="local", model=None, allow_external=False)
        _, gateway, _, deps = self.dependencies({"texts": {"invalid": True}})
        handle_post("/api/embeddings", deps)
        gateway.embed.assert_called_once_with([], provider_id="local", model=None, allow_external=False)

    def test_upload_validation_and_payload(self):
        _, _, _, deps = self.dependencies(headers={"Content-Type": "application/json"})
        result = handle_post("/api/knowledge/upload", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "multipart/form-data required"}, HTTPStatus.BAD_REQUEST))
        deps = self.dependencies(headers={"Content-Type": "multipart/form-data", "Content-Length": "0"})[3]
        self.assertEqual(handle_post("/api/knowledge/upload", deps).payload,
                         {"error": "Content-Length missing or exceeds the configured upload limit"})
        deps = self.dependencies(body=b"x", headers={"Content-Type": "multipart/form-data", "Content-Length": "1"})[3]
        self.assertEqual(handle_post("/api/knowledge/upload", deps).payload,
                         {"error": "No boundary in multipart form"})
        deps = self.dependencies(body=b"--x--", headers={"Content-Type": "multipart/form-data; boundary=x", "Content-Length": "5"})[3]
        self.assertEqual(handle_post("/api/knowledge/upload", deps).payload,
                         {"error": "No file was included in the upload."})

        body = (
            b'--x\r\nContent-Disposition: form-data; name="project"\r\n\r\n core \r\n'
            b'--x\r\nContent-Disposition: form-data; name="embedding_provider"\r\n\r\n hash \r\n'
            b'--x\r\nContent-Disposition: form-data; name="file"; filename="note.txt"\r\n'
            b'Content-Type: text/plain\r\n\r\nhello\r\n--x--\r\n'
        )
        knowledge, _, media, deps = self.dependencies(
            body=body, headers={"Content-Type": 'multipart/form-data; boundary="x"', "Content-Length": str(len(body))}
        )
        media.ingest_bytes.return_value = {
            "id": "a1", "mime_type": "text/plain", "media_type": "document", "size_bytes": 5,
            "analysis": {"extracted_text": "  Evidence  ", "summary": "summary", "findings": {"a": 1}},
        }
        knowledge.ingest_text.return_value = {"id": "d1"}
        result = handle_post("/api/knowledge/upload", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        self.assertEqual(result.payload["filename"], "note.txt")
        self.assertEqual(result.payload["chars_extracted"], 8)
        self.assertEqual(result.payload["document"], {"id": "d1"})
        self.assertEqual(result.payload["attachment"], {
            "id": "a1", "name": "note.txt", "filename": "note.txt", "media_type": "document",
            "mime_type": "text/plain", "size_bytes": 5, "summary": "summary",
            "extracted_text": "Evidence", "findings": {"a": 1}, "document_id": "d1",
        })
        media.ingest_bytes.assert_called_once_with(
            b"hello", filename="note.txt", project="core", source="chat-attachment",
            source_path="note.txt", metadata={"mime_type": "text/plain"}, owner="",
        )
        knowledge.ingest_text.assert_called_once_with(
            title="note.txt", content="Evidence", project="core", source="chat-attachment",
            source_path="note.txt", metadata={"asset_id": "a1", "mime_type": "text/plain"},
            embedding_provider="hash", owner="",
        )

    def test_upload_office_fallback_and_no_text(self):
        body = (b'--x\r\nContent-Disposition: form-data; name="file"; filename="note.md"\r\n'
                b'\r\nraw\r\n--x--\r\n')
        knowledge, _, media, deps = self.dependencies(
            body=body, headers={"Content-Type": "multipart/form-data; boundary=x", "Content-Length": str(len(body))}
        )
        media.ingest_bytes.return_value = {"id": "a1", "analysis": {}}
        with patch("openagent.document_processing.extract_office_text", return_value=" extracted ") as extract:
            result = handle_post("/api/knowledge/upload", deps)
        extract.assert_called_once_with("note.md", b"raw")
        self.assertEqual(result.payload["chars_extracted"], 9)
        knowledge.ingest_text.assert_called_once()
        media.ingest_bytes.return_value = {"id": "a1", "analysis": {"extracted_text": ""}}
        with patch("openagent.document_processing.extract_office_text", return_value=" "):
            result = handle_post("/api/knowledge/upload", deps)
        self.assertIsNone(result.payload["document"])
        self.assertEqual(result.payload["attachment"]["document_id"], "")

    def test_contract_routes_are_handled_and_unknown_paths_are_not(self):
        _, _, _, deps = self.dependencies()
        deps.knowledge.get_document.return_value = {"id": "fixture"}
        deps.knowledge.delete_document.return_value = {"deleted": True}
        for route in contract()["routes"]:
            with self.subTest(route=route):
                result = (handle_get(route["path"], "", deps) if route["method"] == "GET"
                          else handle_delete(route["path"], "", deps) if route["method"] == "DELETE"
                          else handle_put(route["path"], deps) if route["method"] == "PUT"
                          else handle_post(route["path"], deps))
                self.assertIsNotNone(result)
        self.assertIsNone(handle_get("/api/other", "", deps))
        self.assertIsNone(handle_post("/api/other", deps))

    def test_http_ingest_and_upload_use_knowledge_handler(self):
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                ingest = Request(
                    f"{base}/api/knowledge/ingest",
                    data=json.dumps({"title": "Note", "content": "Local research note."}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(ingest, timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.CREATED)
                    document = json.load(response)["document"]
                with urlopen(f"{base}/api/knowledge/documents", timeout=10) as response:
                    self.assertIn(document["id"], {item["id"] for item in json.load(response)["documents"]})
                with urlopen(f"{base}/api/knowledge/documents/{document['id']}?project=default", timeout=10) as response:
                    detail = json.load(response)["document"]
                    self.assertEqual(detail["id"], document["id"])
                    self.assertEqual(detail["content"], "Local research note.")
                update = Request(
                    f"{base}/api/knowledge/documents/{document['id']}",
                    data=json.dumps({"project": "default", "title": "Revised note", "content": "Updated local evidence."}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="PUT",
                )
                with urlopen(update, timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.OK)
                    updated = json.load(response)["document"]
                self.assertEqual(updated["id"], document["id"])
                self.assertEqual(updated["title"], "Revised note")
                with urlopen(f"{base}/api/knowledge/documents/{document['id']}?project=default", timeout=10) as response:
                    detail = json.load(response)["document"]
                    self.assertEqual(detail["content"], "Updated local evidence.")
                with urlopen(f"{base}/api/knowledge/search?project=default&q=Updated%20local", timeout=10) as response:
                    self.assertTrue(json.load(response)["results"])
                delete = Request(f"{base}/api/knowledge/documents/{document['id']}?project=default", method="DELETE")
                with urlopen(delete, timeout=10) as response:
                    self.assertTrue(json.load(response)["deleted"])
                with urlopen(f"{base}/api/knowledge/documents", timeout=10) as response:
                    self.assertNotIn(document["id"], {item["id"] for item in json.load(response)["documents"]})

                body = (
                    b'--test\r\nContent-Disposition: form-data; name="file"; filename="note.md"\r\n'
                    b'Content-Type: text/markdown\r\n\r\nUploaded evidence.\r\n--test--\r\n'
                )
                upload = Request(
                    f"{base}/api/knowledge/upload", data=body,
                    headers={"Content-Type": "multipart/form-data; boundary=test"}, method="POST",
                )
                with urlopen(upload, timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.CREATED)
                    result = json.load(response)
                self.assertEqual(result["filename"], "note.md")
                self.assertEqual(result["attachment"]["extracted_text"], "Uploaded evidence.")
                self.assertTrue(result["document"]["id"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_runtime_has_no_inline_knowledge_dispatch(self):
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/knowledge/', runtime)
        self.assertNotIn('if parsed.path == "/api/embeddings"', runtime)
