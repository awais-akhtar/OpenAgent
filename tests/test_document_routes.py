from __future__ import annotations

import unittest
from dataclasses import replace
from http import HTTPStatus
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from openagent.documents import DocumentManager
from openagent.routes import document_routes
from openagent.routes.runtime import first, parse_bool, parse_int
from openagent.storage import OpenAgentStore


class DocumentRouteContractTests(unittest.TestCase):
    def test_document_lifecycle_routes_are_declared(self) -> None:
        routes = {(route["method"], route["path"]) for route in document_routes.contract()["routes"]}

        self.assertIn(("POST", "/api/document"), routes)
        self.assertIn(("GET", "/api/documents/library"), routes)
        self.assertIn(("POST", "/api/document/{doc_id}/restore/{num}"), routes)


class DocumentRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.documents = DocumentManager(OpenAgentStore(temporary.name))
        self.payload = {}
        self.owner = "alice"
        self.deps = document_routes.DocumentRouteDependencies(
            documents=self.documents,
            read_json=lambda: self.payload,
            auth_scope_owner=lambda: self.owner,
            document_write_owner=lambda data: self.owner or str(data.get("owner", "local")).strip() or "local",
            first=first, parse_int=parse_int, parse_bool=parse_bool,
        )

    def create(self, **values):
        self.payload = {"title": "Notes", "content": "original", **values}
        result = document_routes.handle_post("/api/document", self.deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        return result.payload["document"]

    def get(self, path, query=""):
        return document_routes.handle_get(path, query, self.deps)

    def test_lifecycle_versions_conflict_metadata_archive_restore_delete(self) -> None:
        doc = self.create(owner="bob", tags=[1, "notes"], folder="work")
        self.assertEqual(doc["owner"], "alice")
        self.assertEqual(doc["tags"], ["1", "notes"])
        path = "/api/document/" + doc["doc_id"]
        self.assertEqual(self.get(path).payload, {"document": doc})
        self.payload = {"content": "revised", "if_version": "1"}
        updated = document_routes.handle_put(path, self.deps)
        self.assertEqual(updated.payload["document"]["version"], 2)
        self.assertEqual(updated.payload["document"]["title"], "Notes")
        conflict = document_routes.handle_put(path, self.deps)
        self.assertEqual(conflict.status, HTTPStatus.CONFLICT)
        self.assertEqual(conflict.payload, {"error": "version conflict: expected 2, got 1"})
        self.payload = {"tags": [2], "folder": "new", "title": "Renamed"}
        patched = document_routes.handle_patch(path, self.deps).payload["document"]
        self.assertEqual((patched["version"], patched["tags"], patched["folder"], patched["title"]),
                         (2, ["2"], "new", "Renamed"))
        self.assertEqual(patched["content"], "revised")
        versions = self.get(path + "/versions").payload["versions"]
        self.assertEqual([v["version"] for v in versions], [2, 1])
        self.assertEqual(self.get(path + "/version/1").payload["version"]["content"], "original")
        self.payload = {}
        self.assertEqual(document_routes.handle_post(path + "/archive", self.deps).payload,
                         {"doc_id": doc["doc_id"], "status": "archived"})
        self.payload = {"archived": "false"}
        self.assertEqual(document_routes.handle_post(path + "/archive", self.deps).payload["status"], "active")
        no_body = replace(self.deps, read_json=Mock(side_effect=AssertionError("unexpected body read")))
        restored = document_routes.handle_post(path + "/restore/1", no_body)
        self.assertEqual(restored.payload, {"doc_id": doc["doc_id"], "restored_from": 1, "new_version": 3})
        self.assertEqual(self.get(path).payload["document"]["content"], "original")
        self.assertEqual(len(self.get(path + "/versions").payload["versions"]), 3)
        self.assertEqual(document_routes.handle_delete(path, no_body).payload,
                         {"doc_id": doc["doc_id"], "status": "deleted"})
        self.assertEqual(self.get(path).status, HTTPStatus.NOT_FOUND)
        self.assertEqual(self.get("/api/documents/library").payload, {"items": [], "count": 0})

    def test_missing_and_foreign_documents_are_indistinguishable_for_all_operations(self) -> None:
        doc = self.create()
        self.owner = "bob"
        self.payload = {"content": "stolen", "title": "changed", "owner": "alice", "if_version": 99}
        for doc_id in (doc["doc_id"], "missing"):
            path = "/api/document/" + doc_id
            results = [self.get(path), self.get(path + "/versions"), self.get(path + "/version/1"),
                       document_routes.handle_put(path, self.deps),
                       document_routes.handle_patch(path, self.deps),
                       document_routes.handle_post(path + "/archive", self.deps),
                       document_routes.handle_post(path + "/restore/1", self.deps),
                       document_routes.handle_delete(path, self.deps)]
            for result in results:
                with self.subTest(doc_id=doc_id, result=result):
                    self.assertEqual(result.status, HTTPStatus.NOT_FOUND)
                    self.assertEqual(result.payload, {"error": "'document not found'"})
        self.assertEqual(self.get("/api/documents/library", "owner=alice").payload["count"], 0)
        self.assertEqual(self.documents.get_document(doc["doc_id"]), doc)

    def test_library_filters_limits_and_local_mode(self) -> None:
        doc = self.create(project="p", tags=["tag"], folder="folder")
        self.documents.create_document("Other", owner="alice", project="p")
        self.documents.create_document("Notes", owner="bob", project="p", tags=["tag"], folder="folder")
        query = "project=p&q=original&tag=tag&folder=folder&status=active&limit=invalid&owner=bob"
        result = self.get("/api/documents/library", query).payload
        self.assertEqual([item["doc_id"] for item in result["items"]], [doc["doc_id"]])
        self.assertNotIn("content", result["items"][0])
        self.assertEqual(self.get("/api/documents/library", "project=p&limit=1").payload["count"], 1)
        for query in ("project=other", "project=p&tag=missing", "project=p&folder=missing", "project=p&status=archived"):
            self.assertEqual(self.get("/api/documents/library", query).payload["count"], 0)
        self.owner = ""
        self.assertEqual(self.get("/api/documents/library", "project=p&owner=alice").payload["count"], 3)
        self.assertEqual(self.create(owner=" bob ", project=" ", format=" ")["owner"], "bob")
        defaults = self.create(owner=" ", tags="ignored")
        self.assertEqual((defaults["owner"], defaults["project"], defaults["format"], defaults["tags"]),
                         ("local", "default", "markdown", []))

    def test_validation_missing_versions_and_optional_fields(self) -> None:
        self.payload = {"title": " "}
        invalid = document_routes.handle_post("/api/document", self.deps)
        self.assertEqual((invalid.status, invalid.payload), (HTTPStatus.BAD_REQUEST, {"error": "title is required"}))
        doc = self.create(tags=["keep"], folder="keep")
        path = "/api/document/" + doc["doc_id"]
        for suffix in ("bad", "", "1/extra"):
            for result in (self.get(path + "/version/" + suffix),
                           document_routes.handle_post(path + "/restore/" + suffix, self.deps)):
                self.assertEqual((result.status, result.payload),
                                 (HTTPStatus.BAD_REQUEST, {"error": "invalid version number"}))
        for result in (self.get(path + "/version/99"), document_routes.handle_post(path + "/restore/99", self.deps)):
            self.assertEqual((result.status, result.payload), (HTTPStatus.NOT_FOUND, {"error": "'version not found'"}))
        self.payload = {"if_version": "invalid"}
        invalid = document_routes.handle_put(path, self.deps)
        self.assertEqual(invalid.status, HTTPStatus.BAD_REQUEST)
        self.assertEqual(invalid.payload, {"error": "invalid version number"})
        self.payload = {"tags": "ignored"}
        patched = document_routes.handle_patch(path, self.deps).payload["document"]
        for key in doc.keys() - {"updated_at"}:
            self.assertEqual(patched[key], doc[key])
        self.payload = {}
        updated = document_routes.handle_put(path, self.deps).payload["document"]
        self.assertEqual((updated["content"], updated["title"], updated["version"]), ("original", "Notes", 2))

    def test_encoded_ids_and_unhandled_paths_do_not_consume_body(self) -> None:
        doc = self.create()
        encoded = "/api/document/%20%" + format(ord(doc["doc_id"][0]), "02X") + doc["doc_id"][1:] + "%20"
        self.assertEqual(self.get(encoded).payload["document"], doc)
        self.assertEqual(len(self.get(encoded + "/versions").payload["versions"]), 1)
        self.assertEqual(self.get(encoded + "/version/1").payload["version"]["title"], "Notes")
        self.payload = {}
        self.assertEqual(document_routes.handle_patch(encoded, self.deps).status, HTTPStatus.OK)
        self.assertEqual(document_routes.handle_put(encoded, self.deps).status, HTTPStatus.OK)
        self.assertEqual(document_routes.handle_post(encoded + "/archive", self.deps).status, HTTPStatus.OK)
        self.assertEqual(document_routes.handle_post(encoded + "/restore/1", self.deps).status, HTTPStatus.OK)
        self.assertEqual(document_routes.handle_delete(encoded, self.deps).status, HTTPStatus.OK)
        deps = replace(self.deps, read_json=Mock(side_effect=AssertionError("unexpected body read")))
        for path in ("/api/unrelated", "/api/documents/import", "/api/documentation"):
            self.assertIsNone(document_routes.handle_get(path, "", deps))
            for handler in (document_routes.handle_post, document_routes.handle_put,
                            document_routes.handle_patch, document_routes.handle_delete):
                self.assertIsNone(handler(path, deps))
        self.assertIsNone(document_routes.handle_post("/api/document/id/render-preview", deps))
        self.assertIsNone(self.get("/api/document/"))


if __name__ == "__main__":
    unittest.main()
