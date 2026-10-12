from __future__ import annotations

import json
import threading
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.request import Request, urlopen

from openagent.documents import DocumentManager
from openagent.routes.draft_routes import (
    DraftRouteDependencies,
    contract,
    handle_delete,
    handle_get,
    handle_post,
    handle_put,
)
from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


class DraftRouteTests(unittest.TestCase):
    def test_http_draft_lifecycle(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def send(path: str, method: str, payload: dict[str, object]) -> dict[str, object]:
                request = Request(
                    f"{base}{path}", data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method=method,
                )
                with urlopen(request, timeout=10) as response:
                    return json.load(response)

            try:
                document = send("/api/document", "POST", {"title": "Notes", "content": "original"})["document"]
                created = send(
                    "/api/editor-drafts", "POST",
                    {"doc_id": document["doc_id"], "content": "first", "base_version": 1},
                )
                draft_id = created["draft_id"]
                with urlopen(f"{base}/api/editor-drafts?project=default", timeout=10) as response:
                    self.assertIn(draft_id, {item["draft_id"] for item in json.load(response)["drafts"]})
                updated = send(f"/api/editor-drafts/{draft_id}", "PUT", {"content": "second"})
                self.assertEqual(updated["draft"]["content"], "second")
                with urlopen(f"{base}/api/editor-drafts/{draft_id}", timeout=10) as response:
                    self.assertEqual(json.load(response)["draft"]["content"], "second")
                deleted = send(f"/api/editor-drafts/{draft_id}", "DELETE", {})
                self.assertTrue(deleted)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_runtime_has_no_inline_draft_dispatch(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/editor-drafts"', runtime)
        self.assertNotIn('if parsed.path.startswith("/api/editor-drafts/")', runtime)

    def test_declared_routes_and_unrelated_paths(self) -> None:
        self.assertEqual(
            {(route["method"], route["path"]) for route in contract()["routes"]},
            {
                ("GET", "/api/editor-drafts"),
                ("POST", "/api/editor-drafts"),
                ("GET", "/api/editor-drafts/{draft_id}"),
                ("PUT", "/api/editor-drafts/{draft_id}"),
                ("DELETE", "/api/editor-drafts/{draft_id}"),
            },
        )
        with TemporaryDirectory() as tmp:
            deps = self._deps(DocumentManager(OpenAgentStore(tmp)), {}, "alice")
            self.assertIsNone(handle_get("/api/document", "", deps))
            self.assertIsNone(handle_post("/api/document", deps))
            self.assertIsNone(handle_put("/api/editor-drafts", deps))
            self.assertIsNone(handle_delete("/api/editor-drafts", deps))

    @staticmethod
    def _deps(documents: DocumentManager, payload: dict[str, object], owner: str) -> DraftRouteDependencies:
        return DraftRouteDependencies(
            documents=documents,
            read_json=lambda: payload,
            auth_scope_owner=lambda: owner,
            document_write_owner=lambda data: owner or str(data.get("owner", "local")).strip() or "local",
        )

    def test_create_list_get_update_and_delete_keep_revision_history(self) -> None:
        with TemporaryDirectory() as tmp:
            documents = DocumentManager(OpenAgentStore(tmp))
            doc = documents.create_document("Notes", "original", project="project-a", owner="alice")
            payload: dict[str, object] = {
                "doc_id": doc["doc_id"], "content": "first draft",
                "base_version": 1, "project": "project-a", "owner": "bob",
            }
            deps = self._deps(documents, payload, "alice")
            created = handle_post("/api/editor-drafts", deps)
            self.assertEqual(created.status, HTTPStatus.CREATED)
            draft_id = created.payload["draft_id"]
            self.assertEqual(created.payload["owner"], "alice")

            listing = handle_get("/api/editor-drafts", "project=project-a&doc_id=" + doc["doc_id"], deps)
            self.assertEqual([item["draft_id"] for item in listing.payload["drafts"]], [draft_id])
            self.assertEqual(handle_get("/api/editor-drafts", "project=other", deps).payload, {"drafts": []})
            fetched = handle_get("/api/editor-drafts/" + draft_id, "", deps)
            self.assertEqual(fetched.payload["draft"]["content"], "first draft")
            self.assertEqual(fetched.payload["draft"]["base_version"], 1)

            payload.clear()
            payload.update({"content": "second draft", "base_version": 2})
            updated = handle_put("/api/editor-drafts/" + draft_id, deps)
            self.assertEqual(updated.status, HTTPStatus.OK)
            self.assertEqual(updated.payload["draft"]["base_version"], 2)
            self.assertEqual(updated.payload["draft"]["content"], "second draft")
            self.assertEqual(
                [(version["content"], version["base_version"]) for version in documents.list_draft_versions(draft_id)],
                [("first draft", 1), ("second draft", 2)],
            )
            payload.clear()
            payload["content"] = "third draft"
            self.assertEqual(handle_put("/api/editor-drafts/" + draft_id, deps).payload["draft"]["base_version"], 2)

            deleted = handle_delete("/api/editor-drafts/" + draft_id, deps)
            self.assertEqual(deleted.payload, {"deleted": True, "draft_id": draft_id})
            missing = handle_get("/api/editor-drafts/" + draft_id, "", deps)
            self.assertEqual(missing.status, HTTPStatus.NOT_FOUND)
            self.assertEqual(missing.payload, {"error": "'draft not found'"})

    def test_owner_scope_hides_other_users_documents_and_drafts(self) -> None:
        with TemporaryDirectory() as tmp:
            documents = DocumentManager(OpenAgentStore(tmp))
            doc = documents.create_document("Private", "secret", owner="alice")
            payload: dict[str, object] = {"doc_id": doc["doc_id"], "content": "private draft", "owner": "bob"}
            alice = self._deps(documents, payload, "alice")
            bob = self._deps(documents, payload, "bob")
            self.assertEqual(handle_post("/api/editor-drafts", bob).status, HTTPStatus.NOT_FOUND)
            created = handle_post("/api/editor-drafts", alice)
            self.assertEqual(created.status, HTTPStatus.CREATED)
            draft_id = created.payload["draft_id"]
            self.assertEqual(handle_get("/api/editor-drafts", "", bob).payload, {"drafts": []})
            for result in (
                handle_get("/api/editor-drafts/" + draft_id, "", bob),
                handle_put("/api/editor-drafts/" + draft_id, bob),
                handle_delete("/api/editor-drafts/" + draft_id, bob),
            ):
                self.assertEqual(result.status, HTTPStatus.NOT_FOUND)
                self.assertEqual(result.payload, {"error": "'draft not found'"})
            self.assertEqual(handle_get("/api/editor-drafts/" + draft_id, "", alice).payload["draft"]["content"], "private draft")

    def test_url_decoding_and_default_project_owner(self) -> None:
        with TemporaryDirectory() as tmp:
            documents = DocumentManager(OpenAgentStore(tmp))
            doc = documents.create_document("Notes")
            payload: dict[str, object] = {"doc_id": doc["doc_id"], "content": "draft"}
            deps = self._deps(documents, payload, "")
            created = handle_post("/api/editor-drafts", deps)
            self.assertEqual(created.payload["owner"], "local")
            self.assertEqual(len(handle_get("/api/editor-drafts", "", deps).payload["drafts"]), 1)
            draft_id = created.payload["draft_id"]
            encoded_path = f"/api/editor-drafts/%{ord(draft_id[0]):02X}{draft_id[1:]}"
            self.assertEqual(handle_get(encoded_path, "", deps).payload["draft"]["draft_id"], draft_id)

    def test_invalid_version_matches_runtime_conversion(self) -> None:
        with TemporaryDirectory() as tmp:
            documents = DocumentManager(OpenAgentStore(tmp))
            doc = documents.create_document("Notes")
            payload: dict[str, object] = {"doc_id": doc["doc_id"], "base_version": "invalid"}
            deps = self._deps(documents, payload, "")
            with self.assertRaises(ValueError):
                handle_post("/api/editor-drafts", deps)
            payload["base_version"] = 1
            draft_id = handle_post("/api/editor-drafts", deps).payload["draft_id"]
            payload["base_version"] = "invalid"
            with self.assertRaises(ValueError):
                handle_put("/api/editor-drafts/" + draft_id, deps)


if __name__ == "__main__":
    unittest.main()
