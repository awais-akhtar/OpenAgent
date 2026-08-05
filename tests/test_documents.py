from __future__ import annotations

import json
import os
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager
from openagent.documents import DocumentError, DocumentManager
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class DocumentManagerTests(unittest.TestCase):
    def test_create_update_version_history_and_restore(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = DocumentManager(store)

            doc = manager.create_document("Project Plan", "v1 content", tags=["planning"])
            self.assertEqual(doc["version"], 1)
            self.assertEqual(doc["status"], "active")

            updated = manager.update_document(doc["doc_id"], content="v2 content", if_version=1)
            self.assertEqual(updated["version"], 2)

            with self.assertRaises(DocumentError):
                manager.update_document(doc["doc_id"], content="stale write", if_version=1)

            versions = manager.list_versions(doc["doc_id"])
            self.assertEqual([v["version"] for v in versions], [2, 1])

            first_version = manager.get_version(doc["doc_id"], 1)
            self.assertEqual(first_version["content"], "v1 content")

            restored = manager.restore_version(doc["doc_id"], 1)
            self.assertEqual(restored["new_version"], 3)
            current = manager.get_document(doc["doc_id"])
            self.assertEqual(current["content"], "v1 content")

    def test_patch_archive_and_soft_delete(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = DocumentManager(store)
            doc = manager.create_document("Notes", "hello", tags=["draft"])

            patched = manager.patch_document(doc["doc_id"], tags=["draft", "reviewed"], folder="ops")
            self.assertEqual(set(patched["tags"]), {"draft", "reviewed"})
            self.assertEqual(patched["folder"], "ops")
            self.assertEqual(patched["version"], 1)  # metadata patch does not bump content version

            archived = manager.archive_document(doc["doc_id"], True)
            self.assertEqual(archived["status"], "archived")
            unarchived = manager.archive_document(doc["doc_id"], False)
            self.assertEqual(unarchived["status"], "active")

            deleted = manager.delete_document(doc["doc_id"])
            self.assertEqual(deleted["status"], "deleted")
            with self.assertRaises(KeyError):
                manager.get_document(doc["doc_id"])

    def test_library_listing_filters(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = DocumentManager(store)
            manager.create_document("Alpha doc", "alpha body", project="proj-a", tags=["red"], folder="team")
            manager.create_document("Beta doc", "beta body", project="proj-a", tags=["blue"], folder="team")
            manager.create_document("Gamma doc", "gamma body", project="proj-b", tags=["red"], folder="solo")

            library = manager.list_library(project="proj-a")
            self.assertEqual(library["count"], 2)

            tagged = manager.list_library(project="proj-a", tag="blue")
            self.assertEqual(tagged["count"], 1)
            self.assertEqual(tagged["items"][0]["title"], "Beta doc")

            searched = manager.list_library(project="proj-a", query="alpha")
            self.assertEqual(searched["count"], 1)

    def test_editor_draft_autosave_and_conflict_detection(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = DocumentManager(store)
            doc = manager.create_document("Draft target", "base content")

            draft = manager.create_draft(doc["doc_id"], "typing...", base_version=1)
            self.assertFalse(draft.get("conflict", False) is True)

            fetched = manager.get_draft(draft["draft_id"])
            self.assertFalse(fetched["conflict"])

            manager.update_document(doc["doc_id"], content="server moved on", if_version=1)

            after_bump = manager.get_draft(draft["draft_id"])
            self.assertTrue(after_bump["conflict"])

            updated_draft = manager.update_draft(draft["draft_id"], "more typing", base_version=2)
            self.assertFalse(updated_draft["conflict"])

            manager.delete_draft(draft["draft_id"])
            with self.assertRaises(KeyError):
                manager.get_draft(draft["draft_id"])

            with self.assertRaises(KeyError):
                manager.create_draft("missing-doc-id", "x")


class DocumentHttpRouteTests(unittest.TestCase):
    def test_document_and_draft_lifecycle_over_http(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                create_request = Request(
                    f"{base_url}/api/document",
                    data=json.dumps({"title": "Http Doc", "content": "hello world"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                created = json.loads(urlopen(create_request, timeout=15).read().decode("utf-8"))
                doc_id = created["document"]["doc_id"]
                self.assertEqual(created["document"]["version"], 1)

                fetched = json.loads(urlopen(f"{base_url}/api/document/{doc_id}", timeout=15).read().decode("utf-8"))
                self.assertEqual(fetched["document"]["content"], "hello world")

                update_request = Request(
                    f"{base_url}/api/document/{doc_id}",
                    data=json.dumps({"content": "updated body", "if_version": 1}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="PUT",
                )
                updated = json.loads(urlopen(update_request, timeout=15).read().decode("utf-8"))
                self.assertEqual(updated["document"]["version"], 2)

                patch_request = Request(
                    f"{base_url}/api/document/{doc_id}",
                    data=json.dumps({"folder": "reports"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="PATCH",
                )
                patched = json.loads(urlopen(patch_request, timeout=15).read().decode("utf-8"))
                self.assertEqual(patched["document"]["folder"], "reports")

                library = json.loads(urlopen(f"{base_url}/api/documents/library", timeout=15).read().decode("utf-8"))
                self.assertGreaterEqual(library["count"], 1)

                draft_request = Request(
                    f"{base_url}/api/editor-drafts",
                    data=json.dumps({"doc_id": doc_id, "content": "autosave", "base_version": 2}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                draft = json.loads(urlopen(draft_request, timeout=15).read().decode("utf-8"))
                draft_id = draft["draft_id"]

                draft_get = json.loads(urlopen(f"{base_url}/api/editor-drafts/{draft_id}", timeout=15).read().decode("utf-8"))
                self.assertEqual(draft_get["draft"]["content"], "autosave")

                delete_request = Request(f"{base_url}/api/document/{doc_id}", method="DELETE")
                deleted = json.loads(urlopen(delete_request, timeout=15).read().decode("utf-8"))
                self.assertEqual(deleted["status"], "deleted")

                not_found = Request(f"{base_url}/api/document/{doc_id}")
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(not_found, timeout=15)
                self.assertEqual(ctx.exception.code, 404)
            finally:
                server.shutdown()
                thread.join(timeout=2)

    def test_document_routes_hide_cross_user_records_when_auth_is_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            auth = AuthManager(store)
            alice = auth.register("alice", "alice@example.com", "strong-password-1")
            auth.register("bob", "bob@example.com", "strong-password-2")
            alice_token = auth.login("alice", "strong-password-1")["access_token"]
            bob_token = auth.login("bob", "strong-password-2")["access_token"]

            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                    create_request = Request(
                        f"{base_url}/api/document",
                        data=json.dumps(
                            {
                                "title": "Alice private doc",
                                "content": "private",
                                "owner": "spoofed-owner-is-ignored",
                            }
                        ).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {alice_token}",
                        },
                        method="POST",
                    )
                    created = json.loads(urlopen(create_request, timeout=15).read().decode("utf-8"))
                    doc_id = created["document"]["doc_id"]
                    self.assertEqual(created["document"]["owner"], alice["user_id"])

                    alice_get = Request(
                        f"{base_url}/api/document/{doc_id}",
                        headers={"Authorization": f"Bearer {alice_token}"},
                    )
                    self.assertEqual(urlopen(alice_get, timeout=15).status, 200)

                    bob_get = Request(
                        f"{base_url}/api/document/{doc_id}",
                        headers={"Authorization": f"Bearer {bob_token}"},
                    )
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(bob_get, timeout=15)
                    self.assertEqual(ctx.exception.code, 404)

                    bob_library = Request(
                        f"{base_url}/api/documents/library",
                        headers={"Authorization": f"Bearer {bob_token}"},
                    )
                    library = json.loads(urlopen(bob_library, timeout=15).read().decode("utf-8"))
                    self.assertEqual(library["count"], 0)

                    bob_update = Request(
                        f"{base_url}/api/document/{doc_id}",
                        data=json.dumps({"content": "nope"}).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {bob_token}",
                        },
                        method="PUT",
                    )
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(bob_update, timeout=15)
                    self.assertEqual(ctx.exception.code, 404)

                    draft_request = Request(
                        f"{base_url}/api/editor-drafts",
                        data=json.dumps({"doc_id": doc_id, "content": "alice draft", "base_version": 1}).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {alice_token}",
                        },
                        method="POST",
                    )
                    draft = json.loads(urlopen(draft_request, timeout=15).read().decode("utf-8"))
                    self.assertEqual(draft["owner"], alice["user_id"])

                    bob_draft_get = Request(
                        f"{base_url}/api/editor-drafts/{draft['draft_id']}",
                        headers={"Authorization": f"Bearer {bob_token}"},
                    )
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(bob_draft_get, timeout=15)
                    self.assertEqual(ctx.exception.code, 404)
            finally:
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
