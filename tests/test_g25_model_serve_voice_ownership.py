"""Ownership and scope boundaries for the G25, model-serve and voice-conversation routes.

These three route groups were dispatched by the runtime without being
registered or scope-mapped, so the auth gate used to skip them entirely. They
are now declared and mapped; these tests hold that boundary over real HTTP:
one owner must not read or act on another's records, and a token carrying a
neighbouring scope must not reach them.
"""

from __future__ import annotations

import base64
import json
import threading
import unittest
from unittest import mock
from http import HTTPStatus
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class G25ModelServeVoiceOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self._enforced = mock.patch.dict(
            "os.environ", {"OPENAGENT_AUTH_ENFORCED": "1"}
        )
        self._enforced.start()
        self.addCleanup(self._enforced.stop)

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = OpenAgentStore(self._tmp.name)
        auth = AuthManager(self.store)
        alice = auth.register("g25-alice", "g25-alice@example.test", "strong-password-1")
        bob = auth.register("g25-bob", "g25-bob@example.test", "strong-password-2")
        scopes = ["g25:read", "g25:write", "models:read", "models:write", "voice:read", "voice:write"]
        self.alice = auth.create_api_token(alice["user_id"], "g25-owner", scopes)["token"]
        self.bob = auth.create_api_token(bob["user_id"], "g25-other", scopes)["token"]
        # A neighbouring scope must not unlock these groups.
        self.knowledge_only = auth.create_api_token(
            alice["user_id"], "knowledge-only", ["knowledge:read", "knowledge:write"]
        )["token"]

        self.server = build_http_server("127.0.0.1", 0, self.store)
        worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        worker.start()

        def shutdown() -> None:
            self.server.shutdown()
            self.server.server_close()
            worker.join(timeout=5)

        self.addCleanup(shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def call(self, method: str, path: str, token: str, body: dict | None = None):
        headers = {"Authorization": f"Bearer {token}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read() or b"{}")
        except HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            except json.JSONDecodeError:
                return exc.code, {}

    @staticmethod
    def _rows(payload: dict, key: str) -> list:
        body = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        rows = body.get(key) if isinstance(body, dict) else None
        return rows if isinstance(rows, list) else []

    def test_g25_documents_are_not_visible_or_readable_across_owners(self) -> None:
        status, created = self.call(
            "POST", "/api/g25/documents/upload", self.alice,
            {"filename": "alice.txt", "content": base64.b64encode(b"alice private text").decode()},
        )
        self.assertEqual(status, HTTPStatus.CREATED)
        body = created.get("data") if isinstance(created.get("data"), dict) else created
        document_id = body.get("document_id")
        self.assertTrue(document_id, f"upload did not return a document id: {created}")

        status, mine = self.call("GET", "/api/g25/documents", self.alice)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(
            [str(row.get("id")) for row in self._rows(mine, "documents")], [str(document_id)]
        )

        status, theirs = self.call("GET", "/api/g25/documents", self.bob)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(self._rows(theirs, "documents"), [])

        status, _ = self.call("GET", f"/api/g25/documents/{document_id}", self.bob)
        self.assertEqual(status, HTTPStatus.NOT_FOUND)
        status, _ = self.call("POST", f"/api/g25/documents/{document_id}/analyze", self.bob)
        self.assertEqual(status, HTTPStatus.NOT_FOUND)

        status, _ = self.call("GET", f"/api/g25/documents/{document_id}", self.alice)
        self.assertEqual(status, HTTPStatus.OK)

    def test_voice_conversations_are_not_visible_or_readable_across_owners(self) -> None:
        status, created = self.call("POST", "/api/voice/conversations", self.alice, {})
        self.assertEqual(status, HTTPStatus.CREATED)
        body = created.get("data") if isinstance(created.get("data"), dict) else created
        conversation_id = body.get("conversation_id") or body.get("id")
        self.assertTrue(conversation_id, f"create did not return an id: {created}")

        status, mine = self.call("GET", "/api/voice/conversations", self.alice)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(len(self._rows(mine, "conversations")), 1)

        status, theirs = self.call("GET", "/api/voice/conversations", self.bob)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(self._rows(theirs, "conversations"), [])

        status, _ = self.call("GET", f"/api/voice/conversations/{conversation_id}", self.bob)
        self.assertEqual(status, HTTPStatus.NOT_FOUND)
        status, _ = self.call("POST", f"/api/voice/conversations/{conversation_id}/close", self.bob, {})
        self.assertEqual(status, HTTPStatus.NOT_FOUND)

    def test_model_serve_sessions_are_not_visible_or_controllable_across_owners(self) -> None:
        status, created = self.call(
            "POST", "/api/model-serve/plans", self.alice,
            {"model_id": "synthetic-model", "model_path": "/synthetic/path"},
        )
        self.assertEqual(status, HTTPStatus.CREATED)
        body = created.get("data") if isinstance(created.get("data"), dict) else created
        session_id = body.get("session_id") or body.get("id")
        self.assertTrue(session_id, f"plan did not return a session id: {created}")

        status, mine = self.call("GET", "/api/model-serve/sessions", self.alice)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(len(self._rows(mine, "sessions")), 1)

        status, theirs = self.call("GET", "/api/model-serve/sessions", self.bob)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(self._rows(theirs, "sessions"), [])

        # Starting or stopping another owner's serve session must not succeed.
        for suffix in ("start", "stop"):
            status, _ = self.call(
                "POST", f"/api/model-serve/sessions/{session_id}/{suffix}", self.bob, {}
            )
            self.assertNotIn(
                status, {HTTPStatus.OK, HTTPStatus.CREATED},
                f"bob was allowed to {suffix} alice's serve session",
            )

    def test_groups_reject_missing_and_neighbouring_scopes(self) -> None:
        probes = (
            ("GET", "/api/g25/documents"),
            ("GET", "/api/voice/conversations"),
            ("GET", "/api/model-serve/sessions"),
            ("GET", "/api/ultra-models/models"),
            ("GET", "/api/ultra-models/readiness"),
        )
        for method, path in probes:
            with self.subTest(path=path, credential="none"):
                status, _ = self.call(method, path, "")
                self.assertEqual(status, HTTPStatus.UNAUTHORIZED)
            with self.subTest(path=path, credential="knowledge-only"):
                status, _ = self.call(method, path, self.knowledge_only)
                self.assertEqual(status, HTTPStatus.FORBIDDEN)

        # Write operations are likewise closed to a read-adjacent credential.
        status, _ = self.call(
            "POST", "/api/ultra-models/models/synthetic/download", self.knowledge_only, {}
        )
        self.assertEqual(status, HTTPStatus.FORBIDDEN)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
