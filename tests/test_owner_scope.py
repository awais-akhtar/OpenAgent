from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import auth_enforced_owner
from openagent.storage import OpenAgentStore


class OwnerScopeContractTests(unittest.TestCase):
    def test_owner_scope_uses_authenticated_user_when_enforced(self) -> None:
        user = {"user_id": "usr_a", "role": "member"}

        self.assertEqual(auth_enforced_owner(user, fallback="fallback-owner"), "usr_a")
        self.assertEqual(auth_enforced_owner(None, fallback="fallback-owner"), "fallback-owner")


class KnowledgeOwnerIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        from openagent.knowledge import KnowledgeEngine

        self._tmpdir = tempfile.TemporaryDirectory()
        store = OpenAgentStore(Path(self._tmpdir.name))
        self.knowledge = KnowledgeEngine(store)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_user_b_cannot_list_user_a_documents(self) -> None:
        self.knowledge.ingest_text("Doc A", "content for user A", project="default", owner="user_a")
        docs_b = self.knowledge.list_documents(project="default", owner="user_b")
        self.assertEqual(docs_b, [], "user_b should not see user_a's documents")

    def test_user_b_cannot_get_user_a_document(self) -> None:
        doc = self.knowledge.ingest_text("Doc A", "content for user A", project="default", owner="user_a")
        with self.assertRaises(KeyError):
            self.knowledge.get_document(doc["id"], project="default", owner="user_b")

    def test_user_b_cannot_search_user_a_documents(self) -> None:
        self.knowledge.ingest_text("Secret Doc", "classified content for user A", project="default", owner="user_a")
        results = self.knowledge.search("classified", project="default", owner="user_b")
        self.assertEqual(results, [], "user_b search should return no results for user_a's data")

    def test_owner_sees_own_documents(self) -> None:
        self.knowledge.ingest_text("My Doc", "my content", project="default", owner="user_a")
        docs = self.knowledge.list_documents(project="default", owner="user_a")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["title"], "My Doc")


class MediaOwnerIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        from openagent.media import MediaManager

        self._tmpdir = tempfile.TemporaryDirectory()
        store = OpenAgentStore(Path(self._tmpdir.name))
        self.media = MediaManager(store)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_user_b_cannot_list_user_a_assets(self) -> None:
        self.media.ingest_bytes(
            b"fake image data",
            filename="photo.jpg",
            project="default",
            owner="user_a",
            copy_to_store=False,
        )
        assets_b = self.media.assets(project="default", owner="user_b")
        self.assertEqual(assets_b, [], "user_b should not see user_a's assets")

    def test_owner_sees_own_assets(self) -> None:
        self.media.ingest_bytes(
            b"fake image data",
            filename="photo.jpg",
            project="default",
            owner="user_a",
            copy_to_store=False,
        )
        assets_a = self.media.assets(project="default", owner="user_a")
        self.assertEqual(len(assets_a), 1)

    def test_user_b_cannot_analyze_or_play_user_a_asset(self) -> None:
        private = self.media.ingest_bytes(
            b"private owner text",
            filename="private.txt",
            project="default",
            owner="user_a",
            copy_to_store=True,
        )
        with self.assertRaises(KeyError):
            self.media.analyze_asset(private["id"], project="default", owner="user_b")
        self.assertEqual(
            self.media.analyze_asset(private["id"], project="default", owner="user_a")["asset_id"],
            private["id"],
        )

        audio = self.media.ingest_bytes(
            b"RIFF\x00\x00\x00\x00WAVEfixture",
            filename="private.wav",
            project="default",
            owner="user_a",
            copy_to_store=True,
        )
        self.assertIsNone(self.media.read_audio_asset(audio["id"], project="default", owner="user_b"))
        self.assertIsNotNone(self.media.read_audio_asset(audio["id"], project="default", owner="user_a"))

    def test_chat_attachment_resolution_uses_authenticated_media_owner(self) -> None:
        from openagent.chat import ChatService
        from openagent.model_gateway import ModelGateway

        private = self.media.ingest_bytes(
            b"alice attachment secret", filename="private.txt", project="core",
            owner="user_a", copy_to_store=True,
        )
        chat = ChatService(self.media.store, ModelGateway())
        attachment = {"id": private["id"], "media_type": "document", "name": "private.txt"}
        for_bob = chat._resolve_attachments([attachment], project="core", owner="user_b")[0]
        for_alice = chat._resolve_attachments([attachment], project="core", owner="user_a")[0]
        self.assertNotIn("alice attachment secret", str(for_bob))
        self.assertIn("alice attachment secret", str(for_alice))

    def test_media_status_counts_only_requested_owner(self) -> None:
        self.media.ingest_bytes(b"a", filename="a.txt", project="default", owner="user_a", copy_to_store=False)
        self.media.ingest_bytes(b"b", filename="b.txt", project="default", owner="user_b", copy_to_store=False)
        status = self.media.status(project="default", owner="user_a")
        self.assertEqual(status["assets_by_type"], {"document": 1})

    def test_authenticated_media_http_routes_hide_other_owner_assets(self) -> None:
        from openagent.auth import AuthManager
        from openagent.server import build_http_server

        store = self.media.store
        auth = AuthManager(store)
        alice = auth.register("media-alice", "media-alice@example.test", "strong-password-1")
        bob = auth.register("media-bob", "media-bob@example.test", "strong-password-2")
        scopes = ["media:read", "media:write", "voice:read", "voice:write"]
        alice_token = auth.create_api_token(alice["user_id"], "media-owner", scopes)["token"]
        bob_token = auth.create_api_token(bob["user_id"], "media-other", scopes)["token"]
        alice_voice_write = auth.create_api_token(alice["user_id"], "voice-writer", ["voice:write"])["token"]
        doc = self.media.ingest_bytes(
            b"alice private media", filename="private.txt", project="default",
            owner=alice["user_id"], copy_to_store=True,
        )
        audio = self.media.ingest_bytes(
            b"RIFF\x00\x00\x00\x00WAVEprivate", filename="private.wav", project="default",
            owner=alice["user_id"], copy_to_store=True,
        )
        server = build_http_server("127.0.0.1", 0, store)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def request(path: str, token: str, *, method: str = "GET", body: dict | None = None):
            headers = {"Authorization": f"Bearer {token}"}
            data = None
            if body is not None:
                headers["Content-Type"] = "application/json"
                data = json.dumps(body).encode("utf-8")
            req = Request(base + path, data=data, headers=headers, method=method)
            try:
                with urlopen(req, timeout=10) as response:
                    return response.status, response.headers.get("Content-Type", ""), response.read()
            except HTTPError as exc:
                try:
                    return exc.code, exc.headers.get("Content-Type", ""), exc.read()
                finally:
                    exc.close()

        try:
            with patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                status, _, body = request("/api/media/assets?project=default", alice_token)
                self.assertEqual(status, 200)
                self.assertIn(doc["id"], {item["id"] for item in json.loads(body)["assets"]})
                status, _, body = request("/api/media/assets?project=default", bob_token)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["assets"], [])

                status, _, _ = request(
                    "/api/media/analyze", bob_token, method="POST",
                    body={"asset_id": doc["id"], "project": "default"},
                )
                self.assertEqual(status, 404)
                status, _, body = request(
                    "/api/media/analyze", alice_token, method="POST",
                    body={"asset_id": doc["id"], "project": "default"},
                )
                self.assertEqual(status, 201)
                self.assertEqual(json.loads(body)["analysis"]["asset_id"], doc["id"])

                audio_path = f"/api/voice/assets/{audio['id']}/content?project=default"
                self.assertEqual(request(audio_path, bob_token)[0], 404)
                status, mime, payload = request(audio_path, alice_token)
                self.assertEqual((status, mime), (200, "audio/wav"))
                self.assertTrue(payload.startswith(b"RIFF"))

                status, _, body = request("/api/media/status?project=default", bob_token)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["assets_by_type"], {})

                status, _, body = request(
                    "/api/voice/tts", alice_voice_write, method="POST",
                    body={"project": "default", "text": "Owner-bound playback"},
                )
                self.assertEqual(status, 201)
                voice_asset_id = json.loads(body)["script"]["audio_asset_id"]
                voice_audio_path = f"/api/voice/assets/{voice_asset_id}/content?project=default"
                self.assertEqual(request(voice_audio_path, alice_token)[0], 200)
                self.assertEqual(request(voice_audio_path, bob_token)[0], 404)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


class ResearchOwnerIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        from openagent.research_lifecycle import ResearchLifecycle

        self._tmpdir = tempfile.TemporaryDirectory()
        store = OpenAgentStore(Path(self._tmpdir.name))
        self.lifecycle = ResearchLifecycle(store)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_user_b_cannot_list_user_a_sessions(self) -> None:
        self.lifecycle.create("What is AI?", project="default", owner="user_a")
        sessions_b = self.lifecycle.list(project="default", owner="user_b")
        self.assertEqual(sessions_b, [], "user_b should not see user_a's sessions")

    def test_user_b_cannot_get_user_a_session(self) -> None:
        session = self.lifecycle.create("What is AI?", project="default", owner="user_a")
        result = self.lifecycle.get(session["id"], owner="user_b")
        self.assertIsNone(result, "user_b should not be able to get user_a's session")

    def test_owner_sees_own_sessions(self) -> None:
        self.lifecycle.create("What is AI?", project="default", owner="user_a")
        sessions_a = self.lifecycle.list(project="default", owner="user_a")
        self.assertEqual(len(sessions_a), 1)


if __name__ == "__main__":
    unittest.main()
