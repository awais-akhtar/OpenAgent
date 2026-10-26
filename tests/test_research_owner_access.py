from __future__ import annotations

import json
import os
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager, required_scope_for_path
from openagent.research_lifecycle import ResearchLifecycle
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.worker import WorkerRuntime


class ResearchOwnerAccessTests(unittest.TestCase):
    @staticmethod
    def request(url: str, token: str | None = None, *, method: str = "GET", payload: dict | None = None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=10) as response:
                content = response.read()
                if not content:
                    decoded: object = {}
                else:
                    try:
                        decoded = json.loads(content)
                    except json.JSONDecodeError:
                        decoded = content.decode("utf-8", errors="replace")
                return response.status, decoded
        except HTTPError as error:
            try:
                content = error.read()
                if not content:
                    decoded = {}
                else:
                    try:
                        decoded = json.loads(content)
                    except json.JSONDecodeError:
                        decoded = content.decode("utf-8", errors="replace")
                return error.code, decoded
            finally:
                error.close()

    def test_enforced_research_reads_and_mutations_are_owner_bound(self) -> None:
        self.assertEqual(required_scope_for_path("/api/research/sessions", "GET"), "knowledge:read")
        self.assertEqual(required_scope_for_path("/api/research/sessions", "POST"), "knowledge:write")

        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            auth = AuthManager(store)
            alice = auth.register("research-alice", "research-alice@example.test", "strong-password-1")
            bob = auth.register("research-bob", "research-bob@example.test", "strong-password-2")
            alice_read = auth.create_api_token(alice["user_id"], "research read", ["knowledge:read"])["token"]
            alice_write = auth.create_api_token(alice["user_id"], "research write", ["knowledge:write"])["token"]
            bob_read = auth.create_api_token(bob["user_id"], "research read", ["knowledge:read"])["token"]
            bob_write = auth.create_api_token(bob["user_id"], "research write", ["knowledge:write"])["token"]
            lifecycle = ResearchLifecycle(store)
            owned = lifecycle.create("Alice private report", project="default", owner=alice["user_id"])
            lifecycle.start(owned["id"])
            lifecycle.complete(owned["id"], {
                "plan": "local plan", "report": "owner-only result",
                "sources": [{"id": "source-1", "title": "Private source"}],
            })
            lifecycle.save_source_note(owned["id"], "source-1", "Alice note", project="default")
            legacy = lifecycle.create("Legacy local session", project="default", owner="")

            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}/api/research/sessions"
            try:
                with patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                    self.assertEqual(self.request(f"{base}?project=default")[0], 401)
                    status, listing = self.request(f"{base}?project=default", alice_read)
                    self.assertEqual(status, 200)
                    self.assertEqual([row["id"] for row in listing["sessions"]], [owned["id"]])
                    self.assertNotIn(legacy["id"], {row["id"] for row in listing["sessions"]})
                    self.assertEqual(self.request(f"{base}?project=default", bob_read)[1]["sessions"], [])

                    alice_urls = [
                        f"{base}/{owned['id']}",
                        f"{base}/{owned['id']}/report",
                        f"{base}/{owned['id']}/checkpoints",
                        f"{base}/{owned['id']}/source-graph",
                        f"{base}/{owned['id']}/source-notes?project=default",
                        f"{base}/{owned['id']}/stream?timeout=1",
                    ]
                    for url in alice_urls:
                        self.assertEqual(self.request(url, alice_read)[0], 200, url)
                        self.assertEqual(self.request(url, bob_read)[0], 404, url)
                    self.assertEqual(self.request(f"{base}/{legacy['id']}", alice_read)[0], 404)

                    cross_owner_actions = [
                        (f"{base}/{owned['id']}/cancel", {}),
                        (f"{base}/{owned['id']}/archive", {"archived": True}),
                        (f"{base}/{owned['id']}/delete", {}),
                        (f"{base}/{owned['id']}/source-notes/source-1", {"project": "default", "content": "intruder"}),
                    ]
                    for url, payload in cross_owner_actions:
                        self.assertEqual(self.request(url, bob_write, method="POST", payload=payload)[0], 404, url)
                    self.assertEqual(lifecycle.get(owned["id"], owner=alice["user_id"])["status"], "completed")

                    with patch.object(WorkerRuntime, "enqueue", return_value={"id": "research-job", "state": "queued"}), \
                         patch.object(WorkerRuntime, "run_once", return_value=None):
                        status, created = self.request(base, alice_write, method="POST", payload={
                            "query": "Owned through authenticated route", "project": "default",
                        })
                    self.assertEqual(status, 202)
                    created_id = created["session"]["id"]
                    self.assertIsNotNone(lifecycle.get(created_id, owner=alice["user_id"]))
                    self.assertIsNone(lifecycle.get(created_id, owner=bob["user_id"]))

                    pending = lifecycle.create("Alice cancellable session", owner=alice["user_id"])
                    self.assertEqual(self.request(f"{base}/{pending['id']}/cancel", alice_write, method="POST", payload={})[0], 202)
                    self.assertEqual(lifecycle.get(pending["id"], owner=alice["user_id"])["status"], "cancelled")
                    self.assertIsNone(lifecycle.get(owned["id"], owner=bob["user_id"]))
                    self.assertIsNotNone(lifecycle.get(owned["id"], owner=alice["user_id"]))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_branch_inherits_session_owner_and_owner_filter_is_additive(self) -> None:
        with TemporaryDirectory() as temp:
            lifecycle = ResearchLifecycle(OpenAgentStore(temp))
            alice = lifecycle.create("Owned", project="core", owner="alice-id")
            legacy = lifecycle.create("Legacy", project="core", owner="")
            branch = lifecycle.branch(alice["id"], label="Follow-up", owner="alice-id")
            self.assertEqual(branch["parent_id"], alice["id"])
            self.assertEqual({row["id"] for row in lifecycle.list(project="core", owner="alice-id")}, {branch["id"], alice["id"]})
            self.assertEqual({row["id"] for row in lifecycle.list(project="core", owner="")}, {legacy["id"]})
            self.assertIsNone(lifecycle.get(alice["id"], owner="bob-id"))


if __name__ == "__main__":
    unittest.main()
