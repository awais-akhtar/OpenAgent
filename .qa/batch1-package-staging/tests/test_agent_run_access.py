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
from openagent.agents import AgentRegistry
from openagent.models import AgentResponse
from openagent.routes import route_contracts
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class AgentRunAccessTests(unittest.TestCase):
    def _start_server(self, store: OpenAgentStore):
        server = build_http_server("127.0.0.1", 0, store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    @staticmethod
    def _request(url: str, token: str | None = None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            with urlopen(Request(url, headers=headers), timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            try:
                return error.code, json.loads(error.read().decode("utf-8"))
            finally:
                error.close()

    @staticmethod
    def _post(url: str, token: str, payload: dict[str, object]):
        request = Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            try:
                return error.code, json.loads(error.read().decode("utf-8"))
            finally:
                error.close()

    @staticmethod
    def _run(store: OpenAgentStore, project: str, owner: str, suffix: str) -> None:
        response = AgentResponse(
            agent="executive", goal=f"Goal {suffix}", response=f"Result {suffix}",
            next_actions=[], confidence=0.5, context_used=[],
        )
        store.save_agent_run(response, project=project, owner=owner)

    def test_agent_run_history_requires_project_scope_and_owner(self) -> None:
        self.assertEqual(required_scope_for_path("/api/agent-runs", "GET"), "agents:read")
        self.assertEqual(required_scope_for_path("/api/agents/run", "POST"), "agents:run")
        declared = {
            (route["method"], route["path"])
            for group in route_contracts()
            for route in group["routes"]
        }
        self.assertIn(("GET", "/api/agent-runs"), declared)

        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            auth = AuthManager(store)
            alice = auth.register("alice", "alice@example.test", "strong-password-1")
            bob = auth.register("bob", "bob@example.test", "strong-password-2")
            alice_read = auth.create_api_token(alice["user_id"], "agent history", ["agents:read"])["token"]
            bob_read = auth.create_api_token(bob["user_id"], "agent history", ["agents:read"])["token"]
            run_only = auth.create_api_token(alice["user_id"], "agent runner", ["agents:run"])["token"]
            unrelated = auth.create_api_token(alice["user_id"], "unrelated", ["chat:read"])["token"]
            expired = auth.create_api_token(
                alice["user_id"], "expired", ["agents:read"], expires_at="2000-01-01T00:00:00+00:00",
            )["token"]
            session = auth.login("alice", "strong-password-1")["access_token"]
            self.assertNotIn("agents:read", auth.resolve_bearer(session)["scopes"])

            self.assertTrue(store.claim_agent_run_project("alice-project", alice["user_id"]))
            self.assertFalse(store.claim_agent_run_project("alice-project", bob["user_id"]))
            for index in range(3):
                self._run(store, "alice-project", alice["user_id"], str(index))
            with store.session() as connection:
                connection.execute(
                    "INSERT INTO agent_runs(agent, goal, response_json, project, created_at, owner) VALUES (?, ?, ?, ?, ?, '')",
                    ("legacy", "legacy", json.dumps({"response": "legacy"}), "legacy-project", "2020-01-01T00:00:00+00:00"),
                )

            server, thread = self._start_server(store)
            base = f"http://127.0.0.1:{server.server_address[1]}/api/agent-runs"
            try:
                with patch.object(
                    AgentRegistry, "_run_through_engine",
                    side_effect=lambda _agent, response, **_kwargs: response,
                ), patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                    self.assertEqual(self._request(f"{base}?project=alice-project")[0], 401)
                    self.assertEqual(self._request(f"{base}?project=alice-project", "invalid-token")[0], 401)
                    self.assertEqual(self._request(f"{base}?project=alice-project", expired)[0], 401)
                    self.assertEqual(self._request(f"{base}?project=alice-project", run_only)[0], 403)
                    self.assertEqual(self._request(f"{base}?project=alice-project", unrelated)[0], 403)
                    self.assertEqual(self._request(base, alice_read)[0], 400)
                    self.assertEqual(self._request(f"{base}?project=", alice_read)[0], 400)
                    self.assertEqual(self._request(f"{base}?project=missing", alice_read)[0], 404)
                    self.assertEqual(self._request(f"{base}?project=alice-project", bob_read)[0], 404)
                    self.assertEqual(self._request(
                        f"{base}?project=alice-project&owner={bob['user_id']}", alice_read,
                    )[0], 200)
                    self.assertEqual(self._request(
                        f"{base}?project=alice-project&owner={alice['user_id']}", bob_read,
                    )[0], 404)
                    self.assertEqual(self._request(f"{base}?project=legacy-project", alice_read)[0], 404)

                    status, page = self._request(
                        f"{base}?project=alice-project&limit=2&offset=1", alice_read,
                    )
                    self.assertEqual(status, 200)
                    self.assertEqual(len(page["runs"]), 2)
                    self.assertEqual(page["pagination"], {"limit": 2, "offset": 1, "total": 3})
                    self.assertTrue(all(run["goal"].startswith("Goal") for run in page["runs"]))
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

    def test_agent_run_endpoint_binds_new_project_to_authenticated_owner(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            auth = AuthManager(store)
            owner = auth.register("runner", "runner@example.test", "strong-password-3")
            run_token = auth.create_api_token(owner["user_id"], "runner", ["agents:run"])["token"]
            other = auth.register("project-owner", "project-owner@example.test", "strong-password-5")
            server, thread = self._start_server(store)
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with store.session() as connection:
                    connection.execute(
                        """INSERT INTO codebase_projects
                           (id, project, owner, name, source_path, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        ("owned-project-id", "other-private-project", other["user_id"],
                         "Private project", tmp, "2026-10-18T00:00:00+00:00", "2026-10-18T00:00:00+00:00"),
                    )
                with patch.object(
                    AgentRegistry, "_run_through_engine",
                    side_effect=lambda _agent, response, **_kwargs: response,
                ), patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                    request = Request(
                        f"{base}/api/agents/run",
                        data=json.dumps({"agent": "executive", "goal": "test run", "project": "new-project"}).encode(),
                        headers={"Authorization": f"Bearer {run_token}", "Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=20) as response:
                        self.assertEqual(response.status, 201)
                        run = json.loads(response.read().decode())["run"]
                    self.assertEqual(run["project"], "new-project")
                    self.assertTrue(store.agent_run_project_is_owned("new-project", owner["user_id"]))
                    self.assertEqual(store.list_agent_runs(project="new-project", owner=owner["user_id"])[0]["goal"], "test run")
                    denied, _ = self._post(f"{base}/api/agents/run", run_token, {
                        "agent": "executive", "goal": "claim someone else's existing CodeQ project",
                        "project": "other-private-project",
                    })
                    self.assertEqual(denied, 404)
                    self.assertFalse(store.agent_run_project_is_owned("other-private-project", owner["user_id"]))
                    other = auth.register("intruder", "intruder@example.test", "strong-password-4")
                    intruder = auth.create_api_token(other["user_id"], "runner", ["agents:run"])["token"]
                    denied, _ = self._post(f"{base}/api/agents/run", intruder, {
                        "agent": "executive", "goal": "try to claim", "project": "new-project",
                    })
                    self.assertEqual(denied, 404)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
