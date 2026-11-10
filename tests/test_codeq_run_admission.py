"""CodeQ reports a truthful admission gate for the installed model policy."""

from __future__ import annotations

import unittest
import json
import threading
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager
from openagent.auth import required_scope_for_path
from openagent.codebase.service import CodebaseService
from openagent.routes.codebase_routes import ROUTES, handle_get
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class CodeQRunAdmissionTests(unittest.TestCase):
    @staticmethod
    def response(policy: str):
        service = object.__new__(CodebaseService)
        service.gateway = SimpleNamespace(routing_policy=policy)
        deps = SimpleNamespace(query_owner=lambda _params: "local", codebase=service)
        return handle_get("/api/codebase/run-admission", "", deps)

    def test_cpu_baseline_blocks_autonomous_run_but_keeps_editor_and_questions(self) -> None:
        result = self.response("cpu-baseline")
        self.assertEqual(result.payload["autonomous_run"], "blocked")
        self.assertTrue(result.payload["manual_editor"])
        self.assertTrue(result.payload["read_only_questions"])
        self.assertIn("small CPU baseline", result.payload["reason"])

    def test_other_policy_is_only_eligible_not_promised_success(self) -> None:
        result = self.response("local-first")
        self.assertEqual(result.payload["autonomous_run"], "eligible")
        self.assertIn("checked during execution", result.payload["reason"])

    def test_route_is_declared_and_requires_read_scope(self) -> None:
        self.assertIn(("GET", "/api/codebase/run-admission"), ROUTES)
        self.assertEqual(required_scope_for_path("/api/codebase/run-admission", "GET"), "codebase:read")

    def test_actual_http_requires_read_scope(self) -> None:
        with TemporaryDirectory() as directory, patch.dict("os.environ", {"OPENAGENT_AUTH_ENFORCED": "1", "OPENAGENT_MODEL_ROUTING": "cpu-baseline"}):
            store = OpenAgentStore(directory)
            auth = AuthManager(store)
            owner = auth.register("codeq-admission", "codeq-admission@example.test", "strong-password-1")
            read_token = auth.create_api_token(owner["user_id"], "read", ["codebase:read"])["token"]
            unrelated = auth.create_api_token(owner["user_id"], "other", ["knowledge:read"])["token"]
            server = build_http_server("127.0.0.1", 0, store)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                url = f"http://127.0.0.1:{server.server_address[1]}/api/codebase/run-admission"

                def get(token: str = ""):
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    try:
                        with urlopen(Request(url, headers=headers), timeout=10) as response:
                            return response.status, json.loads(response.read())
                    except HTTPError as error:
                        with error:
                            return error.code, json.loads(error.read())

                self.assertEqual(get()[0], 401)
                self.assertEqual(get(unrelated)[0], 403)
                status, payload = get(read_token)
                self.assertEqual(status, 200)
                self.assertEqual(payload.get("data", payload)["autonomous_run"], "blocked")
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
