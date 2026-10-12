from __future__ import annotations

from http import HTTPStatus
from tempfile import TemporaryDirectory
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.agents import AgentRegistry
from openagent.multi_agent import MultiAgentRuntime
from openagent.routes.agents_routes import AgentRouteDependencies, contract, handle_get, handle_post
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


def parse_bool(value: object, *, default: bool = False) -> bool:
    return default if value is None else str(value).lower() in {"1", "true", "yes"}


class AgentRouteTests(unittest.TestCase):
    def test_declared_agent_routes_have_concrete_handlers(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            payload: dict[str, object] = {
                "agent": "planning", "goal": "Plan a local test", "project": "core",
                "from_agent": "executive", "to_agent": "planning",
                "tasks": [], "scope": "agent.run", "action": "review", "reason": "test",
            }
            deps = AgentRouteDependencies(
                agents=AgentRegistry(store),
                multi_agent=MultiAgentRuntime(store),
                first=lambda params, key: params.get(key, [None])[0],
                parse_bool=parse_bool,
                read_json=lambda: payload,
            )
            for route in contract()["routes"]:
                with self.subTest(route=route):
                    result = (
                        handle_get(route["path"], "project=core", deps)
                        if route["method"] == "GET"
                        else handle_post(route["path"], deps)
                    )
                    self.assertIsNotNone(result)

            self.assertEqual(handle_get("/api/agents", "", deps).status, HTTPStatus.OK)
            self.assertEqual(handle_post("/api/agents/run", deps).status, HTTPStatus.CREATED)
            permissions = handle_get("/api/agents/permissions", "project=core", deps).payload["permissions"]
            self.assertTrue(any(item["project"] == "core" for item in permissions))
            payload["goal"] = ""
            self.assertEqual(handle_post("/api/agents/run", deps).status, HTTPStatus.BAD_REQUEST)
            self.assertIsNone(handle_post("/api/assistant/goals", deps))

    def test_runtime_has_no_inline_agent_dispatch(self) -> None:
        from pathlib import Path

        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/agents/', runtime)

    def test_agent_http_contract_preserves_listing_and_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urlopen(f"{base}/api/agents", timeout=10) as response:
                    listing = json.load(response)
                self.assertIn("planning", {item["id"] for item in listing["agents"]})
                request = Request(
                    f"{base}/api/agents/run",
                    data=json.dumps({"goal": ""}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=10)
                with error.exception as response:
                    self.assertEqual(response.code, HTTPStatus.BAD_REQUEST)
                    self.assertEqual(json.load(response)["error"]["message"], "A goal is required.")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
