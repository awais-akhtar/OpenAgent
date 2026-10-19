from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.digital_twin import DigitalTwin
from openagent.routes.twin_routes import TwinRouteDependencies, contract, handle_get, handle_post
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class TwinRouteTests(unittest.TestCase):
    def test_declared_routes_preserve_payloads_and_status(self) -> None:
        with TemporaryDirectory() as tmp:
            payload: dict[str, object] = {
                "project": "core", "kind": "chat", "content": "Study materials",
                "context": "Plan a study", "title": "Finish report",
            }
            deps = TwinRouteDependencies(
                twin=DigitalTwin(OpenAgentStore(tmp)),
                first=lambda params, key: params.get(key, [None])[0],
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
            first_interaction = handle_post("/api/twin/interaction", deps)
            second_interaction = handle_post("/api/twin/interaction", deps)
            self.assertEqual(first_interaction.status, HTTPStatus.CREATED)
            self.assertNotEqual(first_interaction.payload["id"], second_interaction.payload["id"])
            sources = handle_get("/api/twin/sources", "project=core", deps).payload["sources"]
            self.assertEqual(len(sources), 3)
            self.assertEqual(sources[0]["kind"], "chat")
            self.assertNotIn("content", sources[0])
            self.assertNotIn("context", sources[0])
            self.assertEqual(handle_get("/api/twin/status", "project=core", deps).payload["project"], "core")
            self.assertTrue(handle_get("/api/twin/goals", "project=core", deps).payload["goals"])
            self.assertIsNone(handle_get("/api/chat", "", deps))
            self.assertIsNone(handle_post("/api/chat", deps))

    def test_http_contract_preserves_profile_and_goal(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = Request(
                    f"{base}/api/twin/profile",
                    data=json.dumps({"project": "core", "name": "Ada"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.OK)
                    profile = json.load(response)
                self.assertEqual(profile["profile"]["name"], "Ada")
                with urlopen(f"{base}/api/twin/profile?project=core", timeout=10) as response:
                    self.assertEqual(json.load(response)["profile"]["name"], "Ada")
                request = Request(
                    f"{base}/api/twin/goals",
                    data=json.dumps({"project": "core", "title": "Finish report"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.CREATED)
                    self.assertEqual(json.load(response)["goal"]["title"], "Finish report")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_runtime_has_no_inline_twin_dispatch(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/twin/', runtime)


if __name__ == "__main__":
    unittest.main()
