from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.assistant import AssistantManager
from openagent.routes.assistant_routes import AssistantRouteDependencies, contract, handle_get, handle_post
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class AssistantRouteTests(unittest.TestCase):
    def test_declared_routes_are_handled_and_keep_payloads(self) -> None:
        with TemporaryDirectory() as tmp:
            payload: dict[str, object] = {
                "project": "core", "title": "Plan", "name": "Ada",
                "destination": "Berlin", "transcript": "Discussed deadlines.",
            }
            deps = AssistantRouteDependencies(
                assistant=AssistantManager(OpenAgentStore(tmp)),
                first=lambda params, key: params.get(key, [None])[0],
                parse_int=lambda value, default=0: default if value is None else int(value),
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
            payload = {"project": "core", "title": "Finish review"}
            created = handle_post("/api/assistant/tasks", deps)
            self.assertEqual(created.status, HTTPStatus.CREATED)
            self.assertEqual(created.payload["task"]["title"], "Finish review")
            self.assertTrue(handle_get("/api/assistant/tasks", "project=core&limit=1", deps).payload["tasks"])
            payload = {"title": ""}
            self.assertEqual(handle_post("/api/assistant/tasks", deps).status, HTTPStatus.BAD_REQUEST)
            self.assertIsNone(handle_get("/api/chat", "", deps))
            self.assertIsNone(handle_post("/api/chat", deps))

    def test_http_contract_preserves_task_creation_and_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = Request(
                    f"{base}/api/assistant/tasks",
                    data=json.dumps({"project": "core", "title": "Review"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.CREATED)
                    created = json.load(response)
                with urlopen(f"{base}/api/assistant/tasks?project=core", timeout=10) as response:
                    listing = json.load(response)
                self.assertEqual(listing["tasks"][0]["id"], created["task"]["id"])
                request.data = json.dumps({"title": ""}).encode("utf-8")
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=10)
                with error.exception as response:
                    self.assertEqual(response.code, HTTPStatus.BAD_REQUEST)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_runtime_has_no_inline_assistant_dispatch(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/assistant/', runtime)


if __name__ == "__main__":
    unittest.main()
