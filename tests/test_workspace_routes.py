from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openagent.routes.workspace_routes import WorkspaceRouteDependencies, contract, handle_get, handle_post
from openagent.storage import OpenAgentStore
from openagent.workspace import WorkspaceManager


class WorkspaceRouteTests(unittest.TestCase):
    def test_declared_workspace_routes_preserve_payloads_and_status_codes(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = WorkspaceManager(OpenAgentStore(tmp))
            template = workspace.create_item(
                kind="template", title="Letter", content="Dear {{name}}", project="core",
            )
            payload: dict[str, object] = {"project": "core", "kind": "note", "title": "Route note"}
            deps = WorkspaceRouteDependencies(
                workspace=workspace,
                first=lambda params, key: params.get(key, [None])[0],
                parse_int=lambda value, default=0: default if value is None else int(value),
                read_json=lambda: payload,
            )
            created = handle_post("/api/workspace/items", deps)
            self.assertEqual(created.status, HTTPStatus.CREATED)
            item_id = created.payload["item"]["id"]
            payload = {"item_id": item_id, "content": "Updated content"}
            updated = handle_post("/api/workspace/items/update", deps)
            self.assertEqual(updated.status, HTTPStatus.OK)
            self.assertEqual(updated.payload["item"]["content"], "Updated content")
            payload = {"project": "core", "template_id": template["id"], "title": "Copy", "variables": {"name": "Ada"}}
            instantiated = handle_post("/api/workspace/templates/instantiate", deps)
            self.assertEqual(instantiated.status, HTTPStatus.CREATED)
            self.assertEqual(instantiated.payload["item"]["content"], "Dear Ada")
            for route in contract()["routes"]:
                with self.subTest(route=route):
                    result = (
                        handle_get(route["path"], "project=core", deps)
                        if route["method"] == "GET"
                        else handle_post(route["path"], deps)
                    )
                    self.assertIsNotNone(result)
            self.assertEqual(len(handle_get("/api/workspace/items", "project=core&q=Updated", deps).payload["items"]), 1)
            self.assertEqual(handle_get("/api/workspace/dashboard", "project=core", deps).payload["project"], "core")
            self.assertIsNone(handle_get("/api/chat", "", deps))
            self.assertIsNone(handle_post("/api/chat", deps))

    def test_runtime_has_no_inline_workspace_dispatch(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/workspace/', runtime)


if __name__ == "__main__":
    unittest.main()
