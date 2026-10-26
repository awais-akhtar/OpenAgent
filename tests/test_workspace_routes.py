from __future__ import annotations

from http import HTTPStatus
from tempfile import TemporaryDirectory
from unittest import TestCase

from openagent.routes.workspace_routes import WorkspaceRouteDependencies, contract, handle_delete, handle_post
from openagent.storage import OpenAgentStore
from openagent.workspace import WorkspaceManager


class WorkspaceDeleteRouteTests(TestCase):
    def test_delete_contract_and_project_scoped_leaf_deletion(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = WorkspaceManager(OpenAgentStore(tmp))
            parent = workspace.create_item(kind="note", title="Parent", project="core")
            child = workspace.create_item(kind="note", title="Child", project="core", parent_id=parent["id"])
            deps = WorkspaceRouteDependencies(
                workspace=workspace,
                first=lambda params, key: params.get(key, [None])[0],
                parse_int=lambda value, default=100: int(value) if value else default,
                read_json=lambda: {},
            )

            routes = {(route["method"], route["path"]) for route in contract()["routes"]}
            self.assertIn(("DELETE", "/api/workspace/items/{item_id}"), routes)
            blocked = handle_delete(f"/api/workspace/items/{parent['id']}", "project=core", deps)
            self.assertEqual(blocked.status, HTTPStatus.CONFLICT)
            missing = handle_delete(f"/api/workspace/items/{child['id']}", "project=other", deps)
            self.assertEqual(missing.status, HTTPStatus.NOT_FOUND)
            deleted = handle_delete(f"/api/workspace/items/{child['id']}", "project=core", deps)
            self.assertEqual(deleted.status, HTTPStatus.OK)
            self.assertTrue(deleted.payload["deleted"])
            self.assertIsNone(handle_delete("/api/workspace/items/", "project=core", deps))


class WorkspaceUpdateScopeTests(TestCase):
    def test_update_is_limited_to_explicit_project_and_replaces_search_context(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = WorkspaceManager(OpenAgentStore(tmp))
            item = workspace.create_item(kind="note", title="Original", project="alpha", content="old searchable text")
            payload: dict[str, object] = {"item_id": item["id"], "project": "beta", "title": "Changed", "content": "new"}
            deps = WorkspaceRouteDependencies(
                workspace=workspace,
                first=lambda params, key: params.get(key, [None])[0],
                parse_int=lambda value, default=100: int(value) if value else default,
                read_json=lambda: payload,
            )

            denied = handle_post("/api/workspace/items/update", deps)
            self.assertEqual(denied.status, HTTPStatus.NOT_FOUND)
            self.assertEqual(workspace.get_item(item["id"])["title"], "Original")

            payload["project"] = "alpha"
            updated = handle_post("/api/workspace/items/update", deps)
            self.assertEqual(updated.status, HTTPStatus.OK)
            self.assertEqual(updated.payload["item"]["content"], "new")
            self.assertEqual(workspace.search("old searchable text", project="alpha"), [])
            self.assertEqual(len(workspace.search("new", project="alpha")), 1)

    def test_update_unknown_item_is_not_found(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = WorkspaceManager(OpenAgentStore(tmp))
            payload: dict[str, object] = {"item_id": "missing", "project": "default", "title": "Changed"}
            deps = WorkspaceRouteDependencies(
                workspace=workspace,
                first=lambda params, key: params.get(key, [None])[0],
                parse_int=lambda value, default=100: int(value) if value else default,
                read_json=lambda: payload,
            )
            result = handle_post("/api/workspace/items/update", deps)
            self.assertEqual(result.status, HTTPStatus.NOT_FOUND)
