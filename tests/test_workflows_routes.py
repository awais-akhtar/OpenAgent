from __future__ import annotations

from http import HTTPStatus
from unittest import TestCase
from unittest.mock import Mock

from openagent.routes.workflows_routes import (
    ROUTES,
    WorkflowRouteDependencies,
    handle_delete,
    handle_get,
    handle_post,
    handle_put,
)


class WorkflowRouteTests(TestCase):
    def dependencies(self, payload=None):
        store, workflows = Mock(), Mock()
        read_json = Mock(return_value=payload or {})
        first = lambda params, key: params.get(key, [None])[0]
        parse_int = lambda value, default: int(value) if value and value.isdigit() else default
        parse_bool = lambda value, default: default if value is None else str(value).lower() in {"1", "true", "yes", "on"}
        return store, workflows, read_json, WorkflowRouteDependencies(
            store, workflows, read_json, first, parse_int, parse_bool
        )

    def test_all_contract_verbs_have_dispatchers(self):
        from openagent import routes
        from importlib import import_module

        module = import_module("openagent.routes.workflows_routes")
        self.assertTrue(all(callable(getattr(module, f"handle_{method.lower()}", None)) for method, _ in ROUTES))
        self.assertIn("workflows_routes", routes.DOMAIN_ROUTE_MODULES)

    def test_list_and_visual_get(self):
        store, workflows, _, deps = self.dependencies()
        store.list_workflows.return_value = [{"id": "w1", "steps": [{"action": "notify"}]}]
        workflows.get_trigger.return_value = {"type": "manual"}
        listed = handle_get("/api/workflows", "project=core&limit=8", deps)
        self.assertEqual(listed.payload["workflows"][0]["trigger"], {"type": "manual"})
        store.list_workflows.assert_called_with(project="core", limit=8)
        visual = handle_get("/api/workflows/visual", "workflow_id=w1", deps)
        self.assertEqual(visual.payload["nodes"][-1]["label"], "notify")
        self.assertEqual(visual.payload["edges"], [{"source": "trigger", "target": "s0"}])
        self.assertEqual(handle_get("/api/workflows/layout", "", deps).status, HTTPStatus.BAD_REQUEST)

    def test_create_run_and_event(self):
        _, workflows, _, deps = self.dependencies({"name": "  Daily ", "steps": [{"action": "notify"}, "skip"]})
        workflows.create.return_value = {"id": "w1"}
        created = handle_post("/api/workflows", deps)
        self.assertEqual(created.status, HTTPStatus.CREATED)
        workflows.create.assert_called_once()
        workflows.run.return_value = {"id": "r1"}
        run = handle_post("/api/workflows/w1/run", deps)
        self.assertEqual(run.payload, {"run": {"id": "r1"}})
        workflows.run.assert_called_with("w1", allow_network=False, dry_run=False)
        event_store, workflows, read_json, _ = self.dependencies({"event_name": "hourly"})
        deps = WorkflowRouteDependencies(event_store, workflows, read_json, deps.first, deps.parse_int, deps.parse_bool)
        workflows.trigger_event.return_value = {"triggered": 1}
        event = handle_post("/api/workflows/event", deps)
        self.assertEqual(event.payload, {"triggered": 1})

    def test_update_delete_and_unknown_paths(self):
        _, workflows, _, deps = self.dependencies({"name": "Updated"})
        workflows.update.return_value = {"id": "w1", "name": "Updated"}
        self.assertEqual(handle_put("/api/workflows/w1", deps).payload["workflow"]["name"], "Updated")
        workflows.delete.return_value = {"deleted": True}
        self.assertEqual(handle_delete("/api/workflows/w1", deps).payload, {"deleted": True})
        self.assertIsNone(handle_put("/api/workflows/run", deps))
        self.assertIsNone(handle_delete("/api/workflows/builder", deps))
        self.assertIsNone(handle_get("/api/other", "", deps))


if __name__ == "__main__":
    import unittest

    unittest.main()
