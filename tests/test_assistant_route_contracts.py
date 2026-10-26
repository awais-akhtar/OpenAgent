from __future__ import annotations

import unittest
from types import SimpleNamespace

from openagent.auth import required_scope_for_path
from openagent.routes import assistant_routes


class AssistantLifecycleRouteContractTests(unittest.TestCase):
    def test_task_and_goal_create_routes_reach_the_service(self) -> None:
        created = []
        assistant = SimpleNamespace(
            create_task=lambda **kwargs: created.append(("task", kwargs)) or {"id": "task-1", **kwargs},
            create_goal=lambda **kwargs: created.append(("goal", kwargs)) or {"id": "goal-1", **kwargs},
        )
        payload = {"title": "A local task", "project": "default"}
        deps = assistant_routes.AssistantRouteDependencies(
            assistant=assistant,
            first=lambda values, key: (values.get(key) or [None])[0],
            parse_int=lambda value, default=0: default,
            read_json=lambda: payload,
        )

        for path, expected_key, expected_id in (
            ("/api/assistant/tasks", "task", "task-1"),
            ("/api/assistant/goals", "goal", "goal-1"),
        ):
            result = assistant_routes.handle_post(path, deps)
            self.assertIsNotNone(result)
            self.assertEqual(result.status, 201)
            self.assertEqual(result.payload[expected_key]["id"], expected_id)

        self.assertEqual([kind for kind, _ in created], ["task", "goal"])

    def test_task_and_goal_mutations_require_assistant_write(self) -> None:
        routes = {(route["method"], route["path"]) for route in assistant_routes.contract()["routes"]}
        for method, path in (
            ("PATCH", "/api/assistant/tasks/{task_id}"),
            ("DELETE", "/api/assistant/tasks/{task_id}"),
            ("PATCH", "/api/assistant/goals/{goal_id}"),
            ("DELETE", "/api/assistant/goals/{goal_id}"),
        ):
            self.assertIn((method, path), routes)
        self.assertEqual(required_scope_for_path("/api/assistant/tasks/task-1", "PATCH"), "assistant:write")
        self.assertEqual(required_scope_for_path("/api/assistant/tasks/task-1", "DELETE"), "assistant:write")
        self.assertEqual(required_scope_for_path("/api/assistant/goals/goal-1", "PATCH"), "assistant:write")
        self.assertEqual(required_scope_for_path("/api/assistant/goals/goal-1", "DELETE"), "assistant:write")


if __name__ == "__main__":
    unittest.main()
