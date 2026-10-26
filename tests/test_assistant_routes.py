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
from openagent.auth import required_scope_for_path
from openagent.models import AuditEvent
from openagent.routes.assistant_routes import AssistantRouteDependencies, contract, handle_delete, handle_get, handle_patch, handle_post
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class AssistantRouteTests(unittest.TestCase):
    def test_project_activity_route_is_explicitly_scoped_and_filters_other_audit_events(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            assistant = AssistantManager(store)
            assistant.create_task(title="Project task", project="core")
            assistant.create_goal(title="Project goal", project="core")
            assistant.create_task(title="Other project task", project="other")
            store.add_audit_event(AuditEvent(
                actor="security", action="security.policy.change", resource="policy:demo", project="core",
            ))
            deps = AssistantRouteDependencies(
                assistant=assistant,
                first=lambda params, key: params.get(key, [None])[0],
                parse_int=lambda value, default=0: default if value is None else int(value),
                read_json=lambda: {},
            )
            self.assertEqual(required_scope_for_path("/api/assistant/activity", "GET"), "assistant:read")
            self.assertEqual(handle_get("/api/assistant/activity", "limit=20", deps).status, HTTPStatus.BAD_REQUEST)
            result = handle_get("/api/assistant/activity", "project=core&limit=20", deps)
            self.assertEqual(result.status, HTTPStatus.OK)
            actions = {event["action"] for event in result.payload["events"]}
            self.assertEqual(actions, {"assistant.task.create", "assistant.goal.create"})
            self.assertEqual({event["project"] for event in result.payload["events"]}, {"core"})

    def test_task_creation_rejects_goal_from_another_project(self) -> None:
        with TemporaryDirectory() as tmp:
            assistant = AssistantManager(OpenAgentStore(tmp))
            foreign_goal = assistant.create_goal(title="Private goal", project="alpha")
            payload: dict[str, object] = {"project": "beta", "title": "Cross-project task", "goal_id": foreign_goal["id"]}
            deps = AssistantRouteDependencies(
                assistant=assistant,
                first=lambda params, key: params.get(key, [None])[0],
                parse_int=lambda value, default=0: default if value is None else int(value),
                read_json=lambda: payload,
            )

            response = handle_post("/api/assistant/tasks", deps)

            self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)
            self.assertIn("does not exist in this project", response.payload["error"])
            self.assertEqual(assistant.tasks(project="beta"), [])

    def test_task_completion_requires_matching_project_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            assistant = AssistantManager(OpenAgentStore(tmp))
            task = assistant.create_task(title="Project task", project="alpha")
            payload: dict[str, object] = {"task_id": task["id"]}
            deps = AssistantRouteDependencies(
                assistant=assistant,
                first=lambda params, key: params.get(key, [None])[0],
                parse_int=lambda value, default=0: default if value is None else int(value),
                read_json=lambda: payload,
            )

            missing_scope = handle_post("/api/assistant/tasks/complete", deps)
            self.assertEqual(missing_scope.status, HTTPStatus.BAD_REQUEST)
            payload["project"] = "beta"
            wrong_project = handle_post("/api/assistant/tasks/complete", deps)
            self.assertEqual(wrong_project.status, HTTPStatus.BAD_REQUEST)
            self.assertEqual(assistant.tasks(project="alpha")[0]["status"], "open")

            payload["project"] = "alpha"
            completed = handle_post("/api/assistant/tasks/complete", deps)
            self.assertEqual(completed.status, HTTPStatus.OK)
            self.assertEqual(completed.payload["task"]["project"], "alpha")
            self.assertEqual(completed.payload["task"]["status"], "completed")

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
            task = deps.assistant.create_task(title="Existing task", project="core")
            goal = deps.assistant.create_goal(title="Existing goal", project="core")
            for route in contract()["routes"]:
                with self.subTest(route=route):
                    path = route["path"].replace("{task_id}", str(task["id"])).replace("{goal_id}", str(goal["id"]))
                    result = {
                        "GET": lambda: handle_get(path, "project=core", deps),
                        "POST": lambda: handle_post(path, deps),
                        "PATCH": lambda: handle_patch(path, "project=core", deps),
                        "DELETE": lambda: handle_delete(path, "project=core", deps),
                    }[route["method"]]()
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

    def test_http_contract_rejects_cross_project_goal_link_without_persisting_task(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            foreign_goal = AssistantManager(store).create_goal(title="Alpha-only", project="alpha")
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = Request(
                    f"{base}/api/assistant/tasks",
                    data=json.dumps({"project": "beta", "title": "Invalid link", "goal_id": foreign_goal["id"]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=10)
                with error.exception as response:
                    self.assertEqual(response.code, HTTPStatus.BAD_REQUEST)
                    error_body = json.load(response)
                    error_detail = error_body.get("error", error_body)
                    message = error_detail.get("message", "") if isinstance(error_detail, dict) else str(error_detail)
                    self.assertIn("does not exist in this project", message)
                with urlopen(f"{base}/api/assistant/tasks?project=beta", timeout=10) as response:
                    self.assertEqual(json.load(response)["tasks"], [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_runtime_has_no_inline_assistant_dispatch(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/assistant/', runtime)


if __name__ == "__main__":
    unittest.main()
