from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from openagent.routes import route_contracts
from openagent.routes.science_routes import ScienceRouteDependencies, handle_get, handle_post
from openagent.science.service import ScientificService
from openagent.storage import OpenAgentStore


class ScienceRouteTests(unittest.TestCase):
    def test_science_contract_is_declared(self) -> None:
        contract = next(item for item in route_contracts() if item["group"] == "science")
        paths = {item["path"] for item in contract["routes"]}
        self.assertIn("/api/science/status", paths)
        self.assertIn("/api/science/run", paths)
        self.assertIn("/api/science/tools", paths)
        self.assertIn("/api/science/experiments", paths)
        self.assertIn("/api/science/execute", paths)
        self.assertIn("/api/science/memory", paths)
        self.assertIn("/api/science/materials/datasets", paths)
        self.assertIn("/api/science/materials/analyse", paths)
        self.assertIn("/api/science/verify", paths)
        self.assertIn("/api/science/simulate", paths)

    def test_status_route_is_read_only(self) -> None:
        deps = ScienceRouteDependencies(agent=None, run_agent=None, read_json=lambda: {})  # type: ignore[arg-type]
        result = handle_get("/api/science/status", "", deps)
        self.assertIsNotNone(result)
        self.assertEqual(result.payload["status"], "ready")

    def test_run_route_uses_the_existing_agent_runner(self) -> None:
        calls: list[tuple[str, str]] = []
        deps = ScienceRouteDependencies(
            agent=None,  # type: ignore[arg-type]
            run_agent=lambda goal, project="default": calls.append((goal, project)) or {"agent": "scientific"},
            read_json=lambda: {"goal": "solve heat equation", "project": "lab"},
        )
        result = handle_post("/api/science/run", deps)
        self.assertIsNotNone(result)
        self.assertEqual(calls, [("solve heat equation", "lab")])
        self.assertEqual(result.payload["run"]["agent"], "scientific")

    def test_experiment_proposals_are_saved_and_listed_as_not_executed(self) -> None:
        with TemporaryDirectory() as temp:
            service = ScientificService(OpenAgentStore(temp))
            payload = {"project": "lab", "goal": "Compare thermal materials", "candidates": [{"material": "A", "temperature": 20, "priority": 2}]}
            deps = ScienceRouteDependencies(agent=None, run_agent=None, read_json=lambda: payload, service=service)  # type: ignore[arg-type]
            created = handle_post("/api/science/experiments", deps)
            self.assertEqual(created.status, 201)
            self.assertEqual(created.payload["history"]["status"], "proposed")
            listed = handle_get("/api/science/experiments", "project=lab", deps)
            self.assertEqual(len(listed.payload["experiments"]), 1)
            self.assertEqual(listed.payload["experiments"][0]["execution_status"], "not_started")
            self.assertEqual(listed.payload["experiments"][0]["proposals"][0]["variables"]["material"], "A")


if __name__ == "__main__":
    unittest.main()
