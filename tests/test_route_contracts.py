from __future__ import annotations

from pathlib import Path
import re
import unittest

from openagent.routes import route_contracts
from openagent.routes.health_routes import HealthRouteDependencies, handle_get


class RouteContractTests(unittest.TestCase):
    def test_domain_route_contracts_cover_master_blueprint_api_paths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        blueprint = (root / "MASTER BLUEPRINT.md").read_text(encoding="utf-8")
        blueprint_paths = sorted(set(re.findall(r"/api/[A-Za-z0-9_./{}-]+", blueprint)))
        declared = {
            (route["method"], route["path"])
            for contract in route_contracts()
            for route in contract["routes"]
        }
        declared_paths = {path for _method, path in declared}

        missing = []
        for path in blueprint_paths:
            if "{" in path:
                prefix = path.split("{", 1)[0].rstrip("/")
                if path in declared_paths or any(route.startswith(prefix) for route in declared_paths):
                    continue
            elif path.endswith("/"):
                if any(route.startswith(path) for route in declared_paths):
                    continue
            elif path in declared_paths:
                continue
            missing.append(path)

        self.assertEqual(missing, [])

    def test_each_domain_route_contract_is_declared(self) -> None:
        contracts = route_contracts()
        self.assertGreaterEqual(len(contracts), 20)
        for contract in contracts:
            self.assertIn("group", contract)
            self.assertIn("routes", contract)
            self.assertGreater(contract["route_count"], 0)

    def test_health_routes_preserve_public_contracts(self) -> None:
        deps = HealthRouteDependencies(
            health=lambda: {"status": "ok"},
            system_metrics=lambda: {"status": "ok", "cpu": {"count": 1}},
            ui_build_info=lambda: {"project": "OpenAgent", "assets": {"app.js": {}}},
            module_health=lambda project: {"project": project, "modules": []},
        )

        self.assertEqual(handle_get("/api/health", "", deps).payload["status"], "ok")
        self.assertEqual(handle_get("/api/system/metrics", "", deps).payload["cpu"]["count"], 1)
        self.assertEqual(handle_get("/api/ui/build", "", deps).payload["project"], "OpenAgent")
        self.assertEqual(handle_get("/api/modules/health", "project=core", deps).payload["project"], "core")
        self.assertIsNone(handle_get("/api/chat", "", deps))

    def test_server_module_is_thin_facade(self) -> None:
        root = Path(__file__).resolve().parents[1]
        server_py = root / "openagent" / "server.py"
        runtime_py = root / "openagent" / "routes" / "runtime.py"
        server_text = server_py.read_text(encoding="utf-8")
        self.assertLessEqual(len(server_text.splitlines()), 40)
        self.assertIn("from .routes.runtime import", server_text)
        self.assertTrue(runtime_py.exists())


if __name__ == "__main__":
    unittest.main()
