from __future__ import annotations

import json
import os
from tempfile import TemporaryDirectory
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.enterprise import EnterpriseManager
from openagent.integration_hub import IntegrationHub, IntegrationHubError
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class IntegrationHubTests(unittest.TestCase):
    def test_safe_command_enforces_process_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            enterprise = EnterpriseManager(store)
            hub = IntegrationHub(
                store,
                health_provider=store.health,
                module_health_provider=lambda project: {"status": "ok"},
                metrics_provider=lambda: {"status": "ok"},
                model_provider=lambda: {"status": "ok"},
                voice_provider=lambda project: {"status": "ok"},
                developer_provider=lambda: {"status": "ok"},
                tool_provider=lambda: [],
                backup_provider=enterprise.create_backup,
                backups_provider=enterprise.backups,
            )
            script = "import os; print(os.getenv('OPENAGENT_PROCESS_SECRET')); print('x' * 10000)"
            with patch.object(IntegrationHub, "SAFE_COMMANDS", {
                "bounded": {"label": "Bounded", "description": "Test command", "command": [sys.executable, "-c", script]},
            }), patch.dict(os.environ, {"OPENAGENT_PROCESS_SECRET": "private"}):
                result = hub.run_safe_command(command_key="bounded", project="core")["run"]
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["output"].startswith("None"))
            self.assertNotIn("private", result["output"])
            self.assertLessEqual(len(result["output"].encode("utf-8")), 4000)

            with patch.object(IntegrationHub, "SAFE_COMMANDS", {
                "slow": {"label": "Slow", "description": "Test timeout", "command": [sys.executable, "-c", "import time; time.sleep(5)"]},
            }):
                timed_out = hub.run_safe_command(command_key="slow", project="core", timeout_seconds=1)["run"]
            self.assertEqual(timed_out["status"], "timeout")
            self.assertEqual(timed_out["return_code"], 124)
            self.assertEqual(len(hub.safe_commands(project="core")["runs"]), 2)

    def test_pairing_preferences_diagnostics_and_safe_commands_are_local(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            enterprise = EnterpriseManager(store)
            hub = IntegrationHub(
                store,
                health_provider=store.health,
                module_health_provider=lambda project: {"status": "ok", "project": project, "counts": {"error": 0}},
                metrics_provider=lambda: {"status": "ok", "cpu": {"count": 1}},
                model_provider=lambda: {"status": "ready"},
                voice_provider=lambda project: {"status": "ready", "project": project},
                developer_provider=lambda: {"tools": 1},
                tool_provider=lambda: [{"id": "echo"}],
                backup_provider=enterprise.create_backup,
                backups_provider=enterprise.backups,
            )

            code = hub.create_pairing_code(label="Desktop", project="core", ttl_minutes=5)
            self.assertTrue(code["code"].startswith("OA-"))
            self.assertNotIn("code_hash", code)
            self.assertEqual(len(hub.pairing_codes(project="core")), 1)

            paired = hub.redeem_pairing_code(code=code["code"], device_name="Laptop", project="core")
            self.assertEqual(paired["status"], "paired")
            self.assertTrue(paired["token"].startswith("oa_pair_"))
            with self.assertRaises(IntegrationHubError):
                hub.redeem_pairing_code(code=code["code"], device_name="Laptop", project="core")

            pref = hub.set_preference(project="core", scope="ui", key="sidebar", value={"collapsed": True})
            self.assertEqual(pref["value"]["collapsed"], True)
            self.assertEqual(hub.preferences(project="core", scope="ui")[0]["key"], "sidebar")

            diagnostic = hub.create_diagnostic_report(project="core")
            self.assertEqual(diagnostic["status"], "ok")
            self.assertIn("modules", diagnostic["sections"])

            backup = hub.create_backup(project="core")
            self.assertEqual(backup["backup"]["status"], "completed")
            self.assertEqual(len(enterprise.backups()), 1)

            safe = hub.run_safe_command(command_key="python-version", project="core")
            self.assertIn(safe["run"]["status"], {"completed", "failed"})
            self.assertNotIn("powershell", " ".join(hub.safe_commands(project="core")["commands"][0]["command"]).lower())
            with self.assertRaises(IntegrationHubError):
                hub.run_safe_command(command_key="dir", project="core")

    def test_integration_routes_are_available(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                status = read_json(f"{base_url}/api/integrations/status?project=core")
                self.assertEqual(status["status"], "ready")

                create_code = post_json(
                    f"{base_url}/api/integrations/pairing-codes",
                    {"label": "Desktop", "project": "core", "ttl_minutes": 5},
                )
                self.assertTrue(create_code["code"]["code"].startswith("OA-"))

                codes = read_json(f"{base_url}/api/integrations/pairing-codes?project=core")
                self.assertEqual(len(codes["codes"]), 1)

                pref = post_json(
                    f"{base_url}/api/integrations/preferences",
                    {"project": "core", "scope": "ui", "key": "theme", "value": "dark"},
                )
                self.assertEqual(pref["preference"]["value"], "dark")

                latest = read_json(f"{base_url}/api/integrations/diagnostics?project=core&latest=true")
                self.assertIn(latest["status"], {"ok", "needs_attention"})

                diagnostic = post_json(f"{base_url}/api/integrations/diagnostics", {"project": "core"})
                self.assertIn("sections", diagnostic["diagnostic"])

                safe = post_json(
                    f"{base_url}/api/integrations/safe-shell",
                    {"project": "core", "command_key": "python-version"},
                )
                self.assertIn("run", safe)

                with self.assertRaises(HTTPError) as ctx:
                    post_json(
                        f"{base_url}/api/integrations/safe-shell",
                        {"project": "core", "command_key": "echo hello"},
                    )
                self.assertEqual(ctx.exception.code, 400)
                ctx.exception.close()

                parity = read_json(f"{base_url}/api/integrations/parity?project=core")
                self.assertEqual(parity["status"], "ready")
                self.assertGreaterEqual(len(parity["items"]), 8)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


def read_json(url: str) -> dict[str, object]:
    return json.loads(urlopen(url, timeout=10).read().decode("utf-8"))


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urlopen(request, timeout=20).read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
