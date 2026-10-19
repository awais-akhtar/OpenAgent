from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.local_ai import LocalAIInfrastructure
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class ModelLifecycleTests(unittest.TestCase):
    def test_valid_transitions_and_diagnostics_trail(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            infra = LocalAIInfrastructure(store)
            model = infra.register_model(name="Test Model", provider="local")
            model_id = model["id"]

            step1 = infra.transition_model_lifecycle(model_id, "discovered")
            self.assertEqual(step1["state"], "discovered")
            step2 = infra.transition_model_lifecycle(model_id, "downloading")
            self.assertEqual(step2["state"], "downloading")
            step3 = infra.transition_model_lifecycle(model_id, "ready")
            self.assertEqual(step3["state"], "ready")
            step4 = infra.transition_model_lifecycle(model_id, "serving")
            self.assertEqual(step4["state"], "serving")

            diagnostics = infra.get_model_diagnostics(model_id)
            self.assertEqual(len(diagnostics), 4)
            self.assertEqual(diagnostics[0]["state"], "serving")  # most recent first

    def test_illegal_transition_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            infra = LocalAIInfrastructure(store)
            model = infra.register_model(name="Test Model", provider="local")
            model_id = model["id"]
            infra.transition_model_lifecycle(model_id, "discovered")
            with self.assertRaises(ValueError):
                infra.transition_model_lifecycle(model_id, "serving")  # discovered cannot jump straight to serving

    def test_failure_reason_is_normalized(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            infra = LocalAIInfrastructure(store)
            model = infra.register_model(name="Test Model", provider="local")
            model_id = model["id"]
            infra.transition_model_lifecycle(model_id, "discovered")
            result = infra.transition_model_lifecycle(model_id, "failed", reason="insufficient_memory")
            self.assertEqual(result["reason"], "insufficient_memory")

            model2 = infra.register_model(name="Test Model 2", provider="local")
            infra.transition_model_lifecycle(model2["id"], "discovered")
            unknown_reason = infra.transition_model_lifecycle(model2["id"], "failed", reason="not-a-real-reason-code")
            self.assertEqual(unknown_reason["reason"], "unknown")

    def test_unknown_model_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            infra = LocalAIInfrastructure(store)
            with self.assertRaises(KeyError):
                infra.transition_model_lifecycle("missing-model", "discovered")
            with self.assertRaises(KeyError):
                infra.get_model_diagnostics("missing-model")

    def test_hardware_fit_uses_marketplace_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            infra = LocalAIInfrastructure(store)
            fit = infra.hardware_fit_for_model("qwen2.5-0.5b-q4")  # seeded marketplace entry, ram_gb=4
            self.assertIn(fit["fit"], {"comfortable", "marginal", "insufficient"})
            self.assertEqual(fit["required_ram_gb"], 4)

            with self.assertRaises(KeyError):
                infra.hardware_fit_for_model("totally-unknown-model-id")


class ModelLifecycleHttpRouteTests(unittest.TestCase):
    def test_transition_and_diagnostics_routes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                register_infra = LocalAIInfrastructure(store)
                model = register_infra.register_model(name="HTTP Model", provider="local")
                model_id = model["id"]

                transition_request = Request(
                    f"{base_url}/api/local-ai/models/{model_id}/transition",
                    data=json.dumps({"state": "discovered"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                result = json.loads(urlopen(transition_request, timeout=15).read().decode("utf-8"))
                self.assertEqual(result["state"], "discovered")

                diagnostics = json.loads(
                    urlopen(f"{base_url}/api/local-ai/models/{model_id}/diagnostics", timeout=15).read().decode("utf-8")
                )
                self.assertEqual(len(diagnostics["diagnostics"]), 1)

                bad_request = Request(
                    f"{base_url}/api/local-ai/models/{model_id}/transition",
                    data=json.dumps({"state": "not-a-state"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(bad_request, timeout=15)
                self.assertEqual(ctx.exception.code, 400)
                ctx.exception.close()
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
