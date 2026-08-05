from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openagent.local_ai import LocalAIInfrastructure, ModelDownloadError
from openagent.routes.local_ai_routes import ROUTES
from openagent.storage import OpenAgentStore


class ModelOperationsLifecycleTests(unittest.TestCase):
    def test_checksum_probe_dependencies_and_tool_assignment(self) -> None:
        with TemporaryDirectory() as temp:
            source = Path(temp) / "source.gguf"
            source.write_bytes(b"local model")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            manager = LocalAIInfrastructure(OpenAgentStore(Path(temp) / "data"))
            model = manager.register_model(
                name="local-test", provider="llamacpp", source_url=str(source), metadata={"sha256": expected}
            )
            downloaded = manager.download_model(model["id"])
            self.assertEqual(downloaded["sha256"], expected)
            self.assertTrue(manager.probe_model(model["id"])["checksum_valid"])
            manager.assign_tool(model["id"], "local-search", config={"limit": 5})
            manager.add_model_dependency(model["id"], "role", "research")
            with self.assertRaises(ValueError):
                manager.delete_model(model["id"])

    def test_checksum_mismatch_keeps_partial_artifact(self) -> None:
        with TemporaryDirectory() as temp:
            source = Path(temp) / "source.bin"
            source.write_bytes(b"wrong")
            manager = LocalAIInfrastructure(OpenAgentStore(Path(temp) / "data"))
            model = manager.register_model(name="bad", provider="local", source_url=str(source), metadata={"sha256": "0" * 64})
            with self.assertRaises(ModelDownloadError):
                manager.download_model(model["id"])
            self.assertEqual(manager.get_model(model["id"])["status"], "failed")

    def test_route_contract_exposes_operator_lifecycle(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/local-ai/models/{id}/probe", paths)
        self.assertIn("/api/local-ai/models/{id}/tools", paths)


if __name__ == "__main__":
    unittest.main()
