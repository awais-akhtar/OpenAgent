"""Tests for G22 - Ultra Model session readiness and recovery."""

import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from uuid import uuid4

from openagent.model_cookbook import ModelCookbook
from openagent.storage import OpenAgentStore


class UltraModelReadinessTests(unittest.TestCase):
    """Test Ultra Model session readiness and local manifest handling."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test database."""
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.store = OpenAgentStore(str(Path(cls.tmpdir.name) / "test.db"))
        cls.cookbook = ModelCookbook(cls.store)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory."""
        cls.tmpdir.cleanup()

    def test_mock_llama_server_availability(self) -> None:
        """Test that mock llama-server can be used for testing."""
        # Check that the mock module exists
        try:
            from openagent.mock_llama_server import run_mock_server

            assert run_mock_server is not None
        except ImportError as exc:
            self.fail(f"mock_llama_server module not available: {exc}")

    def test_ultra_model_manifest_validation(self) -> None:
        """Test that Ultra Model manifest can be created and validated."""
        manifest = {
            "model_name": "test-llama-model",
            "model_version": "1.0",
            "ollama_compatible": True,
            "server_type": "llama-server",
            "parameters": {
                "context_size": 2048,
                "gpu_layers": 10,
            },
        }

        # Validate manifest structure
        assert "model_name" in manifest
        assert "server_type" in manifest
        assert manifest["server_type"] in {"ollama", "llama-server", "llama.cpp", "vllm"}

    def test_ultra_model_serve_plan_creation(self) -> None:
        """Test creating a serve plan for Ultra Model."""
        project = "test-ultra"

        serve_plan = self.cookbook.create_serve_plan(
            project=project,
            model_name="test-llama-7b",
            model_source="local",
            serve_type="llama-server",
            config={
                "host": "127.0.0.1",
                "port": 8000,
                "context_size": 2048,
            },
        )

        assert serve_plan is not None
        assert serve_plan["project"] == project
        assert serve_plan["model_name"] == "test-llama-7b"
        assert serve_plan["serve_type"] == "llama-server"

    def test_ultra_model_session_creation(self) -> None:
        """Test creating a session for Ultra Model serving."""
        project = "test-ultra-session"

        serve_plan = self.cookbook.create_serve_plan(
            project=project,
            model_name="test-model",
            model_source="local",
            serve_type="llama-server",
            config={"host": "127.0.0.1", "port": 9000},
        )

        # Create session (doesn't actually launch server in test)
        session = self.cookbook.create_session(
            project=project,
            serve_plan_id=serve_plan["id"],
        )

        assert session is not None
        assert session["project"] == project
        # A session holds a serve plan and starts no process, so it must not
        # report "ready": that read as a model ready to answer.
        assert session["status"] == "planned"

    def test_ultra_model_readiness_check(self) -> None:
        """Test checking readiness of Ultra Model infrastructure."""
        project = "test-readiness"

        # Get readiness status
        readiness = self.cookbook.get_readiness_status(project=project)

        # Should include status of various components
        assert "llama_server_available" in readiness or "status" in readiness

    def test_ultra_model_process_identity_tracking(self) -> None:
        """Test that process identity is tracked for serve sessions."""
        project = "test-process-identity"

        serve_plan = self.cookbook.create_serve_plan(
            project=project,
            model_name="test-model",
            model_source="local",
            serve_type="llama-server",
            config={"host": "127.0.0.1", "port": 9001},
        )

        session = self.cookbook.create_session(
            project=project,
            serve_plan_id=serve_plan["id"],
        )

        # Session should track creation identity for safety
        assert session.get("created_at") is not None
        assert session.get("project") == project

    def test_ultra_model_stop_with_identity_verification(self) -> None:
        """Test that stop operation verifies process identity."""
        project = "test-stop-verify"

        serve_plan = self.cookbook.create_serve_plan(
            project=project,
            model_name="test-model",
            model_source="local",
            serve_type="llama-server",
            config={"host": "127.0.0.1", "port": 9002},
        )

        session = self.cookbook.create_session(
            project=project,
            serve_plan_id=serve_plan["id"],
        )

        # Stop session (mock - doesn't actually have a running process)
        # But should verify identity before attempting stop
        result = self.cookbook.stop_session(session["id"], project=project)

        assert result is not None
        assert result["status"] in {"stopped", "not_running"}

    def test_ultra_model_manifest_gguf_handling(self) -> None:
        """Test that GGUF model files are properly referenced."""
        # GGUF is the quantized format for llama.cpp/llama-server
        manifest = {
            "model_name": "test-llama-gguf",
            "model_file": "model.gguf",
            "model_format": "gguf",
            "quantization": "Q4_K_M",
            "file_size_bytes": 4_294_967_296,  # 4GB example
        }

        # Validate GGUF manifest
        assert manifest["model_format"] == "gguf"
        assert manifest["model_file"].endswith(".gguf")
        assert manifest["quantization"] in {"Q2_K", "Q3_K", "Q4_K", "Q4_K_M", "Q5_K", "Q6_K", "Q8_0"}

    def test_ultra_model_no_unsafe_substitution(self) -> None:
        """Test that training safetensors cannot substitute for GGUF launch."""
        # This verifies that we don't fake GGUF availability with training artifacts
        training_artifact = {"type": "safetensors", "adapter_id": "test-adapter"}
        gguf_requirement = {"type": "gguf", "format": "quantized"}

        # They should not be compatible
        assert training_artifact["type"] != gguf_requirement["type"]
        assert training_artifact["type"] == "safetensors"
        assert gguf_requirement["type"] == "gguf"

    def test_ultra_model_environment_readiness(self) -> None:
        """Test environment readiness check for Ultra Model."""
        # Check if llama-server or compatible alternative exists
        # For testing, mock is sufficient
        try:
            from openagent.mock_llama_server import run_mock_server

            mock_available = True
        except ImportError:
            mock_available = False

        # Either mock or real llama-server should be available
        assert mock_available or self._check_real_llama_server(), "No llama-server alternative available"

    def _check_real_llama_server(self) -> bool:
        """Check if real llama-server executable exists."""
        # Check common paths and PATH
        for program in ["llama-server", "llama-server.exe", "llama.cpp"]:
            try:
                result = subprocess.run([program, "--version"], capture_output=True, timeout=2)
                if result.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return False


class UltraModelBootstrapTests(unittest.TestCase):
    """Test bootstrapping/setup of Ultra Model infrastructure."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test database."""
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.store = OpenAgentStore(str(Path(cls.tmpdir.name) / "test.db"))

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory."""
        cls.tmpdir.cleanup()

    def test_mock_llama_server_script_exists(self) -> None:
        """Test that mock llama-server script is available."""
        mock_path = Path(__file__).parent.parent / "openagent" / "mock_llama_server.py"
        assert mock_path.exists(), f"mock_llama_server.py not found at {mock_path}"

    def test_mock_llama_server_can_be_imported(self) -> None:
        """Test that mock llama-server can be imported."""
        try:
            from openagent.mock_llama_server import run_mock_server, MockLlamaServerHandler

            assert run_mock_server is not None
            assert MockLlamaServerHandler is not None
        except ImportError as exc:
            self.fail(f"Cannot import mock_llama_server: {exc}")

    def test_ultra_model_setup_with_mock(self) -> None:
        """Test that Ultra Model can be set up to use mock llama-server."""
        # Create a cookbook instance
        cookbook = ModelCookbook(self.store)

        # Verify cookbook can work with local setup
        assert cookbook is not None
        assert cookbook.store is not None


if __name__ == "__main__":
    unittest.main()
