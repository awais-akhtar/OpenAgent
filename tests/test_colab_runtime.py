from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from openagent.colab import launch as documented_launch
from openagent.runtimes.colab import detect_colab, recommend_model, launch


class ColabRuntimeTests(unittest.TestCase):
    def test_documented_colab_import_reexports_launcher(self) -> None:
        self.assertIs(documented_launch, launch)

    def test_detect_colab_uses_explicit_environment_marker(self) -> None:
        with patch.dict("os.environ", {"COLAB_RELEASE_TAG": "test"}, clear=True):
            self.assertTrue(detect_colab())

    def test_detect_colab_is_safe_without_google_namespace(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch("openagent.runtimes.colab.importlib.util.find_spec", side_effect=ModuleNotFoundError):
            self.assertFalse(detect_colab())

    def test_model_recommendation_is_hardware_aware_and_conservative(self) -> None:
        self.assertEqual(recommend_model({"gpu": False}), "phi4-mini")
        self.assertEqual(recommend_model({"gpu": True, "vram_mb": 8_000}), "phi4-mini")
        self.assertEqual(recommend_model({"gpu": True, "vram_mb": 16_000}), "qwen2.5:7b")

    def test_launch_uses_existing_http_server_and_returns_actual_port(self) -> None:
        server = MagicMock()
        server.server_address = ("127.0.0.1", 43123)
        with TemporaryDirectory() as temp, patch("openagent.runtimes.colab.build_http_server", return_value=server), patch(
            "openagent.runtimes.colab.detect_hardware", return_value={"gpu": False, "name": "", "vram_mb": 0}
        ):
            started = launch(data_dir=temp, port=0)
            self.assertEqual(started.url, "http://127.0.0.1:43123/")
            self.assertEqual(started.recommended_model, "phi4-mini")
            started.stop()
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()