from __future__ import annotations

from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from openagent.colab import launch as documented_launch
from openagent.process_policy import ProcessPolicy, ProcessResult
from openagent.runtimes.colab import detect_colab, detect_hardware, recommend_model, launch


class ColabRuntimeTests(unittest.TestCase):
    def test_documented_colab_import_reexports_launcher(self) -> None:
        self.assertIs(documented_launch, launch)

    def test_detect_colab_uses_explicit_environment_marker(self) -> None:
        with patch.dict("os.environ", {"COLAB_RELEASE_TAG": "test"}, clear=True):
            self.assertTrue(detect_colab())

    def test_detect_colab_is_safe_without_google_namespace(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch("openagent.runtimes.colab.importlib.util.find_spec", side_effect=ModuleNotFoundError):
            self.assertFalse(detect_colab())

    def test_detect_hardware_parses_probe_output_with_bounded_policy(self) -> None:
        probe = ProcessResult(
            argv=("nvidia-smi",),
            returncode=0,
            stdout="Tesla T4, 15360\nSecond GPU, 8192\n",
            stderr="",
            timed_out=False,
            output_truncated=False,
        )
        with patch("openagent.runtimes.colab.run_process", return_value=probe) as run:
            self.assertEqual(detect_hardware(), {"gpu": True, "name": "Tesla T4", "vram_mb": 15360})
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        self.assertEqual(
            kwargs["policy"],
            ProcessPolicy(timeout_seconds=3, max_output_bytes=4_000, allowed_executables=("nvidia-smi",)),
        )

    def test_detect_hardware_returns_empty_summary_when_tool_is_missing(self) -> None:
        with patch("openagent.runtimes.colab.run_process", side_effect=FileNotFoundError):
            self.assertEqual(detect_hardware(), {"gpu": False, "name": "", "vram_mb": 0})

    def test_detect_hardware_rejects_failed_or_incomplete_probe(self) -> None:
        probe = ProcessResult(
            argv=("nvidia-smi",),
            returncode=0,
            stdout="Tesla T4, 15360\n",
            stderr="",
            timed_out=False,
            output_truncated=False,
        )
        cases = (
            replace(probe, timed_out=True),
            replace(probe, output_truncated=True),
            replace(probe, returncode=1),
            replace(probe, stdout=""),
        )
        for failed_probe in cases:
            with self.subTest(probe=failed_probe), patch("openagent.runtimes.colab.run_process", return_value=failed_probe):
                self.assertEqual(detect_hardware(), {"gpu": False, "name": "", "vram_mb": 0})

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
