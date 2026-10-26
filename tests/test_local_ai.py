from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from io import StringIO
import unittest
from unittest.mock import patch

from openagent.local_ai import LocalAIInfrastructure, ModelDownloadError
from openagent.process_policy import ProcessResult
from openagent.storage import OpenAgentStore


class CompletedPullProcess:
    def __init__(self, output: str = "pulling manifest\n50%\n100%\n") -> None:
        self.stdout = StringIO(output)
        self.returncode = 0

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 1

    def kill(self) -> None:
        self.returncode = 1


class LocalAIInfrastructureTests(unittest.TestCase):
    def test_model_lifecycle_hardware_benchmark_and_cluster_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.bin"
            source.write_bytes(b"local model bytes")
            infra = LocalAIInfrastructure(OpenAgentStore(Path(tmp) / "data"))

            registered = infra.register_model(
                name="Tiny Local Model",
                provider="local",
                source_url=str(source),
            )
            self.assertEqual(registered["status"], "registered")

            downloaded = infra.download_model(registered["id"])
            self.assertEqual(downloaded["status"], "available")
            self.assertGreater(downloaded["size_bytes"], 0)
            self.assertTrue(Path(downloaded["local_path"]).exists())

            quantization = infra.add_quantization(
                downloaded["id"],
                method="q4",
                bits=4,
                format="gguf",
            )
            self.assertEqual(quantization["bits"], 4)
            self.assertEqual(len(infra.quantizations(downloaded["id"])), 1)

            hardware = infra.detect_hardware()
            self.assertGreaterEqual(hardware["cpu_count"], 1)
            self.assertGreater(hardware["ram_gb"], 0)
            self.assertIn("cpu", {adapter["id"] for adapter in infra.adapters()})

            benchmark = infra.benchmark(downloaded["id"], prompt="Benchmark locally")
            self.assertGreater(benchmark["tokens_per_second"], 0)
            history = infra.benchmark_history(model_id=downloaded["id"], limit=10)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["id"], benchmark["id"])
            self.assertEqual(history[0]["model"], "Tiny Local Model")
            self.assertEqual(history[0]["provider"], "local")
            self.assertNotIn("prompt", history[0])
            self.assertNotIn("content", history[0])
            self.assertEqual(infra.benchmark_history(model_id="missing-model"), [])
            self.assertEqual(len(infra.benchmark_history(limit=0)), 1)

            plan = infra.plan_cluster(name="Local cluster", model_id=downloaded["id"], nodes=2)
            self.assertEqual(plan["plan"]["nodes"], 2)
            marketplace = infra.marketplace()
            self.assertGreaterEqual(len(marketplace), 1)
            marketplace_ids = {item["id"] for item in marketplace}
            self.assertIn("phi4-mini-3.8b-q4", marketplace_ids)
            self.assertIn("qwen2.5vl-3b-q4", marketplace_ids)
            self.assertIn("qwen2.5-coder-3b-q4", marketplace_ids)
            self.assertIn("gemma3-2b-q4", marketplace_ids)
            self.assertIn("deepseek-coder-1.3b-q4", marketplace_ids)
            smooth = infra.smooth_runtime_plan(task="chat", target_ram_gb=8)
            self.assertEqual(smooth["profile"]["tier"], "ultra-light-8gb")
            self.assertEqual(smooth["profile"]["recommended_speed_model"], "gemma3:2b")
            auto_quant = infra.auto_quantize(downloaded["id"], task="chat", target_ram_gb=8)
            self.assertEqual(auto_quant["quantization"]["metadata"]["status"], "planned")
            compressed = infra.auto_quantize(downloaded["id"], task="chat", target_ram_gb=3, quantization_mode="below-2bit", parameter_count_b=0.35)
            self.assertLess(compressed["profile"]["effective_bits"], 2)

            deleted = infra.delete_model(downloaded["id"])
            self.assertTrue(deleted["deleted"])
            self.assertIsNone(infra.get_model(downloaded["id"]))

    def test_ollama_pull_registers_pullable_model_name(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            completed = ProcessResult(("ollama", "pull", "phi4-mini"), 0, "pulled", "", False, False)
            with patch("openagent.local_ai.shutil.which", return_value="ollama"), patch(
                "openagent.local_ai.run_process",
                return_value=completed,
            ) as run:
                result = infra.pull_ollama_model(
                    "phi4-mini:3.8b-instruct-q4_K_M",
                    allow_network=True,
                    project="test",
                )

        self.assertEqual(result["ollama_model"], "phi4-mini")
        self.assertEqual(result["model"]["provider"], "ollama")
        self.assertEqual(result["model"]["status"], "available")
        run.assert_called_once()
        policy = run.call_args.kwargs["policy"]
        self.assertEqual(policy.max_output_bytes, 512_000)
        self.assertEqual(policy.allowed_executables, ("ollama",))
        self.assertIn("OLLAMA_HOST", policy.env_keys)
        self.assertNotIn("OPENAI_API_KEY", policy.env_keys)
        self.assertEqual(run.call_args.kwargs["cwd"], infra.model_root)

    def test_ollama_pull_tolerates_missing_captured_output(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            completed = ProcessResult(("ollama", "pull", "qwen2.5vl:3b"), 0, "", "", False, False)
            with patch("openagent.local_ai.shutil.which", return_value="ollama"), patch(
                "openagent.local_ai.run_process", return_value=completed,
            ):
                result = infra.pull_ollama_model("qwen2.5vl:3b", allow_network=True, project="test")

        self.assertEqual(result["ollama_model"], "qwen2.5vl:3b")
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["stderr"], "")

    def test_setup_chat_model_uses_ollama_when_available(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            completed = ProcessResult(("ollama", "pull", "phi4-mini"), 0, "pulled", "", False, False)
            with patch("openagent.local_ai.shutil.which", return_value="ollama"), patch(
                "openagent.local_ai.run_process", return_value=completed
            ):
                result = infra.setup_chat_model(
                    model_name="phi4-mini",
                    allow_network=True,
                    project="test",
                )

        self.assertEqual(result["runner"], "ollama")
        self.assertEqual(result["ollama_model"], "phi4-mini")
        self.assertIn("tool_status", result)

    def test_setup_chat_model_job_records_progress_result_and_benchmark(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            completed = SimpleNamespace(returncode=0, stdout="pulled", stderr="")
            with patch("openagent.local_ai.shutil.which", return_value="ollama"), patch(
                "openagent.local_ai.subprocess.Popen", return_value=CompletedPullProcess()
            ), patch("openagent.local_ai.subprocess.run", return_value=completed):
                job = infra.create_setup_chat_job(
                    model_name="phi4-mini",
                    allow_network=True,
                    project="test",
                    run_inline=True,
                )
            jobs = infra.model_jobs(project="test")

        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"], 100)
        self.assertEqual(job["model_name"], "phi4-mini")
        self.assertIn("setup", job["result"])
        self.assertIn("benchmark", job["result"])
        self.assertEqual(job["result"]["setup"]["runner"], "ollama")
        self.assertGreaterEqual(job["result"]["benchmark"]["tokens_per_second"], 0)
        self.assertEqual(len(jobs), 1)

    def test_setup_code_model_job_records_task_kind(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            completed = SimpleNamespace(returncode=0, stdout="pulled", stderr="")
            with patch("openagent.local_ai.shutil.which", return_value="ollama"), patch(
                "openagent.local_ai.subprocess.Popen", return_value=CompletedPullProcess()
            ), patch("openagent.local_ai.subprocess.run", return_value=completed):
                job = infra.create_setup_chat_job(
                    model_name="qwen2.5-coder:3b",
                    allow_network=True,
                    project="test",
                    task="code",
                    run_inline=True,
                )

        self.assertEqual(job["kind"], "setup-code")
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["model_name"], "qwen2.5-coder:3b")
        self.assertEqual(job["result"]["setup"]["ollama_model"], "qwen2.5-coder:3b")

    def test_delete_ollama_model_removes_runtime_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            model = infra.register_model(
                name="phi4-mini",
                provider="ollama",
                source_url="ollama://phi4-mini",
                local_path="ollama://phi4-mini",
            )
            completed = ProcessResult(("ollama", "rm", "phi4-mini"), 0, "deleted", "", False, False)
            with patch("openagent.local_ai.shutil.which", return_value="ollama"), patch(
                "openagent.local_ai.run_process", return_value=completed
            ) as run:
                result = infra.delete_model(model["id"], delete_file=True)
            self.assertTrue(result["deleted_runtime_artifact"])
            self.assertIsNone(infra.get_model(model["id"]))
            self.assertEqual(run.call_args.args[0], ["ollama", "rm", "phi4-mini"])

    def test_non_streaming_ollama_pull_honours_cancellation(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            cancelled = ProcessResult(
                ("ollama", "pull", "phi4-mini"), 130, "", "", False, False, cancelled=True
            )
            with patch("openagent.local_ai.shutil.which", return_value="ollama"), patch(
                "openagent.local_ai.run_process", return_value=cancelled
            ) as run:
                with self.assertRaisesRegex(ModelDownloadError, "cancelled"):
                    infra.pull_ollama_model(
                        "phi4-mini",
                        allow_network=True,
                        cancel_requested=lambda: True,
                    )
            self.assertTrue(run.call_args.kwargs["cancel_event"].is_set())

    def test_delete_imported_model_preserves_external_file(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "outside-openagent.gguf"
            source.write_bytes(b"external model")
            infra = LocalAIInfrastructure(OpenAgentStore(Path(tmp) / "data"))
            model = infra.register_model(name="External GGUF", provider="llamacpp", local_path=str(source))

            result = infra.delete_model(model["id"], delete_file=True)

            self.assertTrue(result["preserved_external_file"])
            self.assertTrue(source.exists())
            self.assertIsNone(infra.get_model(model["id"]))

    def test_ollama_http_pull_works_when_cli_is_not_on_path(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            with patch("openagent.local_ai.shutil.which", return_value=None), patch(
                "openagent.local_ai.local_http_available",
                return_value=True,
            ), patch.object(
                infra.egress,
                "fetch",
                return_value=SimpleNamespace(body=b'{"status":"success"}'),
            ) as opened:
                result = infra.pull_ollama_model("phi4-mini", allow_network=True)

        self.assertEqual(result["ollama_model"], "phi4-mini")
        self.assertIn("/api/pull", result["command"])
        opened.assert_called_once()

    def test_model_tool_status_reports_installed_ollama_models(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            specs = [
                {
                    "id": "ollama",
                    "available": True,
                    "installed_models": ["llama3.2:3b"],
                    "selected_model": "llama3.2:3b",
                    "default_model_ready": False,
                },
                {"id": "llamacpp", "available": False},
            ]
            with patch("openagent.local_ai.shutil.which", return_value=None), patch(
                "openagent.local_ai.local_http_available",
                return_value=True,
            ), patch("openagent.local_ai.ModelGateway.specs", return_value=specs):
                status = infra.model_tool_status()

        self.assertTrue(status["ollama"]["model_ready"])
        self.assertFalse(status["ollama"]["default_model_ready"])
        self.assertEqual(status["ollama"]["installed_model_count"], 1)
        self.assertEqual(status["ollama"]["selected_model"], "llama3.2:3b")

    def test_llamacpp_gguf_registration_records_start_command(self) -> None:
        with TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "tiny-chat.gguf"
            gguf.write_bytes(b"gguf bytes")
            infra = LocalAIInfrastructure(OpenAgentStore(Path(tmp) / "data"))

            result = infra.register_llamacpp_model(
                local_path=str(gguf),
                name="Tiny GGUF",
                project="test",
            )

        self.assertEqual(result["runner"], "llamacpp")
        self.assertEqual(result["model"]["provider"], "llamacpp")
        self.assertEqual(result["model"]["status"], "available")
        self.assertIn("llama-server -m", result["start_command"])

    def test_network_download_requires_approval(self) -> None:
        with TemporaryDirectory() as tmp:
            infra = LocalAIInfrastructure(OpenAgentStore(tmp))
            registered = infra.register_model(
                name="Remote Model",
                provider="local",
                source_url="https://example.com/model.bin",
            )

            with self.assertRaises(ModelDownloadError):
                infra.download_model(registered["id"], allow_network=False)


if __name__ == "__main__":
    unittest.main()
