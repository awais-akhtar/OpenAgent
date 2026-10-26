"""Tests for the CPU streaming fine-tune engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import MagicMock, patch


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class ArchitectureDetectorTests(TestCase):
    def test_detects_known_model_type(self) -> None:
        from openagent.finetune_cpu_engine import detect_architecture, _ARCH_TARGET_MODULES

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
            }
            (Path(tmp) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            # Create a dummy shard file so shard_files is not empty
            (Path(tmp) / "model.safetensors").write_bytes(b"\x00" * 8)
            arch = detect_architecture(tmp)

        self.assertEqual(arch.model_type, "llama")
        self.assertEqual(arch.target_modules, _ARCH_TARGET_MODULES["llama"])
        self.assertEqual(arch.hidden_size, 4096)
        self.assertEqual(arch.num_layers, 32)
        self.assertEqual(arch.vocab_size, 32000)

    def test_falls_back_to_defaults_for_unknown_type(self) -> None:
        from openagent.finetune_cpu_engine import detect_architecture, _DEFAULT_TARGET_MODULES

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(
                json.dumps({"model_type": "supersecretllm"}), encoding="utf-8"
            )
            arch = detect_architecture(tmp)

        self.assertEqual(arch.target_modules, _DEFAULT_TARGET_MODULES)

    def test_handles_missing_config(self) -> None:
        from openagent.finetune_cpu_engine import detect_architecture

        with tempfile.TemporaryDirectory() as tmp:
            arch = detect_architecture(tmp)

        self.assertEqual(arch.model_type, "unknown")
        self.assertIsInstance(arch.target_modules, list)
        self.assertTrue(len(arch.target_modules) > 0)

    def test_reads_shards_from_index_file(self) -> None:
        from openagent.finetune_cpu_engine import detect_architecture

        with tempfile.TemporaryDirectory() as tmp:
            shard_a = "model-00001-of-00002.safetensors"
            shard_b = "model-00002-of-00002.safetensors"
            index = {
                "metadata": {},
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": shard_a,
                    "model.layers.0.self_attn.v_proj.weight": shard_b,
                },
            }
            (Path(tmp) / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            (Path(tmp) / shard_a).write_bytes(b"\x00")
            (Path(tmp) / shard_b).write_bytes(b"\x00")
            arch = detect_architecture(tmp)

        self.assertEqual(len(arch.shard_files), 2)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", arch.index_map)

    def test_qwen2_normalisation(self) -> None:
        from openagent.finetune_cpu_engine import detect_architecture, _ARCH_TARGET_MODULES

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(
                json.dumps({"model_type": "qwen2_moe"}), encoding="utf-8"
            )
            arch = detect_architecture(tmp)

        self.assertEqual(arch.model_type, "qwen2")
        self.assertEqual(arch.target_modules, _ARCH_TARGET_MODULES["qwen2"])


class StreamingLayerLoaderTests(TestCase):
    def test_is_target_layer(self) -> None:
        from openagent.finetune_cpu_engine import detect_architecture, StreamingLayerLoader

        with tempfile.TemporaryDirectory() as tmp:
            arch = detect_architecture(tmp)

        arch.target_modules = ["q_proj", "v_proj"]
        loader = StreamingLayerLoader(arch, arch.target_modules)

        self.assertTrue(loader.is_target_layer("model.layers.0.self_attn.q_proj.weight"))
        self.assertTrue(loader.is_target_layer("layers.12.v_proj.weight"))
        self.assertFalse(loader.is_target_layer("model.embed_tokens.weight"))
        self.assertFalse(loader.is_target_layer("lm_head.weight"))

    def test_iter_target_layers_skips_non_target(self) -> None:
        """If no target layers exist in shards, iter_target_layers returns nothing."""
        from openagent.finetune_cpu_engine import detect_architecture, StreamingLayerLoader

        with tempfile.TemporaryDirectory() as tmp:
            # Empty shard list -> no layers yielded
            arch = detect_architecture(tmp)
            loader = StreamingLayerLoader(arch, ["q_proj"])
            result = list(loader.iter_target_layers())

        self.assertEqual(result, [])


class LoraModuleTests(TestCase):
    @skipUnless(_has_torch(), "PyTorch not available")
    def test_initialize_shapes(self) -> None:
        import torch
        from openagent.finetune_cpu_engine import LoraModule

        lm = LoraModule(name="test", out_features=128, in_features=64, rank=8, alpha=16.0)
        lm.initialize(torch)

        self.assertEqual(lm.A.shape, (8, 64))
        self.assertEqual(lm.B.shape, (128, 8))
        # B starts at zero
        self.assertTrue(torch.all(lm.B == 0))

    @skipUnless(_has_torch(), "PyTorch not available")
    def test_forward_output_shape(self) -> None:
        import torch
        from openagent.finetune_cpu_engine import LoraModule

        lm = LoraModule(name="test", out_features=128, in_features=64, rank=8, alpha=16.0)
        lm.initialize(torch)
        x = torch.randn(10, 64)
        out = lm.forward(x)
        self.assertEqual(out.shape, (10, 128))

    @skipUnless(_has_torch(), "PyTorch not available")
    def test_scaling(self) -> None:
        from openagent.finetune_cpu_engine import LoraModule

        lm = LoraModule(name="test", out_features=8, in_features=4, rank=4, alpha=8.0)
        self.assertAlmostEqual(lm.scaling, 2.0)

    @skipUnless(_has_torch(), "PyTorch not available")
    def test_adam_step_updates_weights(self) -> None:
        import torch
        from openagent.finetune_cpu_engine import LoraModule

        lm = LoraModule(name="test", out_features=16, in_features=8, rank=4, alpha=8.0)
        lm.initialize(torch)
        A_before = lm.A.clone()
        # Inject a non-zero gradient
        lm.grad_A.fill_(0.1)
        lm.grad_B.fill_(0.1)
        lm.apply_adam(lr=0.01, torch=torch)
        # Weights must have changed
        self.assertFalse(torch.allclose(lm.A, A_before))
        # Gradients reset to zero after update
        self.assertTrue(torch.all(lm.grad_A == 0))
        self.assertTrue(torch.all(lm.grad_B == 0))

    @skipUnless(_has_torch(), "PyTorch not available")
    def test_state_dict_contains_required_keys(self) -> None:
        import torch
        from openagent.finetune_cpu_engine import LoraModule

        lm = LoraModule(name="test", out_features=8, in_features=4, rank=2, alpha=4.0)
        lm.initialize(torch)
        sd = lm.state_dict()
        for key in ("A", "B", "rank", "alpha", "scaling", "step"):
            self.assertIn(key, sd)


class CpuStreamingFineTunerTests(TestCase):
    def test_cancellation_is_checked_before_loading_torch_or_model(self) -> None:
        from openagent.finetune_cpu_engine import CpuStreamingFineTuner, TrainingCancelled

        with tempfile.TemporaryDirectory() as tmp:
            trainer = CpuStreamingFineTuner(
                model_dir=str(Path(tmp) / "missing-model"),
                dataset_examples=[],
                output_dir=str(Path(tmp) / "out"),
                cancel_check=lambda: True,
            )
            with self.assertRaises(TrainingCancelled):
                trainer.run()

    @skipUnless(_has_torch(), "PyTorch not available")
    def test_raises_if_no_target_layers_found(self) -> None:
        from openagent.finetune_cpu_engine import CpuStreamingFineTuner

        with tempfile.TemporaryDirectory() as tmp:
            # config.json with target_modules but no actual weight files
            (Path(tmp) / "config.json").write_text(
                json.dumps({"model_type": "llama", "vocab_size": 100,
                            "hidden_size": 16, "num_hidden_layers": 2}),
                encoding="utf-8",
            )
            trainer = CpuStreamingFineTuner(
                model_dir=tmp,
                dataset_examples=[{"input": "hello", "output": "world"}],
                output_dir=str(Path(tmp) / "out"),
            )
            with self.assertRaises(RuntimeError):
                trainer.run()

    def test_raises_without_torch(self) -> None:
        from openagent.finetune_cpu_engine import CpuStreamingFineTuner

        with tempfile.TemporaryDirectory() as tmp, \
             patch("openagent.finetune_cpu_engine.CpuStreamingFineTuner.run",
                   side_effect=RuntimeError("PyTorch is required")):
            trainer = CpuStreamingFineTuner(
                model_dir=tmp,
                dataset_examples=[],
                output_dir=str(Path(tmp) / "out"),
            )
            with self.assertRaises(RuntimeError, msg="PyTorch is required"):
                trainer.run()

    @skipUnless(_has_torch(), "PyTorch not available")
    def test_run_with_mock_layers_produces_checkpoint(self) -> None:
        """End-to-end run with a mock streaming loader that yields synthetic layers."""
        import torch
        from openagent.finetune_cpu_engine import (
            CpuStreamingFineTuner, ModelArchitecture, StreamingLayerLoader
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "adapters"

            # Build a minimal architecture
            arch = ModelArchitecture(
                model_type="test",
                target_modules=["q_proj"],
                hidden_size=16,
                num_layers=2,
                vocab_size=100,
                shard_files=[],
                index_map={},
            )

            # Patch iter_target_layers to return one synthetic layer on every call
            fake_weight = torch.randn(32, 16)

            def _mock_iter(self_loader):
                # Must yield a fresh clone each call (shape-scan + training passes)
                for _ in range(1):
                    yield "model.layers.0.q_proj.weight", fake_weight.clone()

            progress_log: list[tuple[float, float, str]] = []
            checkpoint_log: list[tuple[int, int, str, float]] = []

            trainer = CpuStreamingFineTuner(
                model_dir=tmp,
                dataset_examples=[
                    {"input": "hello world", "output": "yes"},
                    {"input": "test", "output": "ok"},
                ],
                output_dir=str(out_dir),
                rank=4,
                alpha=8.0,
                epochs=1,
                accumulation_steps=2,
                progress_callback=lambda p, l, m: progress_log.append((p, l, m)),
                checkpoint_callback=lambda ep, st, pa, lo: checkpoint_log.append((ep, st, pa, lo)),
            )

            with patch.object(StreamingLayerLoader, "iter_target_layers", _mock_iter):
                result = trainer.run()

            self.assertIn("adapter_path", result)
            self.assertGreater(result["lora_modules"], 0)
            self.assertTrue(Path(result["adapter_path"]).exists())
            self.assertTrue(len(checkpoint_log) > 0)
            self.assertTrue(len(progress_log) > 0)

            # Adapter JSON must be valid and contain required keys
            payload = json.loads(Path(result["adapter_path"]).read_text(encoding="utf-8"))
            self.assertIn("adapters", payload)
            self.assertIn("rank", payload)


class FinetuningManagerCpuStreamingTests(TestCase):
    def _make_manager(self, tmp: str):
        from openagent.fine_tuning import FineTuningManager
        from openagent.storage import OpenAgentStore

        store = OpenAgentStore(tmp)
        return FineTuningManager(store)

    def test_run_job_cpu_streaming_raises_for_unknown_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            with self.assertRaises(KeyError):
                mgr.run_job_cpu_streaming("no-such-job", "/tmp/model", project="default")

    def test_run_job_cpu_streaming_raises_for_non_runnable_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            ds = mgr.create_dataset("test-dataset", project="default")
            job = mgr.create_job(ds["id"], "base-model", project="default")
            # Cancel the job so it's not runnable
            mgr.cancel_queued_job(job["id"], project="default")
            with self.assertRaises(RuntimeError):
                mgr.run_job_cpu_streaming(job["id"], "/tmp/model", project="default")

    def test_run_job_cpu_streaming_returns_started_for_queued_job(self) -> None:
        import threading as _threading

        captured_thread: list[_threading.Thread] = []
        real_thread = _threading.Thread

        def _capturing_thread(*args, **kwargs):
            t = real_thread(*args, **kwargs)
            captured_thread.append(t)
            return t

        def _mock_run(*args, **kwargs):
            return {
                "epochs": 1, "steps": 2, "final_loss": 0.5,
                "adapter_path": "/tmp/adapter.json", "lora_modules": 1,
            }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("openagent.fine_tuning.run_cpu_finetune", side_effect=_mock_run), \
             patch("openagent.fine_tuning.threading.Thread", side_effect=_capturing_thread):
            mgr = self._make_manager(tmp)
            ds = mgr.create_dataset("test-dataset", project="default")
            job = mgr.create_job(ds["id"], "base-model", project="default")
            result = mgr.run_job_cpu_streaming(job["id"], "/fake/model", project="default")
            self.assertTrue(result["started"])
            self.assertEqual(result["job_id"], job["id"])
            # Wait for the background thread to complete before the tempdir is removed
            for t in captured_thread:
                t.join(timeout=10)

    def test_running_cancel_is_cooperative_and_cannot_finish_as_success(self) -> None:
        import threading
        import time
        from openagent.finetune_cpu_engine import TrainingCancelled

        started = threading.Event()
        cancellation_observed = threading.Event()
        release_worker = threading.Event()
        captured_threads: list[threading.Thread] = []
        real_thread = threading.Thread

        def capture_thread(*args, **kwargs):
            thread = real_thread(*args, **kwargs)
            captured_threads.append(thread)
            return thread

        def wait_for_cancel(*, cancel_check, **_kwargs):
            started.set()
            deadline = time.monotonic() + 5
            while not cancel_check():
                if time.monotonic() >= deadline:
                    raise AssertionError("cancel request was not delivered to the worker")
                time.sleep(0.005)
            cancellation_observed.set()
            if not release_worker.wait(timeout=5):
                raise AssertionError("test did not release the worker after observing cancellation")
            raise TrainingCancelled("cancel requested by test")

        with tempfile.TemporaryDirectory() as tmp, \
             patch("openagent.fine_tuning.run_cpu_finetune", side_effect=wait_for_cancel), \
             patch("openagent.fine_tuning.threading.Thread", side_effect=capture_thread):
            manager = self._make_manager(tmp)
            dataset = manager.create_dataset("cancel dataset", project="local")
            job = manager.create_job(dataset["id"], "local-model", project="local")
            self.assertTrue(manager.run_job_cpu_streaming(job["id"], "/unused", project="local")["started"])
            self.assertTrue(started.wait(timeout=3))
            requested = manager.cancel_queued_job(job["id"], project="local")
            self.assertTrue(cancellation_observed.wait(timeout=3))
            self.assertEqual(requested["status"], "cancelling")
            release_worker.set()
            for thread in captured_threads:
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
            final = manager.get_job(job["id"], project="local")
            self.assertEqual(final["status"], "cancelled")
            self.assertEqual(final["output_adapter_id"], "")
            self.assertEqual(manager.list_adapters(project="local"), [])
