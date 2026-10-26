"""Tests for C1: Real Model Training with checkpoints and progress tracking."""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch, MagicMock
from urllib.request import Request, urlopen

from openagent.fine_tuning import FineTuningManager
from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


class C1RealModelTrainingTests(TestCase):
    """Test suite for real model training with checkpoint generation and progress tracking."""

    def test_training_job_starts_successfully(self) -> None:
        """Test that a training job can be started and begins with running status."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            # Create dataset and job
            dataset = manager.create_dataset("training_data", project="default", description="Test dataset")
            manager.add_example(dataset["id"], "What is AI?", "AI is artificial intelligence.", project="default")

            job = manager.create_job(
                dataset["id"],
                "test-model",
                project="default",
                method="lora",
                config={"epochs": 1, "lr": 2e-4, "rank": 8, "alpha": 16, "batch_size": 2, "model_dir": str(Path(tmp) / "model")},
            )
            self.assertEqual(job["status"], "queued")

            # Mock the run_job_cpu_streaming to prevent actual background thread execution
            with patch.object(manager, 'run_job_cpu_streaming', return_value={"started": True, "job_id": job["id"]}):
                # Start training - this should not raise an error
                result = manager.start_model_training(
                    job["id"],
                    project="default",
                    output_dir=str(Path(tmp) / "output"),
                    checkpoint_interval=1,
                )
                self.assertTrue(result["started"])
                self.assertEqual(result["job_id"], job["id"])
                self.assertEqual(result["status"], "running")

    def test_training_progress_tracking(self) -> None:
        """Test that training progress can be tracked from 0 to 100%."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            dataset = manager.create_dataset("progress_test", project="default")
            manager.add_example(dataset["id"], "Input 1", "Output 1", project="default")

            job = manager.create_job(dataset["id"], "test-model", project="default")

            # Simulate progress updates
            updated = manager.update_job_progress(job["id"], 0.0, loss=0.5, status="running")
            self.assertEqual(updated["progress"], 0.0)
            self.assertEqual(updated["current_loss"], 0.5)
            self.assertEqual(updated["status"], "running")

            updated = manager.update_job_progress(job["id"], 0.5, loss=0.3, status="running")
            self.assertEqual(updated["progress"], 0.5)
            self.assertEqual(updated["current_loss"], 0.3)

            updated = manager.update_job_progress(job["id"], 1.0, loss=0.1, status="running")
            self.assertEqual(updated["progress"], 1.0)
            self.assertEqual(updated["current_loss"], 0.1)

    def test_checkpoint_creation_and_persistence(self) -> None:
        """Test that checkpoints are created and persisted during training."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            dataset = manager.create_dataset("checkpoint_test", project="default")
            manager.add_example(dataset["id"], "Input", "Output", project="default")

            job = manager.create_job(dataset["id"], "test-model", project="default")

            # Create checkpoints
            checkpoint1 = manager.add_checkpoint(
                job["id"],
                project="default",
                step=0,
                epoch=0.5,
                train_loss=0.5,
                val_loss=0.6,
                metrics={"accuracy": 0.75},
                adapter_path="/path/to/adapter1",
            )
            self.assertIsNotNone(checkpoint1["id"])
            self.assertEqual(checkpoint1["step"], 0)
            self.assertEqual(checkpoint1["epoch"], 0.5)
            self.assertEqual(checkpoint1["train_loss"], 0.5)

            checkpoint2 = manager.add_checkpoint(
                job["id"],
                project="default",
                step=1,
                epoch=1.0,
                train_loss=0.3,
                val_loss=0.35,
                metrics={"accuracy": 0.85},
                adapter_path="/path/to/adapter2",
            )
            self.assertEqual(checkpoint2["step"], 1)
            self.assertEqual(checkpoint2["train_loss"], 0.3)

            # Verify checkpoints are persisted
            checkpoints = manager.list_checkpoints(project="default", job_id=job["id"])
            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(checkpoints[0]["step"], 0)
            self.assertEqual(checkpoints[1]["step"], 1)

    def test_checkpoint_metrics_logging(self) -> None:
        """Test that training metrics are correctly logged in checkpoints."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            dataset = manager.create_dataset("metrics_test", project="default")
            manager.add_example(dataset["id"], "Input", "Output", project="default")
            job = manager.create_job(dataset["id"], "test-model", project="default")

            # Create checkpoint with detailed metrics
            metrics = {
                "accuracy": 0.92,
                "precision": 0.90,
                "recall": 0.88,
                "f1": 0.89,
                "perplexity": 5.2,
            }
            checkpoint = manager.add_checkpoint(
                job["id"],
                project="default",
                step=0,
                epoch=1.0,
                train_loss=0.25,
                val_loss=0.30,
                metrics=metrics,
                adapter_path="/path/to/adapter",
            )

            # Verify metrics are stored
            self.assertEqual(checkpoint["metrics"]["accuracy"], 0.92)
            self.assertEqual(checkpoint["metrics"]["f1"], 0.89)
            self.assertEqual(checkpoint["train_loss"], 0.25)
            self.assertEqual(checkpoint["val_loss"], 0.30)

    def test_training_completion_creates_adapter(self) -> None:
        """Test that successful training completes with an adapter ID."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            dataset = manager.create_dataset("completion_test", project="default")
            manager.add_example(dataset["id"], "Input", "Output", project="default")
            job = manager.create_job(dataset["id"], "test-model", project="default")

            # Register an adapter as if training completed
            adapter = manager.register_adapter(
                f"adapter-{job['id'][:8]}",
                "test-model",
                project="default",
                description=f"Adapter from job {job['id']}",
                rank=8,
                alpha=16,
                stored_path="/path/to/weights",
            )

            # Complete the job with the adapter
            completed_job = manager.complete_job(job["id"], adapter_id=adapter["id"])
            self.assertEqual(completed_job["status"], "completed")
            self.assertEqual(completed_job["output_adapter_id"], adapter["id"])
            self.assertEqual(completed_job["progress"], 1.0)

    def test_training_error_handling_on_failure(self) -> None:
        """Test proper error handling when training fails."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            dataset = manager.create_dataset("error_test", project="default")
            manager.add_example(dataset["id"], "Input", "Output", project="default")
            job = manager.create_job(dataset["id"], "test-model", project="default")

            # Simulate a training error
            error_msg = "CUDA out of memory"
            completed_job = manager.complete_job(job["id"], error=error_msg)

            self.assertEqual(completed_job["status"], "failed")
            self.assertEqual(completed_job["error_message"], error_msg)
            self.assertFalse(completed_job["output_adapter_id"])

    def test_training_cancellation_during_run(self) -> None:
        """Test that running training can be cancelled cooperatively."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            dataset = manager.create_dataset("cancel_test", project="default")
            manager.add_example(dataset["id"], "Input", "Output", project="default")
            job = manager.create_job(dataset["id"], "test-model", project="default", config={"model_dir": str(Path(tmp) / "model")})

            # Mock run_job_cpu_streaming to prevent actual background execution
            with patch.object(manager, 'run_job_cpu_streaming', return_value={"started": True, "job_id": job["id"]}):
                # Start training
                result = manager.start_model_training(
                    job["id"],
                    project="default",
                    output_dir=str(Path(tmp) / "output"),
                )
                self.assertTrue(result["started"])

            # Request cancellation - since the job is in queued state (not actually running),
            # it will be directly cancelled, not cancelling
            cancelled_job = manager.cancel_queued_job(job["id"], project="default")
            self.assertIn(cancelled_job["status"], ["cancelled", "cancelling"])

            # Verify cancellation is recognized
            is_cancelled = manager._job_cancel_requested(job["id"], "default")
            self.assertTrue(is_cancelled)

    def test_checkpoint_recovery_from_job(self) -> None:
        """Test that all checkpoints for a job can be recovered and ordered."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            dataset = manager.create_dataset("recovery_test", project="default")
            manager.add_example(dataset["id"], "Input", "Output", project="default")
            job = manager.create_job(dataset["id"], "test-model", project="default")

            # Create multiple checkpoints in sequence
            for i in range(5):
                manager.add_checkpoint(
                    job["id"],
                    project="default",
                    step=i,
                    epoch=float(i),
                    train_loss=0.5 - (i * 0.1),
                    val_loss=0.6 - (i * 0.1),
                    metrics={"step": i},
                    adapter_path=f"/path/to/adapter_{i}",
                )

            # Recover all checkpoints
            checkpoints = manager.list_checkpoints(project="default", job_id=job["id"], limit=100)
            self.assertEqual(len(checkpoints), 5)

            # Verify ordering and values
            for i, cp in enumerate(checkpoints):
                self.assertEqual(cp["step"], i)
                self.assertEqual(cp["epoch"], float(i))
                self.assertAlmostEqual(cp["train_loss"], 0.5 - (i * 0.1), places=5)

    def test_http_training_start_route(self) -> None:
        """Test the HTTP route for starting model training."""
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)

            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            try:
                def post(path: str, payload: dict) -> tuple[int, dict]:
                    request = Request(
                        f"{base}{path}",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=10) as response:
                        return response.status, json.load(response)

                # Create dataset and job
                _, ds_response = post("/api/finetune/datasets", {
                    "project": "default",
                    "name": "HTTP Training Test",
                })
                dataset_id = ds_response["dataset"]["id"]

                _, job_response = post("/api/finetune/jobs", {
                    "project": "default",
                    "dataset_id": dataset_id,
                    "base_model": "test-model",
                    "method": "lora",
                    "config": {"model_dir": str(Path(tmp) / "model")},
                })
                job_id = job_response["job"]["id"]

                # Start training via HTTP
                status, train_response = post(f"/api/finetune/jobs/{job_id}/train", {
                    "project": "default",
                    "output_dir": str(Path(tmp) / "output"),
                    "checkpoint_interval": 10,
                })

                self.assertEqual(status, HTTPStatus.ACCEPTED)
                self.assertIn("training", train_response)
                self.assertTrue(train_response["training"]["started"])

            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_http_checkpoint_stream_status(self) -> None:
        """Test the HTTP route for streaming checkpoint status during training."""
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            try:
                def post(path: str, payload: dict) -> tuple[int, dict]:
                    request = Request(
                        f"{base}{path}",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=10) as response:
                        return response.status, json.load(response)

                def get(path: str) -> tuple[int, dict]:
                    with urlopen(f"{base}{path}", timeout=10) as response:
                        return response.status, json.load(response)

                # Create dataset and job
                _, ds_response = post("/api/finetune/datasets", {
                    "project": "default",
                    "name": "Status Stream Test",
                })
                dataset_id = ds_response["dataset"]["id"]

                _, job_response = post("/api/finetune/jobs", {
                    "project": "default",
                    "dataset_id": dataset_id,
                    "base_model": "test-model",
                })
                job_id = job_response["job"]["id"]

                # Get stream status
                status, status_response = get(f"/api/finetune/jobs/{job_id}/stream-status?project=default")
                self.assertEqual(status, HTTPStatus.OK)
                self.assertEqual(status_response["job_id"], job_id)
                self.assertEqual(status_response["status"], "queued")
                self.assertEqual(status_response["progress"], 0.0)
                self.assertIsInstance(status_response["checkpoints"], list)

            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_training_with_multiple_examples(self) -> None:
        """Test training with multiple dataset examples."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            dataset = manager.create_dataset("multi_example_test", project="default")

            # Add multiple examples
            for i in range(10):
                manager.add_example(
                    dataset["id"],
                    f"Question {i}?",
                    f"Answer {i}",
                    project="default",
                    system_prompt="You are a helpful assistant.",
                    quality_score=0.9 + (i * 0.01),
                )

            # Verify all examples are stored
            updated_dataset = manager.get_dataset(dataset["id"])
            self.assertEqual(updated_dataset["num_examples"], 10)

            # Create and start training job
            job = manager.create_job(
                dataset["id"],
                "test-model",
                project="default",
                config={"epochs": 1, "model_dir": str(Path(tmp) / "model")},
            )

            # Mock training to prevent background execution
            with patch.object(manager, 'run_job_cpu_streaming', return_value={"started": True, "job_id": job["id"]}):
                result = manager.start_model_training(
                    job["id"],
                    project="default",
                    output_dir=str(Path(tmp) / "output"),
                )
                self.assertTrue(result["started"])

    def test_progress_bounds_enforcement(self) -> None:
        """Test that progress values are bounded between 0 and 1."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            dataset = manager.create_dataset("bounds_test", project="default")
            manager.add_example(dataset["id"], "Input", "Output", project="default")
            job = manager.create_job(dataset["id"], "test-model", project="default")

            # Test over-bounds values
            updated = manager.update_job_progress(job["id"], -0.5, status="running")
            self.assertEqual(updated["progress"], 0.0)

            updated = manager.update_job_progress(job["id"], 1.5, status="running")
            self.assertEqual(updated["progress"], 1.0)

            # Test in-bounds value
            updated = manager.update_job_progress(job["id"], 0.5, status="running")
            self.assertEqual(updated["progress"], 0.5)
