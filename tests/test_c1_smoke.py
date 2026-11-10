"""Smoke test for C1: Real Model Training - validates end-to-end functionality."""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from urllib.request import Request, urlopen

from openagent.fine_tuning import FineTuningManager
from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


class C1SmokeTest(TestCase):
    """Smoke test for real model training functionality."""

    def test_c1_full_training_pipeline(self) -> None:
        """
        Smoke test: Full C1 training pipeline
        1. Create dataset with examples
        2. Create training job
        3. Start training
        4. Track progress and loss
        5. Create checkpoints
        6. Complete training with adapter registration
        """
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            # Step 1: Create dataset with examples
            dataset = manager.create_dataset(
                "smoke_test_dataset_full_pipeline",
                project="smoke_test_full",
                description="Smoke test dataset for C1"
            )
            self.assertIsNotNone(dataset["id"])

            # Add training examples
            for i in range(5):
                manager.add_example(
                    dataset["id"],
                    f"Sample input {i}",
                    f"Sample output {i}",
                    project="smoke_test_full"
                )

            updated_dataset = manager.get_dataset(dataset["id"])
            self.assertEqual(updated_dataset["num_examples"], 5)

            # Step 2: Create training job with config
            job = manager.create_job(
                dataset["id"],
                "test-model",
                project="smoke_test_full",
                method="lora",
                config={
                    "epochs": 1,
                    "lr": 2e-4,
                    "rank": 8,
                    "alpha": 16,
                    "batch_size": 2,
                    "model_dir": str(Path(tmp) / "model")
                }
            )
            self.assertEqual(job["status"], "queued")
            self.assertEqual(job["progress"], 0.0)

            # Step 3: Start training job (mocked to prevent actual background execution)
            with patch.object(manager, 'run_job_cpu_streaming', return_value={"started": True, "job_id": job["id"]}):
                result = manager.start_model_training(
                    job["id"],
                    project="smoke_test_full",
                    output_dir=str(Path(tmp) / "output"),
                    checkpoint_interval=10
                )
                self.assertTrue(result["started"])
                self.assertEqual(result["status"], "running")

            # Step 4: Simulate progress tracking
            job_after_start = manager.get_job(job["id"])
            self.assertIn(job_after_start["status"], ["queued", "running"])

            # Simulate progress updates
            manager.update_job_progress(job["id"], 0.25, loss=0.8, status="running")
            manager.update_job_progress(job["id"], 0.5, loss=0.5, status="running")
            manager.update_job_progress(job["id"], 0.75, loss=0.3, status="running")

            progress_job = manager.get_job(job["id"])
            self.assertGreaterEqual(progress_job["progress"], 0.75)

            # Step 5: Create checkpoints
            checkpoint1 = manager.add_checkpoint(
                job["id"],
                project="smoke_test_full",
                step=0,
                epoch=0.5,
                train_loss=0.5,
                val_loss=0.55,
                metrics={"accuracy": 0.85, "perplexity": 8.2},
                adapter_path=str(Path(tmp) / "output" / "adapter_0")
            )
            self.assertIsNotNone(checkpoint1["id"])
            self.assertEqual(checkpoint1["metrics"]["accuracy"], 0.85)

            checkpoint2 = manager.add_checkpoint(
                job["id"],
                project="smoke_test_full",
                step=1,
                epoch=1.0,
                train_loss=0.3,
                val_loss=0.35,
                metrics={"accuracy": 0.92, "perplexity": 4.5},
                adapter_path=str(Path(tmp) / "output" / "adapter_1")
            )

            checkpoints = manager.list_checkpoints(project="smoke_test_full", job_id=job["id"])
            self.assertEqual(len(checkpoints), 2)
            self.assertLess(checkpoints[0]["train_loss"], checkpoints[0]["val_loss"])

            # Step 6: Complete training and register adapter
            adapter = manager.register_adapter(
                f"smoke-adapter-{job['id'][:8]}",
                "test-model",
                project="smoke_test_full",
                description=f"Adapter from smoke test job {job['id']}",
                rank=8,
                alpha=16,
                stored_path=str(Path(tmp) / "output" / "adapter_final")
            )
            self.assertIsNotNone(adapter["id"])

            # Complete the job
            completed_job = manager.complete_job(job["id"], adapter_id=adapter["id"])
            self.assertEqual(completed_job["status"], "completed")
            self.assertEqual(completed_job["output_adapter_id"], adapter["id"])
            self.assertEqual(completed_job["progress"], 1.0)

            # Verify final state
            final_job = manager.get_job(job["id"])
            self.assertEqual(final_job["status"], "completed")
            self.assertIsNotNone(final_job["completed_at"])

            # Activate the adapter
            activated_adapter = manager.activate_adapter(adapter["id"], project="smoke_test_full")
            self.assertTrue(activated_adapter["active"])

            final_adapter = manager.get_adapter(adapter["id"])
            self.assertTrue(final_adapter["active"])

    def test_c1_http_training_endpoint(self) -> None:
        """Smoke test: HTTP training endpoint works end-to-end."""
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

                # Create dataset
                status, ds_resp = post("/api/finetune/datasets", {
                    "project": "smoke_test_http",
                    "name": "HTTP Smoke Test Dataset",
                    "description": "For C1 smoke testing"
                })
                self.assertEqual(status, HTTPStatus.CREATED)
                dataset_id = ds_resp["dataset"]["id"]

                # Add example
                status, ex_resp = post("/api/finetune/examples", {
                    "project": "smoke_test_http",
                    "dataset_id": dataset_id,
                    "user_input": "Test input",
                    "assistant_output": "Test output"
                })
                self.assertEqual(status, HTTPStatus.CREATED)

                # Create job with config
                status, job_resp = post("/api/finetune/jobs", {
                    "project": "smoke_test_http",
                    "dataset_id": dataset_id,
                    "base_model": "test-model",
                    "method": "lora",
                    "config": {"model_dir": str(Path(tmp) / "model")}
                })
                self.assertEqual(status, HTTPStatus.CREATED)
                job_id = job_resp["job"]["id"]

                # Start training
                status, train_resp = post(f"/api/finetune/jobs/{job_id}/train", {
                    "project": "smoke_test_http",
                    "output_dir": str(Path(tmp) / "output"),
                    "checkpoint_interval": 10
                })
                self.assertEqual(status, HTTPStatus.ACCEPTED)
                self.assertTrue(train_resp["training"]["started"])

                # Get training status
                status, status_resp = get(f"/api/finetune/jobs/{job_id}/stream-status?project=smoke_test_http")
                self.assertEqual(status, HTTPStatus.OK)
                self.assertEqual(status_resp["job_id"], job_id)
                self.assertIn(status_resp["status"], ["queued", "running"])
                # The background worker may record its first real milestone before
                # this status request completes; a running job need not remain at 0.
                self.assertGreaterEqual(status_resp["progress"], 0.0)
                self.assertLessEqual(status_resp["progress"], 1.0)

                # Get adapters list
                status, adapters_resp = get(f"/api/finetune/adapters?project=smoke_test_http")
                self.assertEqual(status, HTTPStatus.OK)
                self.assertIsInstance(adapters_resp["adapters"], list)

            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                time.sleep(0.1)  # Allow final cleanup

    def test_c1_error_handling_and_recovery(self) -> None:
        """Smoke test: Error handling and recovery mechanisms."""
        with TemporaryDirectory() as tmp:
            manager = FineTuningManager(OpenAgentStore(tmp))

            # Create dataset and job
            dataset = manager.create_dataset("error_test", project="smoke_test_error")
            manager.add_example(dataset["id"], "Input", "Output", project="smoke_test_error")
            job = manager.create_job(dataset["id"], "test-model", project="smoke_test_error")

            # Test error handling
            error_message = "Simulated training error: GPU memory exhausted"
            completed_job = manager.complete_job(job["id"], error=error_message)

            self.assertEqual(completed_job["status"], "failed")
            self.assertEqual(completed_job["error_message"], error_message)
            self.assertEqual(completed_job["progress"], 0.0)

            # Verify we can still query the failed job
            retrieved_job = manager.get_job(job["id"])
            self.assertEqual(retrieved_job["status"], "failed")

            # Create another job and test cancellation (use different base_model to avoid ID collision)
            job2 = manager.create_job(dataset["id"], "test-model-2", project="smoke_test_error")
            cancelled = manager.cancel_queued_job(job2["id"], project="smoke_test_error")
            self.assertIn(cancelled["status"], ["cancelled", "cancelling"])

            # Verify that cancelling an already-cancelled job raises an error
            from openagent.fine_tuning import FineTuningError
            with self.assertRaises(FineTuningError):
                manager.cancel_queued_job(job2["id"], project="smoke_test_error")
