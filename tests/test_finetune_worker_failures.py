"""Fault-injection tests for truthful worker persistence failures, not training proof."""
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import threading
import unittest
from unittest.mock import patch

from openagent.fine_tuning import FineTuningManager
from openagent.storage import OpenAgentStore


class FineTunePersistenceFailureTests(unittest.TestCase):
    def run_failure(self, failing_method: str, run):
        with TemporaryDirectory() as temporary:
            manager = FineTuningManager(OpenAgentStore(temporary))
            dataset = manager.create_dataset("Failure boundary", project="default")
            manager.add_example(dataset["id"], "A", "B", project="default")
            job = manager.create_job(dataset["id"], "test-only", project="default")
            with patch("openagent.fine_tuning.run_cpu_finetune", side_effect=run), patch.object(
                manager, failing_method, side_effect=RuntimeError("controlled persistence failure")
            ):
                manager.run_job_cpu_streaming(job["id"], str(Path(temporary) / "test-model"))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    result = manager.get_job(job["id"])
                    if result["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.01)
                self.assertEqual(result["status"], "failed")
                self.assertIn("controlled persistence failure", result["error_message"])
                self.assertFalse(result["output_adapter_id"])
                # Terminal SQL state can be visible before the worker's final
                # read closes its connection. Join only this test's worker.
                for thread in threading.enumerate():
                    if thread.name == f"cpu-finetune-{job['id'][:8]}":
                        thread.join(timeout=5)
                        self.assertFalse(thread.is_alive(), "Worker did not release the temporary database")


    def test_checkpoint_manifest_failure_is_not_silently_completed(self):
        def run(**kwargs):
            kwargs["checkpoint_callback"](1, 1, "fault-injection-only.json", 0.2)
        self.run_failure("add_checkpoint", run)

    def test_adapter_registration_failure_is_not_silently_completed(self):
        self.run_failure("register_adapter", lambda **_: {"adapter_path": "fault-injection-only.json"})
