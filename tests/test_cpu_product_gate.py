from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openagent.cpu_product_gate import (
    REQUIRED_CALLERS, bundle_pin, source_pins, validate_cpu_product_observation,
)


class CPUProductGateTests(unittest.TestCase):
    def test_exact_model_and_source_pins_are_required(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            for name in ("config.json", "openvino_model.xml", "openvino_model.bin", "tokenizer.json"):
                (model / name).write_bytes(b"small local checkpoint")
            receipt = Path(tmp) / "observation.json"
            responses = {
                "chat": "Local CPU inference stays on this machine.",
                "chat_followup": "blue otter",
                "coding": "def add(a, b):\n    return a + b\n",
                "research": "1. Test accuracy. 2. Measure latency. Limitation: smaller models vary.",
                "executive": "Test on a 16 GB laptop.",
                "codebase_ask": "greet('Alice') returns Hello, Alice.",
                "codebase_edit_proposal": 'def greet(name):\n    return "Hello, " + name + "!"\n',
            }
            observed = {
                "kind": "local-cpu-product-smoke-not-family-route-receipt",
                "backend": "openvino-direct", "device": "CPU", "model_type": "qwen2",
                "source_sha256": source_pins(root), "model_bundle_sha256": bundle_pin(model),
                "model_bytes": sum(file.stat().st_size for file in model.iterdir()),
                "host_total_ram_bytes": 16 * 1024**3,
                "sampled_process_tree_peak_rss_bytes": 2 * 1024**3,
                "sampled_app_process_tree_peak_rss_bytes": 3 * 1024**3,
                "complete_text": "hello", "stream_text": "ready",
                "cancelled_worker_retired": True, "post_cancel_gateway_reload": True,
                "task_callers": {
                    name: {"provider": "openvino-direct", "elapsed_seconds": 2.0,
                           "response_text": responses[name],
                           **({"source_unchanged": True} if name == "codebase_ask" else {}),
                           **({"same_conversation": True} if name == "chat_followup" else {}),
                           **({"source_unchanged_before_review": True,
                               "saved_after_review": True} if name == "codebase_edit_proposal" else {})}
                    for name in REQUIRED_CALLERS
                },
            }
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            self.assertTrue(validate_cpu_product_observation(root, receipt, model)["valid"])
            self.assertTrue(validate_cpu_product_observation(
                root, receipt, model, require_host_max_bytes=16 * 1024**3)["valid"])
            self.assertIn("enforced Windows job", " ".join(
                validate_cpu_product_observation(
                    root, receipt, model, require_job_max_bytes=8 * 1024**3)["issues"]))
            observed.update({
                "budget_method": "windows-job-object", "job_limit_enforced": True,
                "job_memory_limit_bytes": 8 * 1024**3,
                "job_peak_commit_bytes": 4 * 1024**3,
            })
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            self.assertTrue(validate_cpu_product_observation(
                root, receipt, model, require_job_max_bytes=8 * 1024**3)["valid"])
            observed["host_total_ram_bytes"] = 32 * 1024**3
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            self.assertIn("required low-RAM host", " ".join(
                validate_cpu_product_observation(
                    root, receipt, model, require_host_max_bytes=16 * 1024**3)["issues"]))
            observed["host_total_ram_bytes"] = 16 * 1024**3
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            (model / "openvino_model.bin").write_bytes(b"different checkpoint")
            self.assertIn("model bundle pin", " ".join(validate_cpu_product_observation(root, receipt, model)["issues"]))
            (model / "openvino_model.bin").write_bytes(b"small local checkpoint")
            observed["task_callers"]["coding"]["provider"] = "local"
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            self.assertIn("coding", " ".join(validate_cpu_product_observation(root, receipt, model)["issues"]))
            observed["task_callers"]["coding"]["provider"] = "openvino-direct"
            observed["task_callers"]["coding"]["response_text"] = "I wrote code for you."
            observed["model_bytes"] = 1
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            issues = " ".join(validate_cpu_product_observation(root, receipt, model)["issues"])
            self.assertIn("coding response is not valid Python", issues)
            self.assertIn("recorded model bytes", issues)
            observed["model_bytes"] = sum(file.stat().st_size for file in model.iterdir())
            observed["task_callers"]["coding"]["response_text"] = "def add(a, b):\n    return 0\n"
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            self.assertIn("did not return a + b", " ".join(
                validate_cpu_product_observation(root, receipt, model)["issues"]))
            observed["task_callers"]["chat_followup"]["response_text"] = "forgotten"
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            self.assertIn("two-turn codeword", " ".join(
                validate_cpu_product_observation(root, receipt, model)["issues"]))
            observed["task_callers"]["codebase_edit_proposal"]["saved_after_review"] = False
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            self.assertIn("proposal/review/save", " ".join(
                validate_cpu_product_observation(root, receipt, model)["issues"]))
            observed["sampled_app_process_tree_peak_rss_bytes"] = 1 * 1024**3
            receipt.write_text(json.dumps(observed), encoding="utf-8")
            self.assertIn("whole-app memory cannot be below", " ".join(
                validate_cpu_product_observation(root, receipt, model)["issues"]))


if __name__ == "__main__":
    unittest.main()
