from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore
from openagent.ultra_model_engine import UltraModelEngine, UltraModelError, validate_trusted_nodes


class _FakeProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class UltraModelEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = OpenAgentStore(self.root / "data")
        self.engine = UltraModelEngine(self.store)

    def tearDown(self) -> None:
        self.engine.close()
        self.temp.cleanup()

    def _manifest(self, *, params: float = 200, bits: float = 2) -> dict:
        model_root = self.root / "models"
        model_root.mkdir(exist_ok=True)
        first = model_root / "target-00001-of-00002.gguf"
        second = model_root / "target-00002-of-00002.gguf"
        first.write_bytes(b"GGUF" + b"0" * 128)
        second.write_bytes(b"GGUF" + b"1" * 128)
        return self.engine.register_manifest(
            name="Target 200B",
            shard_paths=[str(first)],
            parameter_count_b=params,
            quantization_bits=bits,
            project="test",
        )

    def test_auto_discovers_contiguous_gguf_shards(self) -> None:
        manifest = self._manifest()
        self.assertEqual(len(manifest["shard_paths"]), 2)
        self.assertTrue(manifest["shard_paths"][0].endswith("00001-of-00002.gguf"))
        self.assertEqual(self.engine.manifests(project="test")[0]["id"], manifest["id"])

    def test_200b_plan_is_honest_about_out_of_core_speed(self) -> None:
        manifest = self._manifest()
        plan = self.engine.plan(
            manifest_id=manifest["id"],
            hardware={
                "total_ram_gb": 32,
                "available_ram_gb": 24,
                "memory_bandwidth_gbps": 30,
                "disk_bandwidth_gbps": 2,
                "cpu_threads": 16,
            },
            context_tokens=2048,
        )
        self.assertEqual(plan["mode"], "out-of-core-mmap")
        self.assertTrue(plan["launchable"])
        self.assertFalse(plan["interactive_predicted"])
        self.assertLess(plan["performance"]["predicted_tokens_per_second"], 1.0)
        self.assertEqual(plan["performance"]["bottleneck"], "storage-bandwidth")
        self.assertTrue(any("200B" in warning for warning in plan["warnings"]))

    def test_moe_plan_uses_active_parameters_for_per_token_bandwidth(self) -> None:
        manifest = self._manifest(params=200, bits=4)
        with self.store.session() as connection:
            connection.execute(
                "UPDATE ultra_model_manifests SET architecture = 'moe', active_parameter_count_b = 20 WHERE id = ?",
                (manifest["id"],),
            )
        plan = self.engine.plan(
            manifest_id=manifest["id"],
            hardware={
                "total_ram_gb": 64,
                "available_ram_gb": 54,
                "memory_bandwidth_gbps": 60,
                "disk_bandwidth_gbps": 4,
                "cpu_threads": 32,
            },
        )
        self.assertEqual(plan["architecture"], "moe")
        self.assertLess(plan["memory"]["active_weight_gb_per_token"], plan["memory"]["effective_weight_gb"])
        self.assertTrue(any("MoE" in warning for warning in plan["warnings"]))

    def test_build_command_uses_safe_current_llama_server_features(self) -> None:
        manifest = self._manifest(params=7, bits=4)
        binary = self.root / "llama-server.exe"
        binary.write_bytes(b"binary")
        plan = self.engine.plan(
            manifest_id=manifest["id"],
            hardware={
                "total_ram_gb": 32,
                "available_ram_gb": 28,
                "memory_bandwidth_gbps": 40,
                "disk_bandwidth_gbps": 2,
                "cpu_threads": 12,
            },
        )
        command = self.engine.build_command(
            manifest["id"], plan=plan, executable=str(binary), port=8181
        )
        self.assertEqual(command[0], str(binary.resolve()))
        self.assertIn("--mmap", command)
        self.assertIn("--cache-prompt", command)
        self.assertIn("--cache-reuse", command)
        self.assertIn("--spec-type", command)
        self.assertIn("ngram-mod", command)
        self.assertIn("--offline", command)
        with self.assertRaises(UltraModelError):
            self.engine.build_command(
                manifest["id"], plan=plan, executable=str(binary), host="0.0.0.0"
            )

    def test_process_lifecycle_is_audited_and_stoppable(self) -> None:
        manifest = self._manifest(params=7, bits=4)
        binary = self.root / "llama-server.exe"
        binary.write_bytes(b"binary")
        fake = _FakeProcess()
        with patch("openagent.ultra_model_engine.subprocess.Popen", return_value=fake):
            launched = self.engine.launch(
                manifest["id"],
                approved=True,
                executable=str(binary),
                keep_warm=False,
                monitor=False,
            )
        session = launched["session"]
        self.assertEqual(session["status"], "starting")
        self.assertEqual(session["pid"], 4242)
        stopped = self.engine.stop(session["id"])
        self.assertEqual(stopped["status"], "stopped")
        actions = [item["action"] for item in self.store.list_audit_events(project="test", limit=20)]
        self.assertIn("ultra_model.launch", actions)
        self.assertIn("ultra_model.stop", actions)

    def test_output_gate_rejects_unsupported_or_invalid_citations(self) -> None:
        verified = self.engine.verify_output(
            "OpenAgent keeps project data local [S1].",
            evidence=["OpenAgent keeps project data local by default."],
            project="test",
        )
        self.assertEqual(verified["status"], "verified")
        rejected = self.engine.verify_output(
            "OpenAgent guarantees every answer is correct [S9].",
            evidence=["OpenAgent verifies grounded claims and can abstain."],
            project="test",
        )
        self.assertEqual(rejected["status"], "rejected")

    def test_public_rpc_endpoint_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_trusted_nodes([{"endpoint": "8.8.8.8:50052", "ram_gb": 64, "trusted": True}])

    def test_http_contract_exposes_status_and_planning(self) -> None:
        server = build_http_server("127.0.0.1", 0, self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            status = json.loads(urlopen(f"{base}/api/ultra-model/status", timeout=10).read())
            self.assertEqual(status["engine"], "openagent-tiered-inference-v1")
            request = Request(
                f"{base}/api/ultra-model/plan",
                data=json.dumps(
                    {
                        "project": "test",
                        "specification": {
                            "name": "Planning 200B",
                            "architecture": "dense",
                            "parameter_count_b": 200,
                            "quantization_bits": 2,
                        },
                        "hardware": {
                            "total_ram_gb": 32,
                            "available_ram_gb": 24,
                            "memory_bandwidth_gbps": 30,
                            "disk_bandwidth_gbps": 2,
                            "cpu_threads": 16,
                        },
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            planned = json.loads(urlopen(request, timeout=10).read())
            self.assertEqual(planned["plan"]["mode"], "planning-only")
            self.assertFalse(planned["plan"]["launchable"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
