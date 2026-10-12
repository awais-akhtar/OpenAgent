from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.device_runtime import DirectHardwareClient, DirectHardwareError
from openagent.adaptive_engine.hardware import HardwareProfiler, accelerator_environment
from openagent.adaptive_engine.worker import _device_config, _process_alive
from openagent.adaptive_engine.worker import _decode_expert_payload, _expert_payload_size


class DirectHardwareClientTests(unittest.TestCase):
    def test_accelerator_environment_does_not_inherit_host_python_packages(self) -> None:
        with patch.dict(
            os.environ,
            {"PYTHONPATH": "host-site-packages", "PYTHONHOME": "host-python"},
            clear=False,
        ):
            environment = accelerator_environment(Path("isolated-source"))
        self.assertEqual(environment["PYTHONPATH"], "isolated-source")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertNotIn("PYTHONHOME", environment)

    def test_worker_parent_watchdog_process_probe(self) -> None:
        self.assertTrue(_process_alive(os.getpid()))
        self.assertFalse(_process_alive(-1))

    def test_device_config_tunes_interactive_gpu_and_npu_execution(self) -> None:
        gpu = _device_config({}, device="GPU.0")
        self.assertEqual(gpu["PERFORMANCE_HINT"], "LATENCY")
        self.assertEqual(gpu["EXECUTION_MODE_HINT"], "PERFORMANCE")
        self.assertEqual(gpu["INFERENCE_PRECISION_HINT"], "f16")
        npu = _device_config({}, device="NPU")
        self.assertEqual(npu["NPU_TURBO"], "YES")
        self.assertIn("optimization-level=2", npu["NPU_COMPILATION_MODE_PARAMS"])

    def test_broken_accelerator_launcher_is_not_reported_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / ("python.exe" if os.name == "nt" else "python")
            fake.write_text("not an interpreter", encoding="ascii")
            profiler = HardwareProfiler(temp, accelerator_python=str(fake))
            profiler.project_root = Path(temp)
            self.assertEqual(profiler.accelerator_python_path(), "")

    def test_status_does_not_start_or_open_a_network_service(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            profiler = HardwareProfiler(temp, accelerator_python=str(Path(temp) / "missing-python"))
            client = DirectHardwareClient(profiler)
            status = client.status()
        self.assertFalse(status["running"])
        self.assertEqual(status["transport"], "inherited-jsonl-pipes")
        self.assertNotIn("port", status)

    def test_stream_protocol_yields_native_events_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            client = DirectHardwareClient(HardwareProfiler(temp))

            def request(action, *, on_event, **payload):
                self.assertEqual(action, "generate_stream")
                on_event({"type": "token", "content": "one"})
                on_event({"type": "token", "content": " two"})
                return {"status": "completed", "text": "one two", "streamed": True}

            with patch.object(client, "request", side_effect=request):
                events = list(client.stream_generate("model", prompt="test"))
        self.assertEqual([item["type"] for item in events], ["token", "token", "done"])
        self.assertEqual(events[-1]["text"], "one two")

    def test_expert_payload_decoder_reconstructs_row_quantized_matrices(self) -> None:
        import numpy as np

        hidden, intermediate = 2, 3
        width = hidden * intermediate
        values = np.arange(-width, 2 * width, dtype=np.int8)
        scales = np.linspace(0.1, 0.6, 2 * intermediate + hidden, dtype="<f4")
        payload = values.tobytes() + scales.tobytes()
        self.assertEqual(len(payload), _expert_payload_size(hidden, intermediate, "int8-row"))
        gate, up, down = _decode_expert_payload(payload, hidden, intermediate, "int8-row")
        np.testing.assert_allclose(gate, values[:width].reshape(intermediate, hidden) * scales[:intermediate, None])
        np.testing.assert_allclose(up, values[width:2 * width].reshape(intermediate, hidden) * scales[intermediate:2 * intermediate, None])
        np.testing.assert_allclose(down, values[2 * width:].reshape(hidden, intermediate) * scales[2 * intermediate:, None])


@unittest.skipUnless(
    os.environ.get("OPENAGENT_RUN_HARDWARE_TESTS") == "1"
    and Path(os.environ.get("OPENAGENT_HARDWARE_TEST_MODEL", "")).is_file(),
    "set OPENAGENT_RUN_HARDWARE_TESTS=1 and OPENAGENT_HARDWARE_TEST_MODEL for native validation",
)
class DirectHardwareIntegrationTests(unittest.TestCase):
    def test_exact_model_is_validated_and_kept_resident_on_the_winner(self) -> None:
        model_path = str(Path(os.environ["OPENAGENT_HARDWARE_TEST_MODEL"]).resolve())
        with tempfile.TemporaryDirectory() as temp:
            profiler = HardwareProfiler(temp)
            client = DirectHardwareClient(profiler)
            try:
                live = client.status(start=True)
                self.assertIn("CPU", live["devices"])
                result = client.calibrate(
                    model_path,
                    devices=["CPU", "GPU", "NPU"],
                    iterations=4,
                    cache_dir=str(Path(temp) / "cache"),
                    keep_model_id="integration-model",
                )
                self.assertEqual(result["status"], "verified")
                self.assertTrue(result["winner"]["valid"])
                resident = client.status(start=True)["models"]
                self.assertTrue(any(item["model_id"] == "integration-model" for item in resident))
                inference = client.infer("integration-model", inputs="auto")
                self.assertEqual(inference["status"], "completed")
            finally:
                client.close()


@unittest.skipUnless(
    os.environ.get("OPENAGENT_RUN_HARDWARE_TESTS") == "1",
    "set OPENAGENT_RUN_HARDWARE_TESTS=1 for native expert validation",
)
class DirectExpertHardwareIntegrationTests(unittest.TestCase):
    def test_tiny_expert_matches_numpy_on_available_accelerators(self) -> None:
        import hashlib
        import numpy as np

        hidden, intermediate = 4, 7
        rng = np.random.default_rng(73)
        weights = rng.normal(0, 0.2, 3 * hidden * intermediate).astype("<f4")
        vector = rng.normal(0, 0.2, hidden).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "expert.bin"
            path.write_bytes(weights.tobytes())
            client = DirectHardwareClient(HardwareProfiler(temp))
            try:
                width = hidden * intermediate
                gate = weights[:width].reshape(intermediate, hidden).astype(np.float32) @ vector
                up = weights[width:2 * width].reshape(intermediate, hidden).astype(np.float32) @ vector
                expected = weights[2 * width:].reshape(hidden, intermediate).astype(np.float32) @ (
                    gate / (1 + np.exp(-gate)) * up
                )
                available = [str(item) for item in client.status(start=True).get("devices", [])]
                devices = [name for name in ("GPU", "NPU") if any(item.startswith(name) for item in available)]
                if not devices:
                    devices = ["CPU"]
                successful = []
                rejected = []
                for device in devices:
                    expert_id = f"integration-expert-{device.lower()}"
                    try:
                        client.load_expert(
                            expert_id=expert_id, path=str(path), offset=0,
                            size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            hidden_size=hidden, intermediate_size=intermediate,
                            dtype="float32", device=device,
                        )
                        actual = np.asarray(client.run_expert(expert_id, vector.tolist())["output"])
                        with self.subTest(device=device):
                            np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0.003)
                        successful.append(device)
                    except DirectHardwareError as exc:
                        rejected.append({"device": device, "error": str(exc)})
                self.assertTrue(successful, f"no OpenVINO device executed the expert; rejected={rejected}")
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
