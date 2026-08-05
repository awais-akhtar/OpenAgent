from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.compiler import AdaptiveCompiler
from openagent.adaptive_engine.frontends import ModelFrontend, ModelFormatError
from openagent.adaptive_engine.hardware import DeviceProfile, HardwareProfile
from openagent.adaptive_engine.ir import ModelIR, OpCode, build_transformer_graph, lower_graph_to_tensor_ir
from openagent.adaptive_engine.memory import VirtualKVCache
from openagent.adaptive_engine.runtime import AdaptiveExecutionEngine
from openagent.adaptive_engine.scheduler import (
    DeadlineBatchScheduler,
    InferenceRequest,
    RequestPhase,
    RequestPriority,
)
from openagent.routes.adaptive_engine_routes import ROUTES
from openagent.model_gateway import ModelResult
from openagent.storage import OpenAgentStore


def hardware_profile() -> HardwareProfile:
    operations = tuple(item.value for item in OpCode)
    cpu = DeviceProfile(
        id="cpu:0",
        kind="cpu",
        name="Test CPU",
        vendor="intel",
        memory_gb=16,
        unified_memory=True,
        compute_tops=2,
        memory_bandwidth_gbps=30,
        precisions=("fp32", "fp16", "int8", "int4", "int3", "int2"),
        supported_ops=operations,
        backends=("openvino", "llama-cpu"),
    )
    gpu = DeviceProfile(
        id="gpu:0",
        kind="gpu",
        name="Test Arc",
        vendor="intel",
        memory_gb=12,
        unified_memory=True,
        compute_tops=12,
        memory_bandwidth_gbps=80,
        precisions=("fp32", "fp16", "int8", "int4"),
        supported_ops=tuple(item for item in operations if item not in {"tokenization", "sampling"}),
        backends=("openvino", "llama-sycl"),
        capabilities={"openvino": {"id": "GPU"}},
    )
    npu = DeviceProfile(
        id="npu:0",
        kind="npu",
        name="Test NPU",
        vendor="intel",
        memory_gb=8,
        unified_memory=True,
        compute_tops=8,
        memory_bandwidth_gbps=40,
        precisions=("fp16", "int8", "int4"),
        supported_ops=("embedding", "matmul", "attention", "rope", "softmax", "activation", "residual_add"),
        backends=("openvino",),
        capabilities={"openvino": {"id": "NPU"}},
    )
    return HardwareProfile(
        id="test-hardware",
        platform="test",
        total_ram_gb=32,
        available_ram_gb=24,
        cpu_threads=16,
        devices=(cpu, gpu, npu),
        runtimes={"openvino": {"available": True}},
        power={"ac_connected": True},
        discovered_at=time.time(),
    )


def transformer_model(source_format: str = "onnx") -> ModelIR:
    graph = build_transformer_graph(
        block_count=4,
        weight_bytes=2_000_000_000,
        parameter_count=1_000_000_000,
        embedding_length=2048,
    )
    return ModelIR(
        id="model-id",
        name="test-model",
        source_path=f"test.{source_format}",
        source_format=source_format,
        fingerprint="fingerprint",
        architecture="llama",
        parameter_count=1_000_000_000,
        weight_bytes=2_000_000_000,
        quantization_bits=16,
        context_length=4096,
        embedding_length=2048,
        block_count=4,
        graph_nodes=graph,
        tensor_nodes=lower_graph_to_tensor_ir(graph),
    )


class AdaptiveCompilerTests(unittest.TestCase):
    def test_coherent_planner_uses_large_regions_and_separate_phases(self) -> None:
        package = AdaptiveCompiler().compile(
            transformer_model(),
            hardware_profile(),
            prompt_tokens=128,
            context_tokens=2048,
            memory_budget_gb=3,
        )
        self.assertEqual(package.status, "planned")
        self.assertLessEqual(package.prefill.transfer_count, 2)
        self.assertLessEqual(package.decode.transfer_count, 2)
        self.assertEqual(package.prefill.placements[0].device_id, "cpu:0")
        self.assertEqual(package.prefill.placements[-1].device_id, "cpu:0")
        compute_devices = {
            item.device_id
            for item in package.prefill.placements
            if item.op not in {"tokenization", "sampling"}
        }
        self.assertEqual(len(compute_devices), 1)
        self.assertNotEqual(package.prefill.estimated_tokens_per_second, package.decode.estimated_tokens_per_second)

    def test_plan_labels_conversion_and_unmet_memory_budget(self) -> None:
        package = AdaptiveCompiler().compile(
            transformer_model(),
            hardware_profile(),
            memory_budget_gb=0.1,
            quality_loss_limit=0.001,
        )
        summary = package.constraints["precision_summary"]
        self.assertFalse(summary["memory_feasible"])
        self.assertTrue(any("memory budget" in warning for warning in package.warnings))

    def test_verified_device_preference_pins_compute_without_moving_host_operations(self) -> None:
        package = AdaptiveCompiler().compile(
            transformer_model(),
            hardware_profile(),
            memory_budget_gb=3,
            preferred_device_id="gpu:0",
        )
        compute = [
            item.device_id
            for item in package.decode.placements
            if item.op not in {"tokenization", "sampling"}
        ]
        self.assertTrue(compute)
        self.assertEqual(set(compute), {"gpu:0"})
        self.assertEqual(package.constraints["measured_preferred_device"], "gpu:0")


class ModelFrontendTests(unittest.TestCase):
    def test_reads_safetensors_without_loading_tensor_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            header = {
                "weight": {"dtype": "F16", "shape": [2, 4], "data_offsets": [0, 16]},
                "__metadata__": {"format": "pt"},
            }
            raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
            path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(16))
            model = ModelFrontend().inspect(path)
        self.assertEqual(model.source_format, "safetensors")
        self.assertEqual(model.parameter_count, 8)
        self.assertEqual(model.tensors[0].size_bytes, 16)

    def test_rejects_unknown_container(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.bin"
            path.write_bytes(b"not a model")
            with self.assertRaises(ModelFormatError):
                ModelFrontend().inspect(path)


class KVCacheTests(unittest.TestCase):
    def test_prefix_blocks_are_shared_then_copy_on_write(self) -> None:
        cache = VirtualKVCache(1024 * 1024, block_tokens=4, bytes_per_token=64)
        cache.create_session("one", prefix_tokens=[1, 2, 3])
        cache.create_session("two", prefix_tokens=[1, 2, 3])
        self.assertEqual(cache.status()["shared_blocks"], 1)
        cache.append("two", [5])
        self.assertEqual(cache.session_status("one")["token_count"], 3)
        self.assertEqual(cache.session_status("two")["token_count"], 4)
        self.assertTrue(cache.release("one"))


class SchedulerTests(unittest.TestCase):
    def test_decode_and_deadline_protect_interactive_latency(self) -> None:
        scheduler = DeadlineBatchScheduler(max_batch_size=4, interactive_reserve=2)
        now = time.monotonic()
        scheduler.submit(InferenceRequest("background", "m", RequestPhase.DECODE, 10, RequestPriority.BACKGROUND))
        scheduler.submit(InferenceRequest("interactive", "m", RequestPhase.DECODE, 10, RequestPriority.INTERACTIVE))
        scheduler.submit(
            InferenceRequest(
                "urgent",
                "m",
                RequestPhase.DECODE,
                10,
                RequestPriority.AGENT,
                deadline_monotonic=now - 1,
            )
        )
        batch = scheduler.next_batch(now=now)
        self.assertIn("interactive", [item.id for item in batch[:2]])
        self.assertIn("urgent", [item.id for item in batch[:2]])


class AdaptiveRuntimeTests(unittest.TestCase):
    def test_runtime_persists_an_honestly_planned_onnx_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            engine = AdaptiveExecutionEngine(store)
            model_path = Path(temp) / "model.onnx"
            model_path.write_bytes(b"opaque-test-model")
            with patch.object(engine.profiler, "profile", return_value=hardware_profile()):
                plan = engine.compile_model(str(model_path), backend_compile=False)
                saved = engine.plans()
            self.assertEqual(plan["status"], "planned")
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["id"], plan["id"])

    def test_route_contract_exposes_hardware_compile_and_tuning(self) -> None:
        self.assertIn(("GET", "/api/adaptive-engine/hardware"), ROUTES)
        self.assertIn(("POST", "/api/adaptive-engine/compile"), ROUTES)
        self.assertIn(("POST", "/api/adaptive-engine/autotune"), ROUTES)
        self.assertIn(("GET", "/api/adaptive-engine/direct/status"), ROUTES)
        self.assertIn(("POST", "/api/adaptive-engine/direct/calibrate"), ROUTES)
        self.assertIn(("POST", "/api/adaptive-engine/direct/generate"), ROUTES)
        self.assertIn(("GET", "/api/adaptive-engine/roles"), ROUTES)
        self.assertIn(("POST", "/api/adaptive-engine/orchestrate"), ROUTES)

    def test_exact_model_calibration_persists_verified_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            engine = AdaptiveExecutionEngine(store)
            model_path = Path(temp) / "model.onnx"
            model_path.write_bytes(b"opaque-test-model")
            evidence = {
                "status": "verified",
                "winner": {"device": "GPU", "valid": True, "latency_ms": {"mean": 1.2}},
                "devices": [
                    {"device": "CPU", "valid": True, "latency_ms": {"mean": 3.8}},
                    {"device": "GPU", "valid": True, "latency_ms": {"mean": 1.2}},
                ],
                "resident_model_id": "adaptive-test",
            }
            with patch.object(engine.profiler, "profile", return_value=hardware_profile()), patch.object(
                engine.direct, "calibrate", return_value=evidence
            ):
                result = engine.calibrate_direct_model(
                    str(model_path), devices=["cpu:0", "gpu:0"], model_id="adaptive-test"
                )
                saved = engine.direct_calibrations()
            self.assertEqual(result["winner_profile_device"], "gpu:0")
            self.assertEqual(saved[0]["winner"]["device"], "GPU")
            actions = [item["action"] for item in store.list_audit_events(limit=20)]
            self.assertIn("adaptive.direct.calibrate", actions)

    def test_role_profiles_and_supervised_model_loop_are_persistent_and_audited(self) -> None:
        class Gateway:
            def __init__(self) -> None:
                self.reviews = 0

            def specs(self):
                return [{"id": "auto", "label": "Auto", "available": True}]

            def discover_models(self):
                return [{"provider": "local", "model": "test-model", "available": True}]

            def complete(self, messages, *, provider_id, model, allow_external):
                system = messages[0]["content"]
                if "strict technical reviewer" in system:
                    self.reviews += 1
                    content = "APPROVED\nAll criteria pass." if self.reviews > 1 else "REVISE\nAdd the missing test."
                elif "implementation engineer" in system:
                    content = "complete implementation with tests"
                else:
                    content = "plan and acceptance checklist"
                return ModelResult("local", model or "test-model", content, {"network": False})

        with tempfile.TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            engine = AdaptiveExecutionEngine(store)
            gateway = Gateway()
            engine.bind_gateway(gateway)
            saved_role = engine.upsert_model_role(
                "code", project="core", provider_id="auto", model_id="test-code",
                system_prompt="You are the implementation engineer. Produce complete work.",
            )
            result = engine.orchestrate_models(
                "Build a tested parser.", project="core", max_rounds=2,
            )
            history = engine.orchestration_runs(project="core")
            actions = {item["action"] for item in store.list_audit_events(project="core", limit=30)}
        self.assertTrue(saved_role["persisted"])
        self.assertEqual(saved_role["model_id"], "test-code")
        self.assertTrue(result["approved"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(len(result["turns"]), 5)
        self.assertEqual(history[0]["status"], "completed")
        self.assertEqual(len(history[0]["turns"]), 5)
        self.assertIn("adaptive.role.upsert", actions)
        self.assertIn("adaptive.orchestration.complete", actions)

    def test_sequence_supervisor_verdict_controls_approval(self) -> None:
        class Gateway:
            def specs(self):
                return [{"id": "auto", "label": "Auto", "available": True}]

            def discover_models(self):
                return [{"provider": "local", "model": "test-model", "available": True}]

            def complete(self, messages, *, provider_id, model, allow_external):
                system = messages[0]["content"]
                content = "REVISE\nThe acceptance evidence is incomplete." if "strict technical reviewer" in system else "candidate"
                return ModelResult("local", model or "test-model", content, {"network": False})

        with tempfile.TemporaryDirectory() as temp:
            engine = AdaptiveExecutionEngine(OpenAgentStore(temp))
            engine.bind_gateway(Gateway())
            result = engine.orchestrate_models(
                "Prove a result.",
                project="core",
                mode="sequence",
                roles=["reasoning", "supervisor"],
            )

        self.assertFalse(result["approved"])
        self.assertTrue(result["final_output"].startswith("REVISE"))


if __name__ == "__main__":
    unittest.main()
