from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openagent.core_runtime import CPU_MODEL_STACK, adaptive_quantization_plan, model_request_options, smooth_runtime_profile
from openagent.local_ai import LocalAIInfrastructure
from openagent.storage import OpenAgentStore


class CoreRuntimeTests(unittest.TestCase):
    def test_eight_gb_profile_prefers_ultra_light_quantized_models(self) -> None:
        profile = smooth_runtime_profile(task="research", hardware={"ram_gb": 8, "cpu_count": 4, "gpu": {"available": False}})

        self.assertEqual(profile.tier, "ultra-light-8gb")
        self.assertLessEqual(profile.context_tokens, 2048)
        self.assertLessEqual(profile.output_tokens, 384)
        self.assertIn("q", profile.quantization_method)
        self.assertEqual(profile.recommended_code_model, CPU_MODEL_STACK["code"])
        self.assertEqual(profile.recommended_extraction_model, CPU_MODEL_STACK["extraction"])
        self.assertIn("faster-whisper", profile.recommended_voice_engine)

    def test_cpu_stack_prefers_phi_qwen_gemma_and_deepseek_roles(self) -> None:
        light = smooth_runtime_profile(task="chat", hardware={"ram_gb": 12, "cpu_count": 8, "gpu": {"available": False}})
        micro = smooth_runtime_profile(task="code", hardware={"ram_gb": 5, "cpu_count": 4, "gpu": {"available": False}})
        high = smooth_runtime_profile(task="research", hardware={"ram_gb": 16, "cpu_count": 8, "gpu": {"available": False}})

        self.assertEqual(light.recommended_chat_model, CPU_MODEL_STACK["chat"])
        self.assertEqual(light.recommended_agent_model, CPU_MODEL_STACK["agent"])
        self.assertEqual(light.recommended_speed_model, CPU_MODEL_STACK["speed"])
        self.assertEqual(micro.recommended_code_model, CPU_MODEL_STACK["low_code"])
        self.assertEqual(high.recommended_research_model, CPU_MODEL_STACK["high_research"])

    def test_model_request_options_cap_interactive_output(self) -> None:
        chat = model_request_options(task="chat", hardware={"ram_gb": 16, "cpu_count": 8, "gpu": {"available": False}})
        research = model_request_options(task="research", hardware={"ram_gb": 16, "cpu_count": 8, "gpu": {"available": False}})

        self.assertLessEqual(chat["num_predict"], 192)
        self.assertLessEqual(research["num_predict"], 240)
        self.assertGreaterEqual(chat["num_predict"], 32)

    def test_explicit_document_budget_can_exceed_interactive_cap(self) -> None:
        options = model_request_options(
            task="research",
            hardware={"ram_gb": 16, "cpu_count": 8, "gpu": {"available": False}},
            prompt_tokens=900,
            output_tokens=900,
        )

        self.assertEqual(options["num_predict"], 900)
        self.assertGreaterEqual(options["num_ctx"], 2048)

    def test_sub_two_bit_profile_for_tiny_cpu_systems(self) -> None:
        plan = adaptive_quantization_plan(
            task="chat",
            target_ram_gb=3,
            quantization_mode="below-2bit",
            parameter_count_b=0.35,
        )

        self.assertEqual(plan["profile"]["quantization_method"], "q1_58_K_S")
        self.assertLess(plan["profile"]["effective_bits"], 2)
        self.assertEqual(plan["cpu_strategy"]["threads"], 1)
        self.assertTrue(plan["fits_memory_budget"])

    def test_local_ai_auto_quantize_records_smooth_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.bin"
            source.write_bytes(b"model")
            infra = LocalAIInfrastructure(OpenAgentStore(Path(tmp) / "data"))
            model = infra.register_model(name="Tiny", provider="local", local_path=str(source))

            result = infra.auto_quantize(model["id"], task="chat", target_ram_gb=8, quantization_mode="maximum-compression")

            self.assertEqual(result["profile"]["tier"], "ultra-light-8gb")
            self.assertEqual(result["quantization"]["bits"], result["profile"]["quantization_bits"])
            self.assertEqual(result["quantization"]["metadata"]["status"], "planned")
            self.assertIn("adaptive_plan", result)
            self.assertEqual(len(infra.quantizations(model["id"])), 1)


if __name__ == "__main__":
    unittest.main()
