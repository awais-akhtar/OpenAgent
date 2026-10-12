from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from openagent.core_runtime import CPU_MODEL_STACK
from openagent.model_gateway import (
    ModelGateway,
    ProviderConfigurationError,
    StreamChunk,
    model_timeout_seconds,
    normalize_model_name,
    runtime_settings_snapshot,
    _olmoe_prompt,
)


class FakeDirectHardware:
    def __init__(self) -> None:
        self.loaded = False
        self.load_calls = 0

    def available(self):
        return True

    def status(self, *, start=False):
        return {
            "running": self.loaded,
            "models": [{"kind": "llm", "model_id": "openagent-direct-chat"}] if self.loaded else [],
        }

    def load_llm(self, path, **kwargs):
        self.loaded = True
        self.load_calls += 1
        return {"status": "resident", "model_id": kwargs["model_id"], "device": kwargs["device"]}

    def generate(self, model_id, **kwargs):
        return {"text": "direct local response", "device": "GPU", "latency_ms": 12.5}

    def stream_generate(self, model_id, **kwargs):
        yield {"type": "token", "content": "direct "}
        yield {"type": "token", "content": "stream"}
        yield {"type": "done", "text": "direct stream", "device": "GPU", "latency_ms": 9.5, "first_token_ms": 2.0, "streamed": True}


class ModelGatewayTests(unittest.TestCase):
    def test_auto_openvino_route_reports_the_resident_model_identity(self) -> None:
        class Provider:
            id = "openvino-direct"
            default_model = "openagent-qwen2.5-1.5b-int4"

        provider = Provider()
        self.assertIsNone(
            ModelGateway._effective_provider_model(
                provider,
                provider_id="auto",
                model="phi4-mini",
            )
        )
        self.assertEqual(
            ModelGateway._effective_provider_model(
                provider,
                provider_id="auto",
                model=provider.default_model,
            ),
            provider.default_model,
        )
        self.assertEqual(
            ModelGateway._effective_provider_model(
                provider,
                provider_id="openvino-direct",
                model="explicit-role-model",
            ),
            "explicit-role-model",
        )

    def test_stage_one_provider_inventory_and_policies(self) -> None:
        gateway = ModelGateway()

        provider_ids = {provider["id"] for provider in gateway.specs()}
        self.assertSetEqual(
            provider_ids,
            {
                "auto", "local", "openvino-direct", "olmoe-direct", "ollama", "llamacpp", "openai-compatible",
                "deepseek", "mistral", "groq", "together", "openrouter", "xai", "fireworks", "perplexity", "qwen",
            },
        )
        self.assertIn("local-first", gateway.policies()["routes"])
        self.assertIn("local-server-first", gateway.policies()["routes"])
        self.assertEqual(
            gateway.policies()["routes"]["local-first"],
            ["openvino-direct", "olmoe-direct", "ollama", "llamacpp", "local"],
        )
        ollama = next(provider for provider in gateway.specs() if provider["id"] == "ollama")
        self.assertEqual(ollama["default_model"], CPU_MODEL_STACK["chat"])
        settings = runtime_settings_snapshot()
        self.assertEqual(settings["chat_model"], CPU_MODEL_STACK["chat"])
        self.assertEqual(settings["code_model"], CPU_MODEL_STACK["code"])
        self.assertEqual(settings["research_model"], CPU_MODEL_STACK["research"])

    def test_olmoe_prompt_preserves_turns_and_system_context(self) -> None:
        prompt = _olmoe_prompt([
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second"},
        ])
        self.assertTrue(prompt.startswith("|||IP_ADDRESS|||<|system|>\nBe precise.\n<|user|>\nFirst"))
        self.assertIn("<|assistant|>\nAnswer|||IP_ADDRESS|||\n<|user|>\nSecond", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))

    def test_olmoe_prompt_rejects_incomplete_turn_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "final user"):
            _olmoe_prompt([{"role": "assistant", "content": "orphan"}])

    def test_compatible_providers_are_external_and_require_explicit_egress_approval(self) -> None:
        gateway = ModelGateway()
        deepseek = next(provider for provider in gateway.specs() if provider["id"] == "deepseek")
        self.assertFalse(deepseek["local"])
        self.assertTrue(deepseek["network"])
        self.assertFalse(deepseek["available"])
        with self.assertRaises(ProviderConfigurationError):
            gateway.complete([{"role": "user", "content": "Use DeepSeek."}], provider_id="deepseek")

    def test_auto_route_falls_back_to_offline_local_when_no_server_is_running(self) -> None:
        with patch("openagent.model_gateway.OllamaProvider.available", return_value=False), patch(
            "openagent.model_gateway.LlamaCppProvider.available",
            return_value=False,
        ):
            gateway = ModelGateway()

            result = gateway.complete(
                [{"role": "user", "content": "Stay local."}],
                provider_id="auto",
            )

        self.assertEqual(result.provider, "local")
        self.assertFalse(result.metadata["network"])
        self.assertTrue(result.metadata["deterministic_fallback"])
        self.assertNotIn("OpenAgent local response", result.content)

    def test_external_provider_requires_explicit_boundary_approval(self) -> None:
        gateway = ModelGateway()

        with self.assertRaises(ProviderConfigurationError):
            gateway.complete(
                [{"role": "user", "content": "Use cloud."}],
                provider_id="openai-compatible",
            )

    def test_local_streaming_and_embeddings(self) -> None:
        gateway = ModelGateway()

        chunks = list(
            gateway.stream_complete(
                [{"role": "user", "content": "Stream locally."}],
                provider_id="local",
            )
        )
        self.assertTrue(chunks[-1].done)
        self.assertGreater(len("".join(chunk.content for chunk in chunks)), 20)

        embeddings = gateway.embed(["alpha", "beta"], provider_id="local")
        self.assertEqual(embeddings.provider, "local")
        self.assertEqual(embeddings.dimensions, 64)
        self.assertEqual(len(embeddings.embeddings), 2)
        self.assertEqual(len(embeddings.embeddings[0]), 64)

    def test_auto_stream_uses_provider_streaming_without_blocking_complete(self) -> None:
        def fake_stream(self, messages, *, model=None):
            yield StreamChunk(
                provider="ollama",
                model=model or "phi4-mini",
                content="hello",
                done=False,
                metadata={"network": True, "local_network": True},
            )
            yield StreamChunk(
                provider="ollama",
                model=model or "phi4-mini",
                content="",
                done=True,
                metadata={"network": True, "local_network": True},
            )

        with patch("openagent.model_gateway.OllamaProvider.available", return_value=True), patch(
            "openagent.model_gateway.OllamaProvider.stream_complete",
            fake_stream,
        ), patch("openagent.model_gateway.OllamaProvider.complete", side_effect=AssertionError("blocking path used")):
            gateway = ModelGateway()
            chunks = list(gateway.stream_complete([{"role": "user", "content": "hello"}], provider_id="auto"))

        self.assertEqual("".join(chunk.content for chunk in chunks), "hello")
        self.assertEqual(chunks[0].provider, "ollama")
        self.assertEqual(chunks[0].metadata["routing_policy"], "local-first")

    def test_auto_route_uses_any_installed_ollama_model_before_fallback(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json(url, payload, *, headers):
            captured["url"] = url
            captured["payload"] = payload
            return {"message": {"content": "real local model response"}}

        with patch("openagent.model_gateway.OllamaProvider.installed_models", return_value=["llama3.2:3b"]), patch(
            "openagent.model_gateway.post_json",
            fake_post_json,
        ):
            gateway = ModelGateway()
            result = gateway.complete([{"role": "user", "content": "hello"}], provider_id="auto")

        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "llama3.2:3b")
        self.assertEqual(result.content, "real local model response")
        self.assertEqual(captured["payload"]["model"], "llama3.2:3b")

    def test_ollama_specs_report_installed_and_selected_model(self) -> None:
        with patch("openagent.model_gateway.OllamaProvider.installed_models", return_value=["gemma3:2b"]):
            gateway = ModelGateway()
            ollama = next(provider for provider in gateway.specs() if provider["id"] == "ollama")

        self.assertTrue(ollama["available"])
        self.assertEqual(ollama["installed_models"], ["gemma3:2b"])
        self.assertEqual(ollama["selected_model"], "gemma3:2b")
        self.assertFalse(ollama["default_model_ready"])

    def test_known_model_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_model_name("phi4-mini:3.8b-instruct-q4_K_M"), "phi4-mini")
        self.assertEqual(normalize_model_name("qwen2.5-coder:3b-instruct-q4_k_m"), "qwen2.5-coder:3b")

    def test_default_model_timeout_allows_cpu_cold_start(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(model_timeout_seconds(), 35.0)
            self.assertEqual(runtime_settings_snapshot()["model_timeout_seconds"], 35.0)

    def test_direct_openvino_provider_loads_once_and_serves_without_network(self) -> None:
        client = FakeDirectHardware()
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ", {"OPENAGENT_OPENVINO_MODEL_DIR": temp}, clear=False
        ), patch("openagent.model_gateway.OllamaProvider.available", return_value=False), patch(
            "openagent.model_gateway.LlamaCppProvider.available", return_value=False
        ):
            gateway = ModelGateway(direct_hardware=client)
            result = gateway.complete([{"role": "user", "content": "hello"}], provider_id="auto")
            second = gateway.complete([{"role": "user", "content": "hello again"}], provider_id="auto")
        self.assertEqual(result.provider, "openvino-direct")
        self.assertEqual(result.content, "direct local response")
        self.assertFalse(result.metadata["network"])
        self.assertEqual(second.provider, "openvino-direct")
        self.assertEqual(client.load_calls, 1)

    def test_direct_openvino_provider_forwards_native_token_stream(self) -> None:
        client = FakeDirectHardware()
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ", {"OPENAGENT_OPENVINO_MODEL_DIR": temp}, clear=False
        ):
            gateway = ModelGateway(direct_hardware=client)
            chunks = list(
                gateway.stream_complete(
                    [{"role": "user", "content": "hello"}],
                    provider_id="openvino-direct",
                )
            )
        self.assertEqual("".join(chunk.content for chunk in chunks), "direct stream")
        self.assertTrue(chunks[-1].done)
        self.assertEqual(chunks[-1].metadata["first_token_ms"], 2.0)
        self.assertTrue(chunks[-1].metadata["streamed"])


if __name__ == "__main__":
    unittest.main()
