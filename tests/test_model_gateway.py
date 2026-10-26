from __future__ import annotations

import tempfile
from threading import Event, Timer, get_ident
from types import SimpleNamespace
import time
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
    def test_nonstreaming_native_completion_receives_cancellation_and_never_falls_back(self) -> None:
        from openagent.adaptive_engine.execution_service import ExecutionCancelled

        class NativeProvider:
            id = "native-publisher"
            local = True
            default_model = "fixture"

            def __init__(self):
                self.received_cancel = None

            def available(self):
                return True

            def complete(self, _messages, *, model=None, cancelled=None):
                self.received_cancel = cancelled
                cancellation_event.set()
                if cancelled is not None and cancelled():
                    raise ExecutionCancelled("native completion cancelled")
                return SimpleNamespace(
                    provider=self.id,
                    model=model or self.default_model,
                    content="unexpected",
                    metadata={},
                )

        gateway = ModelGateway()
        native = NativeProvider()
        cancellation_event = Event()
        gateway.providers[native.id] = native
        fallback_calls = []
        gateway.select_fallback = lambda _provider_id: fallback_calls.append(True) or gateway.providers["local"]
        with self.assertRaisesRegex(ExecutionCancelled, "native completion cancelled"):
            gateway.complete(
                [{"role": "user", "content": "cancel this unique completion"}],
                provider_id=native.id,
                cancelled=cancellation_event.is_set,
            )

        self.assertTrue(callable(native.received_cancel))
        self.assertEqual(fallback_calls, [])

    def test_stream_cancellation_is_not_treated_as_provider_failure(self) -> None:
        from openagent.adaptive_engine.execution_service import ExecutionCancelled

        class Provider:
            id = "local"
            local = True
            default_model = "fixture"

            def stream_complete(self, _messages, *, model=None):
                yield StreamChunk(self.id, model or self.default_model, "hidden", False, {})

        gateway = ModelGateway()
        provider = Provider()
        fallback_calls = []
        checks = iter((False, True))
        gateway.select_provider = lambda *_args, **_kwargs: provider
        gateway.select_fallback = lambda provider_id: fallback_calls.append(provider_id)

        with self.assertRaises(ExecutionCancelled):
            next(gateway.stream_complete(
                [{"role": "user", "content": "cancellation must not fallback"}],
                cancelled=lambda: next(checks, True),
            ))

        self.assertEqual(fallback_calls, [])

    def test_stream_forwards_transport_cancellation_signal_to_native_provider(self) -> None:
        class NativeProvider:
            id = "native-publisher"
            local = True
            default_model = "fixture"

            def __init__(self):
                self.signal = None

            def stream_complete(self, _messages, *, model=None, cancelled=None):
                self.signal = cancelled
                yield StreamChunk(self.id, model or self.default_model, "", True, {})

        gateway = ModelGateway()
        provider = NativeProvider()
        gateway.select_provider = lambda *_args, **_kwargs: provider

        def signal():
            return False

        chunks = list(gateway.stream_complete(
            [{"role": "user", "content": "native cancellation"}], cancelled=signal,
        ))

        self.assertIs(provider.signal, signal)
        self.assertTrue(chunks[-1].done)

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

    def test_closing_text_stream_releases_provider_iterator(self) -> None:
        class ProviderStream:
            def __init__(self):
                self.closed = 0
                self.sent = False

            def __iter__(self):
                return self

            def __next__(self):
                if self.sent:
                    raise StopIteration
                self.sent = True
                return StreamChunk('ollama', 'fixture', 'partial', False, {})

            def close(self):
                self.closed += 1

        source = ProviderStream()
        with patch("openagent.model_gateway.OllamaProvider.available", return_value=True), patch(
            "openagent.model_gateway.OllamaProvider.stream_complete", return_value=source
        ):
            gateway = ModelGateway()
            stream = gateway.stream_complete(
                [{"role": "user", "content": "disconnect"}], provider_id="ollama")
            self.assertEqual(next(stream).content, 'partial')
            stream.close()

        self.assertEqual(source.closed, 1)

    def test_text_stream_cleanup_summary_never_calls_hostile_repr(self) -> None:
        from openagent.model_gateway import _closing_provider_stream

        class UnprintableCleanupError(Exception):
            def __repr__(self):
                raise AssertionError("cleanup repr must not run")

        class Source:
            def __init__(self):
                self.read = False

            def __iter__(self):
                return self

            def __next__(self):
                if not self.read:
                    self.read = True
                    return StreamChunk("fixture", "model", "partial", False, {})
                raise RuntimeError("primary provider stream error")

            def close(self):
                raise UnprintableCleanupError()

        stream = _closing_provider_stream(Source())
        self.assertEqual(next(stream).content, "partial")
        with self.assertRaisesRegex(RuntimeError, "primary provider stream error") as caught:
            next(stream)
        self.assertTrue(any(
            "UnprintableCleanupError: failure details omitted" in note
            for note in getattr(caught.exception, "__notes__", ())))

    def test_openai_compatible_stream_close_releases_sse_response_iterator(self) -> None:
        from openagent.model_gateway import OpenAICompatibleProvider

        class Response:
            def __init__(self):
                self.closed = 0
                self.sent = False

            def __iter__(self):
                return self

            def __next__(self):
                if self.sent:
                    raise StopIteration
                self.sent = True
                return {"choices": [{"delta": {"content": "partial"}}]}

            def close(self):
                self.closed += 1

        provider = OpenAICompatibleProvider()
        provider.base_url = "https://provider.example"
        provider.api_key = "token"
        source = Response()
        with patch("openagent.model_gateway.stream_sse_json", return_value=source):
            stream = provider.stream_complete([{"role": "user", "content": "hello"}])
            self.assertEqual(next(stream).content, "partial")
            stream.close()

        self.assertEqual(source.closed, 1)

    def test_ollama_vision_stream_uses_image_payload_and_cancellation(self) -> None:
        from openagent.model_gateway import OllamaProvider

        cancellation = Event()
        captured = {}
        source = iter((
            {"message": {"content": "vision"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ))

        def fake_stream(url, payload, *, headers, cancelled=None):
            captured.update(url=url, payload=payload, headers=headers, cancelled=cancelled)
            return source

        provider = OllamaProvider()
        provider.base_url = "http://127.0.0.1:11434"
        with patch.object(provider, "resolve_vision_model", return_value="vision-model"), patch(
            "openagent.model_gateway.stream_json_lines", side_effect=fake_stream,
        ):
            chunks = list(provider.stream_complete_vision(
                [{"role": "user", "content": "describe"}], b"image-data",
                media_type="image/png", cancelled=cancellation.is_set,
            ))

        self.assertEqual([chunk.content for chunk in chunks], ["vision", ""])
        self.assertTrue(chunks[-1].done)
        self.assertEqual(captured["cancelled"], cancellation.is_set)
        sent = captured["payload"]["messages"][-1]
        self.assertEqual(sent["content"], "describe")
        self.assertEqual(sent["images"], ["aW1hZ2UtZGF0YQ=="])

    def test_openai_vision_stream_uses_multimodal_payload_and_cancellation(self) -> None:
        from openagent.model_gateway import OpenAICompatibleProvider

        cancellation = Event()
        captured = {}
        source = iter((
            {"choices": [{"delta": {"content": "vision"}}]},
            {"done": True},
        ))

        def fake_stream(url, payload, *, headers, cancelled=None):
            captured.update(url=url, payload=payload, headers=headers, cancelled=cancelled)
            return source

        provider = OpenAICompatibleProvider()
        provider.base_url = "https://provider.example/v1"
        provider.api_key = "secret"
        with patch.object(provider, "require_configured"), patch(
            "openagent.model_gateway.stream_sse_json", side_effect=fake_stream,
        ):
            chunks = list(provider.stream_complete_vision(
                [{"role": "user", "content": "describe"}], b"image-data",
                media_type="image/png", cancelled=cancellation.is_set,
            ))

        self.assertEqual([chunk.content for chunk in chunks], ["vision", ""])
        self.assertTrue(chunks[-1].done)
        self.assertEqual(captured["cancelled"], cancellation.is_set)
        part = captured["payload"]["messages"][-1]["content"][1]
        self.assertEqual(part["type"], "image_url")
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_model_stream_reader_closes_transport_when_cancellation_arrives_without_chunks(self) -> None:
        from openagent.adaptive_engine.execution_service import ExecutionCancelled
        from openagent.model_gateway import _stream_model_lines

        cancellation = Event()
        started = Event()
        transport_cancelled = Event()
        callback_threads = []
        consumer_thread = get_ident()

        def check_cancelled():
            callback_threads.append(get_ident())
            return cancellation.is_set()

        def blocking_fetch(_url, *, cancelled, **_kwargs):
            started.set()
            while not cancelled():
                time.sleep(0.005)
            transport_cancelled.set()
            raise InterruptedError("cancelled test transport")

        timer = Timer(0.05, cancellation.set)
        timer.start()
        try:
            with patch("openagent.model_gateway.EgressClient.fetch", side_effect=blocking_fetch):
                with self.assertRaises(ExecutionCancelled):
                    next(_stream_model_lines(
                        "https://provider.example/stream", {}, headers={},
                        cancelled=check_cancelled,
                    ))
        finally:
            timer.cancel()

        self.assertTrue(started.is_set())
        self.assertTrue(transport_cancelled.wait(timeout=1.0))
        self.assertTrue(transport_cancelled.is_set())
        self.assertTrue(callback_threads)
        self.assertEqual(set(callback_threads), {consumer_thread})

    def test_gateway_passes_disconnect_signal_into_ollama_vision_transport(self) -> None:
        cancellation = Event()
        captured = {}

        def fake_stream(self, _messages, _image, *, media_type, model=None,
                        max_output_tokens=None, cancelled=None):
            captured["cancelled"] = cancelled
            yield StreamChunk(self.id, model or "vision-model", "", True, {"modality": "vision"})

        gateway = ModelGateway()
        with patch("openagent.model_gateway.OllamaProvider.available", return_value=True), patch(
            "openagent.model_gateway.OllamaProvider.resolve_vision_model", return_value="vision-model",
        ), patch("openagent.model_gateway.OllamaProvider.stream_complete_vision", fake_stream):
            chunks = list(gateway.stream_complete_vision(
                [{"role": "user", "content": "describe"}], b"image-data",
                media_type="image/png", provider_id="ollama", cancelled=cancellation.is_set,
            ))

        self.assertTrue(chunks[-1].done)
        self.assertEqual(captured["cancelled"], cancellation.is_set)

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
