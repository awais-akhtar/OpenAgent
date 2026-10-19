from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
import unittest

from openagent.model_gateway import ProviderUnavailableError, StreamChunk
from openagent.routes import StreamRouteResult
from openagent.routes.inference_routes import InferenceRouteDependencies, handle_get, handle_post


class _UnavailableGateway:
    def warmup(self, **kwargs):
        raise ProviderUnavailableError("native worker failed")


class _AuditStore:
    def add_audit_event(self, event) -> None:
        raise AssertionError("failed warmup must not be recorded as successful")


class InferenceRouteTests(unittest.TestCase):
    def test_native_warmup_failure_returns_structured_service_unavailable(self) -> None:
        deps = InferenceRouteDependencies(
            gateway=_UnavailableGateway(),
            store=_AuditStore(),
            read_json=lambda: {"project": "default", "provider": "openvino-direct"},
        )

        result = handle_post("/api/inference/warmup", deps)

        self.assertIsNotNone(result)
        self.assertEqual(result.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(result.payload["status"], "unavailable")
        self.assertIn("native worker failed", result.payload["error"])

    def test_compatibility_routes_fail_closed_for_unimplemented_features(self) -> None:
        cases = (
            ("/v1/messages", {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 8, "stop_sequences": ["stop"]},
             "text stop sequences are not yet adapted"),
            ("/v1/messages", {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 8, "tools": [{"name": "lookup"}]},
             "tool calls and thinking blocks are not admitted"),
            ("/v1/messages", {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 8, "queue": "yes"}, "queue must be boolean"),
            ("/v1/messages", {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 8, "priority": 1}, "unsupported queue controls"),
        )
        for path, payload, message in cases:
            with self.subTest(path=path, payload=payload):
                deps = InferenceRouteDependencies(
                    gateway=object(), store=object(), read_json=lambda payload=payload: payload)
                result = handle_post(path, deps)
                self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
                self.assertIn(message, result.payload["error"]["message"])
                if path == "/v1/messages":
                    self.assertEqual(result.payload["type"], "error")
                    self.assertEqual(result.payload["error"]["type"], "invalid_request_error")

    def test_compatibility_queue_is_opt_in_and_passes_bounded_timeout(self) -> None:
        class Provider:
            received = None

            def _configuration(self):
                return {"model": "local"}

            def complete(self, *_args, **kwargs):
                self.received = kwargs
                return SimpleNamespace(model="local", content="answer", metadata={
                    "runtime": {"prompt_tokens": 1, "generated_ids": [2]},
                    "queue_wait_ms": 17,
                })

        class Gateway:
            def __init__(self, provider):
                self.providers = {"native-publisher": provider}

        for queue, timeout, expected in ((False, None, 0), (True, None, 30.0), (True, 1.5, 1.5)):
            provider = Provider()
            payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 2, "queue": queue}
            if timeout is not None:
                payload["queue_timeout"] = timeout
            result = handle_post("/v1/chat/completions", InferenceRouteDependencies(
                gateway=Gateway(provider), store=object(), read_json=lambda payload=payload: payload))
            self.assertEqual(result.status, HTTPStatus.OK)
            self.assertEqual(provider.received["queue_timeout"], expected)
            self.assertEqual(result.headers["X-Queue-Wait-Ms"], "17")

    def test_openai_streaming_frames_usage_and_queue_admission(self) -> None:
        class Provider:
            received = None

            def _configuration(self):
                return {"model": "local"}

            def stream_complete(self, *_args, **kwargs):
                self.received = kwargs
                return iter((
                    StreamChunk("native-publisher", "local", "hello", False,
                                {"prompt_tokens": 3, "queue_wait_ms": 23}),
                    StreamChunk("native-publisher", "local", "", True, {"committed": True,
                        "runtime": {"prompt_ids": [1, 2, 3], "generated_ids": [4, 5],
                                    "finish_reason": "length"}}),
                ))

        class Gateway:
            def __init__(self, provider):
                self.providers = {"native-publisher": provider}

        provider = Provider()
        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2, "stream": True, "stream_options": {"include_usage": True},
                   "queue": True, "queue_timeout": 1.25}
        result = handle_post("/v1/chat/completions", InferenceRouteDependencies(
            gateway=Gateway(provider), store=object(), read_json=lambda: payload))
        self.assertIsInstance(result, StreamRouteResult)
        self.assertEqual(result.headers["X-Queue-Wait-Ms"], "23")
        frames = list(result.events)
        self.assertEqual(provider.received["queue_timeout"], 1.25)
        self.assertEqual(frames[0].data["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(frames[1].data["choices"][0]["delta"], {"content": "hello"})
        self.assertEqual(frames[2].data["choices"][0]["finish_reason"], "length")
        self.assertEqual(frames[3].data["usage"], {"prompt_tokens": 3, "completion_tokens": 2,
                                                    "total_tokens": 5})
        self.assertEqual(frames[4].data, "[DONE]")
        self.assertTrue(frames[4].raw_data and frames[4].terminal)

    def test_stream_queue_rejection_is_returned_before_sse_headers(self) -> None:
        from openagent.adaptive_engine.admission_scheduler import AdmissionRejected

        class Provider:
            def _configuration(self):
                return {"model": "local"}

            def stream_complete(self, *_args, **_kwargs):
                def rejected():
                    raise AdmissionRejected("request admission capacity exhausted")
                    yield
                return rejected()

        class Gateway:
            providers = {"native-publisher": Provider()}

        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2, "stream": True}
        result = handle_post("/v1/chat/completions", InferenceRouteDependencies(
            gateway=Gateway(), store=object(), read_json=lambda: payload))
        self.assertEqual(result.status, HTTPStatus.TOO_MANY_REQUESTS)
        self.assertEqual(result.headers, {"Retry-After": "1"})

    def test_stream_is_closed_when_first_event_fails_before_response_ownership(self) -> None:
        class BrokenStream:
            closed = False

            def __iter__(self):
                return self

            def __next__(self):
                raise ProviderUnavailableError("stream failed before first event")

            def close(self):
                self.closed = True

        class Provider:
            def __init__(self):
                self.source = BrokenStream()

            def _configuration(self):
                return {"model": "local"}

            def stream_complete(self, *_args, **_kwargs):
                return self.source

        class Gateway:
            def __init__(self, provider):
                self.providers = {"native-publisher": provider}

        provider = Provider()
        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2, "stream": True}
        result = handle_post("/v1/chat/completions", InferenceRouteDependencies(
            gateway=Gateway(provider), store=object(), read_json=lambda: payload))

        self.assertEqual(result.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertIn("stream failed before first event", result.payload["error"]["message"])
        self.assertTrue(provider.source.closed)

    def test_truncated_native_stream_emits_error_not_successful_stop(self) -> None:
        class Provider:
            def _configuration(self):
                return {"model": "local"}

            def stream_complete(self, *_args, **_kwargs):
                yield StreamChunk("native-publisher", "local", "partial", False, {})

        class Gateway:
            providers = {"native-publisher": Provider()}

        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2, "stream": True}
        result = handle_post("/v1/chat/completions", InferenceRouteDependencies(
            gateway=Gateway(), store=object(), read_json=lambda: payload))
        frames = list(result.events)
        self.assertEqual(frames[-2].data["error"]["type"], "server_error")
        self.assertEqual(frames[-1].data, "[DONE]")
        self.assertFalse(any(frame.data.get("choices", [{}])[0].get("finish_reason")
                             for frame in frames[:-2] if type(frame.data) is dict))

    def test_openai_stream_cleanup_failure_is_reported_before_success_terminal(self) -> None:
        class BrokenCleanupStream:
            def __init__(self):
                self.events = iter((
                    StreamChunk("native-publisher", "local", "partial", False, {}),
                    StreamChunk("native-publisher", "local", "", True, {"runtime": {
                        "prompt_ids": [1], "generated_ids": [2], "finish_reason": "eos"}}),
                ))

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.events)

            def close(self):
                raise RuntimeError("upstream close failed")

        class Provider:
            def _configuration(self):
                return {"model": "local"}

            def stream_complete(self, *_args, **_kwargs):
                return BrokenCleanupStream()

        class Gateway:
            providers = {"native-publisher": Provider()}

        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2, "stream": True}
        result = handle_post("/v1/chat/completions", InferenceRouteDependencies(
            gateway=Gateway(), store=object(), read_json=lambda: payload))
        frames = list(result.events)

        self.assertEqual(frames[-2].data["error"]["type"], "server_error")
        self.assertEqual(frames[-1].data, "[DONE]")
        self.assertFalse(any(frame.data.get("choices", [{}])[0].get("finish_reason")
                             for frame in frames[:-2] if type(frame.data) is dict))

    def test_anthropic_streaming_uses_named_protocol_events_and_closes_source(self) -> None:
        class Provider:
            def _configuration(self):
                return {"model": "local"}

            def stream_complete(self, *_args, **_kwargs):
                yield StreamChunk("native-publisher", "local", "answer", False,
                                  {"prompt_tokens": 4})
                yield StreamChunk("native-publisher", "local", "", True, {"runtime": {
                    "prompt_ids": [1, 2, 3, 4], "generated_ids": [5], "finish_reason": "eos"}})

        class Gateway:
            providers = {"native-publisher": Provider()}

        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2, "stream": True}
        result = handle_post("/v1/messages", InferenceRouteDependencies(
            gateway=Gateway(), store=object(), read_json=lambda: payload,
            anthropic_version="2023-06-01"))
        frames = list(result.events)
        self.assertEqual([frame.event for frame in frames], [
            "message_start", "content_block_start", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop"])
        self.assertEqual(frames[0].data["message"]["usage"]["input_tokens"], 4)
        self.assertEqual(frames[2].data["delta"]["text"], "answer")
        self.assertEqual(frames[-2].data["delta"]["stop_reason"], "end_turn")
        self.assertTrue(frames[-1].terminal)

    def test_anthropic_immediate_terminal_stream_reports_runtime_prompt_usage(self) -> None:
        class Provider:
            def _configuration(self):
                return {"model": "local"}

            def stream_complete(self, *_args, **_kwargs):
                yield StreamChunk("native-publisher", "local", "", True, {"runtime": {
                    "prompt_ids": [1, 2, 3], "generated_ids": [], "finish_reason": "eos"}})

        class Gateway:
            providers = {"native-publisher": Provider()}

        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2, "stream": True}
        result = handle_post("/v1/messages", InferenceRouteDependencies(
            gateway=Gateway(), store=object(), read_json=lambda: payload,
            anthropic_version="2023-06-01"))

        frames = list(result.events)
        self.assertEqual(frames[0].data["message"]["usage"]["input_tokens"], 3)
        self.assertEqual(frames[-2].data["delta"]["stop_reason"], "end_turn")
        self.assertTrue(frames[-1].terminal)

    def test_queue_overload_uses_429_and_retry_after(self) -> None:
        from openagent.adaptive_engine.admission_scheduler import AdmissionRejected

        class Provider:
            def _configuration(self):
                return {"model": "local"}

            def complete(self, *_args, **_kwargs):
                raise AdmissionRejected("request admission capacity exhausted")

        class Gateway:
            providers = {"native-publisher": Provider()}

        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2}
        result = handle_post("/v1/chat/completions", InferenceRouteDependencies(
            gateway=Gateway(), store=object(), read_json=lambda: payload))
        self.assertEqual(result.status, HTTPStatus.TOO_MANY_REQUESTS)
        self.assertEqual(result.headers, {"Retry-After": "1"})
        self.assertIn("capacity exhausted", result.payload["error"]["message"])

    def test_unreadable_server_provider_configuration_is_service_unavailable(self) -> None:
        class Provider:
            def _configuration(self):
                raise OSError("configured native checkpoint is unreadable")

        class Gateway:
            providers = {"native-publisher": Provider()}

        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2}
        result = handle_post("/v1/chat/completions", InferenceRouteDependencies(
            gateway=Gateway(), store=object(), read_json=lambda: payload))
        self.assertEqual(result.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(result.payload["error"]["type"], "server_error")

    def test_health_and_prometheus_expose_provider_owned_queue_snapshot(self) -> None:
        class Provider:
            def available(self):
                return True

            def status(self):
                return {"active": True, "stream_worker_active": False,
                    "request_queue": {"active": 1, "waiting": 2, "max_active": 1,
                        "max_waiting": 64, "admitted": 10, "rejected": 3,
                        "timed_out": 1, "cancelled": 2, "wait_seconds": 1.25,
                        "wait_buckets": [0.01, 0.05], "wait_histogram": [4, 2, 1]}}

        class Gateway:
            providers = {"native-publisher": Provider()}

        deps = InferenceRouteDependencies(gateway=Gateway(), store=object(), read_json=lambda: {})
        health = handle_get("/health", "", deps)
        self.assertEqual(health.payload["request_queue"]["waiting"], 2)
        metrics = handle_get("/metrics", "", deps).payload["_prometheus"]
        self.assertIn("openagent_native_request_queue_rejected_total 3", metrics)
        self.assertIn('openagent_native_request_queue_wait_seconds_bucket{le="+Inf"} 7', metrics)
        self.assertIn("openagent_native_request_queue_wait_seconds_sum 1.25", metrics)

    def test_anthropic_usage_and_finish_reason_use_v41_generation_fields(self) -> None:
        class Provider:
            def _configuration(self):
                return {"model": "local"}

            def complete(self, *_args, **_kwargs):
                return SimpleNamespace(model="local", content="answer", metadata={
                    "runtime": {
                        "schema": "openagent.publisher-v41-generation.v1",
                        "prompt_ids": [1, 2, 3], "generated_ids": [4, 5],
                        "finish_reason": "length",
                    },
                })

        class Gateway:
            providers = {"native-publisher": Provider()}

        class Store:
            def add_audit_event(self, _event):
                pass

        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2}
        result = handle_post("/v1/messages", InferenceRouteDependencies(
            gateway=Gateway(), store=Store(), read_json=lambda: payload,
            anthropic_version="2023-06-01"))

        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertEqual(result.payload["usage"], {"input_tokens": 3, "output_tokens": 2})
        self.assertEqual(result.payload["stop_reason"], "max_tokens")

    def test_anthropic_requires_valid_version_before_provider_execution(self) -> None:
        class Gateway:
            providers = {"native-publisher": object()}

        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 2}
        for version in (None, "not-a-date", "2023-02-29"):
            with self.subTest(version=version):
                result = handle_post("/v1/messages", InferenceRouteDependencies(
                    gateway=Gateway(), store=object(), read_json=lambda: payload,
                    anthropic_version=version))
                self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
                self.assertIn("anthropic-version header", result.payload["error"]["message"])


if __name__ == "__main__":
    unittest.main()
