from __future__ import annotations

import unittest
import threading
from unittest.mock import patch

from openagent.adaptive_engine.serving_engine import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    Role,
    ServingEngine,
)
from openagent.adaptive_engine.multi_session_service import (
    Phase2MultiSessionService,
    Phase2SessionHandle,
)
from openagent.adaptive_engine.inference_service import InferenceService
from openagent.adaptive_engine.load_balancer import Phase3LoadBalancedService


class _Session:
    def __init__(self, *, generate_error=None, stream_error=None, close_error=None):
        self.generate_error = generate_error
        self.stream_error = stream_error
        self.close_error = close_error
        self.closed = 0

    def generate(self, **_kwargs):
        if self.generate_error:
            raise self.generate_error
        return [65]

    def stream(self, **_kwargs):
        if self.stream_error:
            raise self.stream_error
        yield 65

    def close(self):
        self.closed += 1
        if self.close_error:
            raise self.close_error


class _StopTokenizer:
    def encode(self, text):
        return [7, 8] if text == "<stop>" else [3]

    def decode(self, ids, **_kwargs):
        return "".join({4: "A", 7: "<", 8: "stop>", 9: "Z"}.get(token, "?")
                       for token in ids)


class _StopTokenSession:
    def __init__(self, tokens, *, streaming=True):
        self.tokens = list(tokens)
        self.streaming = streaming
        self.closed = 0
        self.stream_closed = False
        if not streaming:
            self.stream = None

    def generate(self, **_kwargs):
        return list(self.tokens)

    def stream(self, **_kwargs):
        try:
            yield from self.tokens
        finally:
            self.stream_closed = True

    def close(self):
        self.closed += 1


class _Service:
    def __init__(self, session):
        self.session = session

    def open_session(self):
        return self.session


class _StopOnce:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = 0

    def stop(self):
        self.calls += 1
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure


class _CloseRecorder:
    def __init__(self):
        self.calls = 0

    def close(self):
        self.calls += 1


class _CloseTrackingSource:
    def __init__(self, *, close_error=None):
        self.close_error = close_error
        self.close_calls = 0
        self._tokens = iter((4, 9))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._tokens)

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _SessionBackend(_CloseRecorder):
    def open_session(self, **_kwargs):
        return _Session()


class _FakeTelemetry:
    def __init__(self):
        self.calls = 0

    def stop(self):
        self.calls += 1


class _ConstructorEngine:
    def __init__(self, stop_failure):
        self.scheduler = type("Scheduler", (), {"preempt_one": lambda self: None})()
        self.stop_failure = stop_failure
        self.stop_calls = 0

    def start(self):
        return None

    def stop(self):
        self.stop_calls += 1
        if self.stop_failure is not None:
            failure, self.stop_failure = self.stop_failure, None
            raise failure


class _ConstructorMemoryManager:
    def __init__(self, start_failure):
        self.start_failure = start_failure
        self.stop_calls = 0

    def set_expert_store(self, _store):
        pass

    def set_kv_cache(self, _store):
        pass

    def set_preemption_callback(self, _callback):
        pass

    def start(self):
        raise self.start_failure

    def stop(self):
        self.stop_calls += 1


class _FailOnceOwner:
    def __init__(self):
        self.calls = 0

    def close(self):
        self.calls += 1
        if self.calls == 1:
            raise OSError("transient phase1 session close")


class ServingEngineLifecycleTests(unittest.TestCase):
    def test_lazy_generation_is_consumed_and_released_before_owner_close(self):
        events = []

        class LazySession:
            def generate(self, **_kwargs):
                try:
                    events.append('generate')
                    self.assert_open()
                    yield 65
                finally:
                    events.append('iterator-close')

            def assert_open(self):
                if 'session-close' in events:
                    raise RuntimeError('generation used a closed owner')

            def close(self):
                events.append('session-close')

        engine = ServingEngine(_Service(LazySession()))
        self.assertEqual(engine._run_generate([1], 1, None), [65])
        self.assertEqual(events, ['generate', 'iterator-close', 'session-close'])

    def test_lazy_generation_error_preserved_over_iterator_and_session_cleanup(self):
        primary = ValueError('lazy generation failed')
        events = []

        class Tokens:
            def __iter__(self):
                return self

            def __next__(self):
                raise primary

            def close(self):
                events.append('iterator-close')
                raise RuntimeError('iterator cleanup failed')

        class LazySession:
            def generate(self, **_kwargs):
                return Tokens()

            def close(self):
                events.append('session-close')
                raise RuntimeError('session cleanup failed')

        engine = ServingEngine(_Service(LazySession()))
        with self.assertRaises(ValueError) as caught:
            engine._run_generate([1], 1, None)
        self.assertIs(caught.exception, primary)
        self.assertEqual(events, ['iterator-close', 'session-close'])
        self.assertTrue(any('iterator cleanup failed' in note for note in primary.__notes__))
        self.assertTrue(any('session cleanup failed' in note for note in primary.__notes__))

    def test_generate_failure_preserved_when_close_property_raises(self):
        primary = ValueError('generation failed')

        class Session:
            def generate(self, **_kwargs):
                raise primary

            @property
            def close(self):
                raise RuntimeError('close lookup failed')

        engine = ServingEngine(_Service(Session()))
        with self.assertRaises(ValueError) as caught:
            engine._run_generate([1], 1, None)
        self.assertIs(caught.exception, primary)
        self.assertTrue(any('close lookup failed' in note for note in primary.__notes__))

    def test_streaming_owner_does_not_resolve_unused_generate_property(self):
        class StreamSession(_Session):
            @property
            def generate(self):
                raise RuntimeError('unused generate lookup')

        session = StreamSession()
        engine = ServingEngine(_Service(session))
        self.assertEqual(list(engine._run_stream([1], 1)), [65])
        self.assertEqual(session.closed, 1)

    def test_generated_iterables_are_not_coerced_to_boolean(self):
        class Tokens:
            def __bool__(self):
                raise ValueError('ambiguous token container truth value')

            def __iter__(self):
                return iter([65])

        class Session(_Session):
            stream = None

            def generate(self, **_kwargs):
                return Tokens()

        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                session = Session()
                engine = ServingEngine(_Service(session))
                tokens = (list(engine._run_stream([1], 1)) if streaming else
                          engine._run_generate([1], 1, None))
                self.assertEqual(tokens, [65])
                self.assertEqual(session.closed, 1)

    def test_successful_lazy_generation_cleanup_failure_still_closes_owner(self):
        cleanup_error = RuntimeError('lazy iterator close failed')

        class Tokens:
            def __iter__(self):
                return self

            def __next__(self):
                raise StopIteration

            def close(self):
                raise cleanup_error

        class Session(_Session):
            def generate(self, **_kwargs):
                return Tokens()

        session = Session(close_error=RuntimeError('owner close also failed'))
        engine = ServingEngine(_Service(session))
        with self.assertRaises(RuntimeError) as caught:
            engine._run_generate([1], 1, None)
        self.assertIs(caught.exception, cleanup_error)
        self.assertEqual(session.closed, 1)
        self.assertTrue(any('owner close also failed' in note for note in cleanup_error.__notes__))

    def test_generation_failure_still_closes_session_and_preserves_primary_error(self):
        primary = OSError("forward failed")
        session = _Session(generate_error=primary, close_error=RuntimeError("close failed"))
        engine = ServingEngine(_Service(session))

        response = engine.process_completion(CompletionRequest(model="m", prompt="p"))

        self.assertEqual(session.closed, 1)
        self.assertEqual(response.choices[0].finish_reason, "error")
        self.assertEqual(response.choices[0].text, "")
        self.assertTrue(getattr(primary, "__notes__", ()))

    def test_close_failure_after_success_is_not_silently_discarded(self):
        session = _Session(close_error=OSError("close failed"))
        engine = ServingEngine(_Service(session))

        response = engine.process_completion(CompletionRequest(model="m", prompt="p"))

        self.assertEqual(session.closed, 1)
        self.assertEqual(response.choices[0].finish_reason, "error")

    def test_completion_usage_counts_generated_ids_not_reencoded_text(self):
        completion_session = _StopTokenSession([4, 9])
        completion_engine = ServingEngine(
            _Service(completion_session), tokenizer=_StopTokenizer())
        completion = completion_engine.process_completion(CompletionRequest(
            model="m", prompt="p"))
        self.assertEqual(completion.usage.completion_tokens, 2)

        chat_session = _StopTokenSession([4, 9])
        chat_engine = ServingEngine(_Service(chat_session), tokenizer=_StopTokenizer())
        chat = chat_engine.process_chat_completion(ChatCompletionRequest(
            model="m", messages=[ChatMessage(role=Role.USER, content="hello")]))
        self.assertEqual(chat.usage.completion_tokens, 2)

    def test_configured_tokenizer_decode_failure_is_not_replaced_with_bytes(self):
        class BrokenDecodeTokenizer:
            def encode(self, _text, **_kwargs):
                return [3]

            def decode(self, _ids, **_kwargs):
                raise OSError("configured decode failed")

        tokenizer = BrokenDecodeTokenizer()
        session = _Session()
        engine = ServingEngine(_Service(session), tokenizer=tokenizer)
        response = engine.process_completion(CompletionRequest(model="m", prompt="p"))
        self.assertEqual(response.choices[0].finish_reason, "error")
        self.assertEqual(response.choices[0].text, "")
        self.assertEqual(session.closed, 1)

        stream_session = _Session()
        stream_engine = ServingEngine(_Service(stream_session), tokenizer=tokenizer)
        chunks = list(stream_engine.stream_completion(CompletionRequest(
            model="m", prompt="p", stream=True)))
        self.assertEqual(chunks[-1].finish_reason, "error")
        self.assertEqual(stream_session.closed, 1)

    def test_completion_echo_includes_prompt_without_changing_usage(self):
        session = _Session()
        engine = ServingEngine(_Service(session))
        request = CompletionRequest.from_dict({
            "model": "m", "prompt": "prompt:", "max_tokens": 4, "echo": True,
        })

        response = engine.process_completion(request)

        self.assertEqual(response.choices[0].text, "prompt:A")
        self.assertEqual(response.usage.prompt_tokens, len("prompt:"))
        self.assertEqual(response.usage.completion_tokens, 1)

    def test_streamed_completion_echo_is_first_delta_and_keeps_usage_out_of_prompt(self):
        session = _Session()
        engine = ServingEngine(_Service(session))
        request = CompletionRequest.from_dict({
            "model": "m", "prompt": "prompt:", "max_tokens": 4,
            "echo": True, "stream": True,
        })

        chunks = list(engine.stream_completion(request))

        self.assertEqual([chunk.delta_text for chunk in chunks[:-1]], ["prompt:", "A"])
        self.assertEqual(chunks[-1].finish_reason, "stop")
        self.assertEqual(session.closed, 1)

    def test_unsupported_logprobs_and_malformed_echo_fail_before_session_open(self):
        with self.assertRaisesRegex(ValueError, "echo must be a boolean"):
            CompletionRequest.from_dict({"prompt": "p", "echo": "false"})

        session = _Session()
        engine = ServingEngine(_Service(session))
        with self.assertRaisesRegex(ValueError, "logprobs are not supported"):
            engine.process_completion(CompletionRequest(
                model="m", prompt="p", logprobs=1,
            ))
        self.assertEqual(session.closed, 0)

    def test_sampling_controls_fail_closed_before_session_open(self):
        completion_inputs = (
            {"temperature": 0.7},
            {"top_p": 0.8},
            {"presence_penalty": 0.2},
            {"frequency_penalty": -0.2},
        )
        for options in completion_inputs:
            with self.subTest(options=options):
                session = _Session()
                engine = ServingEngine(_Service(session))
                with self.assertRaisesRegex(ValueError, "not wired to the native session"):
                    engine.process_completion(CompletionRequest.from_dict({
                        "model": "m", "prompt": "p", **options,
                    }))
                self.assertEqual(session.closed, 0)

        for options in ({"temperature": 0.7}, {"top_p": 0.8}):
            with self.subTest(chat_options=options):
                session = _Session()
                engine = ServingEngine(_Service(session))
                with self.assertRaisesRegex(ValueError, "not wired to the native session"):
                    engine.process_chat_completion(ChatCompletionRequest.from_dict({
                        "model": "m", "messages": [{"role": "user", "content": "p"}],
                        **options,
                    }))
                self.assertEqual(session.closed, 0)

        self.assertIsNone(CompletionRequest.from_dict({"prompt": "p"}).temperature)
        self.assertIsNone(ChatCompletionRequest.from_dict({"messages": []}).top_p)

    def test_stream_failure_emits_error_terminal_chunk_and_closes_session(self):
        session = _Session(stream_error=OSError("stream failed"))
        engine = ServingEngine(_Service(session))

        chunks = list(engine.stream_completion(
            CompletionRequest(model="m", prompt="p", stream=True)
        ))

        self.assertEqual(session.closed, 1)
        self.assertEqual(chunks[-1].finish_reason, "error")

    def test_abandoning_public_completion_and_chat_streams_closes_source(self):
        for is_chat in (False, True):
            with self.subTest(is_chat=is_chat):
                source = _CloseTrackingSource()
                engine = ServingEngine(_Service(_Session()), tokenizer=_StopTokenizer())
                engine._run_stream = lambda *_args, **_kwargs: source
                if is_chat:
                    response = engine.stream_chat_completion(ChatCompletionRequest(
                        model="m", messages=[ChatMessage(role=Role.USER, content="hello")],
                        max_tokens=4, stream=True,
                    ))
                else:
                    response = engine.stream_completion(CompletionRequest(
                        model="m", prompt="p", max_tokens=4, stream=True,
                    ))

                self.assertEqual(next(response).delta_text, "A")
                response.close()
                self.assertEqual(source.close_calls, 1)

    def test_inference_service_sse_wrappers_close_serving_stream_on_abandonment(self):
        for method_name, route in (
            ("stream_complete", {"prompt": "p"}),
            ("stream_chat", {"messages": [{"role": "user", "content": "p"}]}),
        ):
            with self.subTest(method=method_name):
                state = {"closed": False}

                def source():
                    try:
                        yield type("Chunk", (), {
                            "to_sse_line": lambda _self: "data: token",
                        })()
                    finally:
                        state["closed"] = True

                serving_stream = source()
                service = object.__new__(InferenceService)
                service._serving = type("Serving", (), {
                    "stream_completion": lambda _self, _request: serving_stream,
                    "stream_chat_completion": lambda _self, _request: serving_stream,
                })()

                response = getattr(service, method_name)(route)
                self.assertEqual(next(response), "data: token")
                response.close()
                self.assertTrue(state["closed"])

    def test_abandoned_stream_close_failure_does_not_mask_consumer_close(self):
        source = _CloseTrackingSource(close_error=OSError("source cleanup failed"))
        engine = ServingEngine(_Service(_Session()), tokenizer=_StopTokenizer())
        engine._run_stream = lambda *_args, **_kwargs: source
        response = engine.stream_completion(CompletionRequest(
            model="m", prompt="p", max_tokens=4, stream=True,
        ))

        self.assertEqual(next(response).delta_text, "A")
        response.close()
        self.assertEqual(source.close_calls, 1)

    def test_stream_method_lookup_failure_still_closes_session_and_preserves_error(self):
        primary = OSError("stream method lookup failed")

        class BrokenLookupSession(_Session):
            @property
            def stream(self):
                raise primary

        session = BrokenLookupSession(close_error=RuntimeError("close failed"))
        engine = ServingEngine(_Service(session))

        with self.assertRaises(OSError) as raised:
            list(engine._run_stream([1], 1))

        self.assertIs(raised.exception, primary)
        self.assertEqual(session.closed, 1)
        self.assertTrue(any("close failed" in note for note in primary.__notes__))

    def test_stream_iterator_close_lookup_failure_still_closes_session(self):
        class BrokenCloseIterator:
            def __iter__(self):
                return self

            def __next__(self):
                raise StopIteration

            @property
            def close(self):
                raise RuntimeError("iterator close lookup failed")

        class BrokenIteratorSession(_Session):
            def stream(self, **_kwargs):
                return BrokenCloseIterator()

        session = BrokenIteratorSession()
        engine = ServingEngine(_Service(session))

        with self.assertRaisesRegex(RuntimeError, "iterator close lookup failed"):
            list(engine._run_stream([1], 1))

        self.assertEqual(session.closed, 1)

    def test_completion_applies_encoded_stop_string_and_reports_stop(self):
        session = _StopTokenSession([4, 7, 8, 9])
        engine = ServingEngine(_Service(session), tokenizer=_StopTokenizer())

        response = engine.process_completion(CompletionRequest(
            model="m", prompt="p", max_tokens=8, stop="<stop>"
        ))

        self.assertEqual(response.choices[0].text, "A")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(session.closed, 1)
        self.assertTrue(session.stream_closed)

    def test_stream_withholds_stop_tokens_and_terminates_backend_iterator(self):
        session = _StopTokenSession([4, 7, 8, 9])
        engine = ServingEngine(_Service(session), tokenizer=_StopTokenizer())

        chunks = list(engine.stream_completion(CompletionRequest(
            model="m", prompt="p", max_tokens=8, stop=["<stop>"], stream=True
        )))

        self.assertEqual([chunk.delta_text for chunk in chunks[:-1]], ["A"])
        self.assertEqual(chunks[-1].finish_reason, "stop")
        self.assertTrue(session.stream_closed)
        self.assertEqual(session.closed, 1)

    def test_generate_only_backend_trims_encoded_stop_after_generation(self):
        session = _StopTokenSession([4, 7, 8, 9], streaming=False)
        engine = ServingEngine(_Service(session), tokenizer=_StopTokenizer())

        response = engine.process_completion(CompletionRequest(
            model="m", prompt="p", max_tokens=8, stop="<stop>"
        ))

        self.assertEqual(response.choices[0].text, "A")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(session.closed, 1)

    def test_chat_completion_applies_stop_string_and_reports_stop(self):
        session = _StopTokenSession([4, 7, 8, 9])
        engine = ServingEngine(_Service(session), tokenizer=_StopTokenizer())
        request = ChatCompletionRequest(
            model="m", messages=[ChatMessage(role=Role.USER, content="hello")],
            max_tokens=8, stop="<stop>",
        )

        response = engine.process_chat_completion(request)

        self.assertEqual(response.choices[0].message.content, "A")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(session.closed, 1)

    def test_invalid_stop_is_rejected_before_opening_backend(self):
        service = _Service(_Session())
        engine = ServingEngine(service, tokenizer=_StopTokenizer())

        with self.assertRaisesRegex(ValueError, "stop requires"):
            engine.process_completion(CompletionRequest(
                model="m", prompt="p", stop=[]
            ))
        self.assertEqual(service.session.closed, 0)

    def test_stop_tokenizer_failure_does_not_fall_back_to_byte_tokenization(self):
        class BrokenTokenizer:
            def encode(self, _text, *, add_special_tokens=False):
                raise RuntimeError("checkpoint tokenizer failed")

        service = _Service(_Session())
        engine = ServingEngine(service, tokenizer=BrokenTokenizer())

        with self.assertRaisesRegex(ValueError, "configured tokenizer failed"):
            engine.process_completion(CompletionRequest(
                model="m", prompt="p", stop="<stop>"
            ))
        self.assertEqual(service.session.closed, 0)

    def test_completion_and_stream_report_length_when_budget_is_exhausted(self):
        complete_session = _StopTokenSession([4, 7, 8, 9])
        complete_engine = ServingEngine(
            _Service(complete_session), tokenizer=_StopTokenizer()
        )
        complete = complete_engine.process_completion(CompletionRequest(
            model="m", prompt="p", max_tokens=4
        ))
        self.assertEqual(complete.choices[0].finish_reason, "length")

        stream_session = _StopTokenSession([4, 7, 8, 9])
        stream_engine = ServingEngine(
            _Service(stream_session), tokenizer=_StopTokenizer()
        )
        chunks = list(stream_engine.stream_completion(CompletionRequest(
            model="m", prompt="p", max_tokens=4, stream=True
        )))
        self.assertEqual(chunks[-1].finish_reason, "length")

    def test_success_after_previous_stream_error_has_normal_stop_reason(self):
        broken = _Session(stream_error=OSError("stream failed"))
        good = _Session()
        engine = ServingEngine(_Service(broken))
        list(engine.stream_completion(
            CompletionRequest(model="m", prompt="p", stream=True)
        ))
        engine._service.session = good

        chunks = list(engine.stream_completion(
            CompletionRequest(model="m", prompt="p", stream=True)
        ))

        self.assertEqual(chunks[-1].finish_reason, "stop")
        self.assertEqual(good.closed, 1)

    def test_chat_stream_failure_is_not_reported_as_success(self):
        session = _Session(stream_error=OSError("stream failed"))
        engine = ServingEngine(_Service(session))
        request = ChatCompletionRequest(
            model="m", messages=[ChatMessage(role=Role.USER, content="hello")], stream=True
        )

        chunks = list(engine.stream_chat_completion(request))

        self.assertEqual(chunks[-1].finish_reason, "error")
        self.assertEqual(session.closed, 1)

    def test_shutdown_timeout_retains_shared_stores_and_close_is_retryable(self):
        service = object.__new__(Phase2MultiSessionService)
        manager = _StopOnce()
        engine = _StopOnce(TimeoutError("worker still running"))
        expert = _CloseRecorder()
        kv = _CloseRecorder()
        service._close_lock = threading.Lock()
        service._lock = threading.Lock()
        service._closed = False
        service._handles_detached = False
        service._handles = {}
        service._memory_stopped = False
        service._engine_stopped = False
        service._expert_store_closed = False
        service._kv_cache_closed = False
        service._memory_manager = manager
        service._engine = engine
        service._expert_store = expert
        service._kv_cache = kv
        service.checkpoint_id = "checkpoint"

        with self.assertRaisesRegex(TimeoutError, "worker still running"):
            service.close()

        self.assertEqual(expert.calls, 0)
        self.assertEqual(kv.calls, 0)
        self.assertTrue(service._closed)

        service.close()

        self.assertEqual(manager.calls, 1)
        self.assertEqual(engine.calls, 2)
        self.assertEqual(expert.calls, 1)
        self.assertEqual(kv.calls, 1)

    def test_load_balancer_retries_a_backend_close_failure(self):
        backend = _CloseRecorder()
        backend.close = _StopOnce(OSError("backend close failed")).stop
        balancer = Phase3LoadBalancedService(health_interval_s=60)
        balancer.register_backend("primary", backend)

        with self.assertRaisesRegex(OSError, "backend close failed"):
            balancer.close()
        balancer.close()

        self.assertEqual(backend.close.__self__.calls, 2)

    def test_backend_lease_is_held_until_session_close(self):
        backend = _SessionBackend()
        balancer = Phase3LoadBalancedService(health_interval_s=60)
        balancer.register_backend("primary", backend)

        session = balancer.open_session(session_id="caller")
        self.assertEqual(balancer.metrics()["lb_instances"][0]["in_flight"], 1)
        self.assertEqual(session.generate(context_ids=[1], max_new_tokens=1), [65])
        session.close()

        self.assertEqual(balancer.metrics()["lb_instances"][0]["in_flight"], 0)
        balancer.close()

    def test_shutdown_timeout_keeps_active_backend_session_open(self):
        backend = _SessionBackend()
        balancer = Phase3LoadBalancedService(health_interval_s=60)
        balancer.register_backend("primary", backend)
        session = balancer.open_session(session_id="caller")

        with self.assertRaisesRegex(TimeoutError, "1 active session lease"):
            balancer.close(timeout=0.01)
        self.assertEqual(backend.calls, 0)

        session.close()
        balancer.close(timeout=0.1)
        self.assertEqual(backend.calls, 1)

    def test_shutdown_retries_a_session_cleanup_that_previously_failed(self):
        backend = _SessionBackend()
        session = _Session(close_error=OSError("transient session close"))
        backend.open_session = lambda **_kwargs: session
        balancer = Phase3LoadBalancedService(health_interval_s=60)
        balancer.register_backend("primary", backend)
        leased = balancer.open_session(session_id="caller")

        with self.assertRaisesRegex(OSError, "transient session close"):
            leased.close()
        session.close_error = None
        balancer.close(timeout=0.1)

        self.assertEqual(session.closed, 2)
        self.assertEqual(backend.calls, 1)

    def test_backend_removal_drains_active_sessions_and_retries(self):
        backend = _SessionBackend()
        balancer = Phase3LoadBalancedService(health_interval_s=60)
        balancer.register_backend("primary", backend)
        session = balancer.open_session(session_id="caller")

        with self.assertRaisesRegex(TimeoutError, "drain timed out"):
            balancer.deregister_backend("primary", timeout=0.01)
        self.assertEqual(backend.calls, 0)
        with self.assertRaisesRegex(RuntimeError, "No healthy backend"):
            balancer.open_session()

        session.close()
        balancer.deregister_backend("primary", timeout=0.1)
        self.assertEqual(backend.calls, 1)
        balancer.close()

    def test_duplicate_and_post_close_backend_registration_fail_closed(self):
        balancer = Phase3LoadBalancedService(health_interval_s=60)
        balancer.register_backend("primary", _CloseRecorder())
        with self.assertRaisesRegex(ValueError, "already registered"):
            balancer.register_backend("primary", _CloseRecorder())
        balancer.close()
        with self.assertRaisesRegex(RuntimeError, "is closed"):
            balancer.register_backend("late", _CloseRecorder())

    def test_backend_open_error_releases_session_lease(self):
        backend = _CloseRecorder()
        backend.open_session = lambda **_kwargs: (_ for _ in ()).throw(OSError("open failed"))
        balancer = Phase3LoadBalancedService(health_interval_s=60)
        balancer.register_backend("primary", backend)

        with self.assertRaisesRegex(OSError, "open failed"):
            balancer.open_session()

        self.assertEqual(balancer.metrics()["lb_instances"][0]["in_flight"], 0)
        balancer.close()

    def test_proxy_done_is_idempotent_and_cannot_release_another_callers_lease(self):
        balancer = Phase3LoadBalancedService(health_interval_s=60)
        balancer.register_backend("primary", _CloseRecorder())
        first = balancer.borrow()
        second = balancer.borrow()

        first.done(latency_ms=1)
        first.done(latency_ms=1)
        self.assertEqual(balancer.metrics()["lb_instances"][0]["in_flight"], 1)
        second.done(latency_ms=1)
        self.assertEqual(balancer.metrics()["lb_instances"][0]["in_flight"], 0)
        balancer.close()

    def test_inference_service_shutdown_is_retryable_and_does_not_hide_errors(self):
        service = object.__new__(InferenceService)
        telemetry = _FakeTelemetry()
        balancer = _StopOnce(OSError("backend close failed"))
        balancer.close = balancer.stop
        service._stop_lock = threading.Lock()
        service._lock = threading.Lock()
        service._stopped = False
        service._stopping = False
        service._telemetry_stopped = False
        service._balancer_stopped = False
        service._telemetry = telemetry
        service._balancer = balancer
        service._config = type("Config", (), {"model_name": "test"})()

        with self.assertRaisesRegex(OSError, "backend close failed"):
            service.stop()
        self.assertFalse(service._stopped)
        service.stop()

        self.assertTrue(service._stopped)
        self.assertEqual(telemetry.calls, 1)
        self.assertEqual(balancer.calls, 2)

    def test_phase2_partial_construction_retains_retryable_owner_cleanup(self):
        from openagent.adaptive_engine import multi_session_service as module

        expert = _CloseRecorder()
        kv = _CloseRecorder()
        construction_failure = OSError("memory manager start failed")
        engine = _ConstructorEngine(TimeoutError("batch worker still stopping"))
        manager = _ConstructorMemoryManager(construction_failure)

        with (
            patch.object(module, "open_shared_expert_store", return_value=expert),
            patch.object(module, "open_shared_kv_cache", return_value=kv),
            patch.object(module, "MultiSessionBatchEngine", return_value=engine),
            patch.object(module, "Phase2MemoryManager", return_value=manager),
        ):
            with self.assertRaisesRegex(OSError, "memory manager start failed") as raised:
                Phase2MultiSessionService("checkpoint", ".", lambda _batch: [])

        self.assertEqual(manager.stop_calls, 1)
        self.assertEqual(engine.stop_calls, 1)
        self.assertEqual(expert.calls, 0)
        self.assertEqual(kv.calls, 0)
        retry_cleanup = getattr(raised.exception, "cleanup")
        retry_cleanup()
        self.assertEqual(engine.stop_calls, 2)
        self.assertEqual(expert.calls, 1)
        self.assertEqual(kv.calls, 1)

    def test_phase2_service_keeps_session_and_stores_reachable_after_close_failure(self):
        service = object.__new__(Phase2MultiSessionService)
        service._lock = threading.Lock()
        service._close_lock = threading.Lock()
        service._closed = False
        service._memory_stopped = False
        service._engine_stopped = False
        service._expert_store_closed = False
        service._kv_cache_closed = False
        service._handles = {}
        service._memory_manager = _StopOnce()
        service._engine = _StopOnce()
        service._expert_store = _CloseRecorder()
        service._kv_cache = _CloseRecorder()
        service.checkpoint_id = "checkpoint"
        owner = _FailOnceOwner()
        handle = Phase2SessionHandle(service, "session", owner)
        service._handles[handle.session_id] = handle

        with self.assertRaisesRegex(OSError, "transient phase1 session close"):
            service.close()
        self.assertFalse(handle._closed)
        self.assertEqual(service._expert_store.calls, 0)
        self.assertEqual(service._kv_cache.calls, 0)

        service.close()

        self.assertTrue(handle._closed)
        self.assertEqual(owner.calls, 2)
        self.assertEqual(service._expert_store.calls, 1)
        self.assertEqual(service._kv_cache.calls, 1)


if __name__ == "__main__":
    unittest.main()
