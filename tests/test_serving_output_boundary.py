"""Direct serving rejects invalid admission and bounds backend output consumption."""
from itertools import count
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.serving_engine import (
    ChatCompletionRequest, ChatMessage, CompletionRequest, Role, ServingEngine,
)


class _Tokens:
    def __init__(self, values):
        self.values = iter(values)
        self.pulls = 0
        self.closed = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.pulls += 1
        return next(self.values)

    def close(self):
        self.closed += 1


class _Session:
    def __init__(self, tokens):
        self.tokens = tokens
        self.closed = 0

    def generate(self, **_kwargs):
        return self.tokens

    def stream(self, **_kwargs):
        return self.tokens

    def close(self):
        self.closed += 1


class _Service:
    def __init__(self, session):
        self.session = session
        self.opens = 0

    def open_session(self):
        self.opens += 1
        return self.session


class ServingOutputBoundaryTests(unittest.TestCase):
    @staticmethod
    def request(chat=False, **kwargs):
        if chat:
            return ChatCompletionRequest(model='m', messages=[ChatMessage(Role.USER, 'p')], **kwargs)
        return CompletionRequest(model='m', prompt='p', **kwargs)

    @staticmethod
    def run_request(engine, request, *, chat, streaming):
        if streaming:
            method = engine.stream_chat_completion if chat else engine.stream_completion
            return list(method(request))
        method = engine.process_chat_completion if chat else engine.process_completion
        return method(request)

    def test_unbounded_backend_consumed_only_to_requested_limit(self):
        for chat in (False, True):
            for streaming in (False, True):
                with self.subTest(chat=chat, streaming=streaming):
                    tokens = _Tokens(count(65))
                    session = _Session(tokens)
                    engine = ServingEngine(_Service(session))
                    result = self.run_request(engine, self.request(chat, max_tokens=2),
                                              chat=chat, streaming=streaming)
                    self.assertEqual(tokens.pulls, 2)
                    self.assertEqual(tokens.closed, 1)
                    self.assertEqual(session.closed, 1)
                    if streaming:
                        self.assertEqual(''.join(chunk.delta_text for chunk in result), 'AB')
                        self.assertEqual(result[-1].finish_reason, 'length')
                    else:
                        self.assertEqual(result.usage.completion_tokens, 2)
                        self.assertEqual(result.choices[0].finish_reason, 'length')

    def test_stop_prefix_at_output_limit_is_flushed_without_extra_pull(self):
        tokens = _Tokens([65, 66, 67])
        session = _Session(tokens)
        engine = ServingEngine(_Service(session))
        state = {}
        self.assertEqual(list(engine._run_stream([1], 2, stop_prefixes=((66, 67),),
                                                stop_state=state)), [65, 66])
        self.assertEqual(tokens.pulls, 2)
        self.assertFalse(state.get('matched', False))
        self.assertEqual(tokens.closed, 1)
        self.assertEqual(session.closed, 1)

    def test_invalid_counts_rejected_before_open_for_every_entrypoint(self):
        for field in ('max_tokens', 'n'):
            for value in (True, False, 0, -1, 1.5, '2', None, 2**31):
                for chat in (False, True):
                    for streaming in (False, True):
                        with self.subTest(field=field, value=value, chat=chat, streaming=streaming):
                            service = _Service(_Session(_Tokens([])))
                            engine = ServingEngine(service)
                            with self.assertRaises(ValueError):
                                self.run_request(engine, self.request(chat, **{field: value}),
                                                 chat=chat, streaming=streaming)
                            self.assertEqual(service.opens, 0)

    def test_wire_counts_are_not_silently_coerced(self):
        for request_type in (CompletionRequest, ChatCompletionRequest):
            for field in ('max_tokens', 'n'):
                for value in (True, 1.5, '2', 0, -1):
                    with self.subTest(request_type=request_type, field=field, value=value):
                        with self.assertRaises(ValueError):
                            request_type.from_dict({field: value})

    def test_wire_stream_flags_rejected_before_wrappers_can_override_them(self):
        for request_type in (CompletionRequest, ChatCompletionRequest):
            for value in ('false', 0, 1, None, []):
                with self.subTest(request_type=request_type, value=value):
                    with self.assertRaises(ValueError):
                        request_type.from_dict({'stream': value})

    def test_multi_choice_stream_and_nonboolean_stream_rejected_before_open(self):
        for chat in (False, True):
            for options in ({'n': 2}, {'stream': 'false'}, {'stream': 1}):
                with self.subTest(chat=chat, options=options):
                    service = _Service(_Session(_Tokens([])))
                    engine = ServingEngine(service)
                    with self.assertRaises(ValueError):
                        self.run_request(engine, self.request(chat, **options),
                                         chat=chat, streaming=True)
                    self.assertEqual(service.opens, 0)

    def test_invalid_backend_tokens_fail_and_close_both_owners(self):
        for token in (True, 1.5, '65', -1, 2**31):
            for streaming in (False, True):
                with self.subTest(token=token, streaming=streaming):
                    tokens = _Tokens([token])
                    session = _Session(tokens)
                    engine = ServingEngine(_Service(session))
                    response = self.run_request(engine, self.request(max_tokens=2),
                                                chat=False, streaming=streaming)
                    finish = response[-1].finish_reason if streaming else response.choices[0].finish_reason
                    self.assertEqual(finish, 'error')
                    self.assertEqual(tokens.closed, 1)
                    self.assertEqual(session.closed, 1)

    def test_unprintable_backend_errors_do_not_break_error_responses(self):
        class UnprintableError(RuntimeError):
            def __str__(self):
                raise RuntimeError('hostile exception formatter')

        class Session(_Session):
            def generate(self, **_kwargs):
                raise UnprintableError()

            def stream(self, **_kwargs):
                raise UnprintableError()

        def strict_log_handler(format_string, *args):
            # Exercise the formatting a synchronous log handler would perform.
            return format_string % args

        for chat in (False, True):
            for streaming in (False, True):
                with self.subTest(chat=chat, streaming=streaming):
                    session = Session(_Tokens([]))
                    engine = ServingEngine(_Service(session))
                    with patch('openagent.adaptive_engine.serving_engine.logger.warning',
                               side_effect=strict_log_handler):
                        result = self.run_request(engine, self.request(chat, max_tokens=2),
                                                  chat=chat, streaming=streaming)
                    finish = result[-1].finish_reason if streaming else result.choices[0].finish_reason
                    self.assertEqual(finish, 'error')
                    self.assertEqual(session.closed, 1)


if __name__ == '__main__':
    unittest.main()
