import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from openagent.adaptive_engine.publisher_provider import (
    NativePublisherProvider,
    ProviderConfigurationError,
)
from openagent.model_gateway import ProviderUnavailableError


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("stop text must not add special tokens")
        return _Encoding(self.mapping.get(text, []))

    def decode(self, ids, *, skip_special_tokens=True):
        return ''.join({4: "answer", 2: "ST", 3: "OP"}.get(token, "?") for token in ids)


class PublisherStopSequenceTests(unittest.TestCase):
    def test_v41_stop_prefix_matching_excludes_prompt_tokens(self):
        from openagent.adaptive_engine.publisher_v41_cli import (
            _finish_reason, _matches_generated_stop_prefix)

        self.assertFalse(_matches_generated_stop_prefix([8], ((7, 8),)))
        self.assertTrue(_matches_generated_stop_prefix([3, 7, 8], ((7, 8),)))
        self.assertEqual(_finish_reason([8], (), ((7, 8),)), 'length')
        self.assertEqual(_finish_reason([3, 7, 8], (), ((7, 8),)), 'stop_prefix')

    def test_recovery_stop_prefix_spanning_k3_restore_boundary_is_never_published(self):
        class RecoveryTokenizer:
            def decode(self, ids, *, skip_special_tokens=True):
                return ''.join({2: 'ST', 3: 'OP', 4: 'answer'}.get(token, '?') for token in ids)

        class Session:
            tokenizer = RecoveryTokenizer()

            def resume_generation(self, _source, *, emit, **_kwargs):
                emit({'event': 'recovered', 'token_ids': [2], 'text': 'ST',
                      'stop_prefixes': [[2, 3]]})
                emit({'event': 'tokens', 'token_ids': [3], 'text': 'STOP'})
                return {'generated_ids': [2, 3], 'resumed_ids': [3], 'text': 'STOP',
                        'stop_prefixes': [[2, 3]]}

            def close(self):
                pass

        provider = NativePublisherProvider()
        args = SimpleNamespace(max_tokens=8, stop_prefix=[])
        config = {'model': 'local', 'family': 'kimi_k3'}
        session = Session()
        emitted = []
        with (patch.object(provider, '_configuration', return_value=config),
              patch.object(provider, '_operator', return_value=(object(), args)),
              patch.object(provider, '_residency_key', return_value='recovery-key'),
              patch.object(provider, '_construct_session', return_value=session)):
            result = provider.resume_checkpoint(
                object(), checkpoint_max_bytes=1024, max_output_tokens=4,
                emit=emitted.append)

        self.assertEqual(result.content, '')
        self.assertEqual(emitted, [])
        provider.close()

    def test_recovery_stop_prefix_uses_token_ids_when_decoded_suffix_differs(self):
        class RecoveryTokenizer:
            def decode(self, ids, *, skip_special_tokens=True):
                # Contextual decoding means the stop's standalone spelling is
                # not a safe way to locate its boundary in streamed text.
                return {(): '', (4,): 'answer', (2,): 'S',
                        (3,): 'T', (4, 2, 3): 'answer<stop>'}.get(tuple(ids), '?')

        class Session:
            tokenizer = RecoveryTokenizer()

            def resume_generation(self, _source, *, emit, **_kwargs):
                emit({'event': 'recovered', 'token_ids': [], 'text': '',
                      'stop_prefixes': [[2, 3]]})
                emit({'event': 'tokens', 'token_ids': [4], 'text': 'answer'})
                emit({'event': 'tokens', 'token_ids': [2, 3], 'text': 'answer<stop>'})
                return {'generated_ids': [4, 2, 3], 'resumed_ids': [4, 2, 3],
                        'text': 'answer<stop>', 'stop_prefixes': [[2, 3]]}

            def close(self):
                pass

        provider = NativePublisherProvider()
        args = SimpleNamespace(max_tokens=8, stop_prefix=[])
        config = {'model': 'local', 'family': 'kimi_k3'}
        emitted = []
        with (patch.object(provider, '_configuration', return_value=config),
              patch.object(provider, '_operator', return_value=(object(), args)),
              patch.object(provider, '_residency_key', return_value='recovery-key-2'),
              patch.object(provider, '_construct_session', return_value=Session())):
            result = provider.resume_checkpoint(
                object(), checkpoint_max_bytes=1024, max_output_tokens=4,
                emit=emitted.append, stop_prefixes=((2, 3),))

        self.assertEqual(result.content, 'answer')
        self.assertEqual(''.join(chunk.content for chunk in emitted), 'answer')
        provider.close()

    def test_nonstreaming_recovery_still_validates_captured_token_events(self):
        class Session:
            tokenizer = _Tokenizer({})

            def resume_generation(self, _source, *, emit, **_kwargs):
                if emit is not None:
                    emit({'event': 'recovered', 'token_ids': [], 'text': '',
                          'stop_prefixes': []})
                    emit({'event': 'tokens', 'token_ids': [4], 'text': 'answer'})
                return {'generated_ids': [4], 'text': 'answer', 'stop_prefixes': []}

            def close(self):
                pass

        provider = NativePublisherProvider()
        args = SimpleNamespace(max_tokens=8, stop_prefix=[])
        config = {'model': 'local', 'family': 'mixtral'}
        with (patch.object(provider, '_configuration', return_value=config),
              patch.object(provider, '_operator', return_value=(object(), args)),
              patch.object(provider, '_residency_key', return_value='recovery-key-3'),
              patch.object(provider, '_construct_session', return_value=Session())):
            result = provider.resume_checkpoint(
                object(), checkpoint_max_bytes=1024, max_output_tokens=4)

        self.assertEqual(result.content, 'answer')
        provider.close()

    def test_v41_recovery_returns_only_new_ids_and_trims_boundary_crossing_stop(self):
        class Session:
            tokenizer = _Tokenizer({})

            def resume_generation(self, _source, *, emit, **_kwargs):
                self.assert_streaming_disabled = emit is None
                return {'generated_ids': [2, 3], 'resumed_ids': [3],
                        'recovered_token_count': 1, 'terminal_stop_start': 0,
                        'stop_prefixes': [[2, 3]], 'text': 'STOP'}

            def close(self):
                pass

        provider = NativePublisherProvider()
        args = SimpleNamespace(max_tokens=8, stop_prefix=[])
        config = {'model': 'local', 'family': 'deepseek_v41'}
        session = Session()
        with (patch.object(provider, '_configuration', return_value=config),
              patch.object(provider, '_operator', return_value=(object(), args)),
              patch.object(provider, '_residency_key', return_value='recovery-key-4'),
              patch.object(provider, '_construct_session', return_value=session)):
            result = provider.resume_checkpoint(
                object(), checkpoint_max_bytes=1024, max_output_tokens=4)

        self.assertTrue(session.assert_streaming_disabled)
        self.assertEqual(result.content, '')
        provider.close()

    def test_unrelated_lifecycle_caller_stays_blocked_by_stream_owner(self):
        provider = NativePublisherProvider()
        provider._stream_lock.acquire()
        try:
            with self.assertRaisesRegex(ProviderUnavailableError, "stream worker owns"):
                with provider._lifecycle_lock(None):
                    self.fail("unrelated caller entered an active stream")
        finally:
            provider._stream_lock.release()

    def test_text_stops_use_active_tokenizer_and_deduplicate_token_aliases(self):
        tokenizer = _Tokenizer({"END": [5, 6], "STOP": [5, 6], "\n\n": [7]})
        self.assertEqual(
            NativePublisherProvider._encode_stop_sequences(
                tokenizer, ("END", "STOP", "\n\n")),
            ((5, 6), (7,)),
        )

    def test_text_stop_that_cannot_be_encoded_is_rejected(self):
        with self.assertRaisesRegex(ProviderConfigurationError, "encode"):
            NativePublisherProvider._encode_stop_sequences(
                _Tokenizer({"EMPTY": []}), ("EMPTY",))

    def test_invalid_and_unbounded_stop_text_is_rejected(self):
        tokenizer = _Tokenizer({"ok": [1]})
        for value in (("",), ("\ud800",), tuple("s" for _ in range(5))):
            with self.subTest(value=value):
                with self.assertRaises(ProviderConfigurationError):
                    NativePublisherProvider._encode_stop_sequences(tokenizer, value)

    def test_tokenizer_cannot_return_out_of_range_or_oversized_prefix(self):
        for ids in ([2**31], [1] * 129, [True]):
            with self.subTest(ids=ids[:2]):
                with self.assertRaises(ProviderConfigurationError):
                    NativePublisherProvider._encode_stop_sequences(
                        _Tokenizer({"STOP": ids}), ("STOP",))

    def test_complete_installs_tokenized_stops_on_native_execution_args(self):
        class Session:
            tokenizer = _Tokenizer({"prompt": [1], "STOP": [2, 3]})

            def run(self, args, **_kwargs):
                self.stop_prefixes = args.stop_prefix
                emit = _kwargs.get("emit")
                if emit is not None:
                    for token in (4, 2, 3):
                        emit({"event": "tokens", "token_ids": [token]})
                return {"text": "answerSTOP", "generated_ids": [4, 2, 3],
                        "finish_reason": "stop"}

            def close(self):
                pass

        provider = NativePublisherProvider()
        args = SimpleNamespace(max_tokens=8, stop_prefix=["9,10", "02,3"])
        session = Session()
        config = {"model": "local", "family": "mixtral", "chat_template": "prompt",
                  "template_variables": {}}
        class Template:
            def render(self, **_kwargs):
                return "prompt"

        class Environment:
            def __init__(self, **_kwargs):
                self.globals = {}

            def from_string(self, _value):
                return Template()

        jinja = SimpleNamespace(StrictUndefined=object, TemplateError=RuntimeError)
        sandbox = SimpleNamespace(SandboxedEnvironment=Environment)
        with (patch.dict(sys.modules, {"jinja2": jinja, "jinja2.sandbox": sandbox}),
              patch.object(provider, "_configuration", return_value=config),
              patch.object(provider, "_operator", return_value=(object(), args)),
              patch.object(provider, "_residency_key", return_value="test-key"),
              patch.object(provider, "_construct_session", return_value=session)):
            result = provider.complete(
                [{"role": "user", "content": "hello"}],
                model="local", stop_sequences=("STOP",), queue_timeout=0)
            streamed = list(provider.stream_complete(
                [{"role": "user", "content": "hello"}],
                model="local", stop_sequences=("STOP",), queue_timeout=0))
        self.assertEqual(result.content, "answer")
        self.assertEqual(result.metadata["runtime"]["finish_reason"], "stop_sequence")
        self.assertEqual(result.metadata["runtime"]["stop_sequence"], "STOP")
        self.assertEqual(session.stop_prefixes, ["9,10", "2,3"])
        self.assertTrue(streamed[-1].done)
        self.assertTrue(streamed[-1].metadata["committed"])
        self.assertEqual(streamed[-1].metadata["runtime"]["finish_reason"], "stop_sequence")
        self.assertEqual(streamed[-1].metadata["runtime"]["stop_sequence"], "STOP")
        self.assertEqual(''.join(chunk.content for chunk in streamed if not chunk.done), "answer")
        provider.close()


if __name__ == "__main__":
    unittest.main()
