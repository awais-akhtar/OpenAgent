import gc
import threading
import unittest
from io import BytesIO
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import numpy as np

from openagent.adaptive_engine.native_conditional_cli import PublisherSession
from openagent.adaptive_engine.failure_reporting import failure_detail, retain_cleanup_failure
from openagent.adaptive_engine.inkling_multimodal import InklingMultimodalDecoder
from openagent.adaptive_engine.native_family_cli import PublisherSession as NativeFamilySession
from openagent.adaptive_engine.publisher_media_processor import PublisherMediaProcessor
from openagent.adaptive_engine.publisher_k3_cli import PublisherSession as K3PublisherSession
from openagent.adaptive_engine.owned_resources import OwnedResources
from openagent.adaptive_engine.publisher_v41_cli import (
    PublisherSession as V41PublisherSession,
    _run_request as run_v41_request,
)


class _Owner:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = 0

    def close(self):
        self.calls += 1
        if self.fail:
            self.fail = False
            raise RuntimeError('injected close failure')


class _UnprintableFailure(RuntimeError):
    def __repr__(self):
        raise RuntimeError('repr must not be called')

    def __str__(self):
        raise RuntimeError('str must not be called')


class ConditionalSessionCleanupTest(unittest.TestCase):
    def test_user_facing_failure_detail_is_bounded_and_unprintable_safe(self):
        self.assertEqual(failure_detail(ValueError('useful diagnostic')),
                         'useful diagnostic')
        self.assertEqual(failure_detail(_UnprintableFailure()),
                         '_UnprintableFailure: failure details omitted')
        self.assertEqual(len(failure_detail(ValueError('x' * 32), limit=8)), 8)

    def test_conditional_close_safely_records_unprintable_secondary_cleanup(self):
        session = object.__new__(PublisherSession)
        session._lock = threading.RLock()
        session._request_active = False
        session._closed = False

        class Owner:
            def __init__(self, failure):
                self.failure = failure
                self.calls = 0

            def close(self):
                self.calls += 1
                raise self.failure

        first = Owner(ValueError('first owner failed'))
        second = Owner(_UnprintableFailure())
        session.service = first
        session.processor = second

        with self.assertRaisesRegex(ValueError, 'first owner failed') as caught:
            session.close()

        self.assertEqual((first.calls, second.calls), (1, 1))
        self.assertTrue(any('_UnprintableFailure: failure details omitted' in note
                            for note in caught.exception.__notes__))

    def test_cleanup_retry_registration_never_formats_failure_objects(self):
        primary = ValueError('construction failed')
        retry = lambda: None
        failure = _UnprintableFailure()
        retain_cleanup_failure(primary, retry, failure, 'cleanup retained: ')
        self.assertIs(primary.cleanup, retry)
        self.assertTrue(any('_UnprintableFailure: failure details omitted' in note
                            for note in primary.__notes__))

    def test_v41_reset_preserves_first_owner_error_if_later_failures_are_unprintable(self):
        class ResetOwner:
            def __init__(self, failure):
                self.failure = failure

            def reset(self):
                raise self.failure

        class CleanupOwner:
            def close(self):
                raise _UnprintableFailure()

        session = object.__new__(V41PublisherSession)
        session.model = ResetOwner(ValueError('target reset failed'))
        session._draft_model = ResetOwner(_UnprintableFailure())
        session._owned = CleanupOwner()
        session._closed = False

        with self.assertRaisesRegex(ValueError, 'target reset failed') as caught:
            session._reset_request()

        self.assertTrue(session._closed)
        notes = getattr(caught.exception, '__notes__', ())
        self.assertEqual(len(notes), 2)
        self.assertTrue(any('additional publisher request reset failure' in note
                            for note in notes))
        self.assertTrue(any('publisher owner cleanup after failed request reset' in note
                            for note in notes))

    def test_v41_generation_reports_actual_prompt_count_and_eos_reason(self):
        class Tokenizer:
            def encode(self,_text):
                return SimpleNamespace(ids=[0, 2])
            def decode(self,tokens):
                return ''.join(str(value) for value in tokens)

        class Model:
            implementation='fixture-v41'
            def forward(self,tokens,**_kwargs):
                count=len(tokens)
                return np.tile(np.asarray([[0.0, 1.0, 0.0]],dtype=np.float32),(count,1))

        args=SimpleNamespace(
            temperature=0.0,seed=0,dspark=False,prompt='fixture',token_ids=None,
            max_tokens=4,context=16,sampling_mib=1,image=[],stop_prefix=[],
            prefill_chunk=8,checkpoint_mib=None)
        session=SimpleNamespace(
            tokenizer=Tokenizer(),checkpoint=SimpleNamespace(config={
                'text_config':{'vocab_size':3,'eos_token_id':1}}),
            arithmetic=object(),model=Model(),load_ms=0.0)

        result=run_v41_request(session,args)

        self.assertEqual(result['prompt_tokens'],2)
        self.assertEqual(result['expanded_prompt_tokens'],2)
        self.assertEqual(result['generated_ids'],[1])
        self.assertEqual(result['finish_reason'],'eos')

    def test_inkling_cache_callbacks_cannot_reenter_the_conditional_owner(self):
        decoder=object.__new__(InklingMultimodalDecoder)
        decoder._lock=threading.RLock();decoder._busy=False;decoder._closed=False

        class Continuation:
            def write_cache(self,_output,*,max_bytes,cancelled):
                return cancelled()
            def restore_cache(self,_source,*,max_bytes,cancelled):
                return cancelled()

        decoder.continuation=Continuation()
        cancel=lambda: decoder.reset()
        with self.assertRaisesRegex(RuntimeError,'closed or executing'):
            decoder.write_cache(BytesIO(),max_bytes=1024,cancelled=cancel)
        self.assertFalse(decoder._busy)
        with self.assertRaisesRegex(RuntimeError,'closed or executing'):
            decoder.restore_cache(BytesIO(),max_bytes=1024,cancelled=cancel)
        self.assertFalse(decoder._busy)

    def test_conditional_cli_releases_decoded_media_after_processor_failure(self):
        session = object.__new__(PublisherSession)
        session._lock = threading.RLock()
        session._request_active = False
        session._closed = False
        session._key = ()
        session._artifacts_unchanged = lambda: None
        session.vocabulary = 16
        session.stops = ()
        session.service = SimpleNamespace(limits=SimpleNamespace(max_new_tokens=8))
        image = np.ones((2, 3, 3), dtype=np.uint8)
        captured = {}

        def fail(_text, *, images, audio, videos, **_kwargs):
            captured['images'] = images
            captured['audio'] = audio
            captured['videos'] = videos
            raise RuntimeError('injected conditional processor failure')

        session.processor = SimpleNamespace(prepare=fail)
        args = SimpleNamespace(
            token_ids='[1]', image=[image], image_mib=1, media_mib=1,
            audio_wav=[], video_manifest=[], family='qwen4_exp',
            include_vision=True, prompt_text='describe this', temperature=0.0,
            top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0,
            frequency_penalty=0.0, presence_penalty=0.0, seed=0,
            stop_token_id=[], max_tokens=2, stop_prefix=[],
            restore_cache=None, save_cache=None, continue_session=False,
        )
        with self.assertRaisesRegex(RuntimeError, 'injected conditional processor failure'):
            session.run(args)

        self.assertEqual(captured['images'], [])
        self.assertEqual(captured['audio'], [])
        self.assertEqual(captured['videos'], [])
        self.assertTrue(np.all(image == 1))

    def test_conditional_cli_releases_prepared_media_after_service_failure(self):
        session = object.__new__(PublisherSession)
        session._lock = threading.RLock()
        session._request_active = False
        session._closed = False
        session._key = ()
        session._artifacts_unchanged = lambda: None
        session.vocabulary = 16
        session.stops = ()
        session.session_id = 'session'
        session.checkpoint_identity = 'fixture'
        retained = {}

        def generate(*_args, media=None, **_kwargs):
            retained['media'] = media
            raise RuntimeError('injected generation failure')

        session.service = SimpleNamespace(
            limits=SimpleNamespace(max_new_tokens=8), reset=lambda _id: None,
            generate=generate,
        )
        payload = {'pixel_values': np.ones((2, 2), dtype=np.float32)}
        session.processor = SimpleNamespace(
            prepare=lambda *_args, **_kwargs: ([1, 2], payload))
        image = np.ones((2, 3, 3), dtype=np.uint8)
        args = SimpleNamespace(
            token_ids='[1]', image=[image], image_mib=1, media_mib=1,
            audio_wav=[], video_manifest=[], family='qwen4_exp',
            include_vision=True, prompt_text='describe this', temperature=0.0,
            top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0,
            frequency_penalty=0.0, presence_penalty=0.0, seed=0,
            stop_token_id=[], max_tokens=2, stop_prefix=[],
            restore_cache=None, save_cache=None, continue_session=False,
        )

        with self.assertRaisesRegex(RuntimeError, 'injected generation failure'):
            session.run(args)

        self.assertEqual(retained['media'], {})

    def test_conditional_cli_releases_decoded_audio_when_later_admission_fails(self):
        session = object.__new__(PublisherSession)
        session._lock = threading.RLock()
        session._request_active = False
        session._closed = False
        session._key = ()
        session._artifacts_unchanged = lambda: None
        captured = {}

        def read_audio(*_args, **_kwargs):
            waveform = np.ones(64, dtype=np.float32)
            captured['waveform'] = weakref.ref(waveform)
            return waveform, 16000

        args = SimpleNamespace(
            token_ids='[1]', image=[], image_mib=1, media_mib=1,
            audio_wav=['audio.wav'], video_manifest=[], family='inkling_mm_model',
            max_audio_samples=1024, include_vision=False, prompt_text=None,
        )
        with patch('openagent.adaptive_engine.publisher_media_files.read_pcm_audio',
                   side_effect=read_audio):
            with self.assertRaisesRegex(ValueError, 'raw media requires explicit media admission'):
                session.run(args)

        gc.collect()
        self.assertIsNone(captured['waveform']())
        self.assertIsNone(session._request_media_cleanup)

    def test_processor_releases_owned_audio_staging_after_failure(self):
        processor = object.__new__(PublisherMediaProcessor)
        processor._lock = threading.RLock()
        processor._closed = False
        processor.family = 'inkling_mm_model'
        processor.max_bytes = 1024 * 1024
        processor.max_audio_samples = 4096
        processor.processor = SimpleNamespace()
        processor._unchanged = lambda: None
        captured = {}

        def fail(**kwargs):
            captured['kwargs'] = kwargs
            captured['waveforms'] = kwargs['audio']
            raise RuntimeError('injected media processor failure')

        processor.processor = fail
        source = np.ones(2048, dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, 'injected media processor failure'):
            processor.prepare('transcribe this', audio=[source], sampling_rate=16000)

        self.assertEqual(captured['kwargs']['audio'], [])
        self.assertEqual(captured['waveforms'], [])
        self.assertTrue(np.all(source == 1))

    def test_v41_speculative_stream_requires_transactional_event_owner(self):
        args = SimpleNamespace(temperature=0.0, seed=0, dspark=True)
        events = []

        with self.assertRaisesRegex(ValueError, 'transactional event sink'):
            run_v41_request(object(), args, emit=events.append)

        self.assertEqual(events, [])

    def test_v41_speculative_recovery_stream_rejects_before_cache_restore(self):
        session = object.__new__(V41PublisherSession)
        session._lock = threading.Lock()
        session._closed = False
        session._admitted_args = SimpleNamespace(max_tokens=1, context=8, sampling_mib=1)
        session._tokenizer_identity = 'tokenizer'
        session.model = SimpleNamespace(cancellation_scope=lambda _request: nullcontext())
        session.arithmetic = SimpleNamespace(cancellation_scope=lambda _request: nullcontext())
        session._reset_request = lambda: None
        cache = SimpleNamespace(closed=False, close=lambda: setattr(cache, 'closed', True))
        header = {'dspark': True}
        events = []

        with patch('openagent.adaptive_engine.v41_generation_recovery.read',
                   return_value=(header, None, cache, 0)):
            with self.assertRaisesRegex(ValueError, 'transactional event sink'):
                session.resume_generation(BytesIO(), max_bytes=64, emit=events.append)

        self.assertTrue(cache.closed)
        self.assertEqual(events, [])

    def test_processor_cleanup_runs_and_failed_service_is_retryable(self):
        session = object.__new__(PublisherSession)
        session._lock = threading.RLock()
        session._request_active = False
        session._closed = False
        session.session_id = 'session'
        session.tokenizer = object()
        session.service = _Owner(fail=True)
        session.processor = _Owner()
        service, processor = session.service, session.processor

        with self.assertRaisesRegex(RuntimeError, 'injected close failure'):
            session.close()

        self.assertEqual(service.calls, 1)
        self.assertIs(session.service, service)
        self.assertEqual(processor.calls, 1)
        self.assertIsNone(session.processor)
        self.assertTrue(session._closed)

        session.close()
        self.assertEqual(service.calls, 2)
        self.assertIsNone(session.service)
        self.assertIsNone(session.tokenizer)
        self.assertIsNone(session.session_id)
        self.assertTrue(session._closed)

    def test_media_processor_close_invokes_and_retains_underlying_owner(self):
        owner = _Owner(fail=True)
        processor = object.__new__(PublisherMediaProcessor)
        processor._lock = threading.RLock()
        processor._closed = False
        processor.processor = owner
        processor.tokenizer = SimpleNamespace(tokenizer=owner)

        with self.assertRaisesRegex(RuntimeError, 'injected close failure'):
            processor.close()

        self.assertIs(processor.processor, owner)
        self.assertFalse(processor._closed)
        processor.close()
        self.assertEqual(owner.calls, 2)
        self.assertIsNone(processor.processor)
        self.assertIsNone(processor.tokenizer)
        self.assertTrue(processor._closed)

    def test_native_family_owner_is_closed_to_requests_before_cleanup_retry(self):
        session = object.__new__(NativeFamilySession)
        session._lock = threading.RLock()
        session._request_active = False
        session._closed = False
        session.service = _Owner(fail=True)
        session.session_id = 'session'
        session.tokenizer = object()
        service = session.service

        with self.assertRaisesRegex(RuntimeError, 'injected close failure'):
            session.close()

        self.assertTrue(session._closed)
        self.assertIs(session.service, service)
        self.assertEqual(service.calls, 1)
        session.close()
        self.assertEqual(service.calls, 2)
        self.assertIsNone(session.service)
        self.assertIsNone(session.session_id)
        self.assertIsNone(session.tokenizer)

    def test_k3_warm_session_is_closed_when_owned_cleanup_needs_retry(self):
        session = object.__new__(K3PublisherSession)
        session._lock = threading.Lock()
        session._closed = False
        session._owned = OwnedResources()
        session._model = object()
        session._tokenizer = object()
        session._checkpoint = object()
        owner = _Owner(fail=True)
        session._owned.callback(owner.close)

        with self.assertRaisesRegex(RuntimeError, 'injected close failure'):
            session.close()

        self.assertTrue(session._closed)
        self.assertEqual(session._owned.pending, 1)
        self.assertIsNotNone(session._model)
        session.close()
        self.assertEqual(owner.calls, 2)
        self.assertEqual(session._owned.pending, 0)
        self.assertIsNone(session._model)
        self.assertIsNone(session._tokenizer)
        self.assertIsNone(session._checkpoint)

    def test_v41_retains_shared_dependencies_until_warm_borrower_closes(self):
        session = object.__new__(V41PublisherSession)
        session._lock = threading.Lock()
        session._closed = False
        session._draft_model = _Owner(fail=True)
        session._vision_model = _Owner()
        session._owned = OwnedResources()
        dependency = _Owner()
        session._owned.callback(dependency.close)
        draft, vision = session._draft_model, session._vision_model

        with self.assertRaisesRegex(RuntimeError, 'warm-owner cleanup remains pending'):
            session.close()

        self.assertTrue(session._closed)
        self.assertIs(session._draft_model, draft)
        self.assertIsNone(session._vision_model)
        self.assertEqual(dependency.calls, 0)

        session.close()
        self.assertEqual(draft.calls, 2)
        self.assertEqual(vision.calls, 1)
        self.assertEqual(dependency.calls, 1)
        self.assertIsNone(session._draft_model)
        self.assertEqual(session._owned.pending, 0)

    def test_invalid_conditional_stop_prefix_does_not_reset_session(self):
        session = object.__new__(PublisherSession)
        session._lock = threading.RLock()
        session._request_active = False
        session._closed = False
        session._key = ()
        session._artifacts_unchanged = lambda: None
        session.vocabulary = 32
        session.stops = ()
        session.session_id = 'session'
        state_changes = []
        session.service = SimpleNamespace(
            limits=SimpleNamespace(max_new_tokens=8),
            reset=lambda session_id: state_changes.append(('reset', session_id)))
        args = SimpleNamespace(
            token_ids='[1]', image=[], audio_wav=[], video_manifest=[], family='qwen4_exp',
            include_vision=False, prompt_text=None, temperature=0.0, top_k=0, top_p=1.0,
            min_p=0.0, repetition_penalty=1.0, frequency_penalty=0.0,
            presence_penalty=0.0, seed=0, restore_cache=None, save_cache=None,
            continue_session=False, stop_prefix=['1,,2'], stop_token_id=[],
            max_tokens=4)

        with self.assertRaisesRegex(ValueError, 'stop prefixes require'):
            session.run(args)

        self.assertEqual(state_changes, [])
        args.stop_prefix = []
        args.stop_token_id = [32]
        with self.assertRaisesRegex(ValueError, 'stop-token inventory'):
            session.run(args)
        self.assertEqual(state_changes, [])


if __name__ == '__main__':
    unittest.main()
