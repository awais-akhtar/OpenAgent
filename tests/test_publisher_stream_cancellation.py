"""Gateway streaming callback thread-ownership regressions."""
import threading
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.publisher_provider import NativePublisherProvider
from openagent.adaptive_engine.execution_service import ExecutionCancelled
from openagent.model_gateway import ModelResult,StreamChunk
from openagent.adaptive_engine.publisher_media_inputs import PublisherVideoInput
from openagent.adaptive_engine.k3_execution.video_input import K3VideoInput


class PublisherStreamCancellationTests(unittest.TestCase):
    def test_streamed_recovery_cancelled_before_first_advance_never_starts_worker(self):
        provider = NativePublisherProvider()
        observed = {"started": False}

        def resume(_source, *, emit, cancelled, **_options):
            observed["started"] = True
            while not cancelled():
                threading.Event().wait(0.01)
            raise ExecutionCancelled("recovery worker stopped")

        provider.resume_checkpoint = resume
        stream = provider.stream_resume_checkpoint(
            object(), checkpoint_max_bytes=1024, max_output_tokens=4,
            cancelled=lambda: True)

        with self.assertRaisesRegex(ExecutionCancelled, "request cancelled"):
            next(stream)

        self.assertFalse(observed["started"])
        self.assertTrue(provider._stream_lock.acquire(timeout=1))
        provider._stream_lock.release()

    def test_k3_media_freezer_adopts_its_cancelled_copy_without_second_constructor_copy(self):
        provider = NativePublisherProvider()
        source = K3VideoInput(
            (np.zeros((4, 5, 3), dtype=np.uint8),), (0.0,), 'k3-selected-frame')

        with patch.object(K3VideoInput, '__post_init__',
                          side_effect=AssertionError('public constructor copied frames again')):
            (frozen,) = provider._freeze_decoded_media((source,))

        self.assertIs(type(frozen), K3VideoInput)
        self.assertFalse(np.shares_memory(frozen.frames[0], source.frames[0]))
        self.assertFalse(frozen.frames[0].flags.writeable)
        self.assertEqual(frozen.timestamps, source.timestamps)

    def test_stream_media_freezer_adopts_its_cancelled_copy_without_second_constructor_copy(self):
        provider = NativePublisherProvider()
        source = PublisherVideoInput(
            (np.zeros((4, 5, 3), dtype=np.uint8),), (0,), 1, 30)

        with patch.object(PublisherVideoInput, '__post_init__',
                          side_effect=AssertionError('public constructor copied frames again')):
            (frozen,) = provider._freeze_decoded_media((source,))

        self.assertIs(type(frozen), PublisherVideoInput)
        self.assertFalse(np.shares_memory(frozen.frames[0], source.frames[0]))
        self.assertFalse(frozen.frames[0].flags.writeable)
        self.assertEqual(frozen.timestamps, source.timestamps)

    def test_user_cancellation_callback_is_never_polled_concurrently(self):
        provider=NativePublisherProvider()
        caller=threading.get_ident();checks=[]

        def cancelled():
            checks.append(threading.get_ident())
            return False

        def run(_messages,*,emit,cancelled,**_options):
            for index in range(8):
                emit(StreamChunk(provider.id,'model',str(index),False,{}))
                cancelled()
            return ModelResult(provider.id,'model','01234567',{})

        provider._run=run
        chunks=list(provider._stream_run(
            [{'role':'user','content':'prompt'}],model='model',cancelled=cancelled))
        self.assertEqual([chunk.content for chunk in chunks if not chunk.done],list(map(str,range(8))))
        self.assertTrue(chunks[-1].done)
        self.assertGreater(len(checks),1)
        self.assertEqual(set(checks),{caller})

    def test_consumer_cancellation_discards_buffered_stream_events(self):
        provider=NativePublisherProvider();checks=0

        def cancelled():
            nonlocal checks
            checks+=1
            # First poll is the pre-worker guard, second permits the initial
            # queued event, and the next poll cancels before another dispatch.
            return checks>=3

        def run(_messages,*,emit,**_options):
            for index in range(8):
                emit(StreamChunk(provider.id,'model',str(index),False,{}))
            return ModelResult(provider.id,'model','01234567',{})

        provider._run=run
        stream=provider._stream_run(
            [{'role':'user','content':'prompt'}],model='model',cancelled=cancelled)
        first=next(stream)
        self.assertEqual(first.content,'0')
        with self.assertRaises(ExecutionCancelled):
            next(stream)
        self.assertTrue(provider._stream_lock.acquire(timeout=1))
        provider._stream_lock.release()

    def test_transport_event_cancels_worker_while_consumer_is_suspended(self):
        provider = NativePublisherProvider()
        caller = threading.get_ident()
        worker_cancelled = threading.Event()
        callback_threads = []

        class TransportCancellation:
            def __init__(self):
                self.worker_event = threading.Event()

            def __call__(self):
                callback_threads.append(threading.get_ident())
                return self.worker_event.is_set()

        signal = TransportCancellation()

        def run(_messages, *, emit, cancelled, **_options):
            emit(StreamChunk(provider.id, 'model', 'first', False, {}))
            deadline = threading.Event()
            while not cancelled() and not deadline.wait(0.01):
                pass
            worker_cancelled.set()
            raise ExecutionCancelled('transport event reached backend worker')

        provider._run = run
        stream = provider._stream_run(
            [{'role': 'user', 'content': 'prompt'}], model='model', cancelled=signal)
        try:
            self.assertEqual(next(stream).content, 'first')
            # The consumer is paused at yield and cannot poll the callback.
            # The backend must observe only the transport-owned Event.
            signal.worker_event.set()
            self.assertTrue(worker_cancelled.wait(1))
            self.assertTrue(callback_threads)
            self.assertEqual(set(callback_threads), {caller})
        finally:
            stream.close()

    def test_stream_forwards_queue_timeout_to_execution_admission(self):
        provider = NativePublisherProvider()
        observed = {}

        def run(_messages, *, emit, queue_timeout, **_options):
            observed["queue_timeout"] = queue_timeout
            emit(StreamChunk(provider.id, "model", "answer", False, {}))
            return ModelResult(provider.id, "model", "answer", {})

        provider._run = run
        chunks = list(provider.stream_complete(
            [{"role": "user", "content": "prompt"}], model="model", queue_timeout=1.75))
        self.assertEqual(observed["queue_timeout"], 1.75)
        self.assertEqual([chunk.content for chunk in chunks], ["answer", ""])
        self.assertTrue(chunks[-1].done)


if __name__=='__main__':
    unittest.main()
