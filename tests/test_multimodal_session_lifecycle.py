"""Lifecycle ownership for service-driven multimodal checkpoint operations."""
from threading import Event, RLock, Thread
import unittest
import numpy as np

from openagent.adaptive_engine.multimodal_session import MultimodalSessionExecutor
from openagent.adaptive_engine.qwen4_multimodal import Qwen4MultimodalDecoder


class MultimodalSessionLifecycleTests(unittest.TestCase):
    @staticmethod
    def wrapper(decoder):
        value=object.__new__(MultimodalSessionExecutor)
        value.decoder=decoder;value._lifecycle=RLock();value._active_thread=None
        value._closed=False;value._close_complete=False
        return value

    def test_close_waits_for_owned_checkpoint_io(self):
        entered=Event();release=Event();closed=Event();close_done=Event();errors=[]

        class Decoder:
            def write_cache(self,_output,*,max_bytes):
                entered.set()
                if not release.wait(2):raise TimeoutError('test checkpoint writer was not released')
                return max_bytes
            def close(self):closed.set()

        wrapper=self.wrapper(Decoder())
        writer=Thread(target=lambda: wrapper.write_cache(object(),max_bytes=64))
        writer.start();self.assertTrue(entered.wait(1))
        def close():
            try:wrapper.close()
            except BaseException as error:errors.append(error)
            finally:close_done.set()
        closer=Thread(target=close);closer.start()
        self.assertFalse(close_done.wait(0.05))
        self.assertFalse(closed.is_set())
        release.set();writer.join(2);closer.join(2)
        self.assertFalse(writer.is_alive());self.assertFalse(closer.is_alive())
        self.assertEqual(errors,[]);self.assertTrue(closed.is_set());self.assertTrue(wrapper._closed)

    def test_checkpoint_callback_reentry_is_rejected_by_lifecycle_owner(self):
        class Decoder:
            def write_cache(self,_output,*,max_bytes,cancelled=None):
                return cancelled()
            def reset(self):return None

        wrapper=self.wrapper(Decoder())
        with self.assertRaisesRegex(RuntimeError,'conditional decoder operation reentered'):
            wrapper.write_cache(object(),max_bytes=64,cancelled=lambda:wrapper.reset())

    def test_cancellation_is_not_silently_dropped_for_unsupported_codec(self):
        class Decoder:
            def restore_cache(self,_source,*,max_bytes):return max_bytes

        wrapper=self.wrapper(Decoder())
        with self.assertRaisesRegex(ValueError,'lacks cooperative cancellation'):
            wrapper.restore_cache(object(),max_bytes=64,cancelled=lambda:False)

    def test_failed_close_blocks_requests_and_retries_underlying_cleanup(self):
        class Decoder:
            def __init__(self):self.close_calls=0
            def close(self):
                self.close_calls+=1
                if self.close_calls==1:raise RuntimeError('injected native close failure')

        decoder=Decoder();wrapper=self.wrapper(decoder)
        with self.assertRaisesRegex(RuntimeError,'injected native close failure'):
            wrapper.close()
        self.assertTrue(wrapper._closed)
        self.assertFalse(wrapper._close_complete)
        with self.assertRaisesRegex(RuntimeError,'executor is closed'):
            wrapper.snapshot()
        wrapper.close()
        self.assertEqual(decoder.close_calls,2)
        self.assertTrue(wrapper._close_complete)
        wrapper.close()
        self.assertEqual(decoder.close_calls,2)

    def test_nonfinite_processor_media_is_rejected_before_decoder_dispatch(self):
        class Decoder:
            def __init__(self):self.calls=0
            def forward(self,*_args,**_kwargs):
                self.calls+=1
                return np.zeros((1,1,2),dtype=np.float32)

        decoder=Decoder();wrapper=self.wrapper(decoder)
        wrapper._fields=frozenset(('pixel_values',));wrapper.max_media_bytes=1024
        for value in (np.array([np.nan],dtype=np.float32),
                      np.array([np.inf],dtype=np.float32)):
            with self.subTest(value=value[0]), self.assertRaisesRegex(
                    ValueError,'nonfinite processor media values'):
                wrapper.forward_media([1],{'pixel_values':value},None)
        self.assertEqual(decoder.calls,0)

    @staticmethod
    def production_wrapper(forward):
        decoder=object.__new__(Qwen4MultimodalDecoder)
        decoder.forward=forward
        decoder.close=lambda:None
        return MultimodalSessionExecutor(decoder,max_media_bytes=1024)

    def test_production_media_call_tracks_and_releases_successful_frame(self):
        wrapper=self.production_wrapper(
            lambda token_ids,**_kwargs: np.zeros((1,len(token_ids),2),dtype=np.float32))

        result=wrapper.forward_media([1],{'pixel_values':np.ones((2,),dtype=np.float32)},None)

        self.assertEqual(result.shape,(1,2))
        self.assertEqual(wrapper._media_lifecycle.active_count(),0)
        self.assertEqual(wrapper._media_lifecycle._frames,{})

    def test_production_media_failure_releases_error_frame(self):
        def fail(*_args,**_kwargs):
            raise RuntimeError('injected vision failure')
        wrapper=self.production_wrapper(fail)

        with self.assertRaisesRegex(RuntimeError,'injected vision failure'):
            wrapper.forward_media([1],{'pixel_values':np.ones((2,),dtype=np.float32)},None)

        self.assertEqual(wrapper._media_lifecycle.active_count(),0)
        self.assertEqual(wrapper._media_lifecycle._frames,{})

    def test_cancellation_observer_failure_does_not_leak_media_frame(self):
        class Cancellation:
            def __init__(self):
                self.fail_on_observation = False

            def is_set(self):
                if self.fail_on_observation:
                    raise LookupError('injected cancellation observer failure')
                return False

        cancellation=Cancellation()

        def fail(*_args,**_kwargs):
            cancellation.fail_on_observation=True
            raise RuntimeError('injected decoder failure')

        wrapper=self.production_wrapper(fail)
        with self.assertRaisesRegex(RuntimeError,'injected decoder failure') as raised:
            wrapper.forward_media([1],{'pixel_values':np.ones((2,),dtype=np.float32)},cancellation)

        self.assertTrue(any('media cancellation observer failed' in note
                            for note in getattr(raised.exception,'__notes__',())))
        self.assertEqual(wrapper._media_lifecycle.active_count(),0)
        self.assertEqual(wrapper._media_lifecycle._frames,{})

    def test_unprintable_cancelled_decoder_error_does_not_leave_active_frame(self):
        class UnprintableError(RuntimeError):
            def __str__(self):
                raise RuntimeError('exception formatting failed')

        cancellation = Event()
        primary = UnprintableError()

        def fail(*_args, **_kwargs):
            cancellation.set()
            raise primary

        wrapper = self.production_wrapper(fail)
        with self.assertRaises(UnprintableError) as caught:
            wrapper.forward_media([1], {'pixel_values': np.ones((2,), dtype=np.float32)}, cancellation)
        self.assertIs(caught.exception, primary)
        self.assertEqual(wrapper._media_lifecycle.active_count(), 0)
        self.assertEqual(wrapper._media_lifecycle._frames, {})


if __name__=='__main__':
    unittest.main()
