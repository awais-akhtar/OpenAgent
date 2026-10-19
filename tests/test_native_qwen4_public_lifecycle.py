"""Public Qwen4 observer callback ownership and cleanup regressions."""
import ctypes as C
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from openagent.adaptive_engine.native_qwen4_public_adapters import (
    NativeQwen4PublicAdapters,
)


class NativeQwen4PublicLifecycleTests(unittest.TestCase):
    def test_checkpoint_change_blocks_public_native_registration(self):
        from openagent.adaptive_engine.native_qwen4_public_adapters import (
            _register_checkpoint_bound_model,
        )
        inventory=SimpleNamespace(assert_unchanged=Mock(side_effect=ValueError(
            'checkpoint changed since planning')))
        register=Mock(return_value=0)
        with self.assertRaisesRegex(ValueError,'checkpoint changed since planning'):
            _register_checkpoint_bound_model(inventory,register,'descriptor')
        register.assert_not_called()

    def owner(self, end):
        owner=object.__new__(NativeQwen4PublicAdapters)
        owner._lock=RLock();owner._closed=False;owner._model_id=1;owner._active_operation=False
        owner._observing=False;owner._pending_observation=None
        owner.layers=1;owner.rows=2;owner.identity='a'*64
        signature=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_uint32,C.c_uint64,C.c_uint64,C.c_uint64,
            C.POINTER(C.c_float),C.POINTER(C.c_int64),C.POINTER(C.c_float))
        callback={}
        def begin(_identity,pointer,_context):
            callback['function']=C.cast(pointer,signature)
            callback['identity']=_identity
            callback['pointer']=pointer
            return 0
        owner.native=SimpleNamespace(library=SimpleNamespace(
            oa_qwen4_public_observation_begin=Mock(side_effect=begin),
            oa_qwen4_public_observation_end=end,
            oa_qwen4_public_unregister=Mock(return_value=0)),close=Mock())
        owner.kernel=None;owner._kernel_release=None;owner._decoder=None
        return owner,callback

    @staticmethod
    def invoke(callback):
        raw=(C.c_float*2)(1.0,2.0)
        chosen=(C.c_int64*1)(0)
        scales=(C.c_float*1)(1.0)
        return callback['function'](None,0,1,2,1,raw,chosen,scales)

    def test_routing_observer_reentry_is_reported_and_scope_is_released(self):
        owner,callback=self.owner(Mock(return_value=0))

        def observe(*_values):
            owner.final_hidden(None,max_host_bytes=1024)

        with self.assertRaisesRegex(RuntimeError,'native routing observation failed'):
            with owner.routing_scope(observe,max_host_bytes=1024):
                self.assertEqual(self.invoke(callback),1)

        self.assertFalse(owner._observing)
        owner.close()

    def test_failed_scope_release_retains_callback_and_allows_owner_thread_retry(self):
        end=Mock(side_effect=[1,0])
        owner,callback=self.owner(end)
        with self.assertRaisesRegex(RuntimeError,'scope release failed'):
            with owner.routing_scope(lambda *_values:None,max_host_bytes=1024):
                self.assertEqual(self.invoke(callback),0)

        self.assertTrue(owner._observing)
        self.assertIsNotNone(owner._pending_observation)
        with self.assertRaisesRegex(RuntimeError,'idle open Qwen4 observation owner'):
            with owner.routing_scope(lambda *_values:None,max_host_bytes=1024):
                pass
        owner.close()
        self.assertIsNone(owner._pending_observation)
        self.assertFalse(owner._observing)
        self.assertTrue(owner._closed)

    def test_hidden_cancellation_callback_cannot_reenter_or_close_native_model(self):
        owner,_callback=self.owner(Mock(return_value=0))
        owner.transport_width=2;owner.hidden_size=2
        invoked=[]

        def hidden(*args):
            invoked.append(args[0])
            cancel=C.cast(args[6],C.CFUNCTYPE(C.c_int,C.c_void_p))
            self.assertEqual(cancel(None),1)
            return 0

        owner.native.library.oa_qwen4_public_hidden=Mock(side_effect=hidden)
        rejected=[]

        def cancellation():
            errors=[]
            try:
                owner.final_hidden(np.zeros((1,2),np.float32),max_host_bytes=1024)
            except RuntimeError as error:
                self.assertIn('operation reentry',str(error))
                errors.append(error)
            try:
                owner.close()
            except RuntimeError as error:
                self.assertIn('active native operation',str(error))
                errors.append(error)
            rejected.append(True)
            raise errors[0]

        with self.assertRaisesRegex(InterruptedError,'cancellation failed'):
            owner.final_hidden(np.zeros((1,2),np.float32),cancelled=cancellation,max_host_bytes=1024)

        self.assertEqual(invoked,[1])
        self.assertEqual(rejected,[True])
        self.assertFalse(owner._active_operation)
        unregister=owner.native.library.oa_qwen4_public_unregister
        owner.close()
        unregister.assert_called_once()


if __name__=='__main__':
    unittest.main()
