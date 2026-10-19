"""Qwen3.5 public handles remain bound to their admitted checkpoint."""
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from openagent.adaptive_engine.qwen35_public_adapters import NativeQwen35PublicAdapters


class Qwen35PublicSourceBindingTests(unittest.TestCase):
    def test_checkpoint_change_before_handle_publication_closes_resource(self):
        adapter=object.__new__(NativeQwen35PublicAdapters)
        adapter._source_snapshot=SimpleNamespace(assert_unchanged=Mock(
            side_effect=ValueError('checkpoint changed since planning')))
        adapter.lock=RLock()
        adapter.objects={}
        adapter.pending={}
        resource=SimpleNamespace(close=Mock())
        output=[None]

        with self.assertRaisesRegex(ValueError,'checkpoint changed since planning'):
            adapter._publish(resource,'engine',output)

        resource.close.assert_called_once_with()
        self.assertIsNone(output[0])
        self.assertEqual(adapter.objects,{})
        self.assertEqual(adapter.pending,{})

    def test_failed_cleanup_after_checkpoint_change_is_retained(self):
        adapter=object.__new__(NativeQwen35PublicAdapters)
        adapter._source_snapshot=SimpleNamespace(assert_unchanged=Mock(
            side_effect=ValueError('checkpoint changed since planning')))
        adapter.lock=RLock()
        adapter.objects={}
        adapter.pending={}
        cleanup=RuntimeError('native resource still active')
        resource=SimpleNamespace(close=Mock(side_effect=cleanup))

        with self.assertRaisesRegex(ValueError,'checkpoint changed since planning') as raised:
            adapter._publish(resource,'engine',[None])

        self.assertIn(id(resource),{-key for key in adapter.pending})
        self.assertIs(adapter.objects[-id(resource)][1],resource)
        self.assertIs(adapter.pending[-id(resource)],cleanup)
        self.assertTrue(any('cleanup retained' in note for note in raised.exception.__notes__))


if __name__=='__main__':
    unittest.main()
