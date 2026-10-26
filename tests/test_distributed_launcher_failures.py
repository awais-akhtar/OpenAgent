"""Fault tests for the owned distributed process-group lifecycle."""
from datetime import timedelta
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.distributed_launcher import (
    _worker,
    launch_distributed_execution,
)


class DistributedLauncherFailureTests(unittest.TestCase):
    def test_hostile_factory_cleanup_cannot_skip_group_teardown_or_mask_primary(self):
        class FactoryError(RuntimeError):
            @property
            def cleanup(self):
                raise LookupError('cleanup property failed')

            def add_note(self, _note):
                raise AssertionError('custom add_note must not run')

        primary = FactoryError('factory failed')
        destroyed = []

        def destroy(group=None):
            destroyed.append(group)
            raise OSError('group shutdown failed')

        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=object()),
            init_process_group=lambda **_kwargs: None,
            new_group=lambda **_kwargs: 'projection-group',
            destroy_process_group=destroy,
        )
        torch = ModuleType('torch')
        torch.__path__ = []
        torch.distributed = distributed

        def factory(*_args):
            raise primary

        with patch.dict(sys.modules, {'torch': torch, 'torch.distributed': distributed}):
            with self.assertRaises(FactoryError) as caught:
                _worker(0, 1, factory, lambda *_args: None,
                        'tcp://127.0.0.1:12345', 'gloo', (), 9.0, 4096, 1024)
        self.assertIs(caught.exception, primary)
        self.assertEqual(destroyed, ['projection-group', None])
        self.assertTrue(any('rank shutdown failures' in note for note in primary.__notes__))

    def test_launcher_rejects_timeouts_above_the_collective_bound(self):
        with self.assertRaisesRegex(ValueError, 'one second and 300 seconds'):
            launch_distributed_execution(
                lambda *_args: None,
                lambda *_args: None,
                world_size=1,
                init_method='tcp://127.0.0.1:12345',
                projection_backend='gloo',
                timeout_seconds=301)

    def test_worker_applies_timeout_to_both_groups_and_destroys_them_on_setup_failure(self):
        class Distributed:
            def __init__(self):
                self.group = SimpleNamespace(WORLD=object())
                self.calls = []

            def init_process_group(self, **kwargs):
                self.calls.append(('init', kwargs))

            def new_group(self, **kwargs):
                self.calls.append(('new_group', kwargs))
                return 'projection-group'

            def destroy_process_group(self, group=None):
                self.calls.append(('destroy', group))

        distributed = Distributed()
        torch = ModuleType('torch')
        torch.__path__ = []
        torch.distributed = distributed
        with patch.dict(sys.modules, {
            'torch': torch,
            'torch.distributed': distributed,
        }):
            with self.assertRaisesRegex(OSError, 'factory failed'):
                _worker(
                    0, 1,
                    lambda *_args: (_ for _ in ()).throw(OSError('factory failed')),
                    lambda *_args: None,
                    'tcp://127.0.0.1:12345', 'gloo', (), 9.0, 4096, 1024,
                    )

        timeout = timedelta(seconds=9)
        init = next(row[1] for row in distributed.calls if row[0] == 'init')
        group = next(row[1] for row in distributed.calls if row[0] == 'new_group')
        self.assertEqual(init['timeout'], timeout)
        self.assertEqual(group['timeout'], timeout)
        self.assertEqual(
            [row for row in distributed.calls if row[0] == 'destroy'],
            [('destroy', 'projection-group'), ('destroy', None)])


if __name__ == '__main__':
    unittest.main()
