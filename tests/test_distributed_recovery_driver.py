from pathlib import Path
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.distributed_recovery import (
    RankRecoveryCheckpoint,
    _RecoveryDriver,
)
from openagent.adaptive_engine.distributed_launcher import _cleanup_failure_message


class _Coordinator:
    def __init__(self, *, fail_load=None, fail_close=()):
        self.sessions = []
        self.operations = []
        self.closed = []
        self.fail_load = fail_load
        self.fail_close = set(fail_close)

    def open_session(self):
        session = object()
        self.sessions.append(session)
        self.operations.append(('open', session))
        return session

    def load_session_cache(self, session, path, *, max_rank_bytes):
        self.operations.append(('load', session, path, max_rank_bytes))
        if len([row for row in self.operations if row[0] == 'load']) == self.fail_load:
            raise OSError('injected rank restore failure')

    def close_session(self, session):
        self.operations.append(('close', session))
        if session in self.fail_close:
            raise OSError('injected rank cleanup failure')
        self.closed.append(session)


class DistributedRecoveryDriverTests(unittest.TestCase):
    def test_rank_teardown_summary_does_not_stringify_arbitrary_failures(self):
        class Unprintable(Exception):
            def __str__(self):
                raise AssertionError('teardown must preserve the primary failure')

        self.assertEqual(_cleanup_failure_message((Unprintable(), OSError('hidden'))),
                         'Unprintable: failure details omitted; OSError: failure details omitted')

    def _checkpoints(self):
        return (
            RankRecoveryCheckpoint('first', 'unused', 'a' * 64, 1024),
            RankRecoveryCheckpoint('second', 'unused', 'b' * 64, 1024),
        )

    def test_application_driver_runs_only_after_all_checkpoint_restores(self):
        coordinator = _Coordinator()
        checkpoints = self._checkpoints()
        driver_calls = []
        driver = lambda owner, restored, attempt: (
            driver_calls.append((owner, restored, attempt)), 'result')[1]
        with patch.object(RankRecoveryCheckpoint, 'admit',
                          side_effect=((Path('first.cache'), {}),
                                       (Path('first.cache'), {}),
                                       (Path('second.cache'), {}),
                                       (Path('second.cache'), {}))) as admit:
            result = _RecoveryDriver(checkpoints, driver, 1, 3)(coordinator)

        self.assertEqual(result, 'result')
        self.assertEqual(len(driver_calls), 1)
        owner, restored, attempt = driver_calls[0]
        self.assertIs(owner, coordinator)
        self.assertEqual(attempt, 3)
        self.assertEqual(dict(restored), {'first': coordinator.sessions[0],
                                         'second': coordinator.sessions[1]})
        self.assertEqual([row[0] for row in coordinator.operations],
                         ['open', 'load', 'open', 'load'])
        self.assertEqual(admit.call_count, 4)  # pin recheck after each restore

    def test_restore_failure_closes_every_opened_session_in_reverse_order(self):
        coordinator = _Coordinator(fail_load=2)
        checkpoints = self._checkpoints()
        driver_calls = []
        driver = lambda *_args: driver_calls.append(True)
        with patch.object(RankRecoveryCheckpoint, 'admit',
                          side_effect=((Path('first.cache'), {}),
                                       (Path('first.cache'), {}),
                                       (Path('second.cache'), {}),
                                       (Path('second.cache'), {}))):
            with self.assertRaisesRegex(OSError, 'injected rank restore failure'):
                _RecoveryDriver(checkpoints, driver, 1, 0)(coordinator)

        self.assertEqual(driver_calls, [])
        self.assertEqual(coordinator.closed, list(reversed(coordinator.sessions)))
        self.assertEqual([row[0] for row in coordinator.operations],
                         ['open', 'load', 'open', 'load', 'close', 'close'])

    def test_cleanup_failure_is_attached_without_skipping_other_sessions(self):
        coordinator = _Coordinator(fail_load=2, fail_close=())
        original_close = coordinator.close_session

        def close_with_one_failure(session):
            if len(coordinator.sessions) >= 2 and session is coordinator.sessions[1]:
                coordinator.operations.append(('close', session))
                raise OSError('injected rank cleanup failure')
            return original_close(session)

        coordinator.close_session = close_with_one_failure
        checkpoints = self._checkpoints()
        driver = lambda *_args: self.fail('application work must not start')
        with patch.object(RankRecoveryCheckpoint, 'admit',
                          side_effect=((Path('first.cache'), {}),
                                       (Path('first.cache'), {}),
                                       (Path('second.cache'), {}),
                                       (Path('second.cache'), {}))):
            with self.assertRaisesRegex(OSError, 'injected rank restore failure') as caught:
                _RecoveryDriver(checkpoints, driver, 1, 0)(coordinator)

        # The failed handle is retained for replacement-owner teardown; cleanup
        # still proceeds to close the other session.
        self.assertEqual(coordinator.closed, [coordinator.sessions[0]])
        self.assertTrue(any('replacement recovery session cleanup retained' in note
                            for note in getattr(caught.exception, '__notes__', ())))

    def test_unprintable_cleanup_failure_does_not_skip_remaining_sessions(self):
        class UnprintableCleanupError(Exception):
            def __repr__(self):
                raise AssertionError('cleanup diagnostics must not execute repr')

        coordinator = _Coordinator(fail_load=2)
        original_close = coordinator.close_session

        def close_with_unprintable_failure(session):
            if session is coordinator.sessions[1]:
                coordinator.operations.append(('close', session))
                raise UnprintableCleanupError()
            return original_close(session)

        coordinator.close_session = close_with_unprintable_failure
        checkpoints = self._checkpoints()
        with patch.object(RankRecoveryCheckpoint, 'admit',
                          side_effect=((Path('first.cache'), {}),
                                       (Path('first.cache'), {}),
                                       (Path('second.cache'), {}),
                                       (Path('second.cache'), {}))):
            with self.assertRaisesRegex(OSError, 'injected rank restore failure') as caught:
                _RecoveryDriver(checkpoints, lambda *_args: self.fail('driver must not run'), 1, 0)(coordinator)

        self.assertEqual(coordinator.closed, [coordinator.sessions[0]])
        self.assertEqual([row[0] for row in coordinator.operations][-2:], ['close', 'close'])
        self.assertTrue(any('UnprintableCleanupError: failure details omitted' in note
                            for note in getattr(caught.exception, '__notes__', ())))


if __name__ == '__main__':
    unittest.main()
