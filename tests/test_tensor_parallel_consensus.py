"""Malformed rank agreement must quarantine distributed projection state."""
import unittest
import threading
from datetime import timedelta
from types import SimpleNamespace
import numpy as np

from openagent.adaptive_engine.tensor_parallel import (
    TensorParallelProjections,
    configured_group_timeout_seconds,
    require_group_timeout,
)
from openagent.adaptive_engine.distributed_execution import _failure_summary


class _MalformedAgreement:
    def __init__(self, remote_descriptor):
        self.remote_descriptor = remote_descriptor

    def all_gather_object(self, records, value, *, group):
        records[:] = [value, (self.remote_descriptor, None)]


class _LoopbackAgreement:
    def __init__(self):
        self.called = False

    def all_gather_object(self, records, value, *, group):
        self.called = True
        records[:] = [value for _ in records]


class _UnprintableError(Exception):
    def __str__(self):
        raise AssertionError('exception formatting must not run before agreement')


class _SpecializedInterruption(InterruptedError):
    def __str__(self):
        raise AssertionError('interruption formatting must not run before agreement')


class TensorParallelConsensusTests(unittest.TestCase):
    def test_caller_group_timeout_must_be_explicit_finite_and_within_admission(self):
        group = SimpleNamespace(options=SimpleNamespace(_timeout=timedelta(seconds=20)))
        self.assertEqual(configured_group_timeout_seconds(group), 20.0)
        self.assertEqual(require_group_timeout(group, 20), 20.0)
        with self.assertRaisesRegex(ValueError, 'exceeds collective timeout admission'):
            require_group_timeout(group, 10)
        with self.assertRaisesRegex(ValueError, 'expose its operation timeout'):
            configured_group_timeout_seconds(object())
        with self.assertRaisesRegex(ValueError, 'finite timedelta'):
            configured_group_timeout_seconds(SimpleNamespace(
                options=SimpleNamespace(_timeout=object())))

    def test_tensor_collective_uses_bounded_async_wait_and_quarantines_timeout(self):
        class Work:
            def __init__(self, completed):
                self.completed = completed
                self.timeouts = []

            def wait(self, timeout):
                self.timeouts.append(timeout)
                return self.completed

        class Distributed:
            def __init__(self, work):
                self.work = work
                self.kwargs = None

            def all_reduce(self, tensor, **kwargs):
                self.kwargs = kwargs
                return self.work

        bank = object.__new__(TensorParallelProjections)
        bank._closed = False
        bank._poisoned = False
        bank.collective_timeout_seconds = 3.0
        bank.group = object()
        bank.distributed = Distributed(Work(True))
        work = bank.distributed.work
        self.assertIs(bank._collective('all_reduce', object(), group=bank.group), work)
        self.assertTrue(bank.distributed.kwargs['async_op'])
        self.assertEqual(work.timeouts, [timedelta(seconds=3)])

        bank._closed = bank._poisoned = False
        bank.distributed = Distributed(Work(False))
        with self.assertRaisesRegex(TimeoutError, 'collective timed out'):
            bank._collective('all_reduce', object(), group=bank.group)
        self.assertTrue(bank._closed)
        self.assertTrue(bank._poisoned)

    def test_non_scalar_peer_descriptor_is_rejected_and_bank_quarantined(self):
        bank = object.__new__(TensorParallelProjections)
        bank.world_size = 2
        bank.distributed = _MalformedAgreement(('remote', np.asarray([1, 2])))
        bank.group = object()
        bank._poisoned = False
        bank._closed = False

        with self.assertRaisesRegex(RuntimeError, 'malformed distributed projection agreement'):
            bank._agree(('local', 'projection'), None)

        self.assertTrue(bank._poisoned)
        self.assertTrue(bank._closed)

    def test_unbounded_integer_peer_descriptor_is_rejected_and_quarantined(self):
        bank = object.__new__(TensorParallelProjections)
        bank.world_size = 2
        bank.distributed = _MalformedAgreement(('remote', 1 << 4096))
        bank.group = object()
        bank._poisoned = False
        bank._closed = False

        with self.assertRaisesRegex(RuntimeError, 'malformed distributed projection agreement'):
            bank._agree(('local', 'projection'), None)

        self.assertTrue(bank._poisoned)
        self.assertTrue(bank._closed)

    def test_exception_stringification_cannot_escape_local_stage_agreement(self):
        bank = object.__new__(TensorParallelProjections)
        bank.world_size = 1
        bank.distributed = _LoopbackAgreement()
        bank.group = object()
        bank._lock = threading.RLock()
        bank._closed = False
        bank._poisoned = False
        bank._active_stages = 0
        bank._cancellation = threading.local()

        with self.assertRaisesRegex(ValueError, 'distributed admission failed'):
            with bank.local_stage(('local', 'fault-injection')):
                raise _UnprintableError()

        self.assertTrue(bank.distributed.called)
        self.assertEqual(bank._active_stages, 0)

    def test_interruption_subclass_keeps_consensus_cancellation_type(self):
        bank = object.__new__(TensorParallelProjections)
        bank.world_size = 1
        bank.distributed = _LoopbackAgreement()
        bank.group = object()
        bank._lock = threading.RLock()
        bank._closed = False
        bank._poisoned = False
        bank._active_stages = 0
        bank._cancellation = threading.local()

        with self.assertRaisesRegex(InterruptedError, 'distributed projection stage cancelled'):
            with bank.local_stage(('local', 'cancel-fault-injection')):
                raise _SpecializedInterruption()

        self.assertTrue(bank.distributed.called)
        self.assertEqual(bank._active_stages, 0)

    def test_distributed_control_error_summary_is_bounded_and_unprintable_safe(self):
        failure = _UnprintableError()
        self.assertEqual(_failure_summary(failure),
                         '_UnprintableError: failure details omitted')
        self.assertEqual(len(_failure_summary(failure, limit=12)), 12)


if __name__ == '__main__':
    unittest.main()
