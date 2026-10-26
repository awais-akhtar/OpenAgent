"""Durable replay lifecycle with a deterministic in-memory executor."""
from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
import unittest
import numpy as np

from openagent.adaptive_engine.execution_service import (
    ExecutionCancelled, ExecutionLimits, NativeExecutionService,
)


@dataclass(frozen=True)
class Snapshot:
    identity: str
    tokens: tuple

    @property
    def position(self):
        return len(self.tokens)


class Executor:
    identity = 'eviction-regression-executor-v1'

    def __init__(self):
        self.tokens = ()
        self.closed = False

    @property
    def position(self):
        return len(self.tokens)

    def reset(self):
        self.tokens = ()

    def forward(self, tokens):
        self.tokens += tuple(tokens)
        result = np.zeros((len(tokens), 8), np.float32)
        result[:, 3] = 1
        return result

    def snapshot(self):
        return Snapshot(self.identity, self.tokens)

    def restore(self, snapshot):
        self.tokens = snapshot.tokens

    def close(self):
        self.closed = True


class CancelAtCommitExecutor(Executor):
    def __init__(self, cancel):
        super().__init__()
        self.cancel = cancel

    @property
    def position(self):
        if len(self.tokens) == 2:
            self.cancel.set()
        return len(self.tokens)


class _UnprintableRecoveryError(RuntimeError):
    def __repr__(self):
        raise RuntimeError('repr must not be called')

    def __str__(self):
        raise RuntimeError('str must not be called')


class RestoreFailsTwiceExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.restore_calls = 0

    def restore(self, snapshot):
        self.restore_calls += 1
        if self.restore_calls == 1:
            raise ValueError('original restore failure')
        raise _UnprintableRecoveryError()


class DurableEvictionTests(unittest.TestCase):
    def service(self):
        service = NativeExecutionService(Executor, ExecutionLimits(
            vocabulary=8, context=32, max_new_tokens=4, max_sessions=2,
            max_concurrent_generations=1, prefill_chunk_tokens=2))
        self.addCleanup(service.close)
        return service

    def test_eviction_recovery_and_continuation(self):
        service = self.service()
        session = service.open_session()
        service.generate(session, [1, 2], max_new_tokens=1)
        before = service.history(session)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'session.bin'
            receipt = service.evict_session(session, path)
            self.assertEqual(receipt['context_tokens'], len(before))
            self.assertFalse(receipt['cache_bytes_preserved'])
            self.assertEqual(service.runtime_status()['sessions'], 0)
            restored = service.open_recovered_session(path)
            self.assertEqual(service.history(restored), before)
            service.generate(restored, [2], max_new_tokens=1)
            self.assertEqual(service.history(restored), before+(2, 3))

    def test_stop_prefix_does_not_match_across_prompt_generation_boundary(self):
        service = self.service()
        session = service.open_session()
        result = service.generate(session, [1], max_new_tokens=2,
                                  stop_prefixes=((1, 3),))
        self.assertEqual(result.token_ids, (3, 3))
        self.assertEqual(result.finish_reason, 'length')

    def test_failed_persistence_keeps_live_session(self):
        service = self.service()
        session = service.open_session()
        service.generate(session, [1], max_new_tokens=1)
        before = service.history(session)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                service.evict_session(session, Path(directory)/'absent'/'session.bin')
        self.assertEqual(service.history(session), before)
        service.generate(session, [2], max_new_tokens=1)

    def test_cancelled_replay_removes_new_executor(self):
        service = self.service()
        session = service.open_session()
        service.generate(session, [1, 2], max_new_tokens=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'session.bin'
            service.evict_session(session, path)
            cancel = threading.Event()
            def progress(event):
                cancel.set()
            with self.assertRaises(ExecutionCancelled):
                service.open_recovered_session(path, cancel=cancel, on_event=progress)
        self.assertEqual(service.runtime_status()['sessions'], 0)
        self.assertEqual(service.runtime_status()['active_generations'], 0)

    def test_cancellation_during_final_position_validation_rolls_back(self):
        cancel = threading.Event()
        service = NativeExecutionService(lambda: CancelAtCommitExecutor(cancel), ExecutionLimits(
            vocabulary=8, context=32, max_new_tokens=1, max_sessions=1,
            prefill_chunk_tokens=2))
        self.addCleanup(service.close)
        session = service.open_session()
        with self.assertRaises(ExecutionCancelled):
            service.generate(session,[1],max_new_tokens=1,cancel=cancel)
        self.assertEqual(service.history(session),())
        self.assertEqual(service.status(session).context_tokens,0)
        self.assertEqual(service._session(session).model.tokens,())

    def test_corruption_rejected_before_executor_creation(self):
        service = self.service()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.bin'
            path.write_bytes(b'not a replay checkpoint')
            with self.assertRaises(ValueError):
                service.open_recovered_session(path)
        self.assertEqual(service.runtime_status()['sessions'], 0)

    def test_unprintable_restore_rollback_does_not_mask_primary_error(self):
        service = NativeExecutionService(RestoreFailsTwiceExecutor, ExecutionLimits(
            vocabulary=8, context=32, max_new_tokens=4, max_sessions=1,
            prefill_chunk_tokens=2))
        self.addCleanup(service.close)
        session = service.open_session()
        service.generate(session, [1], max_new_tokens=1)
        snapshot = service.snapshot(session)

        with self.assertRaisesRegex(ValueError, 'original restore failure') as caught:
            service.restore(session, snapshot)

        self.assertTrue(service.status(session).closed)
        self.assertTrue(any('executor restore rollback failed' in note
                            for note in getattr(caught.exception, '__notes__', ())))
