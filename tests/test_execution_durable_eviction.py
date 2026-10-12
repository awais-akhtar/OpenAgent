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

    def test_corruption_rejected_before_executor_creation(self):
        service = self.service()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.bin'
            path.write_bytes(b'not a replay checkpoint')
            with self.assertRaises(ValueError):
                service.open_recovered_session(path)
        self.assertEqual(service.runtime_status()['sessions'], 0)
