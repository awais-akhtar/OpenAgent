"""Regression coverage for collective checkpoint error envelopes."""
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import timedelta
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.distributed_generation_checkpoint import (
    CollectiveGenerationSink,
    _valid_generation_record,
)
from openagent.adaptive_engine.distributed_execution import (
    DistributedExecutionService,
    _collective_cancel_requested,
    _history_digest,
    _required_service_snapshot,
    _valid_result_summary,
)
from openagent.adaptive_engine.execution_service import ExecutionSessionSnapshot


class _LoopbackCollective:
    def all_gather_object(self, records, value, *, group):
        records[:] = [value for _ in records]


class _CancellationCollective(_LoopbackCollective):
    class ReduceOp:
        MAX = object()

    def __init__(self):
        self.calls = []

    def all_gather_object(self, records, value, *, group):
        self.calls.append('all_gather_object')
        super().all_gather_object(records, value, group=group)

    def all_reduce(self, flag, *, op, group):
        self.calls.append('all_reduce')


class _MalformedReservationCollective:
    def __init__(self):
        self.calls = 0

    def all_gather_object(self, records, value, *, group):
        self.calls += 1
        if self.calls == 1:
            records[:] = [('not-a-generation-name', None)]
        else:
            records[:] = [None for _ in records]


class _UnprintableReservationError(Exception):
    def __str__(self):
        raise AssertionError('collective error handling must not stringify this exception')


class DistributedCheckpointErrorTests(unittest.TestCase):
    def test_cancel_flag_allocation_failure_is_agreed_before_all_reduce(self):
        class Torch:
            int64 = object()

            @staticmethod
            def tensor(*_args, **_kwargs):
                raise MemoryError('injected rank-local tensor allocation failure')

        collective = _CancellationCollective()
        with self.assertRaisesRegex(RuntimeError, 'cancellation flag allocation failed'):
            _collective_cancel_requested(collective, Torch, object(), 1, False)

        self.assertEqual(collective.calls, ['all_gather_object'])

    def test_cancel_flag_read_failure_is_agreed_after_all_reduce(self):
        class Flag:
            def item(self):
                raise OSError('injected rank-local reduction result failure')

        class Torch:
            int64 = object()

            @staticmethod
            def tensor(*_args, **_kwargs):
                return Flag()

        collective = _CancellationCollective()
        with self.assertRaisesRegex(RuntimeError, 'cancellation flag read failed'):
            _collective_cancel_requested(collective, Torch, object(), 1, False)

        self.assertEqual(collective.calls,
                         ['all_gather_object', 'all_reduce', 'all_gather_object'])

    def test_root_collective_failure_retires_service_and_local_owners(self):
        class Service:
            def __init__(self):
                self.close_calls = 0

            def close(self):
                self.close_calls += 1

        service = Service()
        coordinator = object.__new__(DistributedExecutionService)
        coordinator.rank = 0
        coordinator._lock = threading.Lock()
        coordinator._closed = False
        coordinator._failed = False
        coordinator._sequence = 0
        coordinator.max_control_bytes = 4096
        coordinator._active_cancel = None
        coordinator._active_event = None
        coordinator.service = service
        coordinator._sessions = {7: 'local-session'}
        coordinator._snapshots = {11: (7, object())}
        coordinator._generation_boundaries = {7: object()}
        coordinator._broadcast = lambda _command: None

        def fail_during_collective(_command):
            raise OSError('control collective transport failed')

        coordinator._execute = fail_during_collective
        with self.assertRaisesRegex(OSError, 'control collective transport failed'):
            coordinator._request('reset', session=7)

        self.assertTrue(coordinator._failed)
        self.assertEqual(service.close_calls, 1)
        self.assertEqual(coordinator._sessions, {})
        self.assertEqual(coordinator._snapshots, {})
        self.assertEqual(coordinator._generation_boundaries, {})

    def test_control_group_timeout_must_match_across_ranks(self):
        from openagent.adaptive_engine.execution_service import ExecutionLimits, NativeExecutionService

        class DistributedOperations:
            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_backend(group):
                return "gloo"

            @staticmethod
            def get_rank(group):
                return 0

            @staticmethod
            def get_world_size(group):
                return 2

            @staticmethod
            def get_global_rank(group, rank):
                return rank

            @staticmethod
            def all_gather_object(records, value, *, group):
                policy = value[1]
                remote_policy = (*policy[:-1], policy[-1] + 1.0)
                records[:] = [value, ((), remote_policy)]

        distributed = ModuleType("torch.distributed")
        for name in ("is_initialized", "get_backend", "get_rank", "get_world_size",
                     "get_global_rank", "all_gather_object"):
            setattr(distributed, name, getattr(DistributedOperations, name))
        torch = ModuleType("torch")
        torch.distributed = distributed
        service = NativeExecutionService(
            lambda: object(), ExecutionLimits(vocabulary=8, context=16, max_new_tokens=4))
        self.addCleanup(service.close)
        group = SimpleNamespace(options=SimpleNamespace(_timeout=timedelta(seconds=30)))

        with patch.dict(sys.modules, {"torch": torch, "torch.distributed": distributed}):
            with self.assertRaisesRegex(ValueError, "control-group timeouts differ"):
                DistributedExecutionService(service, control_group=group)

    def test_rank_snapshot_must_return_rollback_receipt(self):
        class Service:
            def __init__(self, receipt):
                self.receipt = receipt
                self.sessions = []
                self._identity = 'service-instance'
                self.local = SimpleNamespace(identity='model-identity', tokens=(4, 5))

            def snapshot(self, session):
                self.sessions.append(session)
                return self.receipt

            def _session(self, session):
                if session not in (17, '17'):
                    raise KeyError(session)
                return self.local

        service = Service(None)
        with self.assertRaisesRegex(RuntimeError, 'rollback receipt'):
            _required_service_snapshot(service, 17)
        self.assertEqual(service.sessions, [17])

        receipt = object()
        service.receipt = receipt
        with self.assertRaisesRegex(RuntimeError, 'service-, model-, and history-bound'):
            _required_service_snapshot(service, 17)

        receipt = ExecutionSessionSnapshot(
            'service-instance', '17', 'snapshot-id', 'model-identity', (4, 5))
        service.receipt = receipt
        self.assertIs(_required_service_snapshot(service, '17'), receipt)

        service.receipt = ExecutionSessionSnapshot(
            'foreign-service', '17', 'snapshot-id', 'model-identity', (4, 5))
        with self.assertRaisesRegex(RuntimeError, 'service-, model-, and history-bound'):
            _required_service_snapshot(service, '17')

        service.receipt = ExecutionSessionSnapshot(
            'service-instance', '17', 'snapshot-id', 'foreign-model', (4, 5))
        with self.assertRaisesRegex(RuntimeError, 'service-, model-, and history-bound'):
            _required_service_snapshot(service, '17')

        service.receipt = ExecutionSessionSnapshot(
            'service-instance', '17', 'snapshot-id', 'model-identity', (4,))
        with self.assertRaisesRegex(RuntimeError, 'service-, model-, and history-bound'):
            _required_service_snapshot(service, '17')

        service.receipt = ExecutionSessionSnapshot(
            'service-instance', 'other', 'snapshot-id', 'model-identity', (4, 5))
        with self.assertRaisesRegex(RuntimeError, 'service-, model-, and history-bound'):
            _required_service_snapshot(service, '17')

    def test_generation_receipt_bounds_bytes_before_using_rank_record(self):
        valid = ('a' * 64, None, {
            'bytes': 128,
            'sha256': 'b' * 64,
            'execution_identity': 'executor-id',
        })
        oversized = ('a' * 64, None, {
            'bytes': 1 << 4096,
            'sha256': 'b' * 64,
            'execution_identity': 'executor-id',
        })
        self.assertTrue(_valid_generation_record(valid, max_rank_bytes=1024))
        self.assertFalse(_valid_generation_record(oversized, max_rank_bytes=1024))
        self.assertFalse(_valid_generation_record(
            ('a' * 64, 'write failed', valid[2]), max_rank_bytes=1024))
        self.assertFalse(_valid_generation_record(
            (None, 'write failed', valid[2]), max_rank_bytes=1024))

    def test_result_summary_rejects_unbounded_integer_wire_values(self):
        valid = ('open', 7, 'model-id', ('executor', 'Decoder'))
        oversized = ('open', 1 << 4096, 'model-id', ('executor', 'Decoder'))
        self.assertTrue(_valid_result_summary('open', valid))
        self.assertFalse(_valid_result_summary('open', oversized))

    def test_reset_result_summary_binds_empty_history_boundary(self):
        self.assertTrue(_valid_result_summary(
            'reset', ('reset', 7, 0, _history_digest(()))))
        self.assertFalse(_valid_result_summary('reset', ('reset', 7)))

    def test_reset_that_returns_with_stale_rank_history_fails_and_tears_down(self):
        class Session:
            tokens = (5,)

        class Service:
            def __init__(self):
                self.session = Session()
                self._identity = 'service'
                self.closed = []
                self.restored = []
                self.released = []

            def snapshot(self, _session):
                return ExecutionSessionSnapshot(
                    'service', 'session', 'snapshot-id', 'model', (5,))

            def reset(self, _session):
                # Simulate a faulty local executor which returns before its
                # physical state/history reaches the empty reset boundary.
                return None

            def status(self, _session):
                return SimpleNamespace(context_tokens=len(self.session.tokens))

            def _session(self, _session):
                self.session.identity = 'model'
                return self.session

            def restore(self, _session, _receipt):
                self.restored.append(True)
                self.session.tokens = (5,)

            def release_snapshot(self, _session, _receipt):
                self.released.append(True)

            def close_session(self, session):
                self.closed.append(session)

        service = Service()
        coordinator = object.__new__(DistributedExecutionService)
        coordinator.service = service
        coordinator.distributed = _LoopbackCollective()
        coordinator.group = object()
        coordinator.rank = 0
        coordinator.world_size = 1
        coordinator._sequence = 0
        coordinator._sessions = {1: 'session'}
        coordinator._generation_boundaries = {}
        coordinator._snapshots = {}
        coordinator._failed = False

        with self.assertRaisesRegex(RuntimeError, 'distributed execution failed'):
            coordinator._execute({'sequence': 0, 'action': 'reset', 'session': 1})

        self.assertTrue(coordinator._failed)
        self.assertEqual(service.restored, [True])
        self.assertEqual(service.released, [True])
        self.assertEqual(service.closed, ['session'])
        self.assertNotIn(1, coordinator._sessions)

    def test_partial_cache_restore_rolls_back_retires_receipt_and_tears_down_session(self):
        import json

        class Session:
            def __init__(self):
                self.model = SimpleNamespace(identity='model')
                self.identity = 'model'
                self.tokens = (4, 5)

        class Service:
            _identity = 'service'

            def __init__(self):
                self.session = Session()
                self.restored = []
                self.released = []
                self.closed = []
                self.load_attempts = 0

            def _session(self, _session):
                return self.session

            def snapshot(self, _session):
                return ExecutionSessionSnapshot(
                    'service', 'session', 'rollback', 'model', self.session.tokens)

            def load_session_cache(self, _session, _source, **_kwargs):
                self.load_attempts += 1
                self.session.tokens = (8, 9)
                raise OSError('injected cache install failure')

            def restore(self, _session, receipt):
                self.restored.append(receipt)
                self.session.tokens = receipt.token_ids

            def release_snapshot(self, _session, receipt):
                self.released.append(receipt)

            def close_session(self, session):
                self.closed.append(session)

        service = Service()
        coordinator = object.__new__(DistributedExecutionService)
        coordinator.service = service
        coordinator.distributed = _LoopbackCollective()
        coordinator.group = object()
        coordinator.rank = 0
        coordinator.world_size = 1
        coordinator._sequence = 0
        coordinator._sessions = {1: 'session'}
        coordinator._generation_boundaries = {}
        coordinator._snapshots = {}
        coordinator._failed = False

        rank_receipt = {'rank': 0, 'filename': 'rank-0.cache', 'bytes': 128,
                        'sha256': 'c' * 64, 'identity': 'model'}
        manifest = {'version': 1, 'world_size': 1, 'context_tokens': 2,
            'history_sha256': _history_digest((1, 2)), 'model_identity': 'model',
            'ranks': [rank_receipt]}
        encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()

        def read_manifest(path):
            return Path(path), encoded

        command = {'sequence': 0, 'action': 'load_cache', 'session': 1,
            'directory': 'cache-generation', 'manifest': manifest, 'max_bytes': 1024}
        with (patch('openagent.adaptive_engine.distributed_generation_checkpoint.read_manifest',
                    side_effect=read_manifest),
              patch('openagent.adaptive_engine.cache_file_recovery.verify_service_cache_file')):
            with self.assertRaisesRegex(RuntimeError, 'distributed execution failed') as caught:
                coordinator._execute(command)

        self.assertTrue(coordinator._failed)
        self.assertEqual(service.load_attempts, 1)
        self.assertEqual(service.session.tokens, (4, 5))
        self.assertEqual(len(service.restored), 1)
        self.assertEqual(service.released, service.restored)
        self.assertEqual(service.closed, ['session'])
        self.assertNotIn(1, coordinator._sessions)

    def test_failed_close_session_retries_local_teardown_before_poisoning_owner(self):
        class Service:
            def __init__(self):
                self.attempts = 0
                self.closed = []

            def close_session(self, session):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError('injected close failure')
                self.closed.append(session)

        service = Service()
        coordinator = object.__new__(DistributedExecutionService)
        coordinator.service = service
        coordinator.distributed = _LoopbackCollective()
        coordinator.group = object()
        coordinator.rank = 0
        coordinator.world_size = 1
        coordinator._sequence = 0
        coordinator._sessions = {1: 'session'}
        coordinator._generation_boundaries = {}
        coordinator._snapshots = {}
        coordinator._failed = False

        with self.assertRaisesRegex(RuntimeError, 'distributed execution failed'):
            coordinator._execute({'sequence': 0, 'action': 'close_session', 'session': 1})

        self.assertTrue(coordinator._failed)
        self.assertEqual(service.attempts, 2)
        self.assertEqual(service.closed, ['session'])
        self.assertNotIn(1, coordinator._sessions)

    def test_reservation_stringifier_failure_still_reaches_agreement(self):
        owner = type('Owner', (), {
            'rank': 0,
            'world_size': 1,
            'distributed': _LoopbackCollective(),
            'group': object(),
        })()
        sink = CollectiveGenerationSink(owner, Path('checkpoint-root'), max_rank_bytes=1024)

        with patch.object(Path, 'mkdir', side_effect=_UnprintableReservationError()):
            with self.assertRaisesRegex(RuntimeError, 'failure details omitted'):
                sink(object(), lambda *_args, **_kwargs: None)

    def test_rank_writer_failure_quarantines_uncommitted_generation(self):
        with TemporaryDirectory() as temporary:
            owner = SimpleNamespace(rank=0, world_size=1,
                distributed=_LoopbackCollective(), group=object(), _failed=False)
            root = Path(temporary) / 'generations'
            sink = CollectiveGenerationSink(owner, root, max_rank_bytes=1024)
            boundary = SimpleNamespace(
                tokens=(1,), dtype='float32', logits=b'logits', sampler=b'sampler',
                policy=None, stops=(), stopped=False, stop_prefixes=(),
                vocabulary=3, context=8, application_identity='', rng=None,
                proposal_rng=None, position=1)

            def fail_writer(_writer):
                raise OSError('rank frame writer failed')

            with self.assertRaisesRegex(RuntimeError, 'distributed generation capture failed'):
                sink(boundary, fail_writer)

            self.assertEqual(list(root.iterdir()), [])
            self.assertFalse((root / 'latest.json').exists())

    def test_manifest_readback_failure_removes_uncommitted_generation_directory(self):
        with TemporaryDirectory() as temporary:
            owner = SimpleNamespace(rank=0, world_size=1,
                distributed=_LoopbackCollective(), group=object(), _failed=False)
            root = Path(temporary) / 'generations'
            sink = CollectiveGenerationSink(owner, root, max_rank_bytes=1024)
            boundary = SimpleNamespace(
                tokens=(1,), dtype='float32', logits=b'logits', sampler=b'sampler',
                policy=None, stops=(), stopped=False, stop_prefixes=(),
                vocabulary=3, context=8, application_identity='', rng=None,
                proposal_rng=None, position=1, config_sha256='a' * 64,
                tokenizer_sha256='b' * 64)

            class Store:
                def __init__(self, *_args, **_kwargs):
                    pass

                def __call__(self, _boundary, _writer):
                    return dict(bytes=128, sha256='c' * 64,
                                execution_identity='rank-model')

                def close(self):
                    pass

            with patch('openagent.adaptive_engine.distributed_generation_checkpoint.AtomicGenerationCheckpoint',
                       Store), patch(
                    'openagent.adaptive_engine.distributed_generation_checkpoint.read_manifest',
                    side_effect=OSError('injected manifest readback failure')):
                with self.assertRaisesRegex(RuntimeError,
                                            'distributed generation manifest publication failed'):
                    sink(boundary, lambda *_args: None)

            self.assertEqual(list(root.iterdir()), [])
            self.assertFalse((root / 'latest.json').exists())

    def test_malformed_reservation_record_cleans_partial_directory_and_poisons_owner(self):
        with TemporaryDirectory() as temporary:
            collective = _MalformedReservationCollective()
            owner = type('Owner', (), {
                'rank': 0,
                'world_size': 1,
                'distributed': collective,
                'group': object(),
                '_failed': False,
            })()
            root = Path(temporary) / 'generations'
            sink = CollectiveGenerationSink(owner, root, max_rank_bytes=1024)

            with self.assertRaisesRegex(RuntimeError, 'malformed distributed generation reservation'):
                sink(object(), lambda *_args, **_kwargs: None)

            self.assertTrue(owner._failed)
            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(collective.calls, 2)


if __name__ == '__main__':
    unittest.main()
