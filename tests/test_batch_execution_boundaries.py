"""Real scheduler/bridge boundaries: no synthetic output, skipped prompt or stranded waiter."""
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.continuous_batching import (
    BatchRequest, BatchWorker, ContinuousBatchScheduler, MultiSessionBatchEngine,
    RequestState, SchedulerConfig, wait_for_request,
)
from openagent.adaptive_engine.forward_bridge import Phase2ForwardBridge


class _Session:
    def __init__(self, *, error=None, close_failures=0):
        self.error = error
        self.close_failures = close_failures
        self.contexts = []
        self.closes = 0

    def generate(self, *, context_ids, max_new_tokens):
        self.contexts.append(list(context_ids))
        if self.error is not None:
            raise self.error
        return [9]

    def close(self):
        self.closes += 1
        if self.close_failures:
            self.close_failures -= 1
            raise OSError('slot close failed')


class BatchExecutionBoundaryTests(unittest.TestCase):
    def test_partial_prefill_continues_and_tracks_actual_budget(self):
        scheduler = ContinuousBatchScheduler(SchedulerConfig(
            max_tokens_per_batch=2, prefill_chunk_size=4))
        request = BatchRequest(token_ids=[1, 2, 3, 4, 5], max_new_tokens=1)
        scheduler.submit(request)
        for cursor, extent in ((0, 2), (2, 2), (4, 1)):
            self.assertEqual(scheduler.schedule(), [request])
            self.assertEqual(request.prefill_cursor, cursor)
            self.assertEqual(request.scheduled_prefill_tokens, extent)
            scheduler.commit_step([(request, None)])
        self.assertEqual(request.prefill_cursor, 5)
        self.assertEqual(request.state, RequestState.DECODE)
        self.assertEqual(scheduler.schedule(), [request])
        scheduler.commit_step([(request, 9)])
        self.assertEqual(wait_for_request(request, timeout=0), [9])

    def test_fatal_forward_failure_wakes_requests_outside_current_batch(self):
        scheduler = ContinuousBatchScheduler(SchedulerConfig(max_batch_size=1))
        requests = [BatchRequest(token_ids=[1], max_new_tokens=1) for _ in range(3)]
        for request in requests:
            scheduler.submit(request)
        primary = KeyboardInterrupt('worker interrupted')
        worker = BatchWorker(scheduler, lambda _batch: (_ for _ in ()).throw(primary))
        worker._run()
        self.assertTrue(worker._stop_event.is_set())
        for request in requests:
            self.assertTrue(request._event.is_set())
            self.assertIs(request.error, primary)
            self.assertEqual(request.state, RequestState.FAILED)
        self.assertEqual(scheduler.metrics()['waiting_requests'], 0)
        self.assertEqual(scheduler.metrics()['running_requests'], 0)
        with self.assertRaisesRegex(RuntimeError, 'terminal worker failure') as caught:
            scheduler.submit(BatchRequest(token_ids=[4]))
        self.assertIs(caught.exception.__cause__, primary)

    def test_schedule_failure_is_reported_to_every_waiter(self):
        scheduler = ContinuousBatchScheduler(SchedulerConfig())
        request = BatchRequest(token_ids=[1], max_new_tokens=1)
        scheduler.submit(request)
        worker = BatchWorker(scheduler, lambda _batch: [])
        primary = OSError('scheduler failed')
        with patch.object(scheduler, 'schedule', side_effect=primary):
            worker._run()
        self.assertIs(request.error, primary)
        self.assertTrue(request._event.is_set())
        self.assertTrue(worker._stop_event.is_set())

    def test_engine_rejects_new_submissions_after_worker_fatal_failure(self):
        engine = MultiSessionBatchEngine(lambda _batch: (_ for _ in ()).throw(
            KeyboardInterrupt('worker failed')), config=SchedulerConfig(max_batch_size=1))
        first = engine.submit([1], max_new_tokens=1)
        engine._worker._run()
        self.assertTrue(first._event.is_set())
        with self.assertRaisesRegex(RuntimeError, 'worker is stopped'):
            engine.submit([2], max_new_tokens=1)
        self.assertEqual(engine.scheduler.metrics()['waiting_requests'], 0)
        engine.stop(timeout=1)

    def test_real_bridge_worker_preserves_long_prompt_and_releases_completed_slot(self):
        session = _Session()
        bridge = Phase2ForwardBridge(lambda: session)
        engine = MultiSessionBatchEngine(bridge, config=SchedulerConfig(
            max_tokens_per_batch=2, prefill_chunk_size=4, scheduler_tick_ms=1))
        engine.start()
        try:
            request = engine.submit([1, 2, 3, 4, 5], max_new_tokens=2)
            self.assertEqual(wait_for_request(request, timeout=2), [9, 9])
            self.assertEqual(session.contexts, [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 9]])
            self.assertEqual(session.closes, 1)
            self.assertEqual(bridge.metrics()['bridge_active_slots'], 0)
        finally:
            engine.stop(timeout=2)
            bridge.close_all()

    def test_model_failure_is_not_replaced_with_random_token_and_wakes_waiter(self):
        primary = OSError('actual model failed')
        session = _Session(error=primary)
        bridge = Phase2ForwardBridge(lambda: session)
        engine = MultiSessionBatchEngine(bridge, config=SchedulerConfig(scheduler_tick_ms=1))
        engine.start()
        try:
            request = engine.submit([1], max_new_tokens=1)
            with self.assertRaises(OSError) as caught:
                wait_for_request(request, timeout=2)
            self.assertIs(caught.exception, primary)
            self.assertEqual(request.generated, [])
            self.assertEqual(session.closes, 1)
        finally:
            engine.stop(timeout=2)
            bridge.close_all()

    def test_single_step_error_never_dispatches_second_generation_path(self):
        class Session(_Session):
            def generate_single(self, _context):
                raise OSError('single step failed after mutation')

        session = Session()
        bridge = Phase2ForwardBridge(lambda: session)
        request = BatchRequest(token_ids=[1], state=RequestState.DECODE)
        with self.assertRaisesRegex(OSError, 'single step failed'):
            bridge([request])
        self.assertEqual(session.contexts, [])
        self.assertEqual(session.closes, 1)

    def test_nonfinite_or_wrong_geometry_logits_fail_closed(self):
        for logits in (np.array([np.nan, 0], np.float32), np.zeros((2, 2), np.float32)):
            class Session(_Session):
                def generate_single(self, _context):
                    return logits

            session = Session()
            bridge = Phase2ForwardBridge(lambda: session, vocab_size=2)
            with self.assertRaisesRegex(ValueError, 'finite vocabulary-sized'):
                bridge([BatchRequest(state=RequestState.DECODE)])
            self.assertEqual(session.contexts, [])
            self.assertEqual(session.closes, 1)

    def test_failed_slot_close_remains_owned_until_retry(self):
        session = _Session(close_failures=2)
        bridge = Phase2ForwardBridge(lambda: session)
        request = BatchRequest(state=RequestState.DECODE, max_new_tokens=1)
        with self.assertRaisesRegex(OSError, 'slot close failed'):
            bridge([request])
        self.assertEqual(bridge.metrics()['bridge_active_slots'], 1)
        bridge.close_all()
        self.assertEqual(session.closes, 3)
        self.assertEqual(bridge.metrics()['bridge_active_slots'], 0)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            bridge([BatchRequest(state=RequestState.DECODE)])

    def test_shared_kv_without_physical_binding_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, 'bound physical cache adapter'):
            Phase2ForwardBridge(lambda: _Session(), kv_cache=object())
        with patch.object(_Session, '__init__', return_value=None) as factory:
            bridge = Phase2ForwardBridge(_Session)
            with self.assertRaisesRegex(ValueError, 'bound physical cache adapter'):
                bridge([BatchRequest(state=RequestState.PREFILL, kv_prefix_len=1)])
            factory.assert_not_called()

    def test_failure_callback_interrupt_wakes_other_queued_requests(self):
        scheduler = ContinuousBatchScheduler(SchedulerConfig(max_batch_size=1))
        primary = OSError('forward failed')

        def hostile_callback(_failure):
            raise KeyboardInterrupt('callback interrupted')

        first = BatchRequest(token_ids=[1], max_new_tokens=1, on_error=hostile_callback)
        second = BatchRequest(token_ids=[2], max_new_tokens=1)
        scheduler.submit(first)
        scheduler.submit(second)
        worker = BatchWorker(scheduler, lambda _batch: (_ for _ in ()).throw(primary))
        worker._run()
        self.assertTrue(worker._stop_event.is_set())
        for request in (first, second):
            self.assertTrue(request._event.is_set())
            self.assertIs(request.error, primary)

    def test_duplicate_or_completed_requests_cannot_be_resubmitted(self):
        scheduler = ContinuousBatchScheduler(SchedulerConfig())
        request = BatchRequest(request_id='owned', token_ids=[1], max_new_tokens=1)
        scheduler.submit(request)
        for duplicate in (request, BatchRequest(request_id='owned', token_ids=[2])):
            with self.assertRaisesRegex(ValueError, 'already active'):
                scheduler.submit(duplicate)
        self.assertEqual(scheduler.metrics()['waiting_requests'], 1)
        scheduler.cancel(request.request_id)
        with self.assertRaisesRegex(ValueError, 'fresh request'):
            scheduler.submit(request)

    def test_bridge_foreign_id_cannot_close_or_use_another_request_owner(self):
        session = _Session()
        bridge = Phase2ForwardBridge(lambda: session)
        owner = BatchRequest(request_id='owned', state=RequestState.DECODE, max_new_tokens=3)
        self.assertEqual(bridge([owner]), [(owner, 9)])
        for state in (RequestState.DECODE, RequestState.CANCELLED):
            foreign = BatchRequest(request_id='owned', state=state)
            with self.assertRaisesRegex(ValueError, 'another active request'):
                bridge([foreign])
        self.assertEqual(session.closes, 0)
        self.assertEqual(len(session.contexts), 1)
        bridge.close_all()
        self.assertEqual(session.closes, 1)

    def test_premature_prefill_token_is_rejected_before_commit(self):
        scheduler = ContinuousBatchScheduler(SchedulerConfig(max_tokens_per_batch=1))
        request = BatchRequest(token_ids=[1, 2], max_new_tokens=1)
        scheduler.submit(request)
        batch = scheduler.schedule()
        with self.assertRaisesRegex(ValueError, 'before scheduled prefill completed'):
            BatchWorker._validate_results(batch, [(request, 9)])
        self.assertEqual(request.generated, [])
        self.assertEqual(request.prefill_cursor, 0)

    def test_missing_model_execution_method_fails_without_synthetic_output(self):
        class UnsupportedSession:
            def close(self):
                pass

        bridge = Phase2ForwardBridge(UnsupportedSession)
        with self.assertRaisesRegex(RuntimeError, 'no supported generation method'):
            bridge([BatchRequest(state=RequestState.DECODE)])
        self.assertEqual(bridge.metrics()['bridge_active_slots'], 0)

    def test_bridge_accepts_only_exact_context_plus_one_or_single_token(self):
        for result, valid in (([1, 2, 9], True), ([9], True), ([5, 6, 9], False),
                              ([9, 9], False), ([True], False), ([], False), ([32000], False)):
            class Session(_Session):
                def generate(self, **_kwargs):
                    return result

            session = Session()
            bridge = Phase2ForwardBridge(lambda: session)
            request = BatchRequest(token_ids=[1, 2], state=RequestState.PREFILL,
                                   max_new_tokens=1)
            with self.subTest(result=result):
                if valid:
                    self.assertEqual(bridge([request]), [(request, 9)])
                else:
                    with self.assertRaises(ValueError):
                        bridge([request])
                self.assertEqual(session.closes, 1)


if __name__ == '__main__':
    unittest.main()
