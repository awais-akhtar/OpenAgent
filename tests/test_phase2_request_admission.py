from __future__ import annotations

import time
import threading
import unittest

from openagent.adaptive_engine.continuous_batching import (
    BatchRequest,
    BatchWorker,
    ContinuousBatchScheduler,
    MultiSessionBatchEngine,
    RequestState,
    SchedulerConfig,
    wait_for_request,
)
from openagent.adaptive_engine.request_queue import Phase2RequestQueue, TokenBucket
from openagent.adaptive_engine.multi_session_service import Phase2MultiSessionService


class Phase2RequestAdmissionTests(unittest.TestCase):
    def engine(self, admission, *, max_depth=4):
        return MultiSessionBatchEngine(
            forward_fn=lambda _batch: [],
            config=SchedulerConfig(max_queue_depth=max_depth),
            request_admission=admission,
        )

    def test_small_configured_bucket_starts_with_one_admission_token(self):
        bucket = TokenBucket(rate=0.01, capacity=0.005)
        self.assertTrue(bucket.consume())
        self.assertFalse(bucket.consume())

    def test_real_engine_submission_enforces_global_rate_limit(self):
        admission = Phase2RequestQueue(
            max_size=4, global_rate=0.01, per_session_rate=100, max_wait_s=0,
        )
        engine = self.engine(admission)

        accepted = engine.submit([1], session_id="a", max_new_tokens=1)
        rejected = engine.submit([1], session_id="b", max_new_tokens=1)

        self.assertEqual(accepted.state, RequestState.WAITING)
        self.assertEqual(rejected.state, RequestState.FAILED)
        self.assertTrue(rejected._event.is_set())
        with self.assertRaisesRegex(RuntimeError, "global rate limit"):
            wait_for_request(rejected, timeout=0)
        self.assertEqual(admission.metrics()["queue_admitted"], 1)
        self.assertEqual(admission.metrics()["queue_rate_dropped"], 1)

    def test_per_session_rejection_refunds_global_token(self):
        admission = Phase2RequestQueue(
            max_size=4, global_rate=100, per_session_rate=0.01,
            per_session_burst=1, max_wait_s=0,
        )
        engine = self.engine(admission)
        first = engine.submit([1], session_id="same", max_new_tokens=1)
        second = engine.submit([1], session_id="same", max_new_tokens=1)
        other = engine.submit([1], session_id="other", max_new_tokens=1)

        self.assertEqual(first.state, RequestState.WAITING)
        self.assertEqual(second.state, RequestState.FAILED)
        self.assertEqual(other.state, RequestState.WAITING)
        self.assertEqual(admission.metrics()["queue_admitted"], 2)

    def test_queue_wait_expiry_fails_and_wakes_the_scheduled_request(self):
        admission = Phase2RequestQueue(
            max_size=4, global_rate=100, per_session_rate=100, max_wait_s=0.02,
        )
        engine = self.engine(admission)
        errors = []
        request = engine.submit(
            [1], session_id="a", max_new_tokens=1,
            on_error=errors.append,
        )
        time.sleep(0.03)

        self.assertEqual(engine.scheduler.schedule(), [])
        self.assertEqual(request.state, RequestState.FAILED)
        self.assertTrue(request._event.is_set())
        with self.assertRaisesRegex(TimeoutError, "expired before batch scheduling"):
            wait_for_request(request, timeout=0)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TimeoutError)
        service = object.__new__(Phase2MultiSessionService)
        service._request_queue = admission
        service._engine = engine
        metrics = service.request_queue_metrics()
        self.assertEqual(metrics["queue_expired"], 1)
        self.assertEqual(metrics["queue_scheduler_waiting"], 0)

    def test_scheduler_capacity_rejection_is_counted_by_shared_admission_metrics(self):
        admission = Phase2RequestQueue(
            max_size=5, global_rate=4, per_session_rate=4,
            per_session_burst=2, max_wait_s=0,
        )
        engine = self.engine(admission, max_depth=1)
        first = engine.submit([1], session_id="a", max_new_tokens=1)
        rejected = engine.submit([1], session_id="b", max_new_tokens=1)

        self.assertEqual(first.state, RequestState.WAITING)
        self.assertEqual(rejected.state, RequestState.FAILED)
        self.assertTrue(rejected._event.is_set())
        self.assertEqual(admission.metrics()["queue_full_dropped"], 1)

        engine.cancel(first.request_id)
        replacement = engine.submit([1], session_id="b", max_new_tokens=1)
        self.assertEqual(replacement.state, RequestState.WAITING)
        self.assertEqual(admission.metrics()["queue_rate_dropped"], 0)

    def test_wait_timeout_is_reported_instead_of_returning_partial_output(self):
        admission = Phase2RequestQueue(
            max_size=2, global_rate=100, per_session_rate=100, max_wait_s=0,
        )
        request = self.engine(admission).submit([1], session_id="a", max_new_tokens=1)
        with self.assertRaisesRegex(TimeoutError, "wait deadline"):
            wait_for_request(request, timeout=0)

    def test_forward_failure_retires_request_and_preserves_original_error(self):
        scheduler = ContinuousBatchScheduler(SchedulerConfig(max_queue_depth=2))
        worker = BatchWorker(
            scheduler,
            lambda _batch: (_ for _ in ()).throw(OSError("forward exploded")),
            tick_ms=1,
        )
        request = BatchRequest(token_ids=[1], max_new_tokens=1)
        worker.start()
        try:
            scheduler.submit(request)
            with self.assertRaisesRegex(OSError, "forward exploded"):
                wait_for_request(request, timeout=2)
            self.assertEqual(scheduler.metrics()["running_requests"], 0)
        finally:
            worker.stop(timeout=2)

    def test_cancelled_inflight_request_is_retired_if_forward_call_fails(self):
        entered = threading.Event()
        release = threading.Event()
        returned = threading.Event()

        def forward(_batch):
            entered.set()
            release.wait(2)
            returned.set()
            raise OSError("in-flight forward failed")

        engine = MultiSessionBatchEngine(
            forward, config=SchedulerConfig(scheduler_tick_ms=1, max_queue_depth=2)
        )
        engine.start()
        request = engine.submit([1], session_id="a", max_new_tokens=1)
        try:
            self.assertTrue(entered.wait(2))
            self.assertTrue(engine.cancel(request.request_id))
            release.set()
            with self.assertRaisesRegex(RuntimeError, "was cancelled"):
                wait_for_request(request, timeout=2)
            self.assertTrue(returned.wait(2))
            deadline = time.monotonic() + 2
            while engine.scheduler.metrics()["running_requests"] and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(engine.scheduler.metrics()["running_requests"], 0)
        finally:
            release.set()
            engine.stop(timeout=2)

    def test_incomplete_forward_result_fails_and_wakes_every_scheduled_request(self):
        engine = MultiSessionBatchEngine(
            lambda _batch: [],
            config=SchedulerConfig(scheduler_tick_ms=1, max_queue_depth=2),
        )
        engine.start()
        request = engine.submit([1], session_id="a", max_new_tokens=1)
        try:
            with self.assertRaisesRegex(ValueError, "omitted"):
                wait_for_request(request, timeout=2)
            self.assertEqual(engine.scheduler.metrics()["running_requests"], 0)
        finally:
            engine.stop(timeout=2)

    def test_decode_forward_step_cannot_silently_return_no_token(self):
        calls = []

        def forward(batch):
            calls.append(batch[0].state)
            return [(batch[0], None)]

        engine = MultiSessionBatchEngine(
            forward, config=SchedulerConfig(scheduler_tick_ms=1, max_queue_depth=2)
        )
        engine.start()
        request = engine.submit([1], session_id="a", max_new_tokens=1)
        try:
            with self.assertRaisesRegex(ValueError, "decode forward step"):
                wait_for_request(request, timeout=2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(engine.scheduler.metrics()["running_requests"], 0)
        finally:
            engine.stop(timeout=2)

    def test_worker_process_level_failure_wakes_request_and_stops_worker(self):
        worker = BatchWorker(
            ContinuousBatchScheduler(SchedulerConfig(scheduler_tick_ms=1)),
            lambda _batch: (_ for _ in ()).throw(KeyboardInterrupt("worker interrupted")),
            tick_ms=1,
        )
        request = BatchRequest(token_ids=[1], max_new_tokens=1)
        worker.start()
        try:
            worker._scheduler.submit(request)
            with self.assertRaisesRegex(KeyboardInterrupt, "worker interrupted"):
                wait_for_request(request, timeout=2)
            self.assertEqual(request.state, RequestState.FAILED)
            worker.stop(timeout=2)
            self.assertFalse(worker.metrics()["worker_running"])
        finally:
            worker.stop(timeout=2)

    def test_empty_prompt_enters_decode_instead_of_remaining_queued_forever(self):
        engine = MultiSessionBatchEngine(
            lambda batch: [(batch[0], 17)],
            config=SchedulerConfig(scheduler_tick_ms=1, max_queue_depth=2),
        )
        engine.start()
        request = engine.submit([], session_id="empty", max_new_tokens=1)
        try:
            self.assertEqual(wait_for_request(request, timeout=2), [17])
            self.assertEqual(engine.scheduler.metrics()["waiting_requests"], 0)
        finally:
            engine.stop(timeout=2)

    def test_invalid_scheduler_geometry_and_direct_request_fail_before_admission(self):
        for options in (
            {"max_batch_size": 0},
            {"max_tokens_per_batch": 0},
            {"prefill_chunk_size": 0},
            {"max_decode_len": 0},
            {"scheduler_tick_ms": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SchedulerConfig(**options)

        admission = Phase2RequestQueue(
            global_rate=100, per_session_rate=100, max_wait_s=0,
        )
        engine = self.engine(admission)
        with self.assertRaisesRegex(ValueError, "token budget"):
            engine.submit([1], session_id="a", max_new_tokens=999)
        with self.assertRaisesRegex(ValueError, "token IDs"):
            engine.submit([True], session_id="a", max_new_tokens=1)
        self.assertEqual(admission.metrics()["queue_admitted"], 0)

    def test_zero_output_budget_completes_without_model_work_or_rate_admission(self):
        admission = Phase2RequestQueue(
            global_rate=100, per_session_rate=100, max_wait_s=0,
        )
        engine = self.engine(admission)
        request = engine.submit([], session_id="a", max_new_tokens=0)

        self.assertEqual(request.state, RequestState.FINISHED)
        self.assertTrue(request._event.is_set())
        self.assertEqual(wait_for_request(request, timeout=0), [])
        self.assertEqual(admission.metrics()["queue_admitted"], 0)

    def test_shutdown_waits_for_admission_boundary_before_stopping_worker(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingAdmission:
            def admit(self, _request):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test admission gate timed out")
                return True

        engine = MultiSessionBatchEngine(
            lambda batch: [(request, 9) for request in batch],
            config=SchedulerConfig(scheduler_tick_ms=1, max_queue_depth=2),
            request_admission=BlockingAdmission(),
        )
        engine.start()
        result = []
        submitter = threading.Thread(
            target=lambda: result.append(engine.submit([1], session_id="a", max_new_tokens=1))
        )
        submitter.start()
        self.assertTrue(entered.wait(2))
        try:
            with self.assertRaisesRegex(TimeoutError, "admission boundary"):
                engine.stop(timeout=0.01)
            release.set()
            submitter.join(2)
            self.assertFalse(submitter.is_alive())
            engine.stop(timeout=2)
            self.assertTrue(result[0].is_done())
        finally:
            release.set()
            if submitter.is_alive():
                submitter.join(2)
            if not engine._closed:
                engine.stop(timeout=2)

    def test_standalone_queue_rejection_wakes_waiter_before_reentrant_callback(self):
        admission = Phase2RequestQueue(
            max_size=1, global_rate=100, per_session_rate=100, max_wait_s=0,
        )
        first = BatchRequest(session_id="a")
        admission.submit(first)
        callback_done = []
        rejected = BatchRequest(
            session_id="b",
            on_error=lambda _error: callback_done.append(admission.metrics()),
        )

        with self.assertRaisesRegex(RuntimeError, "Request queue full"):
            admission.submit(rejected)
        self.assertEqual(rejected.state, RequestState.FAILED)
        self.assertTrue(rejected._event.is_set())
        self.assertEqual(len(callback_done), 1)
        with self.assertRaisesRegex(RuntimeError, "queue full"):
            wait_for_request(rejected, timeout=0)


if __name__ == "__main__":
    unittest.main()
