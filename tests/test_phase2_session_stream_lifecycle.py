from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openagent.adaptive_engine.continuous_batching import (
    BatchRequest,
    ContinuousBatchScheduler,
    RequestState,
    SchedulerConfig,
    wait_for_request,
)
from openagent.adaptive_engine.multi_session_service import Phase2SessionHandle


class _Engine:
    def __init__(self, mode: str = "pending") -> None:
        self.scheduler = SimpleNamespace(config=SimpleNamespace(max_decode_len=16))
        self.mode = mode
        self.request = None
        self.cancel_calls = 0

    def submit(self, token_ids, *, max_new_tokens, on_token=None, on_done=None,
               on_error=None, **_kwargs):
        self.request = BatchRequest(token_ids=list(token_ids), max_new_tokens=max_new_tokens,
                                    on_token=on_token, on_done=on_done, on_error=on_error)
        if self.mode == "stream_token":
            on_token(41)
        elif self.mode == "failed":
            self.request.state = RequestState.FAILED
            self.request._event.set()
            on_error(ValueError("backend forward failed"))
        return self.request

    def cancel(self, request_id):
        if self.request is None or self.request.request_id != request_id or self.request.is_done():
            return False
        self.cancel_calls += 1
        self.request.state = RequestState.CANCELLED
        self.request._event.set()
        return True


class Phase2SessionStreamLifecycleTests(unittest.TestCase):
    def handle(self, engine):
        service = SimpleNamespace(
            _engine=engine,
            kv_cache=SimpleNamespace(find_prefix=lambda _tokens: None),
            metrics=lambda: {},
        )
        return Phase2SessionHandle(service, "session")

    def test_stream_surfaces_backend_failure(self):
        stream = self.handle(_Engine("failed")).stream([], max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "backend forward failed"):
            next(stream)

    def test_abandoned_stream_cancels_its_live_request(self):
        engine = _Engine("stream_token")
        stream = self.handle(engine).stream([], max_new_tokens=2)
        self.assertEqual(next(stream), 41)
        stream.close()
        self.assertEqual(engine.cancel_calls, 1)
        self.assertEqual(engine.request.state, RequestState.CANCELLED)

    def test_stream_cancellation_callback_failure_still_cancels_request(self):
        engine = _Engine("stream_token")

        def cancelled():
            raise LookupError("cancel observer failed")

        stream = self.handle(engine).stream([], max_new_tokens=2, cancelled=cancelled)
        with self.assertRaisesRegex(LookupError, "cancel observer failed"):
            next(stream)
        self.assertEqual(engine.cancel_calls, 1)

    def test_stream_cancel_cleanup_failure_does_not_mask_consumer_failure(self):
        class CancelFailureEngine(_Engine):
            def cancel(self, _request_id):
                raise OSError("cancel cleanup failed")

        engine = CancelFailureEngine("stream_token")
        handle = self.handle(engine)

        def cancelled():
            raise LookupError("cancel observer failed")

        stream = handle.stream([], max_new_tokens=2, cancelled=cancelled)
        with self.assertRaisesRegex(LookupError, "cancel observer failed") as raised:
            next(stream)
        self.assertTrue(any("cancel cleanup failed" in note
                            for note in getattr(raised.exception, "__notes__", ())))
        self.assertEqual(handle._active_operations, 0)

    def test_generation_cancellation_callback_failure_is_returned_to_caller(self):
        engine = _Engine()
        callback_threads = []

        def cancelled():
            callback_threads.append(threading.get_ident())
            raise LookupError("cancel observer failed")

        with self.assertRaisesRegex(LookupError, "cancel observer failed"):
            self.handle(engine).generate([], max_new_tokens=1, cancelled=cancelled)
        self.assertEqual(engine.cancel_calls, 1)
        self.assertEqual(callback_threads, [threading.get_ident()])

    def test_full_scheduler_queue_wakes_waiter_before_reentrant_error_callback(self):
        scheduler = ContinuousBatchScheduler(SchedulerConfig(max_queue_depth=0))
        callback_returned = threading.Event()
        request = BatchRequest(
            max_new_tokens=1,
            on_error=lambda _failure: (scheduler.metrics(), callback_returned.set()),
        )

        scheduler.submit(request)
        self.assertTrue(request._event.is_set())
        self.assertTrue(callback_returned.wait(1), "error callback deadlocked on scheduler lock")
        with self.assertRaisesRegex(RuntimeError, "queue full"):
            wait_for_request(request, timeout=0)

    def test_scheduler_token_callback_can_reenter_metrics(self):
        scheduler = ContinuousBatchScheduler(SchedulerConfig(max_queue_depth=1))
        callback_returned = threading.Event()
        request = BatchRequest(
            token_ids=[1],
            max_new_tokens=1,
            on_token=lambda _token: (scheduler.metrics(), callback_returned.set()),
        )
        scheduler.submit(request)
        request.state = RequestState.DECODE
        scheduler._running.append(request)
        scheduler.commit_step([(request, 7)])
        self.assertTrue(callback_returned.wait(1), "token callback deadlocked on scheduler lock")
        self.assertEqual(request.generated, [7])
        self.assertEqual(request.state, RequestState.FINISHED)

    def test_generation_budget_is_bounded_before_request_accounting(self):
        engine = _Engine()
        handle = self.handle(engine)
        with self.assertRaisesRegex(ValueError, "decode limit"):
            handle.generate([], max_new_tokens=17)
        self.assertEqual(handle.metrics()["phase2_session"]["requests_made"], 0)

    def test_unforwarded_generation_options_fail_before_batch_admission(self):
        engine = _Engine()
        handle = self.handle(engine)
        with self.assertRaisesRegex(TypeError, "temperature"):
            handle.generate([], max_new_tokens=1, temperature=0.4)
        self.assertIsNone(engine.request)
        self.assertEqual(handle.metrics()["phase2_session"]["requests_made"], 0)

        stream = handle.stream([], max_new_tokens=1, top_p=0.7)
        with self.assertRaisesRegex(TypeError, "top_p"):
            next(stream)
        self.assertIsNone(engine.request)
        self.assertEqual(handle.metrics()["phase2_session"]["requests_made"], 0)

    def test_close_waits_for_generation_lease_and_timeout_keeps_owner_open(self):
        engine = _Engine()
        service = SimpleNamespace(
            _engine=engine,
            kv_cache=SimpleNamespace(find_prefix=lambda _tokens: None),
            metrics=lambda: {},
            _close_handle=lambda _session_id: None,
        )
        owner = SimpleNamespace(closed=0, close=lambda: setattr(owner, "closed", owner.closed + 1))
        handle = Phase2SessionHandle(service, "session", owner)
        entered = threading.Event()
        release = threading.Event()
        results = []
        failures = []

        def blocking_wait(_request):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test request was not released")
            return [9]

        def generate():
            try:
                results.extend(handle.generate([], max_new_tokens=1))
            except BaseException as failure:
                failures.append(failure)

        with patch("openagent.adaptive_engine.multi_session_service.wait_for_request",
                   side_effect=blocking_wait):
            worker = threading.Thread(target=generate)
            worker.start()
            self.assertTrue(entered.wait(1), "generation did not acquire its operation lease")
            with self.assertRaisesRegex(TimeoutError, "active Phase2 operation"):
                handle.close(timeout=0)
            self.assertEqual(owner.closed, 0)
            self.assertFalse(handle._closed)
            release.set()
            worker.join(1)

        self.assertFalse(worker.is_alive(), "generation lease did not release")
        self.assertEqual(failures, [])
        self.assertEqual(results, [9])
        handle.close()
        self.assertEqual(owner.closed, 1)
        self.assertTrue(handle._closed)

    def test_interrupted_close_wait_reopens_handle_for_retry(self):
        engine = _Engine()
        service = SimpleNamespace(
            _engine=engine,
            kv_cache=SimpleNamespace(find_prefix=lambda _tokens: None),
            metrics=lambda: {},
            _close_handle=lambda _session_id: None,
        )
        owner = SimpleNamespace(closed=0, close=lambda: setattr(owner, "closed", owner.closed + 1))
        handle = Phase2SessionHandle(service, "session", owner)
        with handle._condition:
            handle._active_operations = 1
        with patch.object(handle._condition, "wait", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                handle.close()
        self.assertFalse(handle._closing)
        self.assertFalse(handle._closed)
        with handle._condition:
            handle._active_operations = 0
        handle.close()
        self.assertEqual(owner.closed, 1)
        self.assertTrue(handle._closed)

    def test_forwarded_phase1_method_holds_owner_lease_until_return(self):
        engine = _Engine()
        service = SimpleNamespace(
            _engine=engine,
            kv_cache=SimpleNamespace(find_prefix=lambda _tokens: None),
            metrics=lambda: {},
            _close_handle=lambda _session_id: None,
        )
        entered = threading.Event()
        release = threading.Event()
        owner = SimpleNamespace(closed=0)

        def flush_owner_state():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test owner callback was not released")

        def close_owner():
            owner.closed += 1

        owner.flush_owner_state = flush_owner_state
        owner.close = close_owner
        handle = Phase2SessionHandle(service, "session", owner)
        failures = []

        def invoke():
            try:
                handle.flush_owner_state()
            except BaseException as failure:
                failures.append(failure)

        worker = threading.Thread(target=invoke)
        worker.start()
        self.assertTrue(entered.wait(1), "forwarded method did not acquire its owner lease")
        with self.assertRaisesRegex(TimeoutError, "active Phase2 operation"):
            handle.close(timeout=0)
        self.assertEqual(owner.closed, 0)
        release.set()
        worker.join(1)
        self.assertFalse(worker.is_alive(), "forwarded owner method did not release its lease")
        self.assertEqual(failures, [])
        handle.close()
        self.assertEqual(owner.closed, 1)


if __name__ == "__main__":
    unittest.main()
