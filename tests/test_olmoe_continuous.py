from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_batch import OlmoeBatchSession
from openagent.adaptive_engine.olmoe_continuous import ContinuousOlmoeService
from openagent.adaptive_engine.olmoe_service import (
    OlmoeChatDecoder, OlmoeService, QueueFullError, RequestCancelledError,
    RequestDeadlineError, ServiceClosedError, ServiceRequestError,
)
if __package__:
    from .test_olmoe_batch import checkpoint
else:
    from test_olmoe_batch import checkpoint


class FixtureTokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=[ord(char) % 12 for char in text])

    def decode(self, ids):
        return "".join(chr(97 + token) for token in ids)


class ContinuousOlmoeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        checkpoint(root, True)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        config["max_position_embeddings"] = 256
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        self.manifest = pack_olmoe_experts(root, root / "experts.i8", model_id="continuous-tiny")
        self.root = root

    def model(self):
        budget = max(region.size for layer in self.manifest.layers for region in layer.regions.values())
        model = self.stack.enter_context(OlmoeModel(self.root, self.manifest, expert_cache_bytes=budget))
        model._tokenizer = FixtureTokenizer()
        model.lm_head[0].fill(1)
        model.lm_head[1].fill(-1)
        model.lm_head[12].fill(0)
        return model

    def service(self, model=None, **kwargs):
        return self.stack.enter_context(ContinuousOlmoeService(model or self.model(), **kwargs))

    def snapshot_equal(self, left, right):
        self.assertEqual(left[1], right[1])
        self.assertEqual(left[0]["position"], right[0]["position"])
        for field in ("keys", "values"):
            for a, b in zip(left[0][field], right[0][field]):
                np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5)

    def ids(self, handle):
        events = list(handle.events(timeout=3))
        self.assertEqual(events[-1].kind, "completed")
        self.assertEqual("".join(event.data["text"] for event in events if event.kind == "token"), handle.result())
        return events[-1].data["token_ids"]

    def test_real_batches_match_serial_text_ids_and_followup_snapshots(self):
        model = self.model()
        service = self.service(model, prompt_chunk_tokens=7, microbatch_size=3)
        serial_model = self.model()
        serial = self.stack.enter_context(OlmoeService(OlmoeChatDecoder(serial_model)))
        observed = []
        execute = model.experts.execute_batch

        def record(layer, hidden, routes):
            observed.append((layer, len(hidden), len(routes)))
            return execute(layer, hidden, routes)

        with patch.object(model.experts, "execute_batch", side_effect=record):
            with service._condition:
                handles = [service.submit(prompt, session_id=sid, max_tokens=budget)
                           for sid, prompt, budget in (("a", "short", 1), ("b", "longer prompt", 3), ("c", "third", 2))]
            for handle, budget in zip(handles, (1, 3, 2)):
                expected = serial.submit({"a": "short", "b": "longer prompt", "c": "third"}[handle.session_id],
                                         session_id=handle.session_id, max_tokens=budget).result(3)
                self.assertEqual(handle.result(3), expected)
                self.assertEqual(FixtureTokenizer().decode(self.ids(handle)), expected)
                self.assertEqual(service.session(handle.session_id), serial.session(handle.session_id))
                self.snapshot_equal(service._decoder_states[handle.session_id], serial._decoder_states[handle.session_id])
            with service._condition:
                followups = [service.submit("next", session_id=sid, max_tokens=2) for sid in ("a", "b")]
            for handle in followups:
                self.assertEqual(handle.result(3), serial.submit("next", session_id=handle.session_id, max_tokens=2).result(3))
                self.snapshot_equal(service._decoder_states[handle.session_id], serial._decoder_states[handle.session_id])
        self.assertTrue(any(rows > 7 for _, rows, _ in observed))
        self.assertTrue(all(rows == routes for _, rows, routes in observed))
        self.assertGreaterEqual(service.telemetry()["max_batch_sessions"], 3)
        self.assertEqual(model.position, 0)

    def test_admission_during_decode_and_no_same_session_overlap(self):
        service = self.service(prompt_chunk_tokens=256, microbatch_size=3)
        reached, release = threading.Event(), threading.Event()
        forward = service._batch.forward
        calls = []

        def gate(sessions, chunks):
            calls.append((tuple(session.position for session in sessions), tuple(map(len, chunks))))
            if len(calls) == 2:
                reached.set()
                self.assertTrue(release.wait(3))
            return forward(sessions, chunks)

        with patch.object(service._batch, "forward", side_effect=gate):
            first = service.submit("one", session_id="a", max_tokens=5)
            self.assertTrue(reached.wait(3))
            same = service.submit("followup", session_id="a", max_tokens=1)
            other = service.submit("different", session_id="b", max_tokens=1)
            with self.assertRaises(RuntimeError):
                service.clear_session("a")
            release.set()
            first.result(3)
            other.result(3)
            same.result(3)
        self.assertTrue(any(len(positions) == 2 and 0 in positions and max(positions) > 0
                            for positions, _ in calls))
        self.assertEqual(service.session("a").revision, 2)
        self.assertEqual([message.content for message in service.session("a").messages][::2], ["one", "followup"])

    def test_cancellation_and_deadline_after_forward_do_not_publish(self):
        now = [0.0]
        service = self.service(prompt_chunk_tokens=256, clock=lambda: now[0])
        service.submit("committed", session_id="a", max_tokens=1).result(3)
        saved = service._decoder_states["a"]
        history = service.session("a")
        reached, release = threading.Event(), threading.Event()
        forward = service._batch.forward

        def gate(sessions, chunks):
            result = forward(sessions, chunks)
            reached.set()
            self.assertTrue(release.wait(3))
            return result

        with patch.object(service._batch, "forward", side_effect=gate):
            with service._condition:
                cancelled = service.submit("discard", session_id="a", max_tokens=2)
                expired = service.submit("expired", session_id="b", deadline=1, max_tokens=2)
            self.assertTrue(reached.wait(3))
            self.assertTrue(cancelled.cancel())
            now[0] = 2
            release.set()
            with self.assertRaises(RequestCancelledError):
                cancelled.result(3)
            with self.assertRaises(RequestDeadlineError):
                expired.result(3)
        self.assertEqual(service.session("a"), history)
        self.assertEqual(service.session("b").revision, 0)
        self.assertIs(service._decoder_states["a"], saved)
        service.submit("retry", session_id="a", max_tokens=1).result(3)
        self.assertEqual(service.session("a").revision, 2)

    def test_last_token_consumption_and_eos_boundary_match_chat_decoder(self):
        for eos in (False, True):
            with self.subTest(eos=eos):
                model, reference = self.model(), self.model()
                if eos:
                    for item in (model, reference):
                        item.lm_head.fill(0)
                        item.config = replace(item.config, eos_token_id=0)
                service = self.service(OlmoeChatDecoder(model), prompt_chunk_tokens=8)
                serial = self.stack.enter_context(OlmoeService(OlmoeChatDecoder(reference)))
                for turn in ("first", "second"):
                    handle = service.submit(turn, session_id="a", max_tokens=2)
                    self.assertEqual(handle.result(3), serial.submit(turn, session_id="a", max_tokens=2).result(3))
                    generated = self.ids(handle)
                    self.assertEqual(len(generated), 0 if eos else 2)
                    self.snapshot_equal(service._decoder_states["a"], serial._decoder_states["a"])
                    expected_boundary = tuple(reference.encode("\n") + [reference.config.eos_token_id] + reference.encode("\n"))
                    self.assertEqual(service._decoder_states["a"][1], expected_boundary)

    def test_evicted_and_zero_budget_states_replay_full_history(self):
        for budget in (0, 4500):
            with self.subTest(budget=budget):
                model = self.model()
                service = self.service(model, session_state_capacity_bytes=budget, prompt_chunk_tokens=9)
                serial = self.stack.enter_context(OlmoeService(OlmoeChatDecoder(self.model()),
                                                              session_state_capacity_bytes=budget))
                for sid, prompt in (("a", "one"), ("b", "two"), ("a", "three")):
                    self.assertEqual(service.submit(prompt, session_id=sid, max_tokens=1).result(3),
                                     serial.submit(prompt, session_id=sid, max_tokens=1).result(3))
                    self.assertEqual(service.session(sid), serial.session(sid))
                telemetry = service.telemetry()
                self.assertLessEqual(telemetry["decoder_state_bytes"], budget)
                self.assertEqual(telemetry["decoder_states"], 0 if budget == 0 else 1)
                if budget:
                    self.assertGreater(telemetry["decoder_state_evictions"], 0)
                self.assertTrue(service.clear_session("a"))
                self.assertEqual(service.session("a").revision, 0)

    def test_active_kv_budget_priority_queue_and_shutdown(self):
        model = self.model()
        slot = 2 * 2 * 2 * 256 * 2 * 4
        service = self.service(model, active_kv_capacity_bytes=slot, microbatch_size=4,
                               queue_capacity=3, prompt_chunk_tokens=256)
        reached, release = threading.Event(), threading.Event()
        forward = service._batch.forward
        batch_sizes = []

        def gate(sessions, chunks):
            batch_sizes.append(len(sessions))
            if len(batch_sizes) == 1:
                reached.set()
                self.assertTrue(release.wait(3))
            return forward(sessions, chunks)

        starts = []
        emit = service._emit

        def record(request, kind, **data):
            if kind == "started":
                starts.append(request.prompt)
            return emit(request, kind, **data)

        with patch.object(service._batch, "forward", side_effect=gate), patch.object(service, "_emit", side_effect=record):
            active = service.submit("active", max_tokens=1)
            self.assertTrue(reached.wait(3))
            low = service.submit("low", priority=1, max_tokens=1)
            high = service.submit("high", priority=10, max_tokens=1)
            urgent = service.submit("urgent", deadline=service._clock() + 30, max_tokens=1)
            with self.assertRaises(QueueFullError):
                service.submit("overflow")
            self.assertEqual(service.telemetry()["active_kv_bytes"], slot)
            release.set()
            for handle in (active, urgent, high, low):
                handle.result(3)
        self.assertEqual(starts, ["active", "urgent", "high", "low"])
        self.assertEqual(set(batch_sizes), {1})
        self.assertLessEqual(service.telemetry()["peak_active_kv_bytes"], slot)
        service.shutdown(cancel_pending=False)
        self.assertFalse(service.telemetry()["worker_alive"])
        self.assertEqual(service.telemetry()["active_kv_bytes"], 0)
        with self.assertRaises(ServiceClosedError):
            service.submit("closed")
        with self.assertRaises(ValueError):
            ContinuousOlmoeService(model, active_kv_capacity_bytes=slot - 1)

    def test_shared_batch_failure_preserves_history_and_worker_recovers(self):
        model = self.model()
        service = self.service(model, prompt_chunk_tokens=8)
        service.submit("original", session_id="a", max_tokens=1).result(3)
        history, saved = service.session("a"), service._decoder_states["a"]
        counters = service.telemetry()
        execute = model.experts.execute_batch
        failed = [False]

        def fail(layer, hidden, routes):
            if layer.endswith(".1") and not failed[0]:
                failed[0] = True
                raise RuntimeError("tiny expert failed")
            return execute(layer, hidden, routes)

        with patch.object(model.experts, "execute_batch", side_effect=fail):
            with service._condition:
                a = service.submit("bad", session_id="a", max_tokens=1)
                b = service.submit("also bad", session_id="b", max_tokens=1)
            for handle in (a, b):
                with self.assertRaises(ServiceRequestError) as caught:
                    handle.result(3)
                self.assertEqual(caught.exception.code, "decoder_error")
                self.assertEqual(caught.exception.request_id, handle.request_id)
        self.assertEqual(service.session("a"), history)
        self.assertIs(service._decoder_states["a"], saved)
        self.assertEqual(service.session("b").revision, 0)
        after_failure = service.telemetry()
        for name in ("batch_calls", "batch_token_rows", "max_batch_sessions"):
            self.assertEqual(after_failure[name], counters[name])
        service.submit("recovered", session_id="a", max_tokens=1).result(3)
        self.assertEqual(service.session("a").revision, 2)

    def test_cancellation_during_snapshot_and_shutdown_cancel_all_lanes(self):
        service = self.service(prompt_chunk_tokens=256)
        reached, release = threading.Event(), threading.Event()
        snapshot = OlmoeBatchSession.snapshot_state

        def gate(session):
            result = snapshot(session)
            reached.set()
            self.assertTrue(release.wait(3))
            return result

        with patch.object(OlmoeBatchSession, "snapshot_state", gate):
            with service._condition:
                first = service.submit("first", session_id="a", max_tokens=1)
                second = service.submit("second", session_id="b", max_tokens=1)
            self.assertTrue(reached.wait(3))
            queued = service.submit("queued", session_id="a", max_tokens=1)
            service.shutdown(wait=False)
            release.set()
            for handle in (first, second, queued):
                with self.assertRaises(RequestCancelledError):
                    handle.result(3)
        service.shutdown()
        self.assertEqual(service.telemetry()["sessions"], 0)
        self.assertEqual(service.telemetry()["decoder_states"], 0)
        self.assertEqual(service.telemetry()["active_kv_bytes"], 0)

    def test_context_and_output_limits_fail_only_offending_lane(self):
        service = self.service(context_tokens=64, prompt_chunk_tokens=7, session_capacity_chars=1000)
        with service._condition:
            too_long = service.submit("x" * 90, session_id="bad", max_tokens=1)
            good = service.submit("ok", session_id="good", max_tokens=1)
        with self.assertRaises(ServiceRequestError) as caught:
            too_long.result(3)
        self.assertEqual(caught.exception.code, "resource_limit")
        good.result(3)
        self.assertEqual(service.session("bad").revision, 0)
        self.assertEqual(service.session("good").revision, 1)
        limited = self.service(session_capacity_chars=5, prompt_chunk_tokens=256)
        with self.assertRaises(ServiceRequestError) as caught:
            limited.submit("12345", max_tokens=1).result(3)
        self.assertEqual(caught.exception.code, "resource_limit")

    def test_active_allocations_release_even_when_failure_handles_are_retained(self):
        slot = 2 * 2 * 2 * 64 * 2 * 4
        service = self.service(context_tokens=64, active_kv_capacity_bytes=slot, prompt_chunk_tokens=256)
        references = []
        create = service._batch.create_session

        def tracked(**kwargs):
            self.assertFalse(any(reference() is not None for reference in references))
            session = create(**kwargs)
            self.assertEqual(sum(array.nbytes for array in session._k + session._v), slot)
            references.append(weakref.ref(session))
            return session

        with patch.object(service._batch, "create_session", side_effect=tracked):
            with service._condition:
                handles = [service.submit("x" * 80, session_id="bad", max_tokens=1),
                           service.submit("one", session_id="a", max_tokens=1),
                           service.submit("two", session_id="b", max_tokens=1)]
            with self.assertRaises(ServiceRequestError):
                handles[0].result(3)
            handles[1].result(3)
            handles[2].result(3)
        service.shutdown(cancel_pending=False)
        self.assertEqual(len(references), 3)
        self.assertFalse(any(reference() is not None for reference in references))

    def test_graceful_shutdown_drains_and_stream_cancellation_preserves_commit(self):
        service = self.service(prompt_chunk_tokens=256, microbatch_size=1, queue_capacity=1)
        service.submit("committed", session_id="a", max_tokens=1).result(3)
        committed = service.session("a")
        saved = service._decoder_states["a"]
        reached, release = threading.Event(), threading.Event()
        forward = service._batch.forward
        steps = [0]

        def gate(sessions, chunks):
            steps[0] += 1
            if steps[0] == 2:
                reached.set()
                self.assertTrue(release.wait(3))
            return forward(sessions, chunks)

        with patch.object(service._batch, "forward", side_effect=gate):
            active = service.submit("cancel after emission", session_id="a", max_tokens=5)
            self.assertTrue(reached.wait(3))
            queued = service.submit("remove", session_id="b", max_tokens=1)
            self.assertTrue(queued.cancel())
            replacement = service.submit("finish", session_id="b", max_tokens=1)
            self.assertTrue(active.cancel())
            service.shutdown(wait=False, cancel_pending=False)
            release.set()
            for handle in (active, queued):
                with self.assertRaises(RequestCancelledError):
                    handle.result(3)
            replacement.result(3)
        service.shutdown(cancel_pending=False)
        self.assertEqual(service.session("a"), committed)
        self.assertIs(service._decoder_states["a"], saved)
        events = list(active.events(timeout=3))
        self.assertTrue(any(event.kind == "token" for event in events))
        self.assertEqual(events[-1].kind, "cancelled")
        self.assertEqual(service.session("b").revision, 1)
        self.assertEqual(service._requests, {})


if __name__ == "__main__":
    unittest.main()
