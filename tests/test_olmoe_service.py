from __future__ import annotations

import threading
import time
import unittest

from openagent.adaptive_engine.olmoe_service import (
    ChatMessage,
    OlmoeService,
    OlmoeChatDecoder,
    QueueFullError,
    RequestCancelledError,
    RequestDeadlineError,
    ServiceClosedError,
    ServiceRequestError,
)


class FakeOlmoeModel:
    def __init__(self) -> None:
        self.position = 0
        self.prompts: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.prompts.append(text)
        return [1]

    def decode(self, ids) -> str:
        return "".join("a" if value == 1 else "b" for value in ids)

    def continue_id_stream(self, ids, *, max_tokens: int):
        self.position += len(ids)
        for token in (1, 2)[:max_tokens]:
            self.position += 1
            yield token

    def snapshot_state(self) -> object:
        return self.position

    def restore_state(self, state: object) -> None:
        self.position = int(state)

    def reset(self) -> None:
        self.position = 0


class FakeDecoder:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[tuple[ChatMessage, ...]] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def generate(self, messages, *, max_tokens, on_token, cancelled):
        snapshot = tuple(messages)
        self.calls.append(snapshot)
        self.started.set()
        if self.block:
            while not self.release.wait(0.005):
                if cancelled():
                    return None
        prompt = snapshot[-1].content
        if prompt == "explode":
            raise RuntimeError("fake decoder failed")
        prior = sum(1 for message in snapshot if message.role == "assistant")
        for token in (f"{prompt}:", str(prior)):
            if cancelled():
                return None
            if self.delay:
                time.sleep(self.delay)
            on_token(token)
        return None


class StatefulFakeDecoder(FakeDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.state = 0

    def reset_state(self) -> None:
        self.state = 0

    def snapshot_state(self) -> object:
        return self.state

    def restore_state(self, state: object) -> None:
        self.state = int(state)

    def generate(self, messages, *, max_tokens, on_token, cancelled):
        self.state += 1
        on_token(f"state-{self.state}")
        return None


class ByteStateDecoder(FakeDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.state = b""

    def reset_state(self) -> None:
        self.state = b""

    def snapshot_state(self) -> object:
        return self.state

    def restore_state(self, state: object) -> None:
        self.state = bytes(state)

    def generate(self, messages, *, max_tokens, on_token, cancelled):
        self.state += b"x" * 16
        on_token(str(len(self.state)))
        return None


class OlmoeServiceTests(unittest.TestCase):
    def test_concrete_olmoe_adapter_uses_full_then_incremental_prompt(self):
        model = FakeOlmoeModel()
        with OlmoeService(OlmoeChatDecoder(model)) as service:
            self.assertEqual(service.submit("first", session_id="a").result(1), "ab")
            self.assertEqual(service.submit("second", session_id="a").result(1), "ab")
            self.assertEqual(service.submit("other", session_id="b").result(1), "ab")
        self.assertTrue(model.prompts[0].startswith("|||IP_ADDRESS|||<|user|>\nfirst"))
        turns = [prompt for prompt in model.prompts if "<|user|>" in prompt]
        self.assertEqual(turns[1], "<|user|>\nsecond\n<|assistant|>\n")
        self.assertTrue(turns[2].startswith("|||IP_ADDRESS|||<|user|>\nother"))

    def test_snapshot_failure_does_not_commit_history(self):
        class Decoder(StatefulFakeDecoder):
            def snapshot_state(self):
                raise RuntimeError("snapshot failed")

        with OlmoeService(Decoder()) as service:
            with self.assertRaises(ServiceRequestError):
                service.submit("one", session_id="a").result(1)
            self.assertEqual(service.session("a").revision, 0)
            self.assertEqual(service.session("a").messages, ())

    def test_cancellation_during_snapshot_preserves_committed_state(self):
        class Decoder(StatefulFakeDecoder):
            def snapshot_state(self):
                if self.state == 2:
                    self.started.set()
                    self.release.wait(2)
                return self.state

        decoder = Decoder()
        with OlmoeService(decoder) as service:
            self.assertEqual(service.submit("one", session_id="a").result(1), "state-1")
            handle = service.submit("two", session_id="a")
            self.assertTrue(decoder.started.wait(1))
            self.assertTrue(handle.cancel())
            decoder.release.set()
            with self.assertRaises(RequestCancelledError):
                handle.result(1)
            self.assertEqual(service.session("a").revision, 1)
            self.assertEqual(service.submit("three", session_id="a").result(1), "state-2")

    def test_events_and_completed_registry_are_bounded(self):
        class Decoder(FakeDecoder):
            def generate(self, messages, *, max_tokens, on_token, cancelled):
                for _ in range(100):
                    on_token("x")

        with OlmoeService(Decoder(), event_capacity=8) as service:
            handle = service.submit("one", request_id="reusable")
            self.assertEqual(handle.result(1), "x" * 100)
            events = list(handle.events(timeout=1))
            self.assertLessEqual(len(events), 8)
            self.assertEqual(events[-1].kind, "completed")
            self.assertEqual(service._requests, {})
            replacement = service.submit("two", request_id="reusable")
            self.assertFalse(handle.cancel())
            replacement.result(1)

    def test_zero_budget_does_not_retain_empty_states(self):
        class Decoder(ByteStateDecoder):
            def snapshot_state(self):
                return b""

        with OlmoeService(Decoder(), session_state_capacity_bytes=0) as service:
            service.submit("one").result(1)
            self.assertEqual(service.telemetry()["decoder_states"], 0)

    def test_session_resource_limits_do_not_commit_partial_turns(self):
        with OlmoeService(FakeDecoder(), session_capacity=1, session_capacity_chars=12) as service:
            service.submit("one", session_id="a").result(1)
            for prompt, sid in (("two", "a"), ("one", "b")):
                with self.assertRaises(ServiceRequestError) as caught:
                    service.submit(prompt, session_id=sid).result(1)
                self.assertEqual(caught.exception.code, "resource_limit")
            self.assertEqual(service.session("a").revision, 1)
            self.assertEqual(service.session("b").messages, ())
            with self.assertRaises(ValueError):
                service.submit("x" * 13)

    def test_selected_requests_are_protected_and_cancelled_on_shutdown(self):
        decoder = FakeDecoder()
        decoder.block = True
        service = OlmoeService(decoder, microbatch_size=2)
        try:
            with service._condition:
                active = service.submit("active", session_id="a")
                selected = service.submit("selected", session_id="b")
            self.assertTrue(decoder.started.wait(1))
            with self.assertRaises(RuntimeError):
                service.clear_session("b")
            service.shutdown(wait=False)
            decoder.release.set()
            for handle in (active, selected):
                with self.assertRaises(RequestCancelledError):
                    handle.result(1)
            self.assertEqual(len(decoder.calls), 1)
        finally:
            decoder.release.set()
            service.shutdown()

    def test_mutable_restore_does_not_corrupt_saved_snapshot(self):
        class Decoder(StatefulFakeDecoder):
            def reset_state(self):
                self.state = []

            def snapshot_state(self):
                return self.state

            def restore_state(self, state):
                self.state = state

            def generate(self, messages, *, max_tokens, on_token, cancelled):
                self.state.append(1)
                if messages[-1].content == "fail":
                    raise RuntimeError("failed after mutation")
                on_token(str(len(self.state)))

        with OlmoeService(Decoder()) as service:
            self.assertEqual(service.submit("one", session_id="a").result(1), "1")
            with self.assertRaises(ServiceRequestError):
                service.submit("fail", session_id="a").result(1)
            self.assertEqual(service.submit("two", session_id="a").result(1), "2")

    def test_incremental_stream_consumes_last_token_and_replays_eos(self):
        class Model(FakeOlmoeModel):
            class config:
                eos_token_id = 99

            def __init__(self):
                super().__init__()
                self.consumed = []

            def continue_id_stream(self, ids, *, max_tokens):
                self.consumed.extend(ids)
                self.position += len(ids)
                yield 2
                self.consumed.append(2)
                self.position += 1

        model = Model()
        decoder = OlmoeChatDecoder(model)
        with OlmoeService(decoder) as service:
            service.submit("one", session_id="a").result(1)
            self.assertEqual(model.consumed, [1, 2])
            service.submit("two", session_id="b").result(1)
            service.submit("three", session_id="a").result(1)
            self.assertEqual(model.consumed[-5:], [1, 99, 1, 1, 2])

    def test_stream_close_failure_does_not_mask_sink_failure(self):
        class BrokenCloseStream:
            def __iter__(self):
                return self

            def __next__(self):
                return 1

            def close(self):
                raise RuntimeError("stream close failed")

        class Model(FakeOlmoeModel):
            def continue_id_stream(self, ids, *, max_tokens):
                return BrokenCloseStream()

        decoder = OlmoeChatDecoder(Model())

        def fail_sink(_text):
            raise ValueError("sink failed")

        with self.assertRaisesRegex(ValueError, "sink failed") as caught:
            decoder.generate(
                [ChatMessage("user", "hello")],
                max_tokens=1,
                on_token=fail_sink,
                cancelled=lambda: False,
            )
        self.assertTrue(any("stream close failed" in note for note in caught.exception.__notes__))

    def test_stateful_decoder_snapshots_are_isolated_per_session(self):
        decoder = StatefulFakeDecoder()
        with OlmoeService(decoder) as service:
            self.assertEqual(service.submit("one", session_id="a").result(1), "state-1")
            self.assertEqual(service.submit("one", session_id="b").result(1), "state-1")
            self.assertEqual(service.submit("two", session_id="a").result(1), "state-2")
            self.assertTrue(service.clear_session("a"))
            self.assertEqual(service.submit("new", session_id="a").result(1), "state-1")

    def test_decoder_state_budget_evicts_lru_but_preserves_messages(self):
        decoder = ByteStateDecoder()
        with OlmoeService(decoder, session_state_capacity_bytes=20) as service:
            self.assertEqual(service.submit("one", session_id="a").result(1), "16")
            self.assertEqual(service.submit("one", session_id="b").result(1), "16")
            telemetry = service.telemetry()
            self.assertEqual(telemetry["decoder_states"], 1)
            self.assertEqual(telemetry["decoder_state_bytes"], 16)
            self.assertEqual(telemetry["decoder_state_evictions"], 1)
            self.assertEqual(service.submit("two", session_id="a").result(1), "16")
            self.assertEqual(len(service.session("a").messages), 4)

    def test_streams_events_and_resumes_isolated_sessions(self):
        decoder = FakeDecoder()
        with OlmoeService(decoder) as service:
            first = service.submit("one", session_id="alpha", request_id="r1")
            self.assertEqual(first.result(1), "one:0")
            second = service.submit("two", session_id="alpha", request_id="r2")
            other = service.submit("one", session_id="beta", request_id="r3")
            self.assertEqual(second.result(1), "two:1")
            self.assertEqual(other.result(1), "one:0")

            alpha = service.session("alpha")
            beta = service.session("beta")
            self.assertEqual(alpha.revision, 2)
            self.assertEqual([item.content for item in alpha.messages], ["one", "one:0", "two", "two:1"])
            self.assertEqual(beta.revision, 1)
            self.assertEqual([item.content for item in beta.messages], ["one", "one:0"])
            kinds = [event.kind for event in first.events(timeout=1)]
            self.assertEqual(kinds[0], "queued")
            self.assertIn("selected", kinds)
            self.assertIn("started", kinds)
            self.assertEqual(kinds.count("token"), 2)
            self.assertEqual(kinds[-1], "completed")

    def test_cooperative_cancellation_does_not_commit_session(self):
        decoder = FakeDecoder(delay=0.03)
        with OlmoeService(decoder) as service:
            handle = service.submit("slow", session_id="cancelled")
            self.assertTrue(decoder.started.wait(1))
            self.assertTrue(handle.cancel())
            with self.assertRaises(RequestCancelledError):
                handle.result(1)
            self.assertEqual(service.session("cancelled").messages, ())
            kinds = [event.kind for event in handle.events(timeout=1)]
            self.assertIn("cancellation_requested", kinds)
            self.assertEqual(kinds[-1], "cancelled")

    def test_bounded_queue_priority_and_deadline_selection(self):
        decoder = FakeDecoder()
        decoder.block = True
        service = OlmoeService(decoder, queue_capacity=3, microbatch_size=2)
        try:
            active = service.submit("active")
            self.assertTrue(decoder.started.wait(1))
            low = service.submit("low", priority=1)
            high = service.submit("high", priority=10)
            urgent = service.submit("urgent", priority=0, deadline=time.monotonic() + 2)
            with self.assertRaises(QueueFullError):
                service.submit("overflow")
            decoder.release.set()
            self.assertEqual(active.result(1), "active:0")
            self.assertEqual(urgent.result(1), "urgent:0")
            self.assertEqual(high.result(1), "high:0")
            self.assertEqual(low.result(1), "low:0")
            order = [call[-1].content for call in decoder.calls]
            self.assertEqual(order, ["active", "urgent", "high", "low"])
        finally:
            decoder.release.set()
            service.shutdown()

    def test_cancelling_queued_request_releases_capacity_immediately(self):
        decoder = FakeDecoder()
        decoder.block = True
        service = OlmoeService(decoder, queue_capacity=1)
        try:
            active = service.submit("active")
            self.assertTrue(decoder.started.wait(1))
            cancelled = service.submit("cancelled")
            self.assertTrue(cancelled.cancel())
            replacement = service.submit("replacement")
            with self.assertRaises(RequestCancelledError):
                cancelled.result(1)
            decoder.release.set()
            self.assertEqual(active.result(1), "active:0")
            self.assertEqual(replacement.result(1), "replacement:0")
        finally:
            decoder.release.set()
            service.shutdown()

    def test_expired_and_decoder_errors_are_per_request(self):
        decoder = FakeDecoder()
        decoder.block = True
        with OlmoeService(decoder) as service:
            blocker = service.submit("block")
            self.assertTrue(decoder.started.wait(1))
            expired = service.submit("late", timeout=0.02)
            time.sleep(0.04)
            decoder.release.set()
            blocker.result(1)
            with self.assertRaises(RequestDeadlineError):
                expired.result(1)
            failed = service.submit("explode", session_id="errors")
            healthy = service.submit("healthy", session_id="healthy")
            with self.assertRaises(ServiceRequestError) as caught:
                failed.result(1)
            self.assertEqual(caught.exception.code, "decoder_error")
            self.assertEqual(healthy.result(1), "healthy:0")
            self.assertEqual(service.session("errors").messages, ())

    def test_concurrent_submissions_keep_session_context_separate(self):
        decoder = FakeDecoder()
        with OlmoeService(decoder, microbatch_size=4) as service:
            handles = []
            lock = threading.Lock()

            def submit(index):
                handle = service.submit(f"p{index}", session_id=f"s{index}")
                with lock:
                    handles.append(handle)

            threads = [threading.Thread(target=submit, args=(index,)) for index in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            results = {handle.session_id: handle.result(2) for handle in handles}
            self.assertEqual(len(results), 12)
            for index in range(12):
                self.assertEqual(results[f"s{index}"], f"p{index}:0")
                self.assertEqual(service.session(f"s{index}").revision, 1)

    def test_shutdown_cancels_work_and_reports_telemetry(self):
        decoder = FakeDecoder()
        decoder.block = True
        service = OlmoeService(decoder, queue_capacity=2)
        active = service.submit("active")
        self.assertTrue(decoder.started.wait(1))
        queued = service.submit("queued")
        service.shutdown(wait=False, cancel_pending=True)
        decoder.release.set()
        service.shutdown(wait=True, cancel_pending=True)
        with self.assertRaises(RequestCancelledError):
            active.result(1)
        with self.assertRaises(RequestCancelledError):
            queued.result(1)
        with self.assertRaises(ServiceClosedError):
            service.submit("new")
        telemetry = service.telemetry()
        self.assertEqual(telemetry["submitted"], 2)
        self.assertEqual(telemetry["cancelled"], 2)
        self.assertEqual(telemetry["queued"], 0)
        self.assertFalse(telemetry["worker_alive"])
        self.assertEqual(telemetry["execution_mode"], "single-decoder-worker")


if __name__ == "__main__":
    unittest.main()
