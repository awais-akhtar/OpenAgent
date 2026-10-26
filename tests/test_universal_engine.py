"""Fault-injection coverage for the universal lifecycle boundary.

A deterministic owner records every call so each transaction, receipt and
cleanup rule can be checked exactly.  The real native path is covered by
``test_universal_production_path``.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.universal_engine import (
    GenerationCancelled, ModelRoute, PublicationFailed, UniversalEngine, UniversalReceipt,
    freeze_options)


ROUTE = ModelRoute("mixtral", "mixtral", "native", "fp32", "F32", "float32")


class Receipt:
    def __init__(self, owner, state):
        self.owner, self.state = owner, list(state)


class Owner:
    """Token-appending owner whose state is a list of committed tokens."""

    def __init__(self, directory=None, admission=None):
        self.directory, self.admission = directory, admission
        self.state = []
        self.calls = []
        self.close_failures = 0
        self.close_attempts = 0
        self.restore_failures = 0
        self.closed = 0
        self.released = []

    def snapshot(self):
        self.calls.append("snapshot")
        return Receipt(self, self.state)

    def restore(self, receipt):
        self.calls.append("restore")
        if self.restore_failures:
            self.restore_failures -= 1
            raise OSError("restore failed")
        if receipt.owner is not self:
            raise ValueError("foreign owner receipt")
        self.state = list(receipt.state)

    def release_snapshot(self, receipt):
        self.released.append(receipt)

    def generate(self, tokens, *, cancelled=None, fail_after=None, eos_at=None, context=None):
        self.calls.append("generate")
        produced = []
        for index, token in enumerate(tokens):
            if context is not None and len(self.state) >= context:
                raise ValueError("context limit reached")
            self.state.append(token)
            produced.append(token)
            if fail_after is not None and index + 1 == fail_after:
                raise RuntimeError("mid-token failure")
            if cancelled is not None and cancelled():
                raise InterruptedError("owner observed cancellation")
            if eos_at is not None and token == eos_at:
                break
        return produced

    def stream(self, tokens, *, cancelled=None, fail_after=None):
        self.calls.append("stream")
        for index, token in enumerate(tokens):
            self.state.append(token)
            if fail_after is not None and index + 1 == fail_after:
                raise RuntimeError("stream failure")
            yield token

    def close(self):
        self.close_attempts += 1
        if self.close_failures:
            self.close_failures -= 1
            raise OSError("close failed")
        self.closed += 1


class UniversalEngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "config.json").write_text(json.dumps(
            {"model_type": "mixtral", "architectures": ["MixtralForCausalLM"]}), encoding="utf-8")
        self.owners = []
        self.admitted = []
        self.engine = UniversalEngine(route_admission=self.admitted.append)

        def opener(directory, *, admission):
            owner = Owner(directory, admission)
            self.owners.append(owner)
            return owner

        self.engine.register("mixtral", opener, route_resolver=lambda config, directory: ROUTE)

    def open(self, **options):
        return self.engine.open(self.root, **options)

    # -- admission and warm identity ------------------------------------
    def test_route_admission_runs_before_opening(self):
        self.open()
        self.assertEqual(self.admitted, [ROUTE])

    def test_rejected_owner_cleanup_failure_remains_retryable_on_primary_error(self):
        class InvalidOwner:
            def __init__(self):
                self.close_attempts = 0

            def close(self):
                self.close_attempts += 1
                if self.close_attempts == 1:
                    raise OSError("first close fails")

        owner = InvalidOwner()
        engine = UniversalEngine(route_admission=lambda route: None)
        engine.register("mixtral", lambda directory, *, admission: owner,
                        route_resolver=lambda config, directory: ROUTE)

        with self.assertRaisesRegex(TypeError, r"snapshot\(\) and restore\(\)") as raised:
            engine.open(self.root)

        error = raised.exception
        self.assertEqual(owner.close_attempts, 1)
        self.assertTrue(callable(getattr(error, "cleanup", None)))
        self.assertTrue(any("cleanup remains retryable" in note
                            for note in getattr(error, "__notes__", ())))
        error.cleanup()
        self.assertEqual(owner.close_attempts, 2)
        engine = UniversalEngine(route_admission=lambda route: (_ for _ in ()).throw(ValueError("rejected")))
        opened = []
        engine.register("mixtral", lambda *a, **k: opened.append(1),
                        route_resolver=lambda config, directory: ROUTE)
        with self.assertRaisesRegex(ValueError, "rejected"):
            engine.open(self.root)
        self.assertEqual(opened, [])

    def test_real_matrix_admission_rejects_unadmitted_and_absent_routes(self):
        engine = UniversalEngine()
        engine.register("mixtral", Owner, route_resolver=lambda config, directory: ROUTE)
        with self.assertRaisesRegex(ValueError, "absent from the canonical source matrix"):
            engine.open(self.root, route=replace(ROUTE, persisted_format="no-such-format"))
        with self.assertRaises(ValueError):
            engine.open(self.root, route=replace(ROUTE, family="qwen35"))
        session = engine.open(self.root)
        self.assertEqual(session.manifest.route, ROUTE)

    def test_missing_resolver_and_mismatched_route_fail_closed(self):
        engine = UniversalEngine(route_admission=lambda route: None)
        opened = []
        engine.register("mixtral", lambda *args, **kwargs: opened.append(True))
        with self.assertRaisesRegex(ValueError, "exact route resolver is required"):
            engine.open(self.root)
        with self.assertRaisesRegex(ValueError, "differs from checkpoint identity"):
            engine.open(self.root, route=replace(ROUTE, model_type="olmoe"))
        # A caller-supplied canonical tuple is not source-inventory evidence.
        # Explicit-route opens must still resolve the checkpoint before an
        # adapter without its own source guard can be reached.
        with self.assertRaisesRegex(ValueError, "bind checkpoint inventory"):
            engine.open(self.root, route=ROUTE)
        self.assertEqual(opened, [])

    def test_nested_options_have_canonical_identity(self):
        first = self.open(budget={"b": [1, {"x": 2}], "a": {3, 4}})
        same = self.open(budget={"a": {4, 3}, "b": [1, {"x": 2}]})
        self.assertIs(same, first)
        self.assertIsNot(self.open(budget={"a": {3, 4}, "b": [{"x": 2}, 1]}), first)
        self.assertIsNot(self.open(flag=1), self.open(flag=True))
        self.assertNotEqual(freeze_options(1.0), freeze_options(1))
        with self.assertRaises(TypeError):
            self.open(handle=object())
        with self.assertRaises(TypeError):
            freeze_options(float("nan"))

    def test_failed_duplicate_warm_owner_close_is_retained_for_retry(self):
        engine = UniversalEngine(route_admission=lambda route: None)
        construction_barrier = threading.Barrier(2)
        owners = []
        owners_lock = threading.Lock()

        def opener(directory, *, admission):
            owner = Owner(directory, admission)
            owner.close_failures = 1
            with owners_lock:
                owners.append(owner)
            construction_barrier.wait(timeout=2)
            return owner

        engine.register("mixtral", opener,
                        route_resolver=lambda config, directory: ROUTE)
        outcomes = []

        def open_once():
            try:
                outcomes.append(engine.open(self.root))
            except BaseException as error:
                outcomes.append(error)

        threads = [threading.Thread(target=open_once) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(owners), 2)
        self.assertEqual(sum(isinstance(item, BaseException) for item in outcomes), 1)
        self.assertEqual(len(engine._retired), 1)

        with self.assertRaisesRegex(OSError, "close failed"):
            engine.close_model(self.root)
        # The orphaned duplicate was retired successfully, while the pooled
        # winner's first close remains owned by the pool for a second retry.
        self.assertEqual(engine._retired, [])
        self.assertEqual(len(engine._sessions), 1)
        engine.close_model(self.root)
        self.assertEqual(engine._retired, [])
        self.assertTrue(all(owner.close_attempts == 2 and owner.closed == 1
                            for owner in owners))

    def test_owner_without_rollback_is_rejected_and_closed(self):
        closed = []

        class NoRollback:
            def generate(self, *args, **kwargs):
                return None

            def close(self):
                closed.append(True)

        engine = UniversalEngine(route_admission=lambda route: None)
        engine.register("mixtral", lambda directory, *, admission: NoRollback(),
                        route_resolver=lambda config, directory: ROUTE)
        with self.assertRaisesRegex(TypeError, "snapshot\\(\\) and restore\\(\\)"):
            engine.open(self.root)
        self.assertEqual(closed, [True])

    # -- generate transactions ------------------------------------------
    def test_pre_dispatch_cancellation_never_touches_owner(self):
        session = self.open()
        with self.assertRaises(GenerationCancelled):
            session.generate([1], cancelled=lambda: True)
        self.assertEqual(self.owners[0].calls, [])

    def test_mid_token_cancellation_rolls_back(self):
        session = self.open()
        session.generate([1, 2])
        state = {"n": 0}

        def cancelled():
            state["n"] += 1
            return state["n"] > 2

        with self.assertRaises(GenerationCancelled) as raised:
            session.generate([3, 4, 5], cancelled=cancelled)
        self.assertIsInstance(raised.exception.__cause__, InterruptedError)
        self.assertEqual(self.owners[0].state, [1, 2])

    def test_post_generation_cancellation_rolls_back(self):
        session = self.open()
        state = {"n": 0}

        def cancelled():
            state["n"] += 1
            return state["n"] >= 4

        with self.assertRaisesRegex(GenerationCancelled, "before universal publication"):
            session.generate([7, 8], cancelled=cancelled)
        self.assertEqual(self.owners[0].state, [])

    def test_owner_failure_and_context_limit_roll_back(self):
        session = self.open()
        session.generate([1])
        with self.assertRaisesRegex(RuntimeError, "mid-token failure"):
            session.generate([2, 3, 4], fail_after=2)
        self.assertEqual(self.owners[0].state, [1])
        with self.assertRaisesRegex(ValueError, "context limit"):
            session.generate([2, 3, 4], context=2)
        self.assertEqual(self.owners[0].state, [1])

    def test_eos_commits_partial_generation(self):
        session = self.open()
        self.assertEqual(session.generate([1, 2, 3], eos_at=2), [1, 2])
        self.assertEqual(self.owners[0].state, [1, 2])
        self.assertEqual(session.status()["epoch"], 1)

    def test_sink_failure_rolls_back_and_success_publishes_once(self):
        session = self.open()
        published = []
        with self.assertRaises(PublicationFailed) as raised:
            session.generate([1], publish=lambda result: (_ for _ in ()).throw(OSError("disconnect")))
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertEqual(self.owners[0].state, [])
        session.generate([2], publish=published.append)
        self.assertEqual(published, [[2]])
        self.assertEqual(self.owners[0].state, [2])

    def test_failed_rollback_quarantines_session(self):
        session = self.open()
        self.owners[0].restore_failures = 1
        with self.assertRaises(RuntimeError) as raised:
            session.generate([1], fail_after=1)
        self.assertTrue(any("quarantined" in note for note in raised.exception.__notes__))
        self.assertFalse(session.is_usable())
        with self.assertRaisesRegex(RuntimeError, "close is pending"):
            session.generate([1])

    def test_request_receipts_are_released(self):
        session = self.open()
        session.generate([1])
        with self.assertRaises(RuntimeError):
            session.generate([2], fail_after=1)
        self.assertEqual(len(self.owners[0].released), 2)

    # -- stream transactions --------------------------------------------
    def test_stream_cancellation_is_checked_before_dispatch_and_each_event(self):
        session = self.open()
        with self.assertRaises(GenerationCancelled):
            next(session.stream([1], cancelled=lambda: True))
        self.assertEqual(self.owners[0].calls, [])
        state = {"n": 0}

        def cancelled():
            state["n"] += 1
            return state["n"] > 2

        received = []
        with self.assertRaises(GenerationCancelled):
            for token in session.stream([1, 2, 3], cancelled=cancelled):
                received.append(token)
        self.assertEqual(received, [1])
        self.assertEqual(self.owners[0].state, [])

    def test_stream_failure_and_abandonment_roll_back_completion_commits(self):
        session = self.open()
        with self.assertRaisesRegex(RuntimeError, "stream failure"):
            list(session.stream([1, 2, 3], fail_after=2))
        self.assertEqual(self.owners[0].state, [])
        iterator = session.stream([4, 5, 6])
        self.assertEqual(next(iterator), 4)
        iterator.close()
        self.assertEqual(self.owners[0].state, [])
        self.assertEqual(list(session.stream([7, 8])), [7, 8])
        self.assertEqual(self.owners[0].state, [7, 8])
        self.assertEqual(session.status()["epoch"], 1)

    def test_owner_and_publication_callbacks_cannot_reenter_lifecycle(self):
        session = self.open()
        owner = self.owners[0]
        public_receipt = session.snapshot()

        def reenter_restore(tokens, **_options):
            owner.state.extend(tokens)
            session.restore(public_receipt)

        owner.generate = reenter_restore
        with self.assertRaisesRegex(RuntimeError, "lifecycle reentry"):
            session.generate([1])
        self.assertEqual(owner.state, [])
        owner.generate = Owner.generate.__get__(owner, Owner)
        session.release_snapshot(public_receipt)

        def close_from_publish(_result):
            session.close()

        with self.assertRaisesRegex(PublicationFailed, "publication failed") as raised:
            session.generate([2], publish=close_from_publish)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertIn("lifecycle reentry", str(raised.exception.__cause__))
        self.assertEqual(owner.state, [])
        session.close()

    def test_stream_consumer_cannot_snapshot_during_active_stream(self):
        session = self.open()
        owner = self.owners[0]
        stream = session.stream([7, 8])
        self.assertEqual(next(stream), 7)
        with self.assertRaisesRegex(RuntimeError, "lifecycle reentry"):
            session.snapshot()
        stream.close()
        self.assertEqual(owner.state, [])
        self.assertTrue(session.is_usable())

    def test_paused_stream_can_be_closed_from_cancellation_thread(self):
        session = self.open()
        owner = self.owners[0]
        stream = session.stream([11, 12])
        self.assertEqual(next(stream), 11)
        failures = []

        def close_stream():
            try:
                stream.close()
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=close_stream)
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive(), "cross-thread stream close must not deadlock")
        self.assertEqual(failures, [])
        self.assertEqual(owner.state, [])
        self.assertTrue(session.is_usable())

    def test_stream_iterator_close_failure_is_reported_after_rollback(self):
        session = self.open()
        owner = self.owners[0]

        class BrokenCloseIterator:
            def __init__(self):
                self.values = iter((21, 22))

            def __iter__(self):
                return self

            def __next__(self):
                value = next(self.values)
                owner.state.append(value)
                return value

            def close(self):
                raise OSError("backend iterator close failed")

        owner.stream = lambda *_args, **_kwargs: BrokenCloseIterator()
        stream = session.stream([21, 22])
        self.assertEqual(next(stream), 21)
        with self.assertRaisesRegex(OSError, "backend iterator close failed"):
            stream.close()
        self.assertEqual(owner.state, [])
        self.assertTrue(session.is_usable())

    def test_stream_close_property_failure_cannot_skip_rollback_or_receipt_release(self):
        for backend_failure in (False, True):
            with self.subTest(backend_failure=backend_failure):
                session = self.open()
                owner = self.owners[-1]
                primary = ValueError('backend iteration failed')

                class BrokenCloseProperty:
                    def __init__(self):
                        self.first = True

                    def __iter__(self):
                        return self

                    def __next__(self):
                        if self.first:
                            self.first = False
                            owner.state.append(21)
                            return 21
                        raise primary

                    @property
                    def close(self):
                        raise OSError('iterator close lookup failed')

                owner.stream = lambda *_args, **_kwargs: BrokenCloseProperty()
                source = session.stream([21, 22])
                self.assertEqual(next(source), 21)
                with self.assertRaises(ValueError if backend_failure else OSError) as caught:
                    if backend_failure:
                        next(source)
                    else:
                        source.close()
                if backend_failure:
                    self.assertIs(caught.exception, primary)
                self.assertEqual(owner.state, [])
                self.assertEqual(len(owner.released), 1)
                self.assertTrue(session.is_usable())
                session.close()

    def test_context_manager_preserves_body_error_and_retains_cleanup_retry(self):
        session = self.open()
        owner = self.owners[0]
        owner.close_failures = 1
        primary = ValueError('context body failed')
        with self.assertRaises(ValueError) as caught:
            with session:
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertTrue(any('close failed' in note for note in primary.__notes__))
        primary.cleanup()
        self.assertTrue(session.status()['closed'])

    def test_expired_receipt_can_be_released_but_not_restored(self):
        session = self.open()
        owner = self.owners[0]
        receipt = session.snapshot()
        expired_at = receipt.created_monotonic + receipt.max_age_seconds + 1
        with patch('openagent.adaptive_engine.universal_engine.time.monotonic', return_value=expired_at):
            with self.assertRaisesRegex(ValueError, 'stale'):
                session.restore(receipt)
            session.release_snapshot(receipt)
        self.assertEqual(owner.released, [receipt._owner_receipt])
        self.assertEqual(session.status()['live_receipts'], 0)
        self.assertTrue(session.is_usable())

    def test_expired_receipt_release_still_rejects_forged_and_foreign_objects(self):
        session = self.open()
        receipt = session.snapshot()
        forged = replace(receipt)
        expired_at = receipt.created_monotonic + receipt.max_age_seconds + 1
        with patch('openagent.adaptive_engine.universal_engine.time.monotonic', return_value=expired_at):
            with self.assertRaisesRegex(ValueError, 'forged'):
                session.release_snapshot(forged)
            with self.assertRaisesRegex(ValueError, 'different session'):
                session.release_snapshot(replace(receipt, session_id='foreign'))
        self.assertEqual(self.owners[0].released, [])

    def test_snapshot_release_property_failure_retains_transaction_receipt(self):
        session = self.open()
        owner = self.owners[0]

        class LookupOwner(Owner):
            @property
            def release_snapshot(self):
                self.release_lookups += 1
                if self.release_lookups == 1:
                    raise OSError('release lookup failed')
                return super().release_snapshot

        owner.__class__ = LookupOwner
        owner.release_lookups = 0
        with self.assertRaisesRegex(RuntimeError, 'mid-token failure') as caught:
            session.generate([1], fail_after=1)
        self.assertEqual(owner.state, [])
        self.assertEqual(session.status()['pending_cleanup_receipts'], 1)
        self.assertFalse(session.is_usable())
        caught.exception.cleanup()
        self.assertEqual(session.status()['pending_cleanup_receipts'], 0)
        self.assertEqual(len(owner.released), 1)
        session.close()
        self.assertTrue(session.status()['closed'])

    def test_close_property_failure_is_recorded_and_retryable(self):
        session = self.open()
        owner = self.owners[0]

        class LookupOwner(Owner):
            @property
            def close(self):
                self.lookups += 1
                if self.lookups == 1:
                    raise OSError('owner close lookup failed')
                return super().close

        owner.__class__ = LookupOwner
        owner.lookups = 0
        with self.assertRaisesRegex(OSError, 'owner close lookup failed'):
            session.close()
        self.assertEqual(session.status()['close_error'], 'OSError')
        self.assertTrue(session.status()['closing'])
        self.engine.close_model(self.root)
        self.assertTrue(session.status()['closed'])

    def test_terminal_result_remains_committed_on_close_property_failure(self):
        session = self.open()
        owner = self.owners[0]

        class TerminalIterator:
            def __iter__(self):
                return self

            def __next__(self):
                owner.state.append(21)
                return {'event': 'result', 'result': [21]}

            @property
            def close(self):
                raise OSError('terminal iterator close lookup failed')

        owner.stream = lambda *_args, **_kwargs: TerminalIterator()
        source = session.stream([21])
        self.assertEqual(next(source), {'event': 'result', 'result': [21]})
        with self.assertRaisesRegex(OSError, 'terminal iterator close lookup failed'):
            source.close()
        self.assertEqual(owner.state, [21])
        self.assertEqual(len(owner.released), 1)
        self.assertEqual(session.status()['epoch'], 1)

    def test_new_request_waits_for_paused_stream_transaction(self):
        session = self.open()
        owner = self.owners[0]
        stream = session.stream([31, 32])
        self.assertEqual(next(stream), 31)
        completed = threading.Event()
        failures = []

        def generate_after_stream():
            try:
                session.generate([41])
            except BaseException as error:
                failures.append(error)
            finally:
                completed.set()

        worker = threading.Thread(target=generate_after_stream)
        worker.start()
        self.assertFalse(completed.wait(timeout=0.05), "concurrent request must wait for stream")
        stream.close()
        self.assertTrue(completed.wait(timeout=2), "request should proceed after stream rollback")
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(owner.state, [41])

    def test_owner_snapshot_and_close_callbacks_cannot_reenter_lifecycle(self):
        session = self.open()
        owner = self.owners[0]
        snapshot = owner.snapshot
        owner.snapshot = lambda: session.snapshot()
        with self.assertRaisesRegex(RuntimeError, "lifecycle reentry"):
            session.snapshot()
        owner.snapshot = snapshot

        close = owner.close
        owner.close = lambda: session.close()
        with self.assertRaisesRegex(RuntimeError, "lifecycle reentry"):
            session.close()
        owner.close = close
        session.close()
        self.assertTrue(session.status()["closed"])

    # -- recovery receipts ----------------------------------------------
    def test_receipt_carries_complete_binding(self):
        session = self.open()
        receipt = session.snapshot()
        self.assertIsInstance(receipt, UniversalReceipt)
        binding = receipt.as_dict()
        for field in ("version", "route", "model_identity", "session_id", "epoch",
                      "state_digest", "max_age_seconds"):
            self.assertIn(field, binding)
        self.assertEqual(tuple(binding["route"]), ROUTE.as_tuple())
        self.assertEqual(binding["session_id"], session.session_id)

    def test_restore_rejects_raw_foreign_stale_and_released_receipts(self):
        session = self.open()
        session.generate([1])
        receipt = session.snapshot()
        session.generate([2])
        session.restore(receipt)
        self.assertEqual(self.owners[0].state, [1])
        other = self.open(other=True)
        foreign = other.snapshot()
        cases = {
            "raw": receipt.as_dict(),
            "owner-receipt": receipt._owner_receipt,
            "copy": replace(receipt),
            "foreign-session": foreign,
            "foreign-route": replace(receipt, route=("x",) * 6),
            "foreign-model": replace(receipt, model_identity="0" * 64),
            "future-epoch": replace(receipt, epoch=99),
        }
        for name, value in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                session.restore(value)
        stale = session.snapshot()
        with patch("openagent.adaptive_engine.universal_engine.time.monotonic",
                   return_value=stale.created_monotonic + stale.max_age_seconds + 1):
            with self.assertRaisesRegex(ValueError, "stale"):
                session.restore(stale)
        session.release_snapshot(receipt)
        with self.assertRaisesRegex(ValueError, "unknown, released, or forged"):
            session.restore(receipt)
        self.assertEqual(self.owners[0].calls.count("restore"), 1)

    def test_partial_owner_restore_failure_quarantines_until_retained_receipt_cleanup(self):
        session = self.open()
        owner = self.owners[0]
        session.generate([1])
        receipt = session.snapshot()

        def partially_install_then_fail(_owner_receipt):
            owner.state = [99]
            raise OSError("injected partial physical restore")

        owner.restore = partially_install_then_fail
        with self.assertRaisesRegex(OSError, "partial physical restore") as caught:
            session.restore(receipt)

        self.assertEqual(owner.state, [99])
        status = session.status()
        self.assertTrue(status["closing"])
        self.assertEqual(status["live_receipts"], 1)
        with self.assertRaisesRegex(RuntimeError, "close is pending"):
            session.generate([2])
        cleanup = getattr(caught.exception, "cleanup", None)
        self.assertTrue(callable(cleanup))
        cleanup()
        self.assertTrue(session.status()["closed"])
        self.assertEqual(session.status()["live_receipts"], 0)
        self.assertEqual(owner.closed, 1)

    # -- retryable cleanup ----------------------------------------------
    def test_failed_close_is_retryable_and_blocks_new_work(self):
        session = self.open()
        self.owners[0].close_failures = 1
        with self.assertRaisesRegex(OSError, "close failed"):
            session.close()
        status = session.status()
        self.assertFalse(status["closed"])
        self.assertTrue(status["closing"])
        with self.assertRaises(RuntimeError):
            session.generate([1])
        session.close()
        self.assertTrue(session.status()["closed"])
        self.assertEqual(self.owners[0].closed, 1)
        session.close()
        self.assertEqual(self.owners[0].closed, 1)

    def test_failed_transaction_receipt_release_is_retained_and_retryable(self):
        session = self.open()
        owner = self.owners[0]
        release = owner.release_snapshot
        attempts = []

        def fail_once(receipt):
            attempts.append(receipt)
            if len(attempts) == 1:
                raise OSError("injected transaction receipt release failure")
            release(receipt)

        owner.release_snapshot = fail_once
        with self.assertRaisesRegex(RuntimeError, "mid-token failure") as caught:
            session.generate([1, 2], fail_after=1)

        self.assertEqual(owner.state, [])  # the request failure still rolled back
        self.assertTrue(session.status()["closing"])
        self.assertEqual(session.status()["pending_cleanup_receipts"], 1)
        retry = getattr(caught.exception, "cleanup", None)
        self.assertTrue(callable(retry))
        retry()
        self.assertEqual(len(owner.released), 1)
        self.assertEqual(session.status()["pending_cleanup_receipts"], 0)
        with self.assertRaisesRegex(RuntimeError, "close is pending"):
            session.generate([3])
        session.close()
        self.assertTrue(session.status()["closed"])
        self.assertEqual(owner.closed, 1)

    def test_successful_request_release_failure_is_retried_by_close(self):
        session = self.open()
        owner = self.owners[0]
        release = owner.release_snapshot
        attempts = []

        def fail_once(receipt):
            attempts.append(receipt)
            if len(attempts) == 1:
                raise OSError("injected committed receipt release failure")
            release(receipt)

        owner.release_snapshot = fail_once
        with self.assertRaisesRegex(OSError, "committed receipt release failure"):
            session.generate([1])

        self.assertEqual(owner.state, [1])  # generation committed before cleanup failed
        self.assertTrue(session.status()["closing"])
        self.assertEqual(session.status()["pending_cleanup_receipts"], 1)
        with self.assertRaises(RuntimeError):
            session.generate([2])
        session.close()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(owner.released), 1)
        self.assertEqual(session.status()["pending_cleanup_receipts"], 0)
        self.assertEqual(owner.closed, 1)

    def test_pool_keeps_session_until_close_succeeds(self):
        session = self.open()
        self.owners[0].close_failures = 1
        with self.assertRaises(OSError):
            self.engine.close_all()
        self.assertEqual(self.engine.pooled_sessions(), 1)
        replacement = self.open()
        self.assertIsNot(replacement, session)
        self.assertEqual(self.engine.pooled_sessions(), 2)
        self.engine.close_all()
        self.assertEqual(self.engine.pooled_sessions(), 0)
        self.assertTrue(session.status()["closed"])
        self.assertTrue(replacement.status()["closed"])

    def test_close_model_retains_failed_session(self):
        session = self.open()
        self.owners[0].close_failures = 1
        with self.assertRaises(OSError):
            self.engine.close_model(self.root)
        self.assertEqual(self.engine.pooled_sessions(), 1)
        self.engine.close_model(self.root)
        self.assertTrue(session.status()["closed"])
        self.assertEqual(self.engine.pooled_sessions(), 0)

    def test_unprintable_receipt_cleanup_does_not_skip_later_receipts(self):
        class UnprintableCleanupError(RuntimeError):
            def __str__(self):
                raise AssertionError("cleanup formatter must not replace the primary error")

            def __repr__(self):
                raise AssertionError("cleanup repr must not abort receipt retirement")

        session = self.open()
        session.snapshot()
        session.snapshot()
        owner = self.owners[0]
        attempts = []

        def fail_every_release(_receipt):
            attempts.append(len(attempts))
            raise UnprintableCleanupError("unavailable")

        owner.release_snapshot = fail_every_release
        with self.assertRaises(UnprintableCleanupError):
            session.close()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(owner.closed, 0)
        self.assertFalse(session.status()["closed"])


if __name__ == "__main__":
    unittest.main()
