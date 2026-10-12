"""Focused infrastructure harness. Intentionally not executed in this pass."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from . import (Budget, CacheLayout, Cancelled, DispatchStep, DurablePagedCache,
               ExecutionPlan, Geometry, PackedRegion, PackedStore, ScalableExecutor,
               ScalableSession, StateCodec, StateCodecPlanner)


class InfrastructureContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        # Projection width intentionally differs from hidden width.
        self.geometry = Geometry("fixture-v1", 7, 11, 1, 3, 1, 2, 4, 32, 8,
            frozenset({"fixture"}), (CacheLayout("custom", 4, "fixture-byte-order-v1"),))
        self.cache = DurablePagedCache(self.root / "cache", self.geometry,
            capacity_bytes=128, page_tokens=4, read_budget=Budget(64))
        payload = b"packed-fixture"
        (self.root / "weights").write_bytes(payload)
        self.store = PackedStore(self.root, {"expert": PackedRegion("weights", 0,
            len(payload), hashlib.sha256(payload).hexdigest(), "fixture")}, mapped_budget=Budget(65536))
        self.executor = ScalableExecutor(self.geometry, self.store, self.cache,
                                         scratch_budget=Budget(64))
        self.cache.create_session("a", "owner")

    def tearDown(self):
        self.executor.close()
        self.cache.close()
        self.store.close()
        self.temporary.cleanup()

    def append(self, tokens=(1, 2), *, cancelled=None):
        def step(ctx, packed):
            self.assertEqual(bytes(packed), b"packed-fixture")
            ctx.cache.append_page(tokens, b"x" * (4 * len(tokens)))
        return self.executor.dispatch("a", "owner", tokens, [DispatchStep("expert", step)],
            lambda ctx: "result", scratch_bytes=16, cancelled=cancelled)

    def test_commit_prefix_and_rollback(self):
        self.assertEqual(self.append(), "result")
        prefix = self.cache.publish_prefix("a", "owner")
        self.cache.create_session("b", "owner", prefix_id=prefix)
        self.append((3,))
        self.cache.restore_session("a", "owner", prefix)
        self.assertEqual([s["position"] for s in self.cache.sessions("owner")], [2, 2])
        self.cache.release_prefix(prefix, "owner")

    def test_exception_rolls_back_page_and_position(self):
        def fail(ctx, _):
            ctx.cache.append_page((1,), b"abcd")
            raise RuntimeError("kernel failed")
        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            self.executor.dispatch("a", "owner", (1,), [DispatchStep(None, fail)],
                lambda ctx: None, scratch_bytes=0)
        self.assertEqual(self.cache.sessions("owner")[0]["position"], 0)
        self.assertEqual(self.cache._used(), 0)

    def test_cancelled_finish_rolls_back(self):
        cancelled = [False]
        def step(ctx, _):
            ctx.cache.append_page((1,), b"abcd")
        def finish(ctx):
            cancelled[0] = True
        with self.assertRaises(Cancelled):
            self.executor.dispatch("a", "owner", (1,), [DispatchStep(None, step)], finish,
                scratch_bytes=0, cancelled=lambda: cancelled[0])
        self.assertEqual(self.cache.sessions("owner")[0]["position"], 0)

    def test_owner_and_snapshot_recovery(self):
        self.append()
        with self.assertRaises(PermissionError):
            self.cache.reset("a", "other")
        snapshot = self.cache.snapshot(self.root / "snapshot.sqlite")
        self.cache.release_session("a", "owner")
        self.cache.restore_snapshot(snapshot)
        self.assertEqual(self.cache.sessions("owner")[0]["position"], 2)

    def test_corrupt_shard_rejects_dispatch(self):
        (self.root / "weights").write_bytes(b"broken")
        with self.assertRaises(ValueError):
            self.append()
        self.assertEqual(self.cache.sessions("owner")[0]["position"], 0)

    def test_budget_failure_is_nonmutating(self):
        with self.assertRaises(MemoryError):
            self.executor.dispatch("a", "owner", (1,), (), lambda ctx: None, scratch_bytes=65)
        self.assertEqual(self.cache.sessions("owner")[0]["position"], 0)

    def test_escaped_packed_view_stays_charged(self):
        escaped = None
        try:
            with self.assertRaisesRegex(RuntimeError, "escaped"):
                with self.store.acquire("expert") as payload:
                    escaped = payload[:]
            self.assertGreater(self.store.budget.used_bytes, 0)
            with self.assertRaises(RuntimeError):
                self.store.close()
        finally:
            if escaped is not None:
                escaped.release()
            self.store._pool.reap()
        self.assertEqual(self.store.budget.used_bytes, 0)

    def test_ingestion_reentrancy_cannot_reset_or_restore(self):
        self.append()
        session = ScalableSession(self.executor, "a", "owner", lambda tokens: None,
                                  vocab_size=8, create=False)
        checkpoint = session.snapshot()
        self.append((3,))

        class Iterator:
            def __iter__(inner):
                session.reset()
                yield 1

        class IndexToken:
            def __index__(inner):
                session.restore(checkpoint)
                return 1

        try:
            for source in (Iterator(), [IndexToken()]):
                with self.assertRaisesRegex(RuntimeError, "reentrant"):
                    session.forward(source)
                self.assertEqual(session.position, 3)
            session.reset()  # Ingestion failure must also release the active guard.
            self.assertEqual(session.position, 0)
        finally:
            session.release_snapshot(checkpoint)
            session.close(preserve_durable_session=True)

    def test_nonfinite_logits_rollback_before_commit(self):
        import numpy as np
        for invalid in (np.nan, np.inf, -np.inf):
            def planner(tokens):
                def append(ctx, _):
                    ctx.cache.append_page(tokens, b"abcd" * len(tokens))
                return ExecutionPlan([DispatchStep(None, append)],
                    lambda ctx: np.full((len(tokens), 8), invalid), 16)
            session = ScalableSession(self.executor, "a", "owner", planner,
                                      vocab_size=8, create=False)
            try:
                with self.assertRaisesRegex(ValueError, "finite"):
                    session.forward((1,))
                self.assertEqual(session.position, 0)
                self.assertEqual(self.cache._used(), 0)
            finally:
                session.close(preserve_durable_session=True)

    def test_codec_planner_restores_pages_and_disposes_state(self):
        import numpy as np
        decoded, disposed = [], []
        codec = StateCodec(self.geometry.identity, self.geometry.cache_layouts,
            create=lambda position: {"position": position},
            decode_page=lambda state, view, metadata: decoded.append(
                (metadata.position, metadata.token_ids, bytes(view))),
            encode_pages=lambda state, tokens, page_tokens: [(tokens, b"abcd" * len(tokens))],
            dispose=lambda state: disposed.append(state["position"]))
        planner = StateCodecPlanner(self.geometry, codec, phases=lambda ctx, state: (),
            logits=lambda ctx, state: np.zeros((len(ctx.token_ids), 8)),
            scratch_estimate=lambda tokens, context_limit: 32)
        session = ScalableSession(self.executor, "a", "owner", planner, vocab_size=8, create=False)
        try:
            session.forward((1, 2))
            session.forward((3,))
            self.assertEqual(decoded, [(0, (1, 2), b"abcdabcd")])
            self.assertEqual(disposed, [0, 2])
            self.assertEqual(session.position, 3)
        finally:
            session.close(preserve_durable_session=True)

    def test_cleanup_failure_rolls_back_append(self):
        def cleanup():
            raise RuntimeError("cleanup failed")
        def step(ctx, _):
            ctx.defer(cleanup)
            ctx.cache.append_page((1,), b"abcd")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.executor.dispatch("a", "owner", (1,), [DispatchStep(None, step)],
                lambda ctx: None, scratch_bytes=16)
        self.assertEqual(self.cache.sessions("owner")[0]["position"], 0)
