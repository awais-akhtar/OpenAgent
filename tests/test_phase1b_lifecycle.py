"""Phase 1B lifecycle: threads stop, base closes once, failures surface and retry."""
import asyncio
import hashlib
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.a12_unified_inference import TieredExpertStore
from openagent.adaptive_engine.a14_lookahead_prefetch import LearningCache, LookaheadPrefetcher
from openagent.adaptive_engine.expert_runtime import ExpertLayer, SparseExpertManifest, SparseExpertRuntime
from openagent.adaptive_engine.expert_store import ExpertRegion
from openagent.adaptive_engine.session_optimizer import Phase1BOptimizedSession
from openagent.adaptive_engine.async_io_engine import AdvancedAsyncIOPool
from openagent.adaptive_engine.memory_pinning import JITWeightPinner
from openagent.adaptive_engine.universal_engine import ModelManifest, ModelRoute


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


class ThreadAccounting(unittest.TestCase):
    def setUp(self):
        self.before = {thread.ident for thread in threading.enumerate()}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def assert_no_new_threads(self):
        alive = [thread.name for thread in threading.enumerate()
                 if thread.ident not in self.before and thread.is_alive()]
        self.assertEqual(alive, [])


class ComponentLifecycleTests(ThreadAccounting):
    def test_jit_pinner_requires_loader_pins_real_bytes_and_stops(self):
        store = TieredExpertStore(self.root)
        with self.assertRaisesRegex(RuntimeError, "explicit expert payload loader"):
            JITWeightPinner(store).start_pin_thread()
        payloads = {"e1": b"abc", "e2": b""}
        pinner = JITWeightPinner(store, loader=payloads.__getitem__)
        pinner.start_pin_thread()
        pinner.request_pin({"e1", "e2"}, layer_id=0)
        wait_until(lambda: pinner.total_pins_completed + pinner.pin_errors == 2)
        self.assertEqual(store.ram_hot_store["e1"], b"abc")
        self.assertNotIn("e2", store.ram_hot_store)
        self.assertEqual(pinner.pin_errors, 1)
        pinner.stop_pin_thread()
        self.assertIsNone(pinner.pin_thread)
        self.assert_no_new_threads()

    def test_async_pool_returns_reader_bytes_and_shuts_down(self):
        pool = AdvancedAsyncIOPool(num_workers=3, reader=lambda key: key.encode() * 2)
        pool.start_workers()

        async def load():
            return await asyncio.gather(pool.async_load_expert("a"), pool.async_load_expert("b"))

        self.assertEqual(asyncio.run(load()), [b"aa", b"bb"])
        self.assertEqual(pool.total_bytes_read, 4)
        pool.shutdown()
        self.assertEqual(pool.worker_threads, [])
        self.assert_no_new_threads()
        with self.assertRaisesRegex(RuntimeError, "explicit reader"):
            AdvancedAsyncIOPool().start_workers()

    def test_async_pool_reader_failure_reaches_the_awaiting_caller(self):
        def reader(key):
            raise OSError("disk gone")

        pool = AdvancedAsyncIOPool(num_workers=1, reader=reader)
        pool.start_workers()
        try:
            with self.assertRaisesRegex(OSError, "disk gone"):
                asyncio.run(pool.async_load_expert("a"))
            self.assertEqual(pool.failed_reads, 1)
        finally:
            pool.shutdown()
        self.assert_no_new_threads()

    def test_prefetcher_stop_keeps_its_event_and_joins(self):
        prefetcher = LookaheadPrefetcher(router=None, expert_store=TieredExpertStore(self.root))
        event = prefetcher.stop_prefetch
        prefetcher.start_prefetch_thread()
        prefetcher.queue_prefetch({1: {"e"}})
        prefetcher.stop_prefetch_thread()
        self.assertIs(prefetcher.stop_prefetch, event)
        self.assertIsNone(prefetcher.prefetch_thread)
        self.assert_no_new_threads()

    def test_learning_caches_never_default_to_a_shared_home_file(self):
        self.assertIsNone(TieredExpertStore(self.root).learning_cache_path)
        cache = LearningCache()
        self.assertIsNone(cache.cache_file)
        with self.assertRaisesRegex(RuntimeError, "explicit persistence path"):
            cache.save()
        explicit = LearningCache(self.root / "history.json")
        explicit.record_routing([{"layer_id": 0, "expert_ids": ["e"]}])
        explicit.save()
        self.assertTrue((self.root / "history.json").is_file())


class Base:
    """Universal-session stand-in that records close calls."""

    def __init__(self, root, runtime, close_failures=0):
        route = ModelRoute("olmoe", "olmoe", "native", "fp32", "F32", "float32")
        self.manifest = ModelManifest(root / "checkpoint", "olmoe", "olmoe", route, {}, "c" * 64)
        self.owner = SimpleNamespace(expert_telemetry_sources=lambda: (runtime,) if runtime else (),
                                     checkpoint_identity="d" * 64)
        self.close_failures = close_failures
        self.closed = 0
        self.requests = []

    def generate(self, *args, cancelled=None, **kwargs):
        self.requests.append(args)
        return {"ok": True}

    def status(self):
        return {"closed": bool(self.closed)}

    def close(self):
        if self.close_failures:
            self.close_failures -= 1
            raise OSError("base close failed")
        self.closed += 1


class FacadeLifecycleTests(ThreadAccounting):
    def setUp(self):
        super().setUp()
        weights = [np.full(45, index, dtype="<f4") for index in range(3)]
        (self.root / "experts.bin").write_bytes(b"".join(w.tobytes() for w in weights))
        size = weights[0].nbytes
        regions = {str(i): ExpertRegion("experts.bin", i * size, size,
                                        hashlib.sha256(w.tobytes()).hexdigest())
                   for i, w in enumerate(weights)}
        layer = ExpertLayer(id="block.0", hidden_size=3, intermediate_size=5, experts_used=2,
                            dtype="float32", activation="swiglu", regions=regions)
        self.manifest = SparseExpertManifest(model_id="fixture", architecture="fixture",
                                             fingerprint="f" * 64, layers=(layer,))
        self.runtime = SparseExpertRuntime(self.root, self.manifest, capacity_bytes=3 * size)
        self.addCleanup(self.runtime.close)
        self.cache = self.root / "phase1b"

    def test_facade_starts_no_threads_and_closes_base_exactly_once(self):
        base = Base(self.root, self.runtime)
        facade = Phase1BOptimizedSession(base, cache_root=self.cache)
        facade.generate([1])
        self.assert_no_new_threads()
        facade.close()
        facade.close()
        self.assertEqual(base.closed, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            facade.generate([1])

    def test_cleanup_failures_are_raised_and_retry_skips_completed_steps(self):
        base = Base(self.root, self.runtime, close_failures=1)
        facade = Phase1BOptimizedSession(base, cache_root=self.cache)
        with patch.object(self.runtime, "save_route_profile", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                facade.close()
        self.assertEqual(base.closed, 0)
        with self.assertRaisesRegex(OSError, "base close failed"):
            facade.close()
        self.assertTrue(list(facade.state_directory.glob("route-profile-*.json")))
        with patch.object(self.runtime, "save_route_profile", side_effect=AssertionError("saved twice")):
            facade.close()
        self.assertEqual(base.closed, 1)

    def test_unprintable_secondary_profile_failure_does_not_stop_cleanup_attempts(self):
        class UnprintableError(RuntimeError):
            def __repr__(self):
                raise AssertionError("secondary cleanup repr must not stop later owners")

            def __str__(self):
                raise AssertionError("secondary cleanup text must not mask the first failure")

        facade = Phase1BOptimizedSession(Base(self.root, self.runtime), cache_root=self.cache)
        facade._runtimes = (self.runtime, self.runtime, self.runtime)
        attempts = []

        def save(_path):
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise OSError("first profile failure")
            if len(attempts) == 2:
                raise UnprintableError("second profile failure")
            raise OSError("third profile failure")

        with patch.object(self.runtime, "save_route_profile", side_effect=save):
            with self.assertRaisesRegex(OSError, "first profile failure"):
                facade.close()
        self.assertEqual(len(attempts), 3)
        self.assertEqual(facade.base_session.closed, 0)

    def test_runtime_close_continues_after_unprintable_secondary_failure(self):
        from openagent.adaptive_engine.runtime import AdaptiveExecutionEngine

        class UnprintableError(RuntimeError):
            def __repr__(self):
                raise AssertionError("cleanup repr must not skip later runtime owners")

            def __str__(self):
                raise AssertionError("cleanup text must not mask the primary failure")

        class Closer:
            def __init__(self, failure=None):
                self.failure = failure
                self.calls = 0

            def close(self):
                self.calls += 1
                if self.failure is not None:
                    raise self.failure

        class UniversalCloser(Closer):
            def close_all(self):
                self.close()

        runtime = object.__new__(AdaptiveExecutionEngine)
        runtime._phase1b_lock = threading.RLock()
        first = Closer(OSError("first facade failure"))
        second = Closer(UnprintableError("second facade failure"))
        runtime._phase1b_sessions = {1: first, 2: second}
        runtime.universal = UniversalCloser()
        runtime.direct = Closer()

        with self.assertRaisesRegex(OSError, "first facade failure"):
            runtime.close()
        self.assertEqual((first.calls, second.calls), (1, 1))
        self.assertEqual(runtime.universal.calls, 1)
        self.assertEqual(runtime.direct.calls, 1)

    def test_metrics_are_none_until_observed_and_never_placeholders(self):
        facade = Phase1BOptimizedSession(Base(self.root, self.runtime), cache_root=self.cache)
        metrics = facade.metrics()
        self.assertIsNone(metrics["batch_union_reduction"])
        self.assertIsNone(metrics["async_read_hit_rate"])
        self.runtime.record_external_routes("block.0", [["0", "1"], ["0", "2"]])
        self.assertAlmostEqual(facade.metrics()["batch_union_reduction"], 1 - 3 / 4)
        dense = Phase1BOptimizedSession(Base(self.root, None), cache_root=self.cache)
        self.assertFalse(dense.metrics()["expert_streaming"])

    def test_state_is_isolated_and_never_inside_the_checkpoint(self):
        first = Phase1BOptimizedSession(Base(self.root, None), cache_root=self.cache)
        other = Base(self.root, None)
        other.manifest = ModelManifest(self.root / "checkpoint-b", "olmoe", "olmoe",
                                       other.manifest.route, {}, "c" * 64)
        second = Phase1BOptimizedSession(other, cache_root=self.cache)
        self.assertNotEqual(first.state_directory, second.state_directory)
        with self.assertRaisesRegex(ValueError, "inside the checkpoint"):
            Phase1BOptimizedSession(Base(self.root, None), cache_root=self.root / "checkpoint" / "x")


if __name__ == "__main__":
    unittest.main()
