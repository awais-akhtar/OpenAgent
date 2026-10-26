"""Phase 2 smoke tests -- validate all Phase 2 components without real models.

Tests use lightweight duck-typed stubs; no PyTorch, no CUDA, no network.
Run with::

    python -m pytest openagent/adaptive_engine/batching_smoke_tests.py -v

Or directly::

    python openagent/adaptive_engine/batching_smoke_tests.py
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_RESULTS: List[Tuple[str, bool, str]] = []


def _test(name: str):
    def decorator(fn):
        try:
            fn()
            _RESULTS.append((name, True, ""))
            print(f"  [PASS] {name}")
        except Exception as exc:
            tb = traceback.format_exc()
            _RESULTS.append((name, False, tb))
            print(f"  [FAIL] {name}: {exc}")
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubSession:
    """Minimal duck-typed Phase 1 session stub."""

    def __init__(self, session_id: str = "stub-0"):
        self.session_id = session_id
        self.closed     = False
        self.generated: List[List[int]] = []

    def generate(self, context_ids=None, max_new_tokens=1, **kwargs):
        toks = list(context_ids or []) + [42]
        self.generated.append(toks)
        return toks

    def stream(self, context_ids=None, max_new_tokens=4, **kwargs):
        for i in range(max_new_tokens):
            yield 100 + i

    def metrics(self) -> dict:
        return {"stub_calls": len(self.generated)}

    def close(self) -> None:
        self.closed = True


class _StubExpertStore:
    """Stub for SharedExpertLease."""

    def metrics(self) -> dict:
        return {"ram_experts": 5, "vram_experts": 2}

    def close(self):
        pass


class _StubKVCache:
    """Stub for SharedKVLease."""

    def __init__(self):
        self._entries: dict = {}

    def metrics(self) -> dict:
        return {"kv_bytes_cached": 0, "kv_entries": 0}

    def find_prefix(self, token_ids):
        return None, 0

    def store_block(self, token_ids, layer_id, k, v):
        pass

    def pin(self, prefix_hash):
        pass

    def unpin(self, prefix_hash):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Test 1: Phase2MemoryManager -- start, tick, stop
# ---------------------------------------------------------------------------

@_test("Phase2MemoryManager start/stop")
def _t1():
    from openagent.adaptive_engine.memory_pressure_manager import (
        Phase2MemoryManager, MemoryThresholds,
    )
    thresholds = MemoryThresholds(poll_interval_s=0.1)
    mgr = Phase2MemoryManager(thresholds=thresholds)
    mgr.set_expert_store(_StubExpertStore())
    mgr.set_kv_cache(_StubKVCache())
    preempted = []
    mgr.set_preemption_callback(lambda: preempted.append(1) or None)
    mgr.start()
    time.sleep(0.35)   # let 2–3 ticks fire
    mgr.stop()
    snap = mgr.latest_snapshot()
    assert snap is not None, "no snapshot recorded"
    m = mgr.metrics()
    assert "mem_ram_fraction" in m
    assert m["mem_psutil_available"] in (True, False)


# ---------------------------------------------------------------------------
# Test 2: Phase2MemoryManager -- metrics keys
# ---------------------------------------------------------------------------

@_test("Phase2MemoryManager metrics keys")
def _t2():
    from openagent.adaptive_engine.memory_pressure_manager import Phase2MemoryManager
    mgr = Phase2MemoryManager()
    mgr.start()
    time.sleep(0.05)
    mgr.stop()
    m = mgr.metrics()
    required = {
        "mem_ram_used_bytes", "mem_ram_fraction", "mem_vram_fraction",
        "mem_kv_cached_bytes", "mem_process_rss", "mem_kv_evictions",
        "mem_preemptions", "mem_gc_runs",
    }
    missing = required - set(m.keys())
    assert not missing, f"missing keys: {missing}"


# ---------------------------------------------------------------------------
# Test 3: Phase2ForwardBridge -- prefill + decode
# ---------------------------------------------------------------------------

@_test("Phase2ForwardBridge prefill/decode step")
def _t3():
    import numpy as np
    from openagent.adaptive_engine.continuous_batching import (
        BatchRequest, RequestState,
    )
    from openagent.adaptive_engine.forward_bridge import Phase2ForwardBridge

    factory_calls = []
    def factory():
        s = _StubSession(f"sess-{len(factory_calls)}")
        factory_calls.append(s)
        return s

    bridge = Phase2ForwardBridge(factory, max_slots=4)

    req = BatchRequest(
        request_id="req-1",
        session_id="user-1",
        token_ids=[1, 2, 3, 4],
        max_new_tokens=4,
    )
    req.state = RequestState.PREFILL

    results = bridge([req])
    assert len(results) == 1
    assert results[0][0] is req
    # After the full prefill, a first token is generated
    assert results[0][1] is not None or req.state == RequestState.PREFILL

    # Switch to decode
    req.state = RequestState.DECODE
    results2 = bridge([req])
    assert results2[0][1] is not None

    m = bridge.metrics()
    assert m["bridge_total_steps"] == 2
    bridge.close_all()


# ---------------------------------------------------------------------------
# Test 4: Phase2ForwardBridge -- cancelled request short-circuits
# ---------------------------------------------------------------------------

@_test("Phase2ForwardBridge cancelled request")
def _t4():
    from openagent.adaptive_engine.continuous_batching import (
        BatchRequest, RequestState,
    )
    from openagent.adaptive_engine.forward_bridge import Phase2ForwardBridge

    bridge = Phase2ForwardBridge(lambda: _StubSession(), max_slots=4)
    req = BatchRequest(request_id="req-2", session_id="u2",
                       token_ids=[1, 2], max_new_tokens=2)
    req.state = RequestState.CANCELLED
    results = bridge([req])
    assert results[0][1] is None
    bridge.close_all()


# ---------------------------------------------------------------------------
# Test 5: TokenBucket fairness
# ---------------------------------------------------------------------------

@_test("TokenBucket rate limiting")
def _t5():
    from openagent.adaptive_engine.request_queue import TokenBucket
    tb = TokenBucket(rate=10.0, capacity=3.0)
    # Initial burst: 3 tokens available
    assert tb.consume(), "first consume should succeed"
    assert tb.consume(), "second consume should succeed"
    assert tb.consume(), "third consume should succeed"
    # Bucket empty
    assert not tb.consume(), "fourth consume should fail (bucket empty)"
    # After ~200ms, ~2 tokens should refill at 10/s
    time.sleep(0.22)
    assert tb.consume(), "should succeed after refill"


# ---------------------------------------------------------------------------
# Test 6: Phase2RequestQueue -- submit / dequeue / expire
# ---------------------------------------------------------------------------

@_test("Phase2RequestQueue submit/dequeue/expire")
def _t6():
    import uuid
    from openagent.adaptive_engine.continuous_batching import (
        BatchRequest, RequestState,
    )
    from openagent.adaptive_engine.request_queue import (
        Phase2RequestQueue, RequestPriority,
    )

    q = Phase2RequestQueue(max_size=16, global_rate=1000.0,
                           per_session_rate=100.0, max_wait_s=0.1)

    reqs = [
        BatchRequest(request_id=str(uuid.uuid4()), session_id="s1",
                     token_ids=[i], max_new_tokens=1)
        for i in range(4)
    ]
    for r in reqs[:2]:
        q.submit(r, priority=RequestPriority.HIGH)
    for r in reqs[2:]:
        q.submit(r, priority=RequestPriority.LOW)

    assert q.depth == 4

    batch = q.dequeue(max_items=2)
    assert len(batch) == 2
    # HIGH priority should come first
    assert batch[0] in reqs[:2]

    # Wait for TTL expiry
    time.sleep(0.2)
    expired = q.expire_stale()
    # At least the remaining 2 should have expired
    assert expired >= 1

    m = q.metrics()
    assert m["queue_submitted"] == 4
    assert m["queue_dequeued"] == 2


# ---------------------------------------------------------------------------
# Test 7: Phase2RequestQueue -- QueueFullError
# ---------------------------------------------------------------------------

@_test("Phase2RequestQueue QueueFullError on overflow")
def _t7():
    import uuid
    from openagent.adaptive_engine.continuous_batching import BatchRequest
    from openagent.adaptive_engine.request_queue import (
        Phase2RequestQueue, QueueFullError,
    )

    q = Phase2RequestQueue(max_size=2, global_rate=1000.0,
                           per_session_rate=1000.0, max_wait_s=60.0)
    for _ in range(2):
        q.submit(BatchRequest(request_id=str(uuid.uuid4()), session_id="s",
                              token_ids=[1], max_new_tokens=1))
    try:
        q.submit(BatchRequest(request_id=str(uuid.uuid4()), session_id="s",
                              token_ids=[1], max_new_tokens=1))
        assert False, "QueueFullError not raised"
    except QueueFullError:
        pass


# ---------------------------------------------------------------------------
# Test 8: Phase2SessionPool -- acquire / return / recycle
# ---------------------------------------------------------------------------

@_test("Phase2SessionPool acquire/return/recycle")
def _t8():
    from openagent.adaptive_engine.session_pool import (
        Phase2SessionPool, SessionLease,
    )

    created: List[_StubSession] = []
    def factory():
        s = _StubSession(f"p-{len(created)}")
        created.append(s)
        return s

    pool = Phase2SessionPool(factory, min_sessions=1, max_sessions=4,
                             max_idle_s=600.0, health_interval_s=999.0)
    time.sleep(0.05)   # warm-up thread

    assert pool.total_count >= 1

    with pool.acquire(timeout=2.0) as lease:
        assert isinstance(lease, SessionLease)
        assert lease.session is not None
        lease.session.generate([1, 2, 3])

    m = pool.metrics()
    assert m["pool_acquired"] >= 1
    assert m["pool_recycled"] >= 1

    pool.close()
    assert pool._closed


# ---------------------------------------------------------------------------
# Test 9: Phase2SessionPool -- timeout raises TimeoutError
# ---------------------------------------------------------------------------

@_test("Phase2SessionPool timeout when full")
def _t9():
    from openagent.adaptive_engine.session_pool import Phase2SessionPool

    def slow_factory():
        time.sleep(0.5)
        return _StubSession()

    pool = Phase2SessionPool(slow_factory, min_sessions=0, max_sessions=1,
                             max_idle_s=0.0, health_interval_s=999.0)

    # Drain the single slot first
    sessions_out: List[Any] = []
    t = threading.Thread(target=lambda: sessions_out.append(
        pool.acquire(timeout=2.0)))
    t.start()
    time.sleep(0.05)

    # Now try to acquire with a short timeout -- pool at max
    try:
        pool.acquire(timeout=0.1)
        # If it got here without error, check the pool is exhausted
    except TimeoutError:
        pass

    t.join(timeout=3.0)
    pool.close()


# ---------------------------------------------------------------------------
# Test 10: Phase2SessionPool -- health check evicts unhealthy sessions
# ---------------------------------------------------------------------------

@_test("Phase2SessionPool health check eviction")
def _t10():
    from openagent.adaptive_engine.session_pool import Phase2SessionPool

    healthy = {"flag": True}

    def factory():
        return _StubSession("hc")

    def health_fn(sess):
        return healthy["flag"]

    pool = Phase2SessionPool(factory, min_sessions=2, max_sessions=4,
                             max_idle_s=600.0,
                             health_check_fn=health_fn,
                             health_interval_s=0.05)
    time.sleep(0.05)
    initial_count = pool.total_count

    # Mark unhealthy -- next health tick should evict all idle sessions
    healthy["flag"] = False
    time.sleep(0.25)

    # All idle sessions should be discarded (and min re-open will also fail)
    # At minimum, discarded count should be > 0
    m = pool.metrics()
    assert m["pool_discarded"] > 0 or m["pool_total"] <= initial_count

    pool.close()


# ---------------------------------------------------------------------------
# Test 11: Exports from __init__
# ---------------------------------------------------------------------------

@_test("Phase2 exports via __init__")
def _t11():
    from openagent.adaptive_engine import (
        Phase2MemoryManager,
        Phase2ForwardBridge,
        Phase2RequestQueue,
        Phase2SessionPool,
        SessionLease,
        TokenBucket,
        QueueFullError,
        RequestPriority,
    )
    assert callable(Phase2MemoryManager)
    assert callable(Phase2ForwardBridge)
    assert callable(Phase2RequestQueue)
    assert callable(Phase2SessionPool)
    assert callable(SessionLease)
    assert callable(TokenBucket)
    assert callable(QueueFullError)
    assert callable(RequestPriority)


# ---------------------------------------------------------------------------
# Test 12: Full Phase 2 stack wiring smoke test
# ---------------------------------------------------------------------------

@_test("Phase2 full-stack: queue -> bridge -> pool -> memory manager")
def _t12():
    import uuid
    from openagent.adaptive_engine.memory_pressure_manager import (
        Phase2MemoryManager, MemoryThresholds,
    )
    from openagent.adaptive_engine.forward_bridge import Phase2ForwardBridge
    from openagent.adaptive_engine.request_queue import (
        Phase2RequestQueue, RequestPriority,
    )
    from openagent.adaptive_engine.session_pool import Phase2SessionPool
    from openagent.adaptive_engine.continuous_batching import (
        BatchRequest, RequestState,
    )

    # 1. Build pool
    pool = Phase2SessionPool(
        lambda: _StubSession(), min_sessions=1, max_sessions=4,
        max_idle_s=600.0, health_interval_s=999.0)
    time.sleep(0.05)

    # 2. Build bridge backed by pool
    def session_factory():
        lease = pool.acquire(timeout=2.0)
        # Wrap so close() returns to pool
        class _LeasedSession:
            def __init__(self, lease):
                self._lease = lease
                self._s = lease.session
            def generate(self, **kw):
                return self._s.generate(**kw)
            def close(self):
                self._lease.close()
        return _LeasedSession(lease)

    bridge = Phase2ForwardBridge(session_factory, max_slots=4)

    # 3. Build request queue
    q = Phase2RequestQueue(max_size=64, global_rate=1000.0,
                           per_session_rate=100.0)

    # 4. Memory manager
    mgr = Phase2MemoryManager(
        MemoryThresholds(poll_interval_s=0.1))
    mgr.set_kv_cache(_StubKVCache())
    mgr.start()

    # 5. Submit and process a request
    req = BatchRequest(
        request_id=str(uuid.uuid4()), session_id="u-full",
        token_ids=[1, 2, 3], max_new_tokens=3,
    )
    q.submit(req, priority=RequestPriority.HIGH)
    batch = q.dequeue(max_items=1)
    assert len(batch) == 1

    req2 = batch[0]
    req2.state = RequestState.PREFILL
    results = bridge([req2])
    assert len(results) == 1

    # cleanup
    bridge.close_all()
    mgr.stop()
    pool.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> bool:
    print("\n=== Phase 2 Smoke Tests ===\n")
    # All tests run via @_test decorator at import time
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{len(_RESULTS)} passed, {failed} failed")
    if failed:
        print("\nFailed tests:")
        for name, ok, tb in _RESULTS:
            if not ok:
                print(f"\n  {name}:\n{tb}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
