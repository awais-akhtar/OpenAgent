"""Phase 2 Session Pool — lifecycle management for Phase2SessionHandle objects.

``Phase2MultiSessionService`` (Phase 2 adapter) creates sessions on demand.
``Phase2SessionPool`` wraps it with:

1. **Pool pre-warming** — opens N sessions at startup so the first N requests
   do not pay a session-open latency.

2. **Session recycling** — finished sessions are returned to a free list and
   reused rather than destroyed.  The Phase 1 stack (TieredExpertStore, Markov
   chain, n-gram LM) stays warm across requests.

3. **Health checks** — idle sessions in the free list are probed periodically;
   unhealthy ones are discarded and replaced.

4. **Bounded pool** — at most ``max_sessions`` sessions exist simultaneously;
   excess requests wait or are rejected.

5. **Telemetry** — open / idle / busy / total-created / total-recycled counts.

Architecture
------------
``SessionLease``      -- RAII handle; returned to pool on close()
``Phase2SessionPool`` -- pool manager
``create_session_pool`` -- convenience factory
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SessionLease — RAII wrapper returned to callers
# ---------------------------------------------------------------------------


class SessionLease:
    """A borrowed session from the pool.

    Use as a context manager::

        with pool.acquire(timeout=5.0) as lease:
            tokens = lease.session.generate(...)

    Calling ``close()`` (or exiting the ``with`` block) returns the session
    to the pool rather than destroying it.
    """

    def __init__(
        self,
        session: Any,
        pool: "Phase2SessionPool",
        lease_id: str,
    ) -> None:
        self._session  = session
        self._pool     = pool
        self._lease_id = lease_id
        self._closed   = False
        self._lock     = threading.Lock()
        self._acquired_at = time.monotonic()

    @property
    def session(self) -> Any:
        return self._session

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool._return_session(self._lease_id, self._session)

    def __enter__(self) -> "SessionLease":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# PooledSessionEntry — internal state for one session in the pool
# ---------------------------------------------------------------------------

@dataclass
class _PoolEntry:
    session:     Any
    entry_id:    str
    created_at:  float = field(default_factory=time.monotonic)
    last_used:   float = field(default_factory=time.monotonic)
    use_count:   int   = 0
    healthy:     bool  = True


# ---------------------------------------------------------------------------
# Phase2SessionPool
# ---------------------------------------------------------------------------

class Phase2SessionPool:
    """Pool of Phase 1 sessions (or Phase2SessionHandles) for reuse.

    The pool is backed by a thread-safe free queue.  Callers borrow a
    ``SessionLease``; returning the lease recycles the session.

    Args:
        session_factory:  Callable ``() → session`` that opens a fresh session.
        min_sessions:     Pre-warm to this many sessions at startup.
        max_sessions:     Hard cap on total sessions (open + idle).
        max_idle_s:       Idle sessions older than this are discarded (0 = keep forever).
        health_check_fn:  Optional ``(session) → bool`` for health probing.
        health_interval_s: How often to health-check idle sessions.
    """

    def __init__(
        self,
        session_factory:   Callable[[], Any],
        *,
        min_sessions:      int   = 2,
        max_sessions:      int   = 16,
        max_idle_s:        float = 300.0,
        health_check_fn:   Optional[Callable[[Any], bool]] = None,
        health_interval_s: float = 30.0,
    ) -> None:
        self._factory          = session_factory
        self._min_sessions     = min_sessions
        self._max_sessions     = max_sessions
        self._max_idle_s       = max_idle_s
        self._health_check_fn  = health_check_fn
        self._health_interval_s = health_interval_s

        self._free:  queue.Queue = queue.Queue()
        self._all:   Dict[str, _PoolEntry] = {}   # entry_id → entry
        self._lock   = threading.RLock()
        self._closed = False
        self._next_id = 0

        # Telemetry
        self._total_created  = 0
        self._total_recycled = 0
        self._total_discarded = 0
        self._total_acquired  = 0
        self._wait_timeouts   = 0

        # Background health-check thread
        self._health_stop  = threading.Event()
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="phase2-pool-health")
        self._health_thread.start()

        # Pre-warm
        self._warm(min_sessions)

    # ---- pre-warming --------------------------------------------------------

    def _warm(self, n: int) -> None:
        for _ in range(n):
            try:
                self._open_and_enqueue()
            except Exception as exc:
                logger.warning("phase2: pool warm-up failed: %s", exc)

    def _open_and_enqueue(self) -> Optional[_PoolEntry]:
        with self._lock:
            if len(self._all) >= self._max_sessions:
                return None
            self._next_id += 1
            entry_id = f"slot-{self._next_id}"

        session = self._factory()
        entry = _PoolEntry(session=session, entry_id=entry_id)
        with self._lock:
            self._all[entry_id] = entry
            self._total_created += 1
        self._free.put(entry)
        return entry

    # ---- acquire / return ---------------------------------------------------

    def acquire(self, timeout: float = 10.0) -> "SessionLease":
        """Borrow a session from the pool.  Blocks up to ``timeout`` seconds."""
        deadline = time.monotonic() + timeout

        while True:
            # Try to get an idle session
            try:
                entry = self._free.get(timeout=0.05)
                with self._lock:
                    if not entry.healthy:
                        self._discard_entry(entry)
                        continue
                    entry.last_used = time.monotonic()
                    entry.use_count += 1
                    self._total_acquired += 1
                return SessionLease(entry.session, self, entry.entry_id)
            except queue.Empty:
                pass

            # Try to open a new session
            with self._lock:
                can_open = len(self._all) < self._max_sessions
            if can_open:
                entry = self._open_and_enqueue()
                if entry is not None:
                    continue

            if time.monotonic() >= deadline:
                with self._lock:
                    self._wait_timeouts += 1
                raise TimeoutError(
                    f"Phase2SessionPool: no session available within {timeout}s")

    def _return_session(self, entry_id: str, session: Any) -> None:
        """Called by ``SessionLease.close()``."""
        with self._lock:
            entry = self._all.get(entry_id)
            if entry is None:
                return
            if self._closed or self._max_idle_s > 0:
                age = time.monotonic() - entry.created_at
                if self._closed or age > self._max_idle_s:
                    self._discard_entry(entry)
                    return
            entry.last_used = time.monotonic()
            self._total_recycled += 1
        self._free.put(entry)

    def _discard_entry(self, entry: _PoolEntry) -> None:
        """Close session and remove from pool (called under lock)."""
        self._all.pop(entry.entry_id, None)
        self._total_discarded += 1
        close_fn = getattr(entry.session, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logger.debug("phase2: pool discard close error: %s", exc)

    # ---- health check -------------------------------------------------------

    def _health_loop(self) -> None:
        while not self._health_stop.wait(timeout=self._health_interval_s):
            if self._closed:
                break
            self._run_health_check()
            self._evict_stale()

    def _run_health_check(self) -> None:
        if self._health_check_fn is None:
            return
        # Drain the free queue, check each session, re-enqueue healthy ones
        healthy: List[_PoolEntry] = []
        drained = 0
        while drained < 32:
            try:
                entry = self._free.get_nowait()
                drained += 1
            except queue.Empty:
                break
            try:
                ok = self._health_check_fn(entry.session)
            except Exception:
                ok = False
            if ok:
                entry.healthy = True
                healthy.append(entry)
            else:
                entry.healthy = False
                with self._lock:
                    self._discard_entry(entry)
                logger.debug("phase2: pool health check failed for %s", entry.entry_id)
        for e in healthy:
            self._free.put(e)
        # Re-warm if we dropped below min
        with self._lock:
            deficit = self._min_sessions - len(self._all)
        for _ in range(max(0, deficit)):
            try:
                self._open_and_enqueue()
            except Exception:
                pass

    def _evict_stale(self) -> None:
        if self._max_idle_s <= 0:
            return
        now = time.monotonic()
        stale: List[_PoolEntry] = []
        live:  List[_PoolEntry] = []
        while True:
            try:
                entry = self._free.get_nowait()
            except queue.Empty:
                break
            if now - entry.last_used > self._max_idle_s:
                stale.append(entry)
            else:
                live.append(entry)
        for e in live:
            self._free.put(e)
        with self._lock:
            for e in stale:
                self._discard_entry(e)

    # ---- telemetry ----------------------------------------------------------

    @property
    def idle_count(self) -> int:
        return self._free.qsize()

    @property
    def total_count(self) -> int:
        with self._lock:
            return len(self._all)

    def metrics(self) -> dict:
        with self._lock:
            return {
                "pool_total":           len(self._all),
                "pool_idle":            self._free.qsize(),
                "pool_busy":            len(self._all) - self._free.qsize(),
                "pool_max":             self._max_sessions,
                "pool_created":         self._total_created,
                "pool_recycled":        self._total_recycled,
                "pool_discarded":       self._total_discarded,
                "pool_acquired":        self._total_acquired,
                "pool_wait_timeouts":   self._wait_timeouts,
            }

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._health_stop.set()
        self._health_thread.join(timeout=5.0)
        # Close all idle sessions
        while True:
            try:
                entry = self._free.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._discard_entry(entry)
        logger.info("phase2: SessionPool closed (%d discarded)", self._total_discarded)

    def __enter__(self) -> "Phase2SessionPool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_session_pool(
    session_factory: Callable[[], Any],
    *,
    min_sessions:  int   = 2,
    max_sessions:  int   = 16,
    max_idle_s:    float = 300.0,
) -> Phase2SessionPool:
    """Create and pre-warm a Phase2SessionPool."""
    return Phase2SessionPool(
        session_factory,
        min_sessions=min_sessions,
        max_sessions=max_sessions,
        max_idle_s=max_idle_s,
    )


__all__ = [
    "SessionLease",
    "Phase2SessionPool",
    "create_session_pool",
]
