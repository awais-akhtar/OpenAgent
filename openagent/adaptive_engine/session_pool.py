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
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .failure_reporting import add_failure_note, failure_detail

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
        self._returning = False
        self._return_in_progress = False
        self._lock     = threading.RLock()
        self._acquired_at = time.monotonic()

    @property
    def session(self) -> Any:
        with self._lock:
            if self._closed or self._returning:
                raise RuntimeError('session lease has been returned')
            return self._session

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._return_in_progress:
                raise RuntimeError('session lease return reentry is unsupported')
            self._returning = True
            self._return_in_progress = True
            try:
                self._pool._return_session(self._lease_id, self._session)
            finally:
                self._return_in_progress = False
            self._closed = True

    def __enter__(self) -> "SessionLease":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            add_failure_note(exc, 'session lease return failed: ' + failure_detail(cleanup))
            try:
                exc.cleanup = self.close
            except BaseException:
                pass


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
    busy:        bool  = False
    probing:     bool  = False
    retiring:    bool  = False


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
        if not callable(session_factory):
            raise TypeError('callable session factory required')
        if (type(min_sessions) is not int or type(max_sessions) is not int or
                not 0 <= min_sessions <= max_sessions or max_sessions <= 0):
            raise ValueError('pool bounds require 0 <= min_sessions <= max_sessions and positive capacity')
        for name, value in (('max_idle_s', max_idle_s), ('health_interval_s', health_interval_s)):
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    value < 0 or (name == 'health_interval_s' and value == 0)):
                raise ValueError('finite valid pool time bound required: ' + name)
        if health_check_fn is not None and not callable(health_check_fn):
            raise TypeError('pool health check must be callable')
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
        self._opening = 0
        self._callback_threads: set[int] = set()

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
        try:
            self._warm(min_sessions)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                add_failure_note(primary, 'pool construction cleanup retained: ' + failure_detail(cleanup))
                try:
                    primary.cleanup = self.close
                except BaseException:
                    pass
            raise

    # ---- pre-warming --------------------------------------------------------

    def _warm(self, n: int) -> None:
        for _ in range(n):
            try:
                self._open_and_enqueue()
            except Exception as exc:
                logger.warning("phase2: pool warm-up failed: %s", failure_detail(exc))

    def _open_and_enqueue(self) -> Optional[_PoolEntry]:
        with self._lock:
            if self._closed or len(self._all) + self._opening >= self._max_sessions:
                return None
            self._next_id += 1
            entry_id = f"slot-{self._next_id}"
            self._opening += 1
            self._callback_threads.add(threading.get_ident())
        try:
            session = self._factory()
            if session is None:
                raise TypeError('session factory returned no owner')
            entry = _PoolEntry(session=session, entry_id=entry_id)
        except BaseException:
            with self._lock:
                self._opening -= 1
                self._callback_threads.discard(threading.get_ident())
            raise
        with self._lock:
            self._opening -= 1
            self._callback_threads.discard(threading.get_ident())
            if any(owned.session is session for owned in self._all.values()):
                raise ValueError('session factory returned an already owned session')
            self._all[entry_id] = entry
            self._total_created += 1
            if self._closed:
                self._discard_entry(entry)
                return None
            self._free.put(entry)
        return entry

    # ---- acquire / return ---------------------------------------------------

    def acquire(self, timeout: float = 10.0) -> "SessionLease":
        """Borrow a session with a bounded capacity wait.

        The timeout cannot preempt a synchronous factory or owner teardown.
        Shutdown seals admission; active callbacks remain owned until they exit.
        """
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('finite nonnegative pool acquisition timeout required')
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError('session pool is closed')
                if threading.get_ident() in self._callback_threads:
                    raise RuntimeError('session pool callback reentry is unsupported')
            # Try to get an idle session
            try:
                entry = self._free.get(timeout=min(0.05, max(0, deadline - time.monotonic())))
                with self._lock:
                    if self._closed:
                        self._discard_entry(entry)
                        raise RuntimeError('session pool is closed')
                    if self._all.get(entry.entry_id) is not entry or entry.busy or entry.probing:
                        continue
                    if not entry.healthy or entry.retiring:
                        self._discard_entry(entry)
                        continue
                    entry.last_used = time.monotonic()
                    entry.use_count += 1
                    self._total_acquired += 1
                    entry.busy = True
                return SessionLease(entry.session, self, entry.entry_id)
            except queue.Empty:
                pass

            # Try to open a new session
            with self._lock:
                can_open = not self._closed and len(self._all) + self._opening < self._max_sessions
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
            if entry.session is not session:
                raise ValueError('session lease does not own this pool entry')
            if not entry.busy and not entry.retiring:
                raise RuntimeError('session pool entry has already been returned')
            entry.busy = False
            if self._closed or entry.retiring or not entry.healthy:
                self._discard_entry(entry)
                return
            entry.last_used = time.monotonic()
            self._total_recycled += 1
            self._free.put(entry)

    def _discard_entry(self, entry: _PoolEntry) -> None:
        """Close session and remove from pool (called under lock)."""
        if self._all.get(entry.entry_id) is not entry:
            return
        entry.retiring = True
        entry.healthy = False
        self._callback_threads.add(threading.get_ident())
        try:
            close_fn = getattr(entry.session, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            self._callback_threads.discard(threading.get_ident())
        self._all.pop(entry.entry_id)
        self._total_discarded += 1

    # ---- health check -------------------------------------------------------

    def _health_loop(self) -> None:
        while not self._health_stop.wait(timeout=self._health_interval_s):
            if self._closed:
                break
            try:
                self._run_health_check()
                self._evict_stale()
            except Exception as error:
                logger.warning('phase2: pool maintenance failed: %s', failure_detail(error))

    def _run_health_check(self) -> None:
        if self._health_check_fn is None:
            return
        # Reserve a bounded idle snapshot; callbacks run outside the pool lock.
        # Shutdown must not close a session while its probe still uses it.
        with self._lock:
            count = min(32, self._free.qsize())
        for _ in range(count):
            with self._lock:
                if self._closed:
                    return
                try:
                    entry = self._free.get_nowait()
                except queue.Empty:
                    break
                if self._all.get(entry.entry_id) is not entry or entry.busy or entry.probing:
                    continue
                entry.probing = True
                self._callback_threads.add(threading.get_ident())
            probe_error = None
            try:
                ok = bool(self._health_check_fn(entry.session))
            except BaseException as error:
                probe_error = error
                ok = False
            with self._lock:
                entry.probing = False
                self._callback_threads.discard(threading.get_ident())
                if ok and not self._closed and not entry.retiring:
                    entry.healthy = True
                    self._free.put(entry)
                else:
                    try:
                        self._discard_entry(entry)
                    except BaseException as cleanup:
                        if probe_error is None:
                            raise
                        add_failure_note(probe_error, 'pool probe cleanup retained: ' + failure_detail(cleanup))
                        raise probe_error
            if probe_error is not None and not isinstance(probe_error, Exception):
                raise probe_error
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
        with self._lock:
            if self._closed:
                return
            now = time.monotonic()
            stale: List[_PoolEntry] = []
            for _ in range(self._free.qsize()):
                try:
                    entry = self._free.get_nowait()
                except queue.Empty:
                    break
                if self._all.get(entry.entry_id) is not entry or entry.busy or entry.probing:
                    continue
                if now - entry.last_used > self._max_idle_s:
                    stale.append(entry)
                else:
                    self._free.put(entry)
            primary = None
            for entry in stale:
                try:
                    self._discard_entry(entry)
                except BaseException as cleanup:
                    if primary is None:
                        primary = cleanup
                    else:
                        add_failure_note(primary, 'additional idle session cleanup failed: ' + failure_detail(cleanup))
            if primary is not None:
                raise primary

    # ---- telemetry ----------------------------------------------------------

    @property
    def idle_count(self) -> int:
        with self._lock:
            return sum(not entry.busy and not entry.probing and not entry.retiring
                       for entry in self._all.values())

    @property
    def total_count(self) -> int:
        with self._lock:
            return len(self._all)

    def metrics(self) -> dict:
        with self._lock:
            idle = sum(not entry.busy and not entry.probing and not entry.retiring
                       for entry in self._all.values())
            return {
                "pool_total":           len(self._all),
                "pool_idle":            idle,
                "pool_busy":            sum(entry.busy for entry in self._all.values()),
                "pool_opening":         self._opening,
                "pool_pending_cleanup": sum(entry.retiring for entry in self._all.values()),
                "pool_probing":         sum(entry.probing for entry in self._all.values()),
                "pool_max":             self._max_sessions,
                "pool_created":         self._total_created,
                "pool_recycled":        self._total_recycled,
                "pool_discarded":       self._total_discarded,
                "pool_acquired":        self._total_acquired,
                "pool_wait_timeouts":   self._wait_timeouts,
            }

    # ---- lifecycle ----------------------------------------------------------

    def close(self, timeout: float = 5.0) -> None:
        """Seal admission and close idle owners, retaining failed work for retry.

        ``timeout`` bounds the health-worker join, not arbitrary native close
        callbacks. Borrowed owners are closed when their leases are returned.
        """
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('finite nonnegative pool shutdown timeout required')
        with self._lock:
            if threading.get_ident() in self._callback_threads:
                raise RuntimeError('cannot close session pool from an active pool callback')
            self._closed = True
        self._health_stop.set()
        self._health_thread.join(timeout=timeout)
        if self._health_thread.is_alive():
            raise TimeoutError('pool health worker still active; retry close')
        with self._lock:
            if self._opening or any(entry.probing for entry in self._all.values()):
                raise TimeoutError('pool construction or probe still active; retry close')
            while True:
                try:
                    self._free.get_nowait()
                except queue.Empty:
                    break
            primary = None
            for entry in list(self._all.values()):
                if entry.busy:
                    continue  # Borrowers retain ownership until lease return.
                try:
                    self._discard_entry(entry)
                except BaseException as cleanup:
                    if primary is None:
                        primary = cleanup
                    else:
                        add_failure_note(primary, 'additional pool close failure: ' + failure_detail(cleanup))
            if primary is not None:
                raise primary
        logger.info("phase2: SessionPool closed (%d discarded)", self._total_discarded)

    def __enter__(self) -> "Phase2SessionPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            add_failure_note(exc, 'session pool close failed: ' + failure_detail(cleanup))
            try:
                exc.cleanup = self.close
            except BaseException:
                pass


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
