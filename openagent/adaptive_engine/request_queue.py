"""Phase 2 Request Queue — priority scheduling with fairness and rate limiting.

The ``ContinuousBatchScheduler`` (Phase 2C) uses a simple FIFO wait list.
``Phase2RequestQueue`` replaces that with a production-grade queue that adds:

1. **Multi-level priority** — HIGH / NORMAL / LOW buckets; within each bucket,
   requests are ordered by submission time (FIFO fairness).

2. **Per-session fairness** — each session_id gets a token bucket so one
   aggressive caller cannot starve others.

3. **Rate limiting** — global tokens-per-second cap on admissions.

4. **Deadline / TTL** — requests older than ``max_wait_s`` are expired with
   ``RequestState.FAILED`` so callers time out gracefully.

5. **Back-pressure** — ``submit()`` raises ``QueueFullError`` when the queue
   is at capacity (rather than silently dropping).

Architecture
------------
``RequestPriority``       -- enum: HIGH / NORMAL / LOW
``TokenBucket``           -- leaky-bucket rate limiter (per session or global)
``QueueFullError``        -- raised on submit when over capacity
``Phase2RequestQueue``    -- full-featured priority queue
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

from .continuous_batching import BatchRequest, RequestState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RequestPriority
# ---------------------------------------------------------------------------

class RequestPriority(IntEnum):
    HIGH   = 0
    NORMAL = 1
    LOW    = 2

    @classmethod
    def from_int(cls, value: int) -> "RequestPriority":
        return cls(max(0, min(2, value)))


# ---------------------------------------------------------------------------
# TokenBucket — leaky-bucket rate limiter
# ---------------------------------------------------------------------------

class TokenBucket:
    """Token-bucket rate limiter.

    ``consume()`` returns True if a token is available (request can proceed)
    or False if the bucket is empty (caller should wait or reject).
    """

    def __init__(self, rate: float, capacity: float) -> None:
        """
        Args:
            rate:     tokens per second refill rate.
            capacity: maximum burst capacity.
        """
        if (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 or
                type(capacity) not in (int, float) or not math.isfinite(capacity) or
                capacity <= 0):
            raise ValueError("finite positive token rate and capacity required")
        self._rate     = float(rate)
        self._capacity = max(float(capacity), 1.0)
        # Start full at the admitted capacity. The previous value used the
        # caller's raw capacity even though _capacity is clamped to at least
        # one, leaving small but valid configured rates unable to admit their
        # first request until the refill interval elapsed.
        self._tokens   = self._capacity
        self._last_refill = time.monotonic()
        self._lock     = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity,
                               self._tokens + elapsed * self._rate)
            self._last_refill = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def available(self) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            return min(self._capacity, self._tokens + elapsed * self._rate)

    def refund(self, tokens: float = 1.0) -> None:
        """Return a token when a compound admission fails at a later gate."""
        if type(tokens) not in (int, float) or not math.isfinite(tokens) or tokens < 0:
            raise ValueError("finite nonnegative token refund required")
        with self._lock:
            self._tokens = min(self._capacity, self._tokens + tokens)


# ---------------------------------------------------------------------------
# QueueFullError
# ---------------------------------------------------------------------------

class QueueFullError(RuntimeError):
    """Raised when the queue is at capacity and back-pressure kicks in."""


# ---------------------------------------------------------------------------
# Phase2RequestQueue
# ---------------------------------------------------------------------------

_PRIORITY_BUCKETS = len(RequestPriority)


@dataclass
class _QueueEntry:
    """Internal entry in the priority queue."""
    request:     BatchRequest
    priority:    RequestPriority
    inserted_at: float = field(default_factory=time.monotonic)
    expires_at:  Optional[float] = None
    max_wait_s:  Optional[float] = None


class Phase2RequestQueue:
    """Production-grade priority queue for ``BatchRequest`` objects.

    Usage::

        queue = Phase2RequestQueue(max_size=512, global_rate=100.0)

        # Submit from any session thread
        queue.submit(request, priority=RequestPriority.HIGH)

        # Called by the scheduler each step
        batch = queue.dequeue(max_items=8)
        queue.expire_stale()
    """

    def __init__(
        self,
        max_size:          int   = 512,
        global_rate:       float = 1000.0,  # max requests/s across all sessions
        per_session_rate:  float = 50.0,    # max requests/s per session_id
        per_session_burst: float = 10.0,
        max_wait_s:        float = 60.0,    # TTL: expire after this many seconds
    ) -> None:
        if type(max_size) is not int or max_size <= 0:
            raise ValueError("positive request queue capacity required")
        for name, value in (("global_rate", global_rate),
                            ("per_session_rate", per_session_rate),
                            ("per_session_burst", per_session_burst),
                            ("max_wait_s", max_wait_s)):
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    value < 0 or (name != "max_wait_s" and value == 0)):
                raise ValueError(f"finite positive {name} required" if name != "max_wait_s"
                                 else "finite nonnegative max_wait_s required")
        self._max_size         = max_size
        self._max_wait_s       = max_wait_s

        # One list per priority level, ordered by insertion time
        self._buckets: List[List[_QueueEntry]] = [[] for _ in range(_PRIORITY_BUCKETS)]
        self._lock = threading.Lock()

        # Rate limiters
        self._global_bucket   = TokenBucket(global_rate, global_rate * 0.5)
        self._per_session_rate  = per_session_rate
        self._per_session_burst = per_session_burst
        self._session_buckets: Dict[str, TokenBucket] = {}

        # Telemetry
        self._submitted    = 0
        self._dequeued     = 0
        self._expired      = 0
        self._rate_dropped = 0
        self._full_dropped = 0
        self._admitted = 0

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def max_wait_s(self) -> float:
        return self._max_wait_s

    def admit(self, request: BatchRequest) -> bool:
        """Apply configured rate limits before a request enters the scheduler.

        The continuous-batching scheduler owns queue capacity and dequeue
        order; this shared admission gate owns global/per-session token rates
        and attaches the request's queue-wait deadline.
        """
        if not isinstance(request, BatchRequest):
            raise TypeError("batch request required for queue admission")
        failure = None
        with self._lock:
            if not self._global_bucket.consume():
                failure = RuntimeError("global rate limit exceeded")
            else:
                sid = request.session_id or "_default"
                bucket = self._session_buckets.get(sid)
                if bucket is None:
                    bucket = TokenBucket(self._per_session_rate,
                                         self._per_session_burst)
                    self._session_buckets[sid] = bucket
                if not bucket.consume():
                    # An unsuccessful per-session admission must not also
                    # consume one global request token.
                    self._global_bucket.refund()
                    failure = RuntimeError("per-session rate limit exceeded")
            if failure is not None:
                self._rate_dropped += 1
                request.state = RequestState.FAILED
                request.error = failure
                request._event.set()
            else:
                request.expires_at = (
                    time.monotonic() + self._max_wait_s
                    if self._max_wait_s > 0 else None
                )
                self._admitted += 1
        if failure is not None:
            if request.on_error:
                try:
                    request.on_error(failure)
                except Exception:
                    pass
            return False
        return True

    def record_full_rejection(self, request: BatchRequest) -> None:
        """Record scheduler rejection and refund the admission rate tokens."""
        if not isinstance(request, BatchRequest):
            raise TypeError("batch request required for queue rejection accounting")
        with self._lock:
            self._full_dropped += 1
            self._global_bucket.refund()
            sid = request.session_id or "_default"
            bucket = self._session_buckets.get(sid)
            if bucket is not None:
                bucket.refund()

    # ---- submission ---------------------------------------------------------

    def submit(
        self,
        request: BatchRequest,
        priority: RequestPriority = RequestPriority.NORMAL,
        max_wait_s: Optional[float] = None,
    ) -> None:
        """Enqueue a request.  Raises ``QueueFullError`` if at capacity."""
        if not isinstance(request, BatchRequest):
            raise TypeError("BatchRequest required")
        if type(priority) not in (int, RequestPriority) or not 0 <= int(priority) < _PRIORITY_BUCKETS:
            raise ValueError("valid request priority required")
        ttl = self._max_wait_s if max_wait_s is None else max_wait_s
        if (type(ttl) not in (int, float) or not math.isfinite(ttl) or ttl < 0):
            raise ValueError("finite nonnegative request wait bound required")
        failure = None
        full = False
        with self._lock:
            total = sum(len(b) for b in self._buckets)
            if total >= self._max_size:
                self._full_dropped += 1
                failure = QueueFullError(f"Request queue full ({total}/{self._max_size})")
                full = True
            elif not self._global_bucket.consume():
                self._rate_dropped += 1
                failure = RuntimeError("global rate limit exceeded")
            else:
                sid = request.session_id or "_default"
                bucket = self._session_buckets.get(sid)
                if bucket is None:
                    bucket = TokenBucket(self._per_session_rate, self._per_session_burst)
                    self._session_buckets[sid] = bucket
                if not bucket.consume():
                    self._global_bucket.refund()
                    self._rate_dropped += 1
                    failure = RuntimeError("per-session rate limit exceeded")
            if failure is not None:
                request.state = RequestState.FAILED
                request.error = failure
                request._event.set()
            else:
                entry = _QueueEntry(
                    request=request,
                    priority=RequestPriority(int(priority)),
                    expires_at=time.monotonic() + ttl if ttl > 0 else None,
                    max_wait_s=ttl if ttl > 0 else None,
                )
                self._buckets[int(priority)].append(entry)
                self._submitted += 1
        if failure is not None:
            if request.on_error:
                try:
                    request.on_error(failure)
                except Exception:
                    pass
            if full:
                raise failure

    # ---- dequeue ------------------------------------------------------------

    def dequeue(self, max_items: int = 8) -> List[BatchRequest]:
        """Remove and return up to ``max_items`` requests, highest priority first."""
        if type(max_items) is not int or max_items <= 0:
            raise ValueError("positive dequeue batch size required")
        results: List[BatchRequest] = []
        with self._lock:
            for bucket in self._buckets:
                while bucket and len(results) < max_items:
                    entry = bucket.pop(0)
                    results.append(entry.request)
                    self._dequeued += 1
                if len(results) >= max_items:
                    break
        return results

    def peek(self, max_items: int = 8) -> List[BatchRequest]:
        """Return requests without removing them."""
        results: List[BatchRequest] = []
        with self._lock:
            for bucket in self._buckets:
                for entry in bucket:
                    if len(results) >= max_items:
                        break
                    results.append(entry.request)
        return results

    # ---- TTL expiry ---------------------------------------------------------

    def expire_stale(self) -> int:
        """Expire requests that have waited past their TTL.  Returns count."""
        now = time.monotonic()
        expired = 0
        callbacks = []
        with self._lock:
            for bucket in self._buckets:
                live = []
                for entry in bucket:
                    if entry.expires_at is not None and now > entry.expires_at:
                        entry.request.state = RequestState.FAILED
                        failure = TimeoutError(
                            f"Request {entry.request.request_id} expired after "
                            f"{entry.max_wait_s}s"
                        )
                        entry.request.error = failure
                        entry.request._event.set()
                        if entry.request.on_error:
                            callbacks.append((entry.request.on_error, failure))
                        expired += 1
                        self._expired += 1
                    else:
                        live.append(entry)
                bucket[:] = live
        for callback, failure in callbacks:
            try:
                callback(failure)
            except Exception:
                pass
        return expired

    def cancel(self, request_id: str) -> bool:
        """Remove a specific request from the queue."""
        with self._lock:
            for bucket in self._buckets:
                for i, entry in enumerate(bucket):
                    if entry.request.request_id == request_id:
                        bucket.pop(i)
                        entry.request.state = RequestState.CANCELLED
                        entry.request._event.set()
                        return True
        return False

    # ---- telemetry ----------------------------------------------------------

    @property
    def depth(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._buckets)

    def metrics(self) -> dict:
        with self._lock:
            depths = {
                f"queue_depth_{RequestPriority(i).name.lower()}": len(b)
                for i, b in enumerate(self._buckets)
            }
            return {
                "queue_total_depth":   sum(len(b) for b in self._buckets),
                "queue_submitted":     self._submitted,
                "queue_dequeued":      self._dequeued,
                "queue_expired":       self._expired,
                "queue_rate_dropped":  self._rate_dropped,
                "queue_full_dropped":  self._full_dropped,
                "queue_admitted":      self._admitted,
                "queue_sessions":      len(self._session_buckets),
                **depths,
            }


__all__ = [
    "RequestPriority",
    "TokenBucket",
    "QueueFullError",
    "Phase2RequestQueue",
]
