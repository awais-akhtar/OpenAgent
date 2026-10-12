"""Shared HTTP request classification and in-process rate limiting."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True, slots=True)
class RateLimit:
    requests: int
    window_seconds: float


DEFAULT_RATE_LIMITS: dict[str, RateLimit] = {
    "auth": RateLimit(12, 60.0),
    "upload": RateLimit(30, 60.0),
    "external": RateLimit(45, 60.0),
    "model_job": RateLimit(20, 60.0),
    "general": RateLimit(360, 60.0),
}


def request_rate_class(path: str, method: str = "GET") -> str:
    normalized = str(path).casefold()
    verb = str(method).upper()
    if normalized.startswith("/api/auth/") or normalized.startswith("/api/identity/"):
        return "auth" if verb != "GET" else "general"
    if any(token in normalized for token in ("upload", "ingest", "transcribe", "/images/add")):
        return "upload"
    if normalized.startswith(("/api/research", "/api/scraper", "/api/calendar/sync", "/api/communication/sync")):
        return "external"
    if normalized.startswith(("/api/local-ai/models", "/api/finetune", "/api/model-cookbook")):
        return "model_job" if verb != "GET" else "general"
    return "general"


class SlidingWindowRateLimiter:
    """Thread-safe fixed-capacity sliding windows keyed by client and class."""

    def __init__(self, limits: dict[str, RateLimit] | None = None) -> None:
        self.limits = dict(limits or DEFAULT_RATE_LIMITS)
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def consume(self, client: str, category: str, *, now: float | None = None) -> tuple[bool, int]:
        limit = self.limits.get(category, self.limits["general"])
        timestamp = time.monotonic() if now is None else float(now)
        key = (client or "unknown", category)
        cutoff = timestamp - limit.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit.requests:
                retry_after = max(1, int(limit.window_seconds - (timestamp - events[0]) + 0.999))
                return False, retry_after
            events.append(timestamp)
            if len(self._events) > 10_000:
                self._discard_idle(cutoff)
        return True, 0

    def _discard_idle(self, cutoff: float) -> None:
        for key, events in list(self._events.items()):
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                self._events.pop(key, None)
