"""Serialize borrowed packed-expert leases across sessions in one service."""
from contextlib import contextmanager
import math
import threading
import time


class SerializedExpertProvider:
    """Share one immutable provider while keeping each model's caches separate.

    The underlying provider remains caller-owned. This gate does not extend a
    borrowed array's lifetime, cache an expert, or coordinate independent gates
    wrapping the same provider. Callers sharing across services should pass the
    same gate to each service.
    """

    def __init__(self, provider, *, lease_timeout_seconds=30.0):
        if (type(lease_timeout_seconds) not in (int, float)
                or not math.isfinite(lease_timeout_seconds)
                or not 0 < lease_timeout_seconds <= threading.TIMEOUT_MAX):
            raise ValueError("finite positive expert lease timeout required")
        if not callable(getattr(provider, "lease", None)):
            raise ValueError("expert provider must expose lease(layer, expert)")
        self._provider = provider
        self._timeout = float(lease_timeout_seconds)
        self._gate = threading.Lock()
        self._local = threading.local()

    @property
    def identity(self):
        return self._provider.identity

    @property
    def max_expert_bytes(self):
        return self._provider.max_expert_bytes

    @property
    def tensor_shapes(self):
        return self._provider.tensor_shapes

    @property
    def format(self):
        return getattr(self._provider, "format", None)

    @contextmanager
    def cancellation_scope(self, cancel):
        """Bind cancellation to this thread without mutating the shared provider."""
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('expert cancellation must provide is_set')
        previous = getattr(self._local, 'cancel', None)
        self._local.cancel = cancel
        try:
            yield
        finally:
            self._local.cancel = previous

    def _cancelled(self):
        cancel = getattr(self._local, 'cancel', None)
        if cancel is not None and cancel.is_set():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('selected-expert execution cancelled')

    @contextmanager
    def lease(self, layer, expert):
        if getattr(self._local, "active", False):
            raise RuntimeError("nested expert leases are not supported")
        deadline = time.monotonic()+self._timeout
        while True:
            self._cancelled()
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('shared expert provider lease deadline exceeded')
            if self._gate.acquire(timeout=min(remaining, 0.05)):
                break
        self._local.active = True
        try:
            self._cancelled()
            with self._provider.lease(layer, expert) as arrays:
                self._cancelled()
                yield arrays
                self._cancelled()
        finally:
            self._local.active = False
            self._gate.release()


def shared_expert_provider(provider):
    """Preserve an explicit shared gate; otherwise create one borrowed gate."""
    if provider is None or isinstance(provider, SerializedExpertProvider):
        return provider
    return SerializedExpertProvider(provider)
