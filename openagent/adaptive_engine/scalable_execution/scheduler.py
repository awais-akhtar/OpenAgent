"""Bounded synchronous generic dispatch with real packed leases and KV commit."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import threading
import sys
from typing import Callable, Iterable

from .resources import check_cancel, positive


@dataclass(frozen=True)
class DispatchStep:
    """One synchronous backend operation; None denotes a non-packed phase."""

    key: str | None
    callback: Callable


class DispatchContext:
    """Ephemeral backend access to a transactional, paged payload cache.

    Callbacks may use state for scratch admitted by the dispatch reservation.
    Neither state nor views may escape dispatch except the finish result.
    """

    def __init__(self, transaction, token_ids, cancelled, scratch_bytes):
        self.cache = transaction
        self.token_ids = token_ids
        self.position = 0 if transaction is None else transaction.position
        self.scratch_bytes = scratch_bytes
        self.state = {}
        self._cancelled = cancelled
        self._open = True
        self._cleanup_callbacks = []
        self._cleaning = False
        self._thread=threading.get_ident()

    def check_cancel(self):
        if not self._open:
            raise RuntimeError("dispatch context expired")
        if threading.get_ident()!=self._thread:
            raise RuntimeError('dispatch context belongs to another execution thread')
        check_cancel(self._cancelled)

    def defer(self, callback):
        """Register request-local cleanup that runs before durable commit."""
        # Register disposal even if cancellation arrived just after acquisition.
        if not self._open or threading.get_ident()!=self._thread:
            raise RuntimeError('available owner-thread dispatch context required')
        if self._cleaning:
            raise RuntimeError('cannot register disposal during cleanup')
        if not callable(callback):
            raise TypeError("cleanup callback required")
        self._cleanup_callbacks.append(callback)

    def bind_cancellation(self, owner):
        """Keep decoder/provider scopes active until request disposal.

        Register the decoder's close first: scope exit must run before its
        resource teardown and restore any previously borrowed cancellation.
        """
        self.check_cancel()
        scope_factory = getattr(owner, 'cancellation_scope', None)
        if not callable(scope_factory):
            return
        context = self
        class Event:
            def is_set(self):
                if not context._open:
                    raise RuntimeError('expired dispatch cancellation predicate')
                return context._cancelled is not None and context._cancelled()
        scope = scope_factory(Event())
        scope.__enter__()
        def leave():
            if threading.get_ident()!=self._thread:
                raise RuntimeError('retained cancellation scope requires its acquisition thread')
            scope.__exit__(None, None, None)
        try:
            self.defer(leave)
        except BaseException as primary:
            try:leave()
            except BaseException as cleanup:
                self._cleanup_callbacks.append(leave)
                primary.add_note('dispatch cancellation scope release retained: '+repr(cleanup))
            raise

    def _cleanup(self):
        if self._cleaning:
            raise RuntimeError('reentrant dispatch cleanup')
        self._cleaning = True
        try:
            while self._cleanup_callbacks:
                callback = self._cleanup_callbacks[-1]
                callback()
                self._cleanup_callbacks.pop()
        finally:
            self._cleaning = False


class ScalableExecutor:
    """Fail-fast admission, bounded step streaming, transactional completion.

    This executor owns no math kernel and removes no native kernel limit.
    The backend supplies callbacks and accurate scratch/output reservations.
    A returned result means the corresponding cache transaction committed.
    Cancellation after the commit point cannot undo that committed result.
    """

    def __init__(self, geometry, packed_store, cache, *, scratch_budget,
                 max_inflight=1, max_steps=65536, own_resources=False):
        if cache.geometry != geometry:
            raise ValueError("executor/cache geometry mismatch")
        if type(own_resources) is not bool:
            raise ValueError('explicit boolean scalable resource ownership required')
        self.geometry = geometry
        self.store, self.cache = packed_store, cache
        self.scratch_budget = scratch_budget
        self.max_steps = positive(max_steps, "max_steps")
        self._slots = threading.BoundedSemaphore(positive(max_inflight, "max_inflight"))
        self._lock = threading.RLock()
        self._closed = False
        self._active = 0
        self._own_resources = own_resources
        self._pending_cleanup = []
        self._recovering = False
        self._quarantined = False

    @contextmanager
    def _scratch(self, amount, holder):
        reservation = self.scratch_budget.reserve(amount)
        reservation.__enter__()
        try:
            yield
        finally:
            context = holder[0]
            if context is not None and context._cleanup_callbacks:
                context._open = False
                with self._lock:
                    self._pending_cleanup.append((context, reservation))
                    self._quarantined=True
            else:
                primary=sys.exc_info()[1]
                try:reservation.__exit__(None, None, None)
                except BaseException as cleanup:
                    if context is None:context=DispatchContext(None,(),None,amount)
                    context._open=False
                    with self._lock:
                        self._pending_cleanup.append((context,reservation))
                        self._quarantined=True
                    if primary is None:raise
                    primary.add_note('scratch reservation release requires retry: '+repr(cleanup))

    def retry_cleanup(self):
        """Release only successfully disposed resources and their reservations.

        Failed request cleanup quarantines admission. The request transaction
        has already rolled back; retries perform disposal, never cache replay.
        Successful callbacks are removed and never repeated.
        """
        with self._lock:
            if not self._pending_cleanup and not self._recovering and not self._quarantined:
                return
            if self._active or self._recovering:
                raise RuntimeError('idle executor required for cleanup recovery')
            self._recovering = True
            pending = self._pending_cleanup
            self._pending_cleanup = []
        retained = []
        first_error = None
        try:
            for context, reservation in pending:
                try:
                    context._cleanup()
                    context.state.clear()
                    context.cache = None
                    reservation.__exit__(None, None, None)
                except BaseException as failure:
                    retained.append((context, reservation))
                    if first_error is None:
                        first_error = failure
        finally:
            with self._lock:
                self._pending_cleanup.extend(retained)
                self._recovering = False
                self._quarantined = bool(self._pending_cleanup)
        if first_error is not None:
            raise first_error

    def status(self):
        with self._lock:
            return dict(active=self._active, closed=self._closed,
                        recovering=self._recovering,
                        quarantined=self._quarantined,
                        pending_cleanup=len(self._pending_cleanup))

    def dispatch(self, session_id, owner, token_ids, steps: Iterable[DispatchStep],
                 finish: Callable, *, scratch_bytes, cancelled=None):
        """Run callbacks and finish inside a single atomic cache append.

        callback(context, packed_view_or_none) -> None
        finish(context) -> result

        Each callback finishes before its packed lease is released. Steps may
        be a generator (including routes selected by an earlier callback), so
        the executor never materializes a complete expert bank or route list.
        """
        if type(scratch_bytes) is not int or scratch_bytes < 0:
            raise ValueError("scratch_bytes must be nonnegative")
        if not callable(finish):
            raise TypeError("finish callback required")
        # Bound input consumption even if caller supplies an unbounded iterator.
        tokens = []
        for token in token_ids:
            check_cancel(cancelled)
            if len(tokens) >= self.geometry.max_batch_tokens:
                raise ValueError("batch token admission exceeded")
            if type(token) is not int or not 0 <= token < 2**63:
                raise ValueError("token must be a nonnegative signed-64-bit integer")
            tokens.append(token)
        positive(len(tokens), "token count")
        tokens = tuple(tokens)
        check_cancel(cancelled)
        with self._lock:
            if self._closed:
                raise RuntimeError("executor closed")
            if self._pending_cleanup or self._recovering or self._quarantined:
                raise RuntimeError('executor requires cleanup recovery before dispatch')
            if not self._slots.acquire(blocking=False):
                raise MemoryError("executor inflight admission rejected")
            self._active += 1
        context = None
        holder = [None]
        try:
            with self._scratch(scratch_bytes, holder):
                with self.cache.transaction(session_id, owner, tokens, cancelled=cancelled) as tx:
                    # Another admitted request may have waited on this cache's
                    # transaction lock while the preceding request failed disposal.
                    with self._lock:
                        if self._quarantined or self._pending_cleanup or self._recovering:
                            raise RuntimeError('executor disposal failed while this dispatch was waiting')
                    context = DispatchContext(tx, tokens, cancelled, scratch_bytes)
                    holder[0] = context
                    iterator = None
                    try:
                        try:iterator=iter(steps)
                        except BaseException as primary:
                            cleanup=getattr(primary,'cleanup',None)
                            if callable(cleanup):context.defer(cleanup)
                            raise
                        for index, step in enumerate(iterator):
                            context.check_cancel()
                            if index >= self.max_steps:
                                raise ValueError("dispatch step admission exceeded")
                            if not isinstance(step, DispatchStep) or not callable(step.callback):
                                raise TypeError("invalid dispatch step")
                            if step.key is None:
                                step.callback(context, None)
                            else:
                                region = self.store.regions[step.key]
                                self.geometry.admit(tokens=len(tokens), position=tx.position,
                                                    packed_format=region.packed_format)
                                with self.store.acquire(step.key, cancelled=cancelled) as payload:
                                    step.callback(context, payload)
                            context.check_cancel()
                        result = finish(context)
                        context.check_cancel()
                    finally:
                        failure = sys.exc_info()[1]
                        try:
                            close = getattr(iterator, "close", None)
                            if close is not None and not callable(close):
                                if failure is None:
                                    raise TypeError('dispatch iterator close must be callable')
                                failure.add_note('dispatch iterator close is not callable')
                            elif close is not None:
                                try:close()
                                except BaseException as cleanup:
                                    context._cleanup_callbacks.append(close)
                                    if failure is None:raise
                                    failure.add_note('dispatch iterator close requires retry: '+repr(cleanup))
                        finally:
                            if failure is None:failure = sys.exc_info()[1]
                            try:
                                context._cleanup()
                            except BaseException as cleanup_failure:
                                with self._lock:self._quarantined=True
                                if failure is None:
                                    raise
                                failure.add_note('request disposal requires retry: '+str(cleanup_failure))
                return result
        finally:
            if context is not None:
                context._open = False
                if not context._cleanup_callbacks:
                    context.state.clear()
                    context.cache = None
            with self._lock:
                self._active -= 1
                self._slots.release()

    def close(self):
        self.retry_cleanup()
        with self._lock:
            if self._closed:
                return
            if self._active:
                raise RuntimeError("cannot close executor during dispatch")
            if self._own_resources:
                # Reject independently active consumers before either close.
                with self.store._lock, self.cache._lock:
                    if self.store._active or self.cache._active:
                        raise RuntimeError("owned resources still active")
                    self.store._pool.close()
                    self.cache._pool.close()
                    self.store.close()
                    self.cache.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
