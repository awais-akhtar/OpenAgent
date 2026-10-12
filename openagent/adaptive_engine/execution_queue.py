"""Bounded serialized generation scheduling; not continuous batching."""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from itertools import islice
import math
import threading
import time

from .execution_service import ExecutionCancelled, NativeExecutionService
from .sampling import SamplingPolicy


__all__ = ['NativeExecutionQueue', 'GenerationFuture', 'GenerationDeadlineExceeded',
           'ExecutionQueueClosed']


class GenerationDeadlineExceeded(TimeoutError):
    """A monotonic request deadline expired before transaction completion."""


class ExecutionQueueClosed(RuntimeError):
    pass


class GenerationFuture(Future):
    """Future with cooperative active cancellation via cancel_event.

    cancel() cancels queued work normally. For running work it returns False,
    but still sets the Event so the service can roll back at its next boundary.
    """
    def __init__(self, wake):
        super().__init__()
        self.cancel_event = threading.Event()
        self._wake = wake

    def cancel(self):
        self.cancel_event.set()
        try:
            return super().cancel()
        finally:
            self._wake()


class _Cancellation:
    def __init__(self, owned, external, deadline):
        self.owned, self.external, self.deadline = owned, external, deadline

    def is_set(self):
        return (self.owned.is_set() or (self.external is not None and self.external.is_set())
                or (self.deadline is not None and time.monotonic() >= self.deadline))


@dataclass
class _Request:
    sequence: int
    session_id: str
    prompt: tuple
    stops: tuple
    count: int
    priority: int
    deadline: float | None
    enqueued: float
    future: GenerationFuture
    cancel: _Cancellation
    on_event: object
    sampling: SamplingPolicy | None = None
    started: float | None = None

    @property
    def stored_tokens(self):
        return len(self.prompt) + len(self.stops)

    @property
    def reservation(self):
        return len(self.prompt) + self.count


class NativeExecutionQueue:
    """Single generation worker around an existing NativeExecutionService.

    Admission reserves worst-case continuation context for all queued requests
    of a session. Only session heads compete for priority/deadline scheduling.
    Callers must route mutations of these sessions through this queue; direct
    service reset/close/generate calls cannot be coordinated by a queue wrapper.
    Deadlines are absolute time.monotonic() values; cancellation is cooperative,
    not a guarantee that native computation ends at an exact wall-clock time.
    The optional owned service is closed only after generation has stopped.
    """
    def __init__(self, service, *, max_requests, max_pending_tokens,
                 own_service=False, cancellation_poll_seconds=0.05):
        if not isinstance(service, NativeExecutionService):
            raise ValueError('NativeExecutionService required')
        for name, value in (('max_requests', max_requests), ('max_pending_tokens', max_pending_tokens)):
            if type(value) is not int or value <= 0:
                raise ValueError('positive queue limit required: ' + name)
        if type(own_service) is not bool:
            raise ValueError('own_service must be boolean')
        if (type(cancellation_poll_seconds) not in (int, float)
                or not math.isfinite(cancellation_poll_seconds) or cancellation_poll_seconds <= 0):
            raise ValueError('positive finite cancellation polling interval required')
        self.service = service
        self.max_requests, self.max_pending_tokens = max_requests, max_pending_tokens
        self._own_service, self._poll = own_service, cancellation_poll_seconds
        self._condition = threading.Condition()
        self._sessions = {}
        self._reserved = {}
        self._committed = {}
        self._active = None
        self._pending_count = self._stored_tokens = self._sequence = 0
        self._history_epoch = 0
        self._stopping = self._finished = False
        self._cleanup_error = None
        self._service_released = not own_service
        self._cleanup_lock = threading.Lock()
        self._created = time.monotonic()
        self._counters = dict(submitted=0, rejected=0, succeeded=0, failed=0,
                              cancelled=0, expired=0, callback_errors=0,
                              queue_wait_seconds=0.0, execution_seconds=0.0,
                              turnaround_seconds=0.0, completed=0)
        self._worker = threading.Thread(target=self._work, name='native-generation-queue', daemon=True)
        self._monitor = threading.Thread(target=self._monitor_cancellations,
                                         name='native-generation-cancellation', daemon=True)
        try:
            self._worker.start()
            self._monitor.start()
        except BaseException as primary:
            def cleanup_startup():
                if self._worker.ident is not None:
                    self.shutdown(wait=True)
                else:
                    with self._condition:
                        self._stopping=self._finished=True
                    if self._own_service and not self._service_released:
                        self.service.close();self._service_released=True
            try:cleanup_startup()
            except BaseException as cleanup:
                primary.cleanup=cleanup_startup
                primary.add_note('execution queue startup cleanup retained: '+repr(cleanup))
            raise

    def _wake(self):
        with self._condition:
            self._condition.notify_all()

    def _tokens(self, values, maximum, *, empty=False):
        if maximum < 0:
            raise ValueError('generation cannot fit session context')
        result = tuple(islice(iter(values), maximum + 1))
        if (len(result) > maximum or (not result and not empty)
                or any(type(t) is not int or not 0 <= t < self.service.limits.vocabulary for t in result)):
            raise ValueError('invalid/beyond-budget vocabulary token inventory')
        return result

    def submit(self, session_id, prompt_ids, *, max_new_tokens=None, stop_ids=(),
               priority=0, deadline=None, cancel=None, on_event=None, sampling=None):
        """Admit a request and return GenerationFuture; deadlines use monotonic time."""
        try:
            if type(session_id) is not str or not session_id:
                raise ValueError('nonempty service session identifier required')
            if type(priority) is not int or not -(2**31) <= priority < 2**31:
                raise ValueError('signed 32-bit integer priority required')
            if deadline is not None:
                if type(deadline) not in (int, float) or not math.isfinite(deadline):
                    raise ValueError('finite absolute monotonic deadline required')
                deadline = float(deadline)
                if deadline <= time.monotonic():
                    raise GenerationDeadlineExceeded('generation deadline already expired')
            if cancel is not None and type(cancel) is not threading.Event:
                raise ValueError('cancellation must be a threading.Event')
            if cancel is not None and cancel.is_set():
                raise ExecutionCancelled('generation cancelled before admission')
            if on_event is not None and not callable(on_event):
                raise ValueError('event handler must be callable')
            if sampling is not None and type(sampling) is not SamplingPolicy:
                raise ValueError('SamplingPolicy required')
            limits = self.service.limits
            if sampling is not None and sampling.top_k > limits.vocabulary:
                raise ValueError('top-k exceeds vocabulary')
            if sampling is not None and limits.vocabulary*64 > limits.max_sampling_buffer_bytes:
                raise MemoryError('sampling buffers exceed vocabulary admission')
            count = limits.max_new_tokens if max_new_tokens is None else max_new_tokens
            if type(count) is not int or not 1 <= count <= limits.max_new_tokens:
                raise ValueError('invalid generation length')
            with self._condition:
                if self._stopping:
                    raise ExecutionQueueClosed('execution queue is shutting down')
                if self._pending_count >= self.max_requests:
                    raise MemoryError('generation request queue is full')
                cached_base = self._committed.get(session_id)
            # Service/session lookup and caller iterators run outside queue locks.
            # Already-admitted sessions use their tracked committed prefix so
            # submission need not wait for an active service transaction's lock.
            base = len(self.service.history(session_id)) if cached_base is None else cached_base
            prompt = self._tokens(prompt_ids, min(limits.context - base - count, self.max_pending_tokens))
            stops = self._tokens(stop_ids, min(limits.vocabulary, self.max_pending_tokens - len(prompt)), empty=True)
            future = GenerationFuture(self._wake)
            cancellation = _Cancellation(future.cancel_event, cancel, deadline)
            while True:
                with self._condition:
                    if self._stopping:
                        raise ExecutionQueueClosed('execution queue is shutting down')
                    epoch = self._history_epoch
                    cached_base = self._committed.get(session_id)
                base = len(self.service.history(session_id)) if cached_base is None else cached_base
                with self._condition:
                    if self._stopping:
                        raise ExecutionQueueClosed('execution queue is shutting down')
                    if cancellation.is_set():
                        if deadline is not None and time.monotonic() >= deadline:
                            raise GenerationDeadlineExceeded('generation deadline expired during admission')
                        raise ExecutionCancelled('generation cancelled during admission')
                    # A commit between history lookup and this lock invalidates
                    # that lookup, even when its session has no queued successors.
                    if epoch != self._history_epoch:
                        continue
                    if self._pending_count >= self.max_requests:
                        raise MemoryError('generation request queue is full')
                    if len(prompt) + len(stops) > self.max_pending_tokens - self._stored_tokens:
                        raise MemoryError('queued token storage exceeds admission')
                    committed = max(base, self._committed.get(session_id, 0))
                    reservation = len(prompt) + count
                    if committed + self._reserved.get(session_id, 0) + reservation > limits.context:
                        raise MemoryError('queued session continuations exceed context capacity')
                    self._sequence += 1
                    request = _Request(self._sequence, session_id, prompt, stops, count, priority,
                                       deadline, time.monotonic(), future, cancellation, on_event, sampling)
                    self._sessions.setdefault(session_id, deque()).append(request)
                    self._committed[session_id] = committed
                    self._reserved[session_id] = self._reserved.get(session_id, 0) + reservation
                    self._pending_count += 1
                    self._stored_tokens += request.stored_tokens
                    self._counters['submitted'] += 1
                    self._condition.notify_all()
                    break
            return future
        except BaseException:
            with self._condition:
                self._counters['rejected'] += 1
            raise

    def _reason(self, request, now):
        if request.deadline is not None and now >= request.deadline:
            return 'expired', GenerationDeadlineExceeded('generation deadline expired')
        if request.future.cancelled() or request.cancel.is_set():
            return 'cancelled', ExecutionCancelled('queued generation cancelled')
        return None

    def _release_locked(self, request, outcome, now):
        self._pending_count -= 1
        self._stored_tokens -= request.stored_tokens
        sid = request.session_id
        remaining = self._reserved[sid] - request.reservation
        if remaining:
            self._reserved[sid] = remaining
        else:
            self._reserved.pop(sid, None)
            self._committed.pop(sid, None)
        self._counters[outcome] += 1
        self._counters['completed'] += 1
        self._counters['turnaround_seconds'] += max(0.0, now - request.enqueued)
        if request.started is None:
            self._counters['queue_wait_seconds'] += max(0.0, now - request.enqueued)
        else:
            self._counters['queue_wait_seconds'] += max(0.0, request.started - request.enqueued)
            self._counters['execution_seconds'] += max(0.0, now - request.started)

    def _sweep_locked(self, now):
        removed = []
        for sid, requests in tuple(self._sessions.items()):
            retained = deque()
            for request in requests:
                reason = self._reason(request, now)
                if reason is None:
                    retained.append(request)
                else:
                    outcome, error = reason
                    self._release_locked(request, outcome, now)
                    removed.append((request, error))
            if retained:
                self._sessions[sid] = retained
            else:
                self._sessions.pop(sid, None)
        return removed

    def _complete(self, request, *, result=None, error=None):
        # Future callbacks execute synchronously here, never under queue locks.
        try:
            if not request.future.done():
                if error is None:
                    request.future.set_result(result)
                else:
                    request.future.set_exception(error)
        except BaseException:
            with self._condition:
                self._counters['callback_errors'] += 1

    def _monitor_cancellations(self):
        while True:
            with self._condition:
                if self._finished:
                    return
                removed = self._sweep_locked(time.monotonic())
                if self._active is not None and self._active.cancel.is_set():
                    self._active.future.cancel_event.set()
                self._condition.notify_all()
            for request, error in removed:
                self._complete(request, error=error)
            with self._condition:
                if not self._finished:
                    self._condition.wait(self._poll)

    def _work(self):
        try:
            while True:
                with self._condition:
                    removed = self._sweep_locked(time.monotonic())
                    request = None
                    if self._sessions:
                        request = min((requests[0] for requests in self._sessions.values()),
                            key=lambda r: (-r.priority, math.inf if r.deadline is None else r.deadline, r.sequence))
                        requests = self._sessions[request.session_id]
                        requests.popleft()
                        if not requests:
                            self._sessions.pop(request.session_id)
                        request.started = time.monotonic()
                        self._active = request
                    stop = self._stopping and request is None
                for cancelled_request, error in removed:
                    self._complete(cancelled_request, error=error)
                if stop:
                    break
                if request is None:
                    with self._condition:
                        if not self._sessions and not self._stopping:
                            self._condition.wait(self._poll)
                    continue
                result = error = None
                outcome = 'succeeded'
                try:
                    if not request.future.set_running_or_notify_cancel():
                        raise ExecutionCancelled('generation Future cancelled before execution')
                    reason = self._reason(request, time.monotonic())
                    if reason is not None:
                        raise reason[1]
                    result = self.service.generate(request.session_id, request.prompt,
                        max_new_tokens=request.count, stop_ids=request.stops,
                        cancel=request.cancel, on_event=request.on_event, sampling=request.sampling)
                except BaseException as caught:
                    error = caught
                    if isinstance(caught, (GenerationDeadlineExceeded, ExecutionCancelled)):
                        if request.deadline is not None and time.monotonic() >= request.deadline:
                            outcome = 'expired'
                            error = GenerationDeadlineExceeded('active generation deadline expired; transaction cancelled')
                            error.__cause__ = caught
                            cleanup = getattr(caught, 'cleanup', None)
                            if callable(cleanup):
                                error.cleanup = cleanup
                        else:
                            outcome = 'cancelled'
                    else:
                        outcome = 'failed'
                with self._condition:
                    if result is not None:
                        self._history_epoch += 1
                        self._committed[request.session_id] = result.context_tokens
                    self._active = None
                    self._release_locked(request, outcome, time.monotonic())
                    self._condition.notify_all()
                self._complete(request, result=result, error=error)
        finally:
            # Resolve every published Future even if the worker exits unexpectedly.
            with self._condition:
                stranded = [r for requests in self._sessions.values() for r in requests]
                if self._active is not None:
                    stranded.append(self._active)
                self._sessions.clear()
                self._active = None
                self._stopping = True
                for request in stranded:
                    request.future.cancel_event.set()
                    self._release_locked(request, 'failed', time.monotonic())
            for request in stranded:
                self._complete(request, error=ExecutionQueueClosed('generation worker stopped'))
            try:
                if self._own_service:
                    self.service.close()
                    self._service_released = True
            except BaseException as error:
                with self._condition:
                    self._cleanup_error = error
            finally:
                with self._condition:
                    self._finished = True
                    self._condition.notify_all()

    def shutdown(self, *, wait=True, cancel_pending=True, cancel_active=True, timeout=None):
        """Stop admission, optionally cancel/drain, then close an owned service.

        A timeout raises TimeoutError; cancellation and shutdown remain in effect.
        Native calls cannot be forcibly interrupted. Callback code must use
        wait=False when shutting down from either queue thread.
        """
        if any(type(flag) is not bool for flag in (wait, cancel_pending, cancel_active)):
            raise ValueError('shutdown flags must be boolean')
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0):
            raise ValueError('finite nonnegative shutdown timeout required')
        removed = []
        with self._condition:
            self._stopping = True
            if cancel_active and self._active is not None:
                self._active.future.cancel_event.set()
            if cancel_pending:
                for requests in self._sessions.values():
                    for request in requests:
                        request.future.cancel_event.set()
                removed = self._sweep_locked(time.monotonic())
            self._condition.notify_all()
        for request, error in removed:
            self._complete(request, error=error)
        if wait:
            if threading.current_thread() in (self._worker, self._monitor):
                raise RuntimeError('queue callbacks must call shutdown(wait=False)')
            deadline = None if timeout is None else time.monotonic() + timeout
            self._worker.join(timeout)
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if self._monitor.ident is not None:
                self._monitor.join(remaining)
            if self._worker.is_alive() or self._monitor.is_alive():
                raise TimeoutError('native execution queue shutdown is still finishing')
            if self._own_service and not self._service_released:
                with self._cleanup_lock:
                    if not self._service_released:
                        try:
                            self.service.close()
                        except BaseException as error:
                            self._cleanup_error = error
                        else:
                            self._service_released = True
                            self._cleanup_error = None
            if self._cleanup_error is not None:
                raise RuntimeError('owned execution service cleanup failed') from self._cleanup_error

    close = shutdown

    def stats(self):
        with self._condition:
            now = time.monotonic()
            return dict(self._counters, pending_requests=self._pending_count,
                queued_requests=sum(len(requests) for requests in self._sessions.values()),
                active_requests=int(self._active is not None), stored_tokens=self._stored_tokens,
                reserved_context_tokens=sum(self._reserved.values()),
                active_seconds=0.0 if self._active is None else max(0.0, now - self._active.started),
                uptime_seconds=max(0.0, now - self._created), accepting=not self._stopping,
                finished=self._finished, cleanup_error=None if self._cleanup_error is None else
                type(self._cleanup_error).__name__, generation_workers=1, continuous_batching=False)

    def __enter__(self):
        with self._condition:
            if self._stopping:
                raise ExecutionQueueClosed('execution queue is shutting down')
        return self

    def __exit__(self, *exc):
        self.shutdown()
