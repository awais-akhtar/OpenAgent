"""Phase 2C: Continuous batching engine for concurrent sessions.

In Phase 1 each session processes one request at a time.  Phase 2C runs a
**continuous batching scheduler** that dynamically groups requests from
multiple concurrent sessions into one forward pass, maximising GPU/CPU
utilisation.

Key ideas (from Orca / vLLM):

1. **Iteration-level scheduling** — after each forward step new requests are
   admitted and finished sequences are evicted.  Batch composition changes
   every step, not every request.

2. **Chunked prefill** — long prompts are processed in fixed-size chunks so
   they do not starve decoding requests.

3. **Preemption** — when memory is tight, decode sequences are paused and
   their KV state is swapped to the shared KV cache (Phase 2B); they resume
   when space returns.

Architecture
------------
``BatchRequest``          -- one pending generation request
``SchedulerConfig``       -- tuning parameters
``ContinuousBatchScheduler`` -- core scheduler; produces batches each step
``BatchWorker``           -- background thread that drives the scheduler loop
``MultiSessionBatchEngine``  -- high-level engine; sessions submit here
"""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

from .failure_reporting import add_failure_note, failure_detail

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request state machine
# ---------------------------------------------------------------------------

class RequestState(str, Enum):
    WAITING   = "waiting"    # queued, not yet in a batch
    PREFILL   = "prefill"    # processing input tokens
    DECODE    = "decode"     # auto-regressive decode loop
    PAUSED    = "paused"     # KV swapped out; waiting for memory
    FINISHED  = "finished"
    CANCELLED = "cancelled"
    FAILED    = "failed"


# ---------------------------------------------------------------------------
# BatchRequest
# ---------------------------------------------------------------------------

@dataclass
class BatchRequest:
    """One generation request submitted by a session."""
    request_id:      str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id:      str = ""
    token_ids:       List[int] = field(default_factory=list)   # prompt tokens
    generated:       List[int] = field(default_factory=list)   # output so far
    max_new_tokens:  int = 128
    stop_token_id:   Optional[int] = None
    state:           RequestState = RequestState.WAITING
    # Callbacks fired per-token and on completion
    on_token:        Optional[Callable[[int], None]] = None
    on_done:         Optional[Callable[[List[int]], None]] = None
    on_error:        Optional[Callable[[Exception], None]] = None
    # Scheduling metadata
    submitted_at:    float = field(default_factory=time.monotonic)
    expires_at:      Optional[float] = None
    prefill_cursor:  int   = 0    # how many prompt tokens have been prefilled
    scheduled_prefill_tokens: int = 0  # exact prompt extent reserved for this step
    priority:        int   = 0    # higher = sooner; not yet used
    # KV reference (set by engine when prefix KV is found)
    kv_prefix_len:   int   = 0
    error:           Optional[BaseException] = None
    # Internal
    _event:          threading.Event = field(default_factory=threading.Event)

    def is_done(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.CANCELLED,
                              RequestState.FAILED)

    def needs_more_tokens(self) -> bool:
        if len(self.generated) >= self.max_new_tokens:
            return False
        if self.stop_token_id is not None and self.generated:
            if self.generated[-1] == self.stop_token_id:
                return False
        return True


# ---------------------------------------------------------------------------
# SchedulerConfig
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    max_batch_size:        int   = 8     # max concurrent sequences per step
    max_tokens_per_batch:  int   = 2048  # total tokens (prefill + decode) per step
    prefill_chunk_size:    int   = 256   # prompt tokens per prefill step
    max_decode_len:        int   = 512   # max total length of a decode sequence
    swap_threshold_bytes:  int   = 2 << 30  # KV memory threshold for preemption (2 GB)
    scheduler_tick_ms:     float = 10.0  # ms between scheduler ticks when idle
    max_queue_depth:       int   = 256   # max queued requests

    def __post_init__(self) -> None:
        positive = (
            "max_batch_size", "max_tokens_per_batch", "prefill_chunk_size",
            "max_decode_len",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("swap_threshold_bytes", "max_queue_depth"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (type(self.scheduler_tick_ms) not in (int, float) or
                not math.isfinite(self.scheduler_tick_ms) or self.scheduler_tick_ms <= 0):
            raise ValueError("scheduler_tick_ms must be finite and positive")


# ---------------------------------------------------------------------------
# ContinuousBatchScheduler
# ---------------------------------------------------------------------------

class ContinuousBatchScheduler:
    """Selects which requests to process in each iteration step.

    The scheduler is called once per forward-pass iteration by the engine.
    It returns a ``batch`` (list of ``BatchRequest``) that the engine processes
    and then calls ``commit_step`` with per-request token outputs.
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._waiting:  List[BatchRequest] = []
        self._running:  List[BatchRequest] = []
        self._paused:   List[BatchRequest] = []
        self._lock      = threading.Lock()
        self._step      = 0
        self._total_scheduled = 0
        self._total_preempted = 0
        self._total_expired = 0
        self._terminal_failure: BaseException | None = None

    # ---- submission ---------------------------------------------------------

    def submit(self, req: BatchRequest) -> None:
        failure = None
        finished = False
        with self._lock:
            if self._terminal_failure is not None:
                raise RuntimeError('scheduler stopped after terminal worker failure') from self._terminal_failure
            self.validate_request(req)
            if (req.state != RequestState.WAITING or req._event.is_set() or req.generated or
                    req.prefill_cursor != 0 or req.error is not None):
                raise ValueError('scheduler submission requires a fresh request')
            if any(other is req or other.request_id == req.request_id
                   for other in (*self._waiting, *self._running, *self._paused)):
                raise ValueError('scheduler request identity is already active')
            if req.max_new_tokens == 0:
                req.state = RequestState.FINISHED
                req._event.set()
                finished = True
            elif len(self._waiting) >= self.config.max_queue_depth:
                req.state = RequestState.FAILED
                failure = RuntimeError("scheduler queue full")
                req.error = failure
                req._event.set()
            else:
                req.state = RequestState.WAITING
                self._waiting.append(req)
        if finished and req.on_done:
            try:
                req.on_done([])
            except Exception:
                pass
        # User callbacks may re-enter the engine (or wait on another request).
        # Invoke them after releasing the scheduler lock, and signal the waiter
        # first so rejection cannot strand a synchronous caller.
        if failure is not None and req.on_error:
            try:
                req.on_error(failure)
            except Exception:
                pass

    def validate_request(self, req: BatchRequest) -> None:
        if not isinstance(req, BatchRequest):
            raise TypeError("BatchRequest required")
        if type(req.request_id) is not str or not 0 < len(req.request_id) <= 256:
            raise ValueError('bounded nonempty request identity required')
        if (type(req.max_new_tokens) is not int or req.max_new_tokens < 0 or
                req.max_new_tokens > self.config.max_decode_len):
            raise ValueError("request token budget must fit the scheduler decode limit")
        if (not isinstance(req.token_ids, list) or
                any(type(token) is not int or token < 0 for token in req.token_ids)):
            raise ValueError("request token IDs must be a list of nonnegative integers")
        if (req.stop_token_id is not None and
                (type(req.stop_token_id) is not int or req.stop_token_id < 0)):
            raise ValueError("request stop token must be a nonnegative integer")
        if type(req.session_id) is not str:
            raise ValueError("request session ID must be a string")
        if (not isinstance(req.priority, int) or type(req.priority) is bool or
                not 0 <= req.priority <= 2):
            raise ValueError("request priority must be an integer from 0 to 2")
        for name in ("on_token", "on_done", "on_error"):
            callback = getattr(req, name)
            if callback is not None and not callable(callback):
                raise TypeError(f"request {name} callback must be callable")

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            for lst in (self._waiting, self._running, self._paused):
                for req in lst:
                    if req.request_id == request_id:
                        req.state = RequestState.CANCELLED
                        req._event.set()
                        # Requests that have not entered a forward call can be
                        # retired immediately; an in-flight request stays in
                        # _running until its worker returns and commit_step
                        # observes cancellation.
                        if lst is self._waiting or lst is self._paused:
                            lst.remove(req)
                        return True
        return False

    def fail_requests(self, requests: List[BatchRequest], failure: BaseException) -> None:
        """Fail a forward batch and retire every request from scheduler state."""
        callbacks = []
        with self._lock:
            for req in requests:
                if req.state == RequestState.FINISHED:
                    continue
                if req.state == RequestState.CANCELLED:
                    for collection in (self._waiting, self._running, self._paused):
                        _list_discard(collection, req)
                    continue
                req.state = RequestState.FAILED
                req.error = failure
                for collection in (self._waiting, self._running, self._paused):
                    _list_discard(collection, req)
                req._event.set()
                if req.on_error:
                    callbacks.append(req.on_error)
        for callback in callbacks:
            try:
                callback(failure)
            except Exception:
                pass

    def fail_all(self, failure: BaseException) -> None:
        """Retire all queued/paused/running requests after a clean worker stop."""
        with self._lock:
            # Seal admission before collecting requests, otherwise a submitter
            # can enqueue after this snapshot and outlive the failed worker.
            if self._terminal_failure is None:
                self._terminal_failure = failure
            requests = []
            seen = set()
            for req in (*self._waiting, *self._running, *self._paused):
                if id(req) not in seen:
                    seen.add(id(req))
                    requests.append(req)
        self.fail_requests(requests, failure)

    # ---- scheduling ---------------------------------------------------------

    def schedule(self) -> List[BatchRequest]:
        """Return the batch for the current step."""
        expired = []
        with self._lock:
            if self._terminal_failure is not None:
                return []
            self._step += 1
            batch: List[BatchRequest] = []
            token_budget = self.config.max_tokens_per_batch

            now = time.monotonic()
            for req in tuple(self._waiting):
                if req.expires_at is not None and now >= req.expires_at:
                    self._waiting.remove(req)
                    req.state = RequestState.FAILED
                    req.error = TimeoutError("request expired before batch scheduling")
                    req._event.set()
                    self._total_expired += 1
                    if req.on_error:
                        expired.append((req.on_error, req.error))

            # 1. Admit waiting requests (prefill first, sorted by submission time)
            admit_order = sorted(
                [r for r in self._waiting if r.state == RequestState.WAITING],
                key=lambda r: (r.priority, r.submitted_at),
            )
            for req in admit_order:
                if len(batch) >= self.config.max_batch_size:
                    break
                if not req.token_ids:
                    if token_budget <= 0:
                        continue
                    req.state = RequestState.DECODE
                    self._waiting.remove(req)
                    self._running.append(req)
                    batch.append(req)
                    token_budget -= 1
                    self._total_scheduled += 1
                    continue
                chunk = min(
                    self.config.prefill_chunk_size,
                    len(req.token_ids) - req.prefill_cursor,
                    token_budget,
                )
                if chunk <= 0:
                    continue
                token_budget -= chunk
                req.state = RequestState.PREFILL
                req.scheduled_prefill_tokens = chunk
                self._waiting.remove(req)
                self._running.append(req)
                batch.append(req)
                self._total_scheduled += 1

            # Continue partially processed prompts; they remain owned by
            # _running and must not disappear between prefill chunks.
            for req in self._running:
                if req.state != RequestState.PREFILL or any(req is item for item in batch):
                    continue
                if len(batch) >= self.config.max_batch_size or token_budget <= 0:
                    break
                chunk = min(self.config.prefill_chunk_size,
                            len(req.token_ids) - req.prefill_cursor, token_budget)
                if chunk <= 0:
                    continue
                req.scheduled_prefill_tokens = chunk
                token_budget -= chunk
                batch.append(req)

            # 2. Resume paused requests if there's room
            resume_order = sorted(self._paused, key=lambda r: r.submitted_at)
            for req in resume_order:
                if len(batch) >= self.config.max_batch_size or token_budget <= 0:
                    break
                req.state = RequestState.DECODE
                self._paused.remove(req)
                self._running.append(req)
                batch.append(req)
                token_budget -= 1

            # 3. Continue decode for already-running sequences
            decode_candidates = [
                r for r in self._running
                if r.state == RequestState.DECODE and r not in batch
            ]
            for req in decode_candidates:
                if len(batch) >= self.config.max_batch_size or token_budget <= 0:
                    break
                batch.append(req)
                token_budget -= 1

        for callback, failure in expired:
            try:
                callback(failure)
            except Exception:
                pass
        return batch

    def commit_step(self, results: List[Tuple[BatchRequest, Optional[int]]]) -> None:
        """Commit one validated step; callbacks run outside the scheduler lock."""
        _commit_step(self, results)

    def preempt_one(self) -> Optional[BatchRequest]:
        """Pause the lowest-priority running decode request (LRU swap)."""
        with self._lock:
            decode_running = [r for r in self._running if r.state == RequestState.DECODE]
            if not decode_running:
                return None
            victim = min(decode_running, key=lambda r: r.submitted_at)
            victim.state = RequestState.PAUSED
            self._running.remove(victim)
            self._paused.append(victim)
            self._total_preempted += 1
            return victim

    def metrics(self) -> dict:
        with self._lock:
            return {
                "scheduler_step":         self._step,
                "waiting_requests":       len(self._waiting),
                "running_requests":       len(self._running),
                "paused_requests":        len(self._paused),
                "total_scheduled":        self._total_scheduled,
                "total_preempted":        self._total_preempted,
                "expired_requests":        self._total_expired,
            }


def _list_discard(lst: List, item: Any) -> None:
    try:
        lst.remove(item)
    except ValueError:
        pass

def _commit_step(self: ContinuousBatchScheduler, results: List[Tuple[BatchRequest, Optional[int]]]) -> None:
    callbacks = []
    with self._lock:
        for req, token in results:
            if req.state in (RequestState.CANCELLED, RequestState.FAILED):
                _list_discard(self._running, req)
                continue

            if req.state == RequestState.PREFILL:
                req.prefill_cursor = min(
                    req.prefill_cursor + req.scheduled_prefill_tokens, len(req.token_ids))
                req.scheduled_prefill_tokens = 0
                if req.prefill_cursor >= len(req.token_ids):
                    req.state = RequestState.DECODE

            if token is not None:
                req.generated.append(token)
                req.state = RequestState.DECODE
                if req.on_token:
                    callbacks.append((req.on_token, token))

            if not req.needs_more_tokens():
                req.state = RequestState.FINISHED
                _list_discard(self._running, req)
                req._event.set()
                if req.on_done:
                    callbacks.append((req.on_done, list(req.generated)))

    # User callbacks may submit/cancel another request and must not run while
    # the scheduler's state lock is held. Preserve the historical best-effort
    # callback contract while avoiding re-entry deadlocks in serving callers.
    for callback, value in callbacks:
        try:
            callback(value)
        except Exception:
            pass




# ---------------------------------------------------------------------------
# BatchWorker — background thread
# ---------------------------------------------------------------------------

class BatchWorker:
    """Background thread that drives the scheduler and calls forward_fn each step.

    ``forward_fn(batch: List[BatchRequest]) → List[Tuple[BatchRequest, Optional[int]]]``
    must be provided by the engine.  It runs one forward step and returns
    ``(request, next_token_or_None)`` for each request in the batch.
    """

    def __init__(
        self,
        scheduler: ContinuousBatchScheduler,
        forward_fn: Callable[[List[BatchRequest]], List[Tuple[BatchRequest, Optional[int]]]],
        tick_ms: float = 10.0,
        name: str = "phase2-batch-worker",
    ) -> None:
        self._scheduler  = scheduler
        self._forward_fn = forward_fn
        self._tick       = tick_ms / 1000.0
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True, name=name)
        self._steps      = 0
        self._errors     = 0
        self._lock       = threading.Lock()

    def start(self) -> None:
        self._thread.start()
        logger.info("phase2: BatchWorker started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread.ident is None:
            return
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("batch worker did not stop before the shutdown deadline")
        logger.info("phase2: BatchWorker stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            batch = []
            try:
                batch = self._scheduler.schedule()
                if not batch:
                    self._stop_event.wait(self._tick)
                    continue
                results = self._forward_fn(batch)
                results = self._validate_results(batch, results)
                self._scheduler.commit_step(results)
                with self._lock:
                    self._steps += 1
            except BaseException as exc:
                with self._lock:
                    self._errors += 1
                logger.warning("phase2: BatchWorker step error: %s", failure_detail(exc))
                try:
                    if not isinstance(exc, Exception) or not batch:
                        self._stop_event.set()
                        self._scheduler.fail_all(exc)
                    else:
                        self._scheduler.fail_requests(batch, exc)
                except BaseException as callback_error:
                    # A process-level user callback failure must not kill the
                    # only worker while unrelated queued callers still wait.
                    add_failure_note(exc, 'batch failure callback failed: ' + failure_detail(callback_error))
                    self._stop_event.set()
                    try:
                        self._scheduler.fail_all(exc)
                    except BaseException as cleanup:
                        add_failure_note(exc, 'additional batch failure callback failed: ' + failure_detail(cleanup))

    @staticmethod
    def _validate_results(batch, results):
        """Reject malformed or incomplete forward results before scheduler commit."""
        if type(results) is not list:
            raise TypeError("batch forward_fn must return a list of request/token pairs")
        expected = {id(request): request for request in batch}
        seen = set()
        for record in results:
            if type(record) is not tuple or len(record) != 2:
                raise TypeError("batch forward result must be a (request, token) pair")
            request, token = record
            request_id = id(request)
            if request_id not in expected or expected[request_id] is not request:
                raise ValueError("batch forward result contains a foreign request")
            if request_id in seen:
                raise ValueError("batch forward result contains a duplicate request")
            seen.add(request_id)
            if token is not None and (type(token) is not int or token < 0):
                raise ValueError("batch forward token must be a nonnegative integer or None")
            if (request.state == RequestState.PREFILL and token is not None and
                    request.prefill_cursor + request.scheduled_prefill_tokens < len(request.token_ids)):
                raise ValueError('batch forward produced a token before scheduled prefill completed')
            if request.state == RequestState.DECODE and token is None:
                raise ValueError("decode forward step must return a token")
        if seen != set(expected):
            raise ValueError("batch forward result omitted one or more scheduled requests")
        return results

    def metrics(self) -> dict:
        with self._lock:
            return {
                "worker_steps":  self._steps,
                "worker_errors": self._errors,
                "worker_running": self._thread.is_alive(),
            }


# ---------------------------------------------------------------------------
# MultiSessionBatchEngine — high-level engine
# ---------------------------------------------------------------------------

class MultiSessionBatchEngine:
    """Engine that accepts requests from multiple concurrent sessions and
    batches them through a shared forward pass.

    Usage::

        def my_forward(batch):
            results = []
            for req in batch:
                # call your model; return next token
                next_tok = model_step(req.token_ids + req.generated, req.prefill_cursor)
                results.append((req, next_tok))
            return results

        engine = MultiSessionBatchEngine(forward_fn=my_forward)
        engine.start()

        # From any session:
        req = engine.submit(token_ids=[1, 2, 3], max_new_tokens=64)
        tokens = req.wait()   # blocks until done

        engine.stop()
    """

    def __init__(
        self,
        forward_fn: Callable[[List[BatchRequest]], List[Tuple[BatchRequest, Optional[int]]]],
        config: Optional[SchedulerConfig] = None,
        request_admission=None,
    ) -> None:
        self._config    = config or SchedulerConfig()
        self._request_admission = request_admission
        self._scheduler = ContinuousBatchScheduler(self._config)
        self._worker    = BatchWorker(
            self._scheduler, forward_fn, tick_ms=self._config.scheduler_tick_ms
        )
        self._closed = False
        self._started = False
        self._submissions = 0
        self._lock   = threading.Lock()
        self._condition = threading.Condition(self._lock)

    @property
    def scheduler(self) -> "ContinuousBatchScheduler":
        return self._scheduler

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("MultiSessionBatchEngine is closed")
            if self._worker._stop_event.is_set():
                raise RuntimeError('batch worker is stopped; close this engine before replacement')
            if self._started:
                return
            self._worker.start()
            self._started = True

    def stop(self, timeout: float = 10.0) -> None:
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or
                timeout < 0):
            raise ValueError("finite nonnegative batch shutdown timeout required")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._closed = True
            while self._submissions:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "batch submissions did not leave the admission boundary before shutdown"
                    )
                self._condition.wait(timeout=remaining)
        remaining = max(0.0, deadline - time.monotonic())
        self._worker.stop(timeout=remaining)
        self._scheduler.fail_all(RuntimeError("batch engine stopped before request completion"))

    def submit(
        self,
        token_ids:      List[int],
        *,
        session_id:     str = "",
        max_new_tokens: int = 128,
        stop_token_id:  Optional[int] = None,
        on_token:       Optional[Callable[[int], None]] = None,
        on_done:        Optional[Callable[[List[int]], None]] = None,
        on_error:       Optional[Callable[[Exception], None]] = None,
        priority:       int = 0,
    ) -> BatchRequest:
        """Submit a generation request.  Returns a BatchRequest; call ``.wait()``."""
        req = BatchRequest(
            session_id=session_id,
            token_ids=list(token_ids),
            max_new_tokens=max_new_tokens,
            stop_token_id=stop_token_id,
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
            priority=priority,
        )
        self._scheduler.validate_request(req)
        with self._condition:
            if self._closed:
                raise RuntimeError("MultiSessionBatchEngine is closed")
            if self._worker._stop_event.is_set():
                raise RuntimeError('batch worker is stopped; close this engine before replacement')
            self._submissions += 1
        try:
            if max_new_tokens == 0:
                self._scheduler.submit(req)
                return req
            if (self._request_admission is not None and
                    not self._request_admission.admit(req)):
                return req
            self._scheduler.submit(req)
            if (req.state == RequestState.FAILED and
                    self._request_admission is not None):
                self._request_admission.record_full_rejection(req)
            return req
        finally:
            with self._condition:
                self._submissions -= 1
                self._condition.notify_all()

    def cancel(self, request_id: str) -> bool:
        return self._scheduler.cancel(request_id)

    def metrics(self) -> dict:
        return {
            **self._scheduler.metrics(),
            **self._worker.metrics(),
        }

    def __enter__(self) -> "MultiSessionBatchEngine":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Convenience: wait helper on BatchRequest
# ---------------------------------------------------------------------------

def wait_for_request(req: BatchRequest, timeout: Optional[float] = None) -> List[int]:
    """Block until the request finishes; return generated token list."""
    if not req._event.wait(timeout=timeout):
        raise TimeoutError(f"Request {req.request_id} did not finish before the wait deadline")
    if req.state == RequestState.FAILED:
        if req.error is not None:
            raise req.error
        raise RuntimeError(f"Request {req.request_id} failed")
    if req.state == RequestState.CANCELLED:
        raise RuntimeError(f"Request {req.request_id} was cancelled")
    return req.generated


BatchRequest.wait = lambda self, timeout=None: wait_for_request(self, timeout)  # type: ignore[attr-defined]


__all__ = [
    "RequestState",
    "BatchRequest",
    "SchedulerConfig",
    "ContinuousBatchScheduler",
    "BatchWorker",
    "MultiSessionBatchEngine",
    "wait_for_request",
]
