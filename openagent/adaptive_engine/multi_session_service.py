"""Phase 2 integration adapter: multi-session batching with shared caches.

Wires together all Phase 2 components:

1. ``SharedExpertLease``      (Phase 2A) — one warm expert pool per checkpoint
2. ``SharedKVLease``          (Phase 2B) — one prefix KV cache per checkpoint
3. ``MultiSessionBatchEngine``(Phase 2C) — continuous batching across sessions
4. ``Phase2MemoryManager``    — background RAM/VRAM pressure monitor + eviction
5. ``Phase2RequestQueue``     — priority queue with per-session rate limiting

``Phase2MultiSessionService`` is the single entry point.  It:
  - Opens one shared expert store and one shared KV cache per checkpoint.
  - Manages a ``MultiSessionBatchEngine`` backed by a user-supplied forward_fn.
  - Starts a ``Phase2MemoryManager`` wired to both shared stores.
  - Accepts an optional ``Phase2RequestQueue`` for priority admission control.
  - Opens per-session handles (``Phase2SessionHandle``) that wrap Phase 1 sessions
    and route generation through the batch engine.

Session stack overview::

    Phase1H/G/E/D/C/B session   (single-session Phase 1 optimisations)
         └── Phase2SessionHandle  ← this module; submits to batch engine

The Phase 1 session is still used for expert kernel dispatch, placement hints,
and speculative decode.  The batch engine handles scheduling; the forward_fn
drives the actual model.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .failure_reporting import add_failure_note, failure_detail
from .shared_expert_store import (
    SharedExpertLease,
    open_shared_expert_store,
    SharedExpertStoreRegistry,
)
from .shared_kv_cache import (
    SharedKVLease,
    open_shared_kv_cache,
    SharedKVCacheRegistry,
    PrefixKVEntry,
)
from .continuous_batching import (
    BatchRequest,
    SchedulerConfig,
    MultiSessionBatchEngine,
    RequestState,
    wait_for_request,
)
from .memory_pressure_manager import Phase2MemoryManager, MemoryThresholds
from .request_queue import Phase2RequestQueue, RequestPriority

logger = logging.getLogger(__name__)

PHASE2_STATE_FORMAT = "openagent.phase2-multi-session.v1"


# ---------------------------------------------------------------------------
# Phase2MultiSessionService — process-wide service for one checkpoint
# ---------------------------------------------------------------------------

class Phase2MultiSessionService:
    """Process-wide batching service for one checkpoint.

    One service per checkpoint.  Multiple ``Phase2SessionHandle`` instances
    share it; each handle is one concurrent user.

    Args:
        checkpoint_id:   Unique string identifying the checkpoint (e.g. sha256).
        checkpoint_dir:  Root directory of the model checkpoint.
        forward_fn:      ``(batch) → results`` — drives the actual forward pass.
        config:          ``SchedulerConfig`` tuning parameters.
        expert_registry: Override process-wide expert store registry.
        kv_registry:     Override process-wide KV cache registry.
    """

    def __init__(
        self,
        checkpoint_id: str,
        checkpoint_dir: Path,
        forward_fn: Callable[[List[BatchRequest]], List[Tuple[BatchRequest, Optional[int]]]],
        *,
        config: Optional[SchedulerConfig] = None,
        expert_registry: Optional[SharedExpertStoreRegistry] = None,
        kv_registry: Optional[SharedKVCacheRegistry] = None,
        memory_thresholds: Optional[MemoryThresholds] = None,
        request_queue: Optional[Phase2RequestQueue] = None,
    ) -> None:
        self.checkpoint_id  = checkpoint_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._memory_stopped = False
        self._engine_stopped = False
        self._expert_store_closed = False
        self._kv_cache_closed = False
        self._handles: Dict[str, "Phase2SessionHandle"] = {}
        self._expert_store = None
        self._kv_cache = None
        self._engine = None
        self._memory_manager = None
        self._request_queue: Optional[Phase2RequestQueue] = request_queue

        def cleanup_failed_construction() -> None:
            first: BaseException | None = None

            def record(error: BaseException, label: str) -> None:
                nonlocal first
                if first is None:
                    first = error
                else:
                    add_failure_note(first, label + ": " + failure_detail(error))

            if self._memory_manager is not None and not self._memory_stopped:
                try:
                    self._memory_manager.stop()
                    self._memory_stopped = True
                except BaseException as error:
                    record(error, "construction memory-manager stop")
            if self._engine is not None and not self._engine_stopped:
                try:
                    self._engine.stop()
                    self._engine_stopped = True
                except BaseException as error:
                    record(error, "construction batch-engine stop")
            if first is not None:
                add_failure_note(
                    first,
                    "shared stores retained because construction workers did not stop; retry cleanup",
                )
                raise first

            if self._kv_cache is not None and not self._kv_cache_closed:
                try:
                    self._kv_cache.close()
                    self._kv_cache_closed = True
                except BaseException as error:
                    record(error, "construction KV-cache close")
            if self._expert_store is not None and not self._expert_store_closed:
                try:
                    self._expert_store.close()
                    self._expert_store_closed = True
                except BaseException as error:
                    record(error, "construction expert-store close")
            if first is not None:
                raise first

        try:
            # Shared caches and background owners are assembled in stages so
            # failure at any later stage can retire only owners already opened.
            self._expert_store = open_shared_expert_store(
                checkpoint_id, checkpoint_dir, registry=expert_registry)
            self._kv_cache = open_shared_kv_cache(
                checkpoint_id, registry=kv_registry)

            scheduler_config = config
            if request_queue is not None:
                if scheduler_config is None:
                    scheduler_config = SchedulerConfig(max_queue_depth=request_queue.max_size)
                elif scheduler_config.max_queue_depth > request_queue.max_size:
                    scheduler_config = replace(
                        scheduler_config,
                        max_queue_depth=request_queue.max_size,
                    )
            self._engine = MultiSessionBatchEngine(
                forward_fn=forward_fn,
                config=scheduler_config,
                request_admission=request_queue,
            )
            self._engine.start()

            self._memory_manager = Phase2MemoryManager(
                thresholds=memory_thresholds or MemoryThresholds())
            self._memory_manager.set_expert_store(self._expert_store)
            self._memory_manager.set_kv_cache(self._kv_cache)
            self._memory_manager.set_preemption_callback(self._engine.scheduler.preempt_one)
            self._memory_manager.start()
        except BaseException as failure:
            try:
                cleanup_failed_construction()
            except BaseException as cleanup:
                add_failure_note(
                    failure,
                    "Phase 2 construction cleanup failed: " + failure_detail(cleanup),
                )
                try:
                    setattr(failure, "cleanup", cleanup_failed_construction)
                except BaseException:
                    add_failure_note(failure, "retryable cleanup callback could not be attached")
            raise

        logger.info("phase2: MultiSessionService started for %s...", checkpoint_id[:16])

    # ---- session handle management ------------------------------------------

    def open_session(
        self,
        phase1_session: Optional[Any] = None,
        *,
        session_id: Optional[str] = None,
    ) -> "Phase2SessionHandle":
        """Open a new per-session handle to this service."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Phase2MultiSessionService is closed")
            sid = session_id or str(uuid.uuid4())[:8]
            handle = Phase2SessionHandle(
                service=self,
                session_id=sid,
                phase1_session=phase1_session,
            )
            self._handles[sid] = handle
        logger.debug("phase2: opened session handle %s", sid)
        return handle

    def _close_handle(self, session_id: str) -> None:
        with self._lock:
            self._handles.pop(session_id, None)

    # ---- shared resource accessors ------------------------------------------

    @property
    def expert_store(self) -> SharedExpertLease:
        return self._expert_store

    @property
    def kv_cache(self) -> SharedKVLease:
        return self._kv_cache

    # ---- metrics ------------------------------------------------------------

    def metrics(self) -> dict:
        m = {
            "phase2_sessions": len(self._handles),
            "expert_store":    self._expert_store.metrics(),
            "kv_cache":        self._kv_cache.metrics(),
            "batch_engine":    self._engine.metrics(),
            "memory":          self._memory_manager.metrics(),
        }
        if self._request_queue is not None:
            m["request_queue"] = self.request_queue_metrics()
        return m

    def request_queue_metrics(self) -> dict:
        """Combine rate-admission counters with the scheduler's live queue."""
        if self._request_queue is None:
            return {}
        metrics = self._request_queue.metrics()
        scheduler = self._engine.scheduler.metrics()
        metrics["queue_scheduler_waiting"] = scheduler["waiting_requests"]
        metrics["queue_scheduler_active"] = (
            scheduler["running_requests"] + scheduler["paused_requests"]
        )
        metrics["queue_total_depth"] += scheduler["waiting_requests"]
        metrics["queue_expired"] += scheduler["expired_requests"]
        return metrics

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        # Serialize retries: a timed-out worker must retain its stores, and a
        # later close() call resumes only the cleanup stages still outstanding.
        with self._close_lock:
            with self._lock:
                self._closed = True
                handles = list(self._handles.values())

            primary: BaseException | None = None

            def record(err: BaseException, label: str) -> None:
                nonlocal primary
                if primary is None:
                    primary = err
                else:
                    add_failure_note(primary, label + ": " + failure_detail(err))

            for handle in handles:
                try:
                    handle.close()
                except BaseException as err:
                    record(err, "session handle close failure")

            if not self._memory_stopped:
                try:
                    self._memory_manager.stop()
                    self._memory_stopped = True
                except BaseException as err:
                    record(err, "memory manager stop failure")
            if not self._engine_stopped:
                try:
                    self._engine.stop()
                    self._engine_stopped = True
                except BaseException as err:
                    record(err, "batch engine stop failure")

            # A still-running manager or forward worker can touch both shared
            # stores. Keep them alive until both worker stops have succeeded;
            # the per-stage flags make this operation safely retryable.
            if primary is not None:
                add_failure_note(
                    primary,
                    "unfinished session or worker cleanup retained shared stores; retry close()",
                )
                raise primary

            if not self._expert_store_closed:
                try:
                    self._expert_store.close()
                    self._expert_store_closed = True
                except BaseException as err:
                    record(err, "expert store close failure")
            if not self._kv_cache_closed:
                try:
                    self._kv_cache.close()
                    self._kv_cache_closed = True
                except BaseException as err:
                    record(err, "kv cache close failure")

            if primary is not None:
                raise primary
            logger.info("phase2: MultiSessionService closed for %s...", self.checkpoint_id[:16])

    def __enter__(self) -> "Phase2MultiSessionService":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Phase2SessionHandle — per-session interface
# ---------------------------------------------------------------------------

class Phase2SessionHandle:
    """Per-session handle to the Phase2MultiSessionService.

    Exposes the same ``generate`` / ``stream`` / ``close`` / ``metrics``
    interface as Phase 1 sessions.  Routes generation through the batch engine.

    KV prefix reuse: before submitting, checks the shared KV cache for the
    longest cached prefix of the prompt; passes ``kv_prefix_len`` to the
    forward_fn via the BatchRequest so the model can skip recomputing those
    tokens.

    Phase 1 pass-through: all other calls (``on_token_routed``, expert kernel
    registration, etc.) are forwarded to the wrapped Phase 1 session.
    """

    def __init__(
        self,
        service: Phase2MultiSessionService,
        session_id: str,
        phase1_session: Optional[Any] = None,
    ) -> None:
        self._service       = service
        self.session_id     = session_id
        self._phase1        = phase1_session
        self._lock          = threading.RLock()
        self._condition     = threading.Condition(self._lock)
        self._closed        = False
        self._closing       = False
        self._active_operations = 0
        self._requests_made = 0
        self._tokens_generated = 0

    def _check_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError(f"Phase2SessionHandle {self.session_id} is closed")

    def _begin_operation(self, *, count_request: bool = True) -> None:
        with self._condition:
            self._check_open()
            if count_request:
                self._requests_made += 1
            self._active_operations += 1

    def _end_operation(self) -> None:
        with self._condition:
            if self._active_operations <= 0:
                raise RuntimeError("Phase2 session operation lease underflow")
            self._active_operations -= 1
            self._condition.notify_all()

    # ---- KV prefix lookup ---------------------------------------------------

    def _find_kv_prefix(self, token_ids: List[int]) -> int:
        """Return the number of tokens already cached in the shared KV cache."""
        entry = self._service.kv_cache.find_prefix(token_ids)
        if entry is not None:
            return entry.prefix_len
        return 0

    # ---- generation ---------------------------------------------------------

    def generate(
        self,
        *args: Any,
        context_ids: Optional[List[int]] = None,
        max_new_tokens: int = 128,
        stop_token_id: Optional[int] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> List[int]:
        """Submit to the batch engine and wait for completion."""
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported Phase 2 generation options: {unsupported}")
        self._validate_generation(max_new_tokens, cancelled)
        self._begin_operation()
        req = None
        try:
            token_ids = context_ids or (list(args[0]) if args else [])
            kv_prefix_len = self._find_kv_prefix(token_ids)

            req = self._service._engine.submit(
                token_ids=token_ids,
                session_id=self.session_id,
                max_new_tokens=max_new_tokens,
                stop_token_id=stop_token_id,
            )
            req.kv_prefix_len = kv_prefix_len

            if cancelled is None:
                tokens = wait_for_request(req)
            else:
                while not req._event.wait(0.01):
                    if cancelled():
                        self._service._engine.cancel(req.request_id)
                        break
                tokens = wait_for_request(req, timeout=0)
            with self._lock:
                self._tokens_generated += len(tokens)
            return tokens
        except BaseException as failure:
            if req is not None and not req.is_done():
                try:
                    self._service._engine.cancel(req.request_id)
                except BaseException as cleanup:
                    add_failure_note(
                        failure,
                        "batch request cancellation cleanup failed: " + failure_detail(cleanup),
                    )
            raise
        finally:
            self._end_operation()

    def _validate_generation(self, max_new_tokens: int, cancelled) -> None:
        scheduler = getattr(self._service._engine, "scheduler", None)
        config = getattr(scheduler, "config", None)
        maximum = getattr(config, "max_decode_len", None)
        if (type(max_new_tokens) is not int or max_new_tokens < 0 or
                (type(maximum) is int and max_new_tokens > maximum)):
            raise ValueError("generation budget must fit the batch scheduler's decode limit")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("generation cancellation callback must be callable")

    def stream(
        self,
        *args: Any,
        context_ids: Optional[List[int]] = None,
        max_new_tokens: int = 128,
        stop_token_id: Optional[int] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> Iterator[int]:
        """Submit and yield tokens as they are generated."""
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported Phase 2 generation options: {unsupported}")
        self._validate_generation(max_new_tokens, cancelled)
        self._begin_operation()
        req = None
        failure = None
        try:
            token_ids = context_ids or (list(args[0]) if args else [])
            kv_prefix_len = self._find_kv_prefix(token_ids)

            # At most max_new_tokens token callbacks plus one terminal record
            # can be produced. This bounds memory even when the consumer is
            # slow or abandons the iterator.
            token_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue(
                maxsize=max_new_tokens + 1
            )

            def _on_token(t: int) -> None:
                token_queue.put_nowait(("token", t))

            def _on_done(tokens: List[int]) -> None:
                token_queue.put_nowait(("done", None))

            def _on_error(err: Exception) -> None:
                token_queue.put_nowait(("error", err))

            req = self._service._engine.submit(
                token_ids=token_ids,
                session_id=self.session_id,
                max_new_tokens=max_new_tokens,
                stop_token_id=stop_token_id,
                on_token=_on_token,
                on_done=_on_done,
                on_error=_on_error,
            )
            req.kv_prefix_len = kv_prefix_len

            while True:
                if cancelled is not None and cancelled():
                    self._service._engine.cancel(req.request_id)
                    break
                try:
                    kind, value = token_queue.get(timeout=0.05)
                except queue.Empty:
                    if req.is_done():
                        if req.state == RequestState.FAILED:
                            raise RuntimeError(f"Request {req.request_id} failed")
                        if req.state == RequestState.CANCELLED:
                            return
                        break
                    continue
                if kind == "error":
                    raise value
                if kind == "done":
                    break
                with self._lock:
                    self._tokens_generated += 1
                yield value
        except BaseException as primary:
            failure = primary
            raise
        finally:
            # GeneratorExit (consumer abandonment) and callback failures must
            # not leave this request occupying a scheduler slot or advancing
            # the model after its caller has gone away.
            try:
                if req is not None and not req.is_done():
                    try:
                        self._service._engine.cancel(req.request_id)
                    except BaseException as cleanup:
                        if failure is None:
                            raise
                        add_failure_note(
                            failure,
                            "batch stream cancellation cleanup failed: " + failure_detail(cleanup),
                        )
            finally:
                self._end_operation()

    # ---- Phase 1 pass-through -----------------------------------------------

    def on_token_routed(self, layer_id: int, expert_ids: List[str]) -> None:
        """Forward Phase 1F routing hook to Phase 1 session."""
        self._begin_operation(count_request=False)
        try:
            if self._phase1 is not None:
                hook = getattr(self._phase1, "on_token_routed", None)
                if callable(hook):
                    hook(layer_id, expert_ids)
            # Also record heat in shared expert store
            self._service.expert_store.record_routing(expert_ids)
        finally:
            self._end_operation()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._phase1 is not None:
            self._begin_operation(count_request=False)
            try:
                value = getattr(self._phase1, name)
            finally:
                self._end_operation()
            if callable(value):
                def leased_call(*args: Any, **kwargs: Any) -> Any:
                    self._begin_operation(count_request=False)
                    try:
                        return value(*args, **kwargs)
                    finally:
                        self._end_operation()
                return leased_call
            return value
        raise AttributeError(name)

    # ---- telemetry ----------------------------------------------------------

    def metrics(self) -> dict:
        base: dict = {}
        if self._phase1 is not None:
            get_m = getattr(self._phase1, "metrics", None)
            if callable(get_m):
                base = get_m()
        with self._lock:
            phase2_session = {
                "session_id":       self.session_id,
                "requests_made":    self._requests_made,
                "tokens_generated": self._tokens_generated,
                "closed":           self._closed,
            }
        return {**base, "phase2_session": phase2_session,
                "phase2_service": self._service.metrics()}

    # ---- lifecycle ----------------------------------------------------------

    def close(self, timeout: float = 10.0) -> None:
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
                timeout < 0 or timeout > 3600 or not math.isfinite(timeout)):
            raise ValueError("session close timeout must be finite and between zero and 3600 seconds")
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError("Phase2SessionHandle close is already in progress")
            self._closing = True
            try:
                while self._active_operations:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"timed out waiting for {self._active_operations} active Phase2 operation(s)"
                        )
                    self._condition.wait(remaining)
            except BaseException:
                # Interrupted waits must not leave the handle permanently
                # rejecting work or make future close attempts impossible.
                self._closing = False
                self._condition.notify_all()
                raise
        try:
            if self._phase1 is not None:
                close_fn = getattr(self._phase1, "close", None)
                if callable(close_fn):
                    close_fn()
        except BaseException:
            with self._condition:
                self._closing = False
                self._condition.notify_all()
            raise
        with self._condition:
            self._closed = True
            self._closing = False
            self._condition.notify_all()
        self._service._close_handle(self.session_id)

    def __enter__(self) -> "Phase2SessionHandle":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_multi_session_service(
    checkpoint_id: str,
    checkpoint_dir: Path,
    forward_fn: Callable[[List[BatchRequest]], List[Tuple[BatchRequest, Optional[int]]]],
    *,
    config: Optional[SchedulerConfig] = None,
) -> Phase2MultiSessionService:
    """Create and start a Phase2MultiSessionService for a checkpoint."""
    return Phase2MultiSessionService(
        checkpoint_id=checkpoint_id,
        checkpoint_dir=Path(checkpoint_dir),
        forward_fn=forward_fn,
        config=config,
    )


__all__ = [
    "PHASE2_STATE_FORMAT",
    "Phase2MultiSessionService",
    "Phase2SessionHandle",
    "create_multi_session_service",
]
