"""Phase 2 Forward Bridge — connects batch requests to Phase 1 session execution.

The ``MultiSessionBatchEngine`` (Phase 2C) requires a ``forward_fn`` that
processes a batch of ``BatchRequest`` objects and returns per-request next
tokens.  ``Phase2ForwardBridge`` provides that function backed by actual
Phase 1 sessions.

Problem it solves
-----------------
Phase 1 sessions are single-request, synchronous: ``session.generate(token_ids)``
blocks until all tokens are produced.  The batch engine needs one-token-at-a-time
forward steps across a mixed batch of prefill and decode sequences.

Solution
--------
The bridge maintains a pool of Phase 1 sessions (one per concurrently running
request) and dispatches each step:

  - **Prefill step** — stages exactly the scheduler's prompt chunk. At the
    final chunk, the session's generation API evaluates the assembled prompt.
    Staging bounds are not a claim of physically chunked model computation.
  - **Decode step** — calls ``session.generate`` with the accumulated context
    (prompt + generated so far) capped at one new token.
  - **KV prefix reuse** — explicitly rejected here until a physical cache
    adapter binds cache bytes, schema and processed positions. Other engine
    owners' cache support is unaffected by this bridge-specific restriction.

Each ``BatchRequest`` is mapped to a ``ForwardSlot`` that holds its dedicated
Phase 1 session handle for the duration of the request.

Architecture
------------
``ForwardSlot``         -- one active request's execution context
``SessionFactory``      -- callable that produces a fresh Phase 1 session
``Phase2ForwardBridge`` -- implements the forward_fn protocol
"""
from __future__ import annotations

import logging
import threading
import time
from operator import index as integer_index
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .continuous_batching import BatchRequest, RequestState
from .failure_reporting import add_failure_note, failure_detail

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Produces a fresh Phase 1 session (Phase1HNPUSession or any duck-typed equiv)
SessionFactory = Callable[[], Any]

# forward_fn protocol: List[BatchRequest] → List[Tuple[BatchRequest, Optional[int]]]
ForwardFn = Callable[[List[BatchRequest]], List[Tuple[BatchRequest, Optional[int]]]]


# ---------------------------------------------------------------------------
# ForwardSlot — per-request execution context
# ---------------------------------------------------------------------------

@dataclass
class ForwardSlot:
    """Holds one active request's Phase 1 session and execution state."""
    request:        BatchRequest
    session:        Any            # Phase 1 session (duck-typed)
    context_ids:    List[int] = field(default_factory=list)
    prefill_done:   bool = False
    kv_prefix_used: bool = False
    created_at:     float = field(default_factory=time.monotonic)
    steps:          int   = 0
    last_token:     Optional[int] = None


# ---------------------------------------------------------------------------
# Phase2ForwardBridge
# ---------------------------------------------------------------------------

class Phase2ForwardBridge:
    """Implements the ``forward_fn`` protocol for ``MultiSessionBatchEngine``.

    Maintains a pool of Phase 1 session slots — one slot per active
    ``BatchRequest``.  Each call to ``__call__(batch)`` performs one forward
    step across all requests in the batch, returning one token per decode
    request and ``None`` for prefill steps.

    Args:
        session_factory: Callable that returns a new Phase 1 session.
            Called once per admitted request; the session is closed when
            the request finishes or is cancelled.
        max_slots: Maximum concurrent request slots (caps memory usage).
        kv_cache: Must be None until this bridge has a bound physical cache adapter.
        vocab_size: Vocabulary bound for model-produced tokens (no synthetic fallback).
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        max_slots:   int = 32,
        kv_cache:    Any = None,
        vocab_size:  int = 32000,
    ) -> None:
        if not callable(session_factory):
            raise TypeError('callable Phase 1 session factory required')
        if type(max_slots) is not int or max_slots <= 0:
            raise ValueError('positive forward slot capacity required')
        if type(vocab_size) is not int or not 0 < vocab_size < 2**31:
            raise ValueError('positive signed 32-bit vocabulary bound required')
        if kv_cache is not None:
            raise ValueError('forward bridge shared KV requires a bound physical cache adapter')
        self._factory   = session_factory
        self._max_slots = max_slots
        self._kv_cache  = kv_cache
        self._vocab_size = vocab_size

        self._slots:    Dict[str, ForwardSlot] = {}   # request_id → slot
        self._lock      = threading.RLock()
        self._active = False
        self._closed = False

        # Telemetry
        self._total_steps      = 0
        self._prefill_steps    = 0
        self._decode_steps     = 0
        self._kv_reuse_hits    = 0
        self._session_opens    = 0
        self._session_closes   = 0
        self._errors           = 0

    # ---- forward_fn protocol ------------------------------------------------

    def __call__(
        self, batch: List[BatchRequest]
    ) -> List[Tuple[BatchRequest, Optional[int]]]:
        """Process one batch step; return (request, next_token_or_None) per request."""
        results: List[Tuple[BatchRequest, Optional[int]]] = []
        with self._lock:
            if self._closed:
                raise RuntimeError('forward bridge is closed')
            if self._active:
                raise RuntimeError('forward bridge dispatch reentry is unsupported')
            if (type(batch) is not list or any(not isinstance(req, BatchRequest) for req in batch) or
                    any(type(req.request_id) is not str or not req.request_id for req in batch) or
                    len({req.request_id for req in batch}) != len(batch)):
                raise ValueError('distinct exact batch request identities required')
            for req in batch:
                owned = self._slots.get(req.request_id)
                if owned is not None and owned.request is not req:
                    raise ValueError('forward request ID belongs to another active request')
            self._total_steps += 1
            self._active = True
            try:
                for req in batch:
                    if req.state in (RequestState.CANCELLED, RequestState.FAILED):
                        results.append((req, None))
                        self._release_slot(req.request_id)
                        continue
                    token = self._step_request(req)
                    results.append((req, token))
                    if token is not None and (len(req.generated) + 1 >= req.max_new_tokens or
                                              token == req.stop_token_id):
                        self._release_slot(req.request_id)
            except BaseException as primary:
                self._errors += 1
                # Nothing is committed by the batch worker until all results
                # validate. Retire every affected physical owner on failure.
                for req in batch:
                    try:
                        owned = self._slots.get(req.request_id)
                        if owned is not None and owned.request is req:
                            self._release_slot(req.request_id)
                    except BaseException as cleanup:
                        add_failure_note(primary, 'forward slot cleanup retained: ' + failure_detail(cleanup))
                raise
            finally:
                self._active = False

        return results

    # ---- per-request step ---------------------------------------------------

    def _step_request(self, req: BatchRequest) -> Optional[int]:
        if req.state not in (RequestState.PREFILL, RequestState.DECODE):
            raise ValueError('forward request must have a scheduled prefill or decode step')
        if req.kv_prefix_len != 0:
            raise ValueError('forward bridge KV prefix reuse requires a bound physical cache adapter')
        slot = self._get_or_open_slot(req)
        if slot is None:
            raise RuntimeError('forward bridge slot capacity exhausted')

        with self._lock:
            slot.steps += 1

        # ---- prefill phase --------------------------------------------------
        if req.state == RequestState.PREFILL:
            return self._prefill_step(req, slot)

        # ---- decode phase ---------------------------------------------------
        if req.state == RequestState.DECODE:
            return self._decode_step(req, slot)

        return None

    def _prefill_step(self, req: BatchRequest, slot: ForwardSlot) -> Optional[int]:
        """Process one prefill chunk; return None (no token yet)."""
        chunk_end = min(
            req.prefill_cursor + (req.scheduled_prefill_tokens or 256),
            len(req.token_ids),
        )
        chunk = req.token_ids[req.prefill_cursor:chunk_end]

        slot.context_ids.extend(chunk)
        with self._lock:
            self._prefill_steps += 1

        # When prefill is complete, generate first token
        if chunk_end >= len(req.token_ids):
            slot.prefill_done = True
            return self._generate_one(slot, req)

        return None   # still prefilling

    def _decode_step(self, req: BatchRequest, slot: ForwardSlot) -> Optional[int]:
        """Generate one decode token."""
        with self._lock:
            self._decode_steps += 1
        return self._generate_one(slot, req)

    def _generate_one(self, slot: ForwardSlot, req: BatchRequest) -> Optional[int]:
        """Call the Phase 1 session to produce exactly one next token."""
        context = list(slot.context_ids)

        # Try generate_single (fastest — one-token path)
        gen_single = getattr(slot.session, "generate_single", None)
        if callable(gen_single):
            import numpy as np
            logits = gen_single(context)
            if (not isinstance(logits, np.ndarray) or
                    logits.shape not in ((self._vocab_size,), (1, self._vocab_size)) or
                    logits.dtype.kind != 'f' or not np.isfinite(logits).all()):
                raise ValueError('single-step generation requires finite vocabulary-sized logits')
            token = int(np.argmax(logits))
            slot.context_ids.append(token)
            slot.last_token = token
            return token

        # Use generate only when the single-step API is absent, never as
        # recovery from a failed call whose physical state is unknown.
        gen = getattr(slot.session, "generate", None)
        if callable(gen):
            result = gen(context_ids=context, max_new_tokens=1)
            if type(result) in (list, tuple) and len(result) == len(context) + 1 and list(result[:-1]) == context:
                result = result[-1:]
            if type(result) not in (list, tuple) or len(result) != 1 or isinstance(result[0], bool):
                raise ValueError('Phase 1 generate must return exactly one model-produced token')
            try:
                token = integer_index(result[0])
            except TypeError as error:
                raise ValueError('Phase 1 generated token must be an integer') from error
            if not 0 <= token < self._vocab_size:
                raise ValueError('Phase 1 generated token is outside the vocabulary')
            slot.context_ids.append(token)
            slot.last_token = token
            return token
        raise RuntimeError('Phase 1 session has no supported generation method')

    # ---- slot management ----------------------------------------------------

    def _get_or_open_slot(self, req: BatchRequest) -> Optional[ForwardSlot]:
        with self._lock:
            slot = self._slots.get(req.request_id)
            if slot is not None:
                if slot.request is not req:
                    raise ValueError('forward request ID belongs to another active request')
                return slot
            if len(self._slots) >= self._max_slots:
                logger.warning("phase2: max slots reached (%d)", self._max_slots)
                return None

        # Open a new Phase 1 session for this request
        try:
            session = self._factory()
            slot = ForwardSlot(
                request=req,
                session=session,
                context_ids=[],
            )
            with self._lock:
                self._slots[req.request_id] = slot
                self._session_opens += 1
            return slot
        except Exception as exc:
            logger.warning("phase2: session factory failed: %s", failure_detail(exc))
            raise

    def _release_slot(self, request_id: str) -> None:
        with self._lock:
            slot = self._slots.get(request_id)
            if slot is None:
                return
            close_fn = getattr(slot.session, "close", None)
            if callable(close_fn):
                close_fn()
            self._slots.pop(request_id)
            self._session_closes += 1

    def release_slot(self, request_id: str) -> None:
        """Public release — call when a request finishes or is cancelled."""
        with self._lock:
            if self._active:
                raise RuntimeError('cannot release a forward slot from its active callback')
            self._release_slot(request_id)

    def close_all(self) -> None:
        """Release all active slots (call on service shutdown)."""
        with self._lock:
            if self._active:
                raise RuntimeError('cannot close forward bridge from its active callback')
            self._closed = True
            ids = list(self._slots.keys())
            primary = None
            for rid in ids:
                try:
                    self._release_slot(rid)
                except BaseException as cleanup:
                    if primary is None:
                        primary = cleanup
                    else:
                        add_failure_note(primary, 'additional forward slot cleanup failure: ' + failure_detail(cleanup))
            if primary is not None:
                raise primary

    # ---- telemetry ----------------------------------------------------------

    def metrics(self) -> dict:
        with self._lock:
            return {
                "bridge_active_slots":   len(self._slots),
                "bridge_total_steps":    self._total_steps,
                "bridge_prefill_steps":  self._prefill_steps,
                "bridge_decode_steps":   self._decode_steps,
                "bridge_kv_reuse_hits":  self._kv_reuse_hits,
                "bridge_session_opens":  self._session_opens,
                "bridge_session_closes": self._session_closes,
                "bridge_errors":         self._errors,
            }


__all__ = [
    "ForwardSlot",
    "SessionFactory",
    "ForwardFn",
    "Phase2ForwardBridge",
]
