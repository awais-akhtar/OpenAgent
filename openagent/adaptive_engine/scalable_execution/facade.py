"""Opt-in session protocol for coordinator-owned model factory integration."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import threading
import operator

from .resources import check_cancel, positive


@dataclass(frozen=True)
class ExecutionPlan:
    steps: object
    finish: object
    scratch_bytes: int


@dataclass(frozen=True)
class SessionCheckpoint:
    prefix_id: str
    owner: str
    backend_identity: str


class ScalableSession:
    """forward/snapshot/restore/reset facade, with explicit planner injection.

    planner(tuple[int,...]) -> ExecutionPlan. Planner/callbacks must keep ALL
    persistent model state in the supplied cache transaction, not in mutable
    hidden backend state. This facade neither selects nor certifies a model.
    """

    def __init__(self, executor, session_id, owner, planner, *, vocab_size,
                 create=True, prefix_id=None):
        if not callable(planner):
            raise TypeError("backend planner required")
        if type(create) is not bool:
            raise ValueError('explicit boolean scalable session creation required')
        self.executor = executor
        self.session_id, self.owner = session_id, owner
        self.planner = planner
        self.vocab_size = positive(vocab_size, "vocab_size")
        self._closed = False
        self._lock = threading.RLock()
        self._active = False
        self._cancellation = threading.local()
        if create:
            executor.cache.create_session(session_id, owner, prefix_id=prefix_id)
        elif prefix_id is not None:
            raise ValueError("prefix_id only applies when creating a session")
        else:
            with executor.cache._lock:
                executor.cache._check()
                executor.cache._session(session_id, owner)

    def _check(self):
        if self._closed:
            raise RuntimeError("session facade closed")
        if self._active:
            raise RuntimeError("reentrant session operation during forward")

    @property
    def identity(self):
        return self.executor.geometry.identity

    @contextmanager
    def cancellation_scope(self, event):
        if event is not None and not callable(getattr(event, 'is_set', None)):
            raise ValueError('event-like scalable session cancellation required')
        previous = getattr(self._cancellation, 'event', None)
        self._cancellation.event = event
        try:
            yield self
        finally:
            self._cancellation.event = previous

    def _cancel_predicate(self, explicit):
        if explicit is not None and not callable(explicit):
            raise TypeError('callable scalable session cancellation required')
        event = getattr(self._cancellation, 'event', None)
        if event is None:
            return explicit
        return lambda: event.is_set() or (explicit is not None and explicit())

    def forward(self, token_ids, *, cancelled=None):
        cancelled = self._cancel_predicate(cancelled)
        with self._lock:
            self._check()
            self._active = True
            try:
                tokens = []
                for token in token_ids:
                    check_cancel(cancelled)
                    if len(tokens) >= self.executor.geometry.max_batch_tokens:
                        raise ValueError("backend batch token limit exceeded")
                    # Accept numpy integer scalars without accepting floats/bools.
                    if isinstance(token, bool) or getattr(token, "dtype", None) == "bool":
                        raise ValueError("boolean is not a token ID")
                    try:
                        value = operator.index(token)
                    except TypeError as exc:
                        raise ValueError("token ID must be an integer") from exc
                    if not 0 <= value < self.vocab_size:
                        raise ValueError("token ID outside vocabulary")
                    tokens.append(value)
                if not tokens:
                    raise ValueError("empty forward batch")
                plan = self.planner(tuple(tokens))
                if not isinstance(plan, ExecutionPlan):
                    raise TypeError("planner must return ExecutionPlan")

                def finish(context):
                    result = plan.finish(context)
                    # Enforce the coordinator's concrete output protocol before
                    # committing cache, not after a successful durable append.
                    import numpy as np
                    if not isinstance(result, np.ndarray):
                        raise ValueError("backend must return floating numpy [N,vocab] logits")
                    # Strip ndarray subclass hooks while preserving the array data.
                    result = np.asarray(result)
                    if (
                            result.shape != (len(tokens), self.vocab_size) or
                            result.dtype not in (np.dtype('float32'),np.dtype('float64'))):
                        raise ValueError("backend must return floating numpy [N,vocab] logits")
                    finite_scratch=min(4096,result.size)*(result.dtype.itemsize+1)
                    if result.nbytes*2+finite_scratch > plan.scratch_bytes:
                        raise MemoryError('detached logits and finite checks exceed planner scratch reservation')
                    # Keep finite-check scratch bounded even for large vocabularies.
                    with np.nditer(result, flags=["external_loop", "buffered", "zerosize_ok"],
                                   op_flags=["readonly"], buffersize=4096) as iterator:
                        for chunk in iterator:
                            context.check_cancel()
                            if not np.isfinite(chunk).all():
                                raise ValueError("backend logits must be finite")
                    return np.array(result, order='C', copy=True)

                return self.executor.dispatch(self.session_id, self.owner, tokens,
                    plan.steps, finish, scratch_bytes=plan.scratch_bytes, cancelled=cancelled)
            finally:
                self._active = False

    def snapshot(self):
        with self._lock:
            self._check()
            prefix = self.executor.cache.publish_prefix(self.session_id, self.owner)
            return SessionCheckpoint(prefix, self.owner, self.executor.geometry.identity)

    def write_cache(self, output, *, max_bytes, cancelled=None):
        return self._transport(output, max_bytes=max_bytes, cancelled=cancelled)

    def restore_cache(self, source, *, max_bytes, cancelled=None):
        return self._transport(source, max_bytes=max_bytes, cancelled=cancelled, restoring=True)

    def _transport(self, stream, *, max_bytes, cancelled, restoring=False):
        from .session_transport import transport
        cancelled = self._cancel_predicate(cancelled)
        with self._lock:
            self._check()
            self._active = True
            try:
                return transport(self.executor.cache, self.session_id, self.owner, stream,
                                 max_bytes=max_bytes, cancelled=cancelled, restoring=restoring,
                                 vocabulary=self.vocab_size)
            finally:
                self._active = False

    def retry_cleanup(self):
        with self._lock:
            self._check()
            self.executor.retry_cleanup()

    def restore(self, checkpoint):
        with self._lock:
            self._check()
            self._checkpoint(checkpoint)
            self.executor.cache.restore_session(self.session_id, self.owner, checkpoint.prefix_id)

    def release_snapshot(self, checkpoint):
        """Release explicitly after generation success or rollback; never by GC."""
        with self._lock:
            self._check()
            self._checkpoint(checkpoint)
            self.executor.cache.release_prefix(checkpoint.prefix_id, self.owner)

    def _checkpoint(self, checkpoint):
        if (not isinstance(checkpoint, SessionCheckpoint) or checkpoint.owner != self.owner
                or checkpoint.backend_identity != self.executor.geometry.identity):
            raise ValueError("checkpoint owner/backend mismatch")

    def reset(self):
        with self._lock:
            self._check()
            self.executor.cache.reset(self.session_id, self.owner)

    @property
    def position(self):
        with self._lock:
            self._check()
            with self.executor.cache._lock:
                self.executor.cache._check()
                return self.executor.cache._session(self.session_id, self.owner)[1]

    def close(self, *, preserve_durable_session=False):
        if type(preserve_durable_session) is not bool:
            raise ValueError('explicit boolean durable session preservation required')
        with self._lock:
            if self._closed:
                return
            self._check()
            self.executor.retry_cleanup()
            if not preserve_durable_session:
                self.executor.cache.release_session(self.session_id, self.owner)
            self._closed = True
