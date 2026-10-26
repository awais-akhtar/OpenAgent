"""Transactional sessions for explicitly configured native model executors."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
import math
import threading
from typing import Callable
from uuid import uuid4

import numpy as np

from .failure_reporting import add_failure_note, failure_summary


class ExecutionCancelled(RuntimeError):
    """The request was cancelled before its session transaction committed."""


def _admit_checkpoint_publication(sink,max_bytes):
    if sink is None:
        if type(max_bytes) is not int or max_bytes!=0:
            raise ValueError('checkpoint publication budget requires a sink')
    elif not callable(sink) or type(max_bytes) is not int or not 0<max_bytes<=2**63-1:
        raise ValueError('checkpoint sink requires explicit complete frame admission')


def _admit_stop_prefixes(value, vocabulary):
    if type(value) not in (tuple, list) or len(value) > 64:
        raise ValueError('bounded stop-prefix inventory required')
    prefixes = tuple(tuple(prefix) for prefix in value)
    if (any(not prefix or len(prefix) > 128 or
            any(type(token) is not int or not 0 <= token < vocabulary for token in prefix)
            for prefix in prefixes) or len(set(prefixes)) != len(prefixes)):
        raise ValueError('finite valid stop-token prefixes required')
    return prefixes


def _matches_stop_prefix(history, prefixes):
    return any(len(history) >= len(prefix) and tuple(history[-len(prefix):]) == prefix
               for prefix in prefixes)


@dataclass(frozen=True)
class ExecutionLimits:
    vocabulary: int
    context: int
    max_new_tokens: int = 256
    max_sessions: int = 4
    prefill_chunk_tokens: int = 512
    max_snapshots_per_session: int = 4
    max_sampling_buffer_bytes: int = 128*1024**2
    max_logit_buffer_bytes: int = 256*1024**2
    max_concurrent_generations: int | None = None
    max_waiting_generations: int = 64
    max_aggregate_generation_bytes: int | None = None
    admission_timeout_seconds: float | None = None
    max_checkpoint_buffer_bytes: int = 16*1024**2

    def __post_init__(self):
        for name in ('vocabulary', 'context', 'max_new_tokens', 'max_sessions',
                     'prefill_chunk_tokens', 'max_snapshots_per_session', 'max_sampling_buffer_bytes',
                     'max_logit_buffer_bytes','max_checkpoint_buffer_bytes'):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= 2**31 - 1:
                raise ValueError('invalid execution limit: ' + name)
        if self.max_new_tokens > self.context:
            raise ValueError('generation limit exceeds context capacity')
        if self.vocabulary*8 > self.max_logit_buffer_bytes:
            raise MemoryError('one float64 vocabulary row exceeds logit admission')
        if self.max_concurrent_generations is not None and (
                type(self.max_concurrent_generations) is not int
                or not 1 <= self.max_concurrent_generations <= self.max_sessions):
            raise ValueError('generation concurrency must fit admitted session count')
        if type(self.max_waiting_generations) is not int or self.max_waiting_generations<=0:
            raise ValueError('positive generation queue capacity required')
        if self.max_aggregate_generation_bytes is not None and (
                type(self.max_aggregate_generation_bytes) is not int or
                self.max_aggregate_generation_bytes<self.max_logit_buffer_bytes+self.max_sampling_buffer_bytes+self.max_checkpoint_buffer_bytes):
            raise ValueError('aggregate generation bytes must admit one complete transaction')
        if self.admission_timeout_seconds is not None:
            import math
            if type(self.admission_timeout_seconds) not in (float,int) or not math.isfinite(self.admission_timeout_seconds) or self.admission_timeout_seconds<=0:
                raise ValueError('positive finite generation admission timeout required')

    @property
    def admitted_prefill_tokens(self):
        """Limit returned logits; native scratch and model memory are separate."""
        return min(self.prefill_chunk_tokens, self.max_logit_buffer_bytes//(self.vocabulary*8))


@dataclass(frozen=True)
class GenerationResult:
    session_id: str
    token_ids: tuple[int, ...]
    context_tokens: int
    finish_reason: str


@dataclass(frozen=True)
class ExecutionSessionStatus:
    session_id: str
    context_tokens: int
    context_capacity: int
    active: bool
    closed: bool
    snapshot_count: int
    identity: str | None
    pending_cleanup_count: int = 0
    shutdown_requested: bool = False


@dataclass(frozen=True)
class ExecutionSessionSnapshot:
    service_id: str
    session_id: str
    snapshot_id: str
    identity: str
    token_ids: tuple[int, ...]


class _Session:
    def __init__(self, model):
        self.model = model
        self.tokens = ()
        self.lock = threading.RLock()
        self.active = False
        self.closed = False
        self.released = False
        self.identity = None
        self.snapshots = {}
        self.cleanup_snapshots = {}
        self.shutdown_requested = threading.Event()

    @contextmanager
    def transaction(self):
        with self.lock:
            if self.closed or self.shutdown_requested.is_set():
                raise RuntimeError('execution session is closed')
            if self.active:
                raise RuntimeError('reentrant execution session mutation')
            self.active = True
            try:
                yield
            finally:
                self.active = False


class _SessionCancellation:
    def __init__(self, session, external):
        self.shutdown = session.shutdown_requested
        self.external = external

    def is_set(self):
        return self.shutdown.is_set() or (self.external is not None and self.external.is_set())


class NativeExecutionService:
    """Coordinate trusted executors without asserting checkpoint certification.

    Each factory invocation must return a fresh empty executor with forward,
    snapshot, restore and reset methods. Shared immutable weights are permitted;
    shared mutable model sessions are not. Greedy sampling is the default;
    seeded stochastic sampling is explicit. Logits belong to the executor.
    """
    def __init__(self, factory: Callable, limits: ExecutionLimits, *, factory_cleanup=None):
        if not callable(factory) or type(limits) is not ExecutionLimits:
            raise ValueError('explicit executor factory and limits required')
        if factory_cleanup is not None and not callable(factory_cleanup):
            raise ValueError('factory cleanup must be callable')
        self.factory, self.limits = factory, limits
        self._factory_cleanup = factory_cleanup
        self._sessions = {}
        self._lock = threading.RLock()
        self._opening = False
        self._closed = False
        self._identity = uuid4().hex
        self._orphaned = {}
        self._cleanup_lock = threading.RLock()
        self._closing = False
        self._generation_capacity = limits.max_concurrent_generations or limits.max_sessions
        from .admission_scheduler import AdmissionScheduler
        self._generation_bytes=limits.max_logit_buffer_bytes+limits.max_sampling_buffer_bytes+limits.max_checkpoint_buffer_bytes
        self._generation_gate = AdmissionScheduler(max_active=self._generation_capacity,
            max_bytes=limits.max_aggregate_generation_bytes or self._generation_capacity*self._generation_bytes,
            max_waiting=limits.max_waiting_generations)
        self._generation_local = threading.local()
        self._waiting_generations = self._active_generations = 0

    @contextmanager
    def _generation_slot(self, session, cancel):
        """Bound live transaction scratch without holding the registry lock."""
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('cancellation must provide is_set')
        if getattr(self._generation_local, 'active', False):
            raise RuntimeError('reentrant generation through an execution callback is unsupported')
        cancellation = _SessionCancellation(session, cancel)
        with self._lock:
            if self._closed:
                raise RuntimeError('execution service is closed')
            self._waiting_generations += 1
        acquired = None
        counted_active = False
        try:
            try:
                acquired=self._generation_gate.acquire(self._generation_bytes,cancelled=cancellation.is_set,
                    timeout=self.limits.admission_timeout_seconds)
            except InterruptedError as error:
                raise ExecutionCancelled('generation cancelled during admission') from error
            self._cancelled(cancellation)
            with self._lock:
                if self._closed:
                    raise RuntimeError('execution service closed before generation admission')
                self._waiting_generations -= 1
                self._active_generations += 1
                counted_active = True
            self._generation_local.active = True
            yield
        finally:
            with self._lock:
                if counted_active:
                    self._active_generations -= 1
                else:
                    self._waiting_generations -= 1
            if acquired:
                self._generation_local.active = False
                self._generation_gate.release(acquired)

    def runtime_status(self):
        """Report admissions, not RSS bounds or hardware utilization."""
        with self._lock:
            return {'closed': self._closed, 'sessions': len(self._sessions),
                    'session_capacity': self.limits.max_sessions,
                    'generation_capacity': self._generation_capacity,
                    'active_generations': self._active_generations,
                    'waiting_generations': self._waiting_generations,
                    'orphaned_executors': len(self._orphaned),
                    'scheduler':self._generation_gate.status()}

    def open_session(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('execution service is closed')
            if self._opening:
                raise RuntimeError('reentrant executor creation')
            if self._orphaned:
                raise RuntimeError('failed executor cleanup is pending; close the service')
            if len(self._sessions) >= self.limits.max_sessions:
                raise MemoryError('native session limit reached')
            self._opening = True
        model = None
        shared = False
        try:
            # Factory/reset may invoke application code. Do not hold the
            # registry lock while invoking callbacks or another model's lock.
            model = self.factory()
            with self._lock:
                shared = any(s.model is model for s in self._sessions.values())
            if shared:
                raise ValueError('factory returned an existing mutable executor')
            for name in ('forward', 'snapshot', 'restore', 'reset'):
                if not callable(getattr(model, name, None)):
                    raise ValueError('executor is missing ' + name)
            model.reset()
            session_id = uuid4().hex
            with self._lock:
                if self._closed:
                    raise RuntimeError('execution service closed during executor creation')
                self._sessions[session_id] = _Session(model)
            return session_id
        except BaseException as error:
            if model is not None and not shared and callable(getattr(model, 'close', None)):
                try:
                    model.close()
                except BaseException as cleanup_error:
                    with self._lock:
                        self._orphaned[id(model)] = model
                    add_failure_note(error, 'executor cleanup is pending: '+failure_summary(cleanup_error))
            raise
        finally:
            with self._lock:
                self._opening = False

    def _session(self, session_id):
        if type(session_id) is not str:
            raise ValueError('invalid session identifier')
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as error:
                raise ValueError('unknown execution session') from error

    def _tokens(self, values, capacity):
        result = tuple(islice(iter(values), capacity + 1))
        if not result or len(result) > capacity or any(
            type(t) is not int or not 0 <= t < self.limits.vocabulary for t in result
        ):
            raise ValueError('invalid token IDs or context capacity')
        return result

    def _history(self, tokens):
        if (type(tokens) is not tuple or len(tokens) > self.limits.context or
                any(type(t) is not int or not 0 <= t < self.limits.vocabulary for t in tokens)):
            raise ValueError('invalid execution session history')
        return tokens

    @staticmethod
    def _model_position(session, history, *, required=False):
        identity = getattr(session.model, 'identity', None)
        if session.identity is not None and identity is not None and identity != session.identity:
            raise ValueError('execution session model identity changed')
        found = False
        for name in ('length', 'position'):
            value = getattr(session.model, name, None)
            if value is None:
                continue
            found = True
            if type(value) is not int or value != len(history):
                raise ValueError('executor position differs from service history')
        if required and not found:
            raise ValueError('executor lacks public snapshot history/position metadata')
        return found

    def _validate_model_snapshot(self, session, snapshot, history, *, current):
        self._history(history)
        identity = getattr(snapshot, 'identity', None)
        if identity is None:
            identity = getattr(snapshot, 'backend_identity', None)
        if type(identity) is not str or not identity:
            raise ValueError('executor snapshot requires explicit identity metadata')
        model_identity = getattr(session.model, 'identity', None)
        if model_identity is not None and model_identity != identity:
            raise ValueError('executor snapshot identity differs from model')
        if session.identity is not None and identity != session.identity:
            raise ValueError('execution session model identity changed')
        found = False
        tokens = getattr(snapshot, 'tokens', None)
        if tokens is not None:
            if self._history(tokens) != history:
                raise ValueError('executor snapshot differs from service history')
            found = True
        for name in ('length', 'position'):
            position = getattr(snapshot, name, None)
            if position is not None:
                if type(position) is not int or position != len(history):
                    raise ValueError('executor snapshot position differs from service history')
                found = True
        if current:
            self._model_position(session, history, required=not found)
        session.identity = identity
        return identity

    def _capture_snapshot(self, session, history):
        snapshot = session.model.snapshot()
        try:
            self._validate_model_snapshot(session, snapshot, history, current=True)
        except BaseException as primary:
            self._release_transaction_snapshot(session, snapshot, primary)
            raise
        return snapshot

    def _owned_snapshot(self, session_id, session, snapshot):
        if (type(snapshot) is not ExecutionSessionSnapshot or
                any(type(value) is not str or not value for value in (
                    snapshot.service_id, snapshot.session_id, snapshot.snapshot_id, snapshot.identity)) or
                snapshot.service_id != self._identity or snapshot.session_id != session_id):
            raise ValueError('foreign execution session snapshot')
        owned = session.snapshots.get(snapshot.snapshot_id)
        if owned is None or owned[0] is not snapshot:
            raise ValueError('unknown, altered or released execution session snapshot')
        self._history(snapshot.token_ids)
        if (snapshot.identity != session.identity or snapshot.identity != owned[2] or
                snapshot.token_ids != owned[3]):
            raise ValueError('execution session snapshot identity mismatch')
        return owned[1]

    def status(self, session_id):
        session = self._session(session_id)
        with session.lock:
            return ExecutionSessionStatus(session_id, len(session.tokens), self.limits.context,
                                          session.active, session.closed or session.shutdown_requested.is_set(),
                                          len(session.snapshots), session.identity,
                                          len(session.cleanup_snapshots), session.shutdown_requested.is_set())

    def snapshot(self, session_id):
        session = self._session(session_id)
        with session.transaction():
            if len(session.snapshots) >= self.limits.max_snapshots_per_session:
                raise MemoryError('execution session snapshot limit reached')
            model_snapshot = self._capture_snapshot(session, session.tokens)
            try:
                handle = ExecutionSessionSnapshot(self._identity, session_id, uuid4().hex,
                                                   session.identity, session.tokens)
                session.snapshots[handle.snapshot_id] = (
                    handle, model_snapshot, session.identity, session.tokens)
            except BaseException as primary:
                self._release_transaction_snapshot(session, model_snapshot, primary)
                raise
            return handle

    def restore(self, session_id, snapshot):
        session = self._session(session_id)
        with session.transaction():
            model_snapshot = self._owned_snapshot(session_id, session, snapshot)
            self._validate_model_snapshot(session, model_snapshot, snapshot.token_ids, current=False)
            rollback = self._capture_snapshot(session, session.tokens);primary=None
            try:
                try:
                    session.model.restore(model_snapshot)
                    if not self._model_position(session, snapshot.token_ids):
                        verification = self._capture_snapshot(session, snapshot.token_ids)
                        self._release_snapshot(session, verification)
                except BaseException as error:
                    primary=error
                    try:
                        session.model.restore(rollback)
                        self._model_position(session, session.tokens)
                    except BaseException as recovery:
                        session.closed = True
                        add_failure_note(error, 'executor restore rollback failed; session quarantined: '+failure_summary(recovery))
                    raise
                session.tokens = snapshot.token_ids
            finally:
                self._release_transaction_snapshot(session,rollback,primary)

    def release_snapshot(self, session_id, snapshot):
        session = self._session(session_id)
        with session.lock:
            if session.active:
                raise RuntimeError('cannot release a snapshot during an active transaction')
            model_snapshot = self._owned_snapshot(session_id, session, snapshot)
            session.active = True
            try:
                # Remove ownership before invoking a callback that may fail or reenter.
                del session.snapshots[snapshot.snapshot_id]
                self._release_snapshot(session, model_snapshot)
            finally:
                session.active = False

    def _logits(self, value, count):
        if type(value) is not np.ndarray or value.shape != (count, self.limits.vocabulary):
            raise ValueError('executor returned an invalid logit shape')
        if value.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('executor returned invalid logit values')
        if value.nbytes > self.limits.max_logit_buffer_bytes:
            raise MemoryError('executor logit output exceeds admission')
        if not np.isfinite(value).all():
            raise ValueError('executor returned invalid logit values')
        return value

    @staticmethod
    def _cancelled(cancel):
        if cancel is not None and cancel.is_set():
            raise ExecutionCancelled('native generation cancelled; transaction rolled back')

    def _forward_model(self, session, tokens, cancel):
        self._cancelled(cancel)
        scope = getattr(session.model, 'cancellation_scope', None)
        if callable(scope):
            with scope(cancel):
                value = session.model.forward(tokens)
        else:
            value = session.model.forward(tokens)
        self._cancelled(cancel)
        return self._logits(value, len(tokens))

    def prefill(self, session_id, prompt_ids, *, cancel=None):
        """Append a bounded prompt transactionally without sampling a token."""
        session = self._session(session_id)
        incoming = self._tokens(prompt_ids, self.limits.context)
        if not incoming:
            raise ValueError('nonempty tokenized prefill required')
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('cancellation must provide is_set')
        with self._generation_slot(session, cancel), session.transaction():
            cancellation = _SessionCancellation(session, cancel)
            previous = session.tokens
            if len(previous) + len(incoming) > self.limits.context:
                raise ValueError('prefill exceeds session context')
            boundary = self._capture_snapshot(session, previous)
            primary = None
            committed = False
            try:
                for offset in range(0, len(incoming), self.limits.admitted_prefill_tokens):
                    chunk = incoming[offset:offset + self.limits.admitted_prefill_tokens]
                    self._forward_model(session, list(chunk), cancellation)
                target = previous + incoming
                self._model_position(session, target)
                self._cancelled(cancellation)
                session.tokens = target
                committed = True
                return len(incoming)
            except BaseException as error:
                primary = error
                if not committed:
                    try:
                        session.model.restore(boundary)
                        self._model_position(session, previous)
                    except BaseException as recovery:
                        session.closed = True
                        add_failure_note(error, 'prefill rollback failed; session quarantined: ' + failure_summary(recovery))
                raise
            finally:
                self._release_transaction_snapshot(session, boundary, primary)

    def score_continuations(self, session_id, prompt_ids, continuations, *,
                            normalize='mean', cancel=None):
        """Score a bounded closed set of candidate continuations without committing it.

        Candidate log probabilities are computed from the target executor's
        unmodified logits. The prompt is evaluated once; each continuation is
        evaluated from a snapshot of that prompt boundary. Both the session
        history and executor state are restored before returning.
        """
        session = self._session(session_id)
        if normalize not in ('mean', 'sum'):
            raise ValueError("score normalization must be 'mean' or 'sum'")
        if type(prompt_ids) not in (tuple, list) or not prompt_ids:
            raise ValueError('nonempty tokenized scoring prompt required')
        if (type(continuations) not in (tuple, list) or
                not 2 <= len(continuations) <= 64):
            raise ValueError('closed-set scoring requires 2 to 64 candidates')
        admitted = []
        total_read = 0
        for candidate in continuations:
            if (type(candidate) not in (tuple, list) or not candidate or
                    len(candidate) > self.limits.max_new_tokens):
                raise ValueError('each scoring candidate must be a bounded nonempty token sequence')
            tokens = tuple(candidate)
            if any(type(token) is not int or not 0 <= token < self.limits.vocabulary
                   for token in tokens):
                raise ValueError('scoring candidate contains an invalid token ID')
            total_read += len(tokens)
            admitted.append(tokens)
        if total_read > 65536:
            raise MemoryError('closed-set scoring token budget exceeded')
        incoming = self._tokens(prompt_ids, self.limits.context)
        if self.limits.vocabulary * 8 > self.limits.max_logit_buffer_bytes:
            raise MemoryError('scoring logit row exceeds admission')
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('cancellation must provide is_set')

        with self._generation_slot(session, cancel), session.transaction():
            cancellation = _SessionCancellation(session, cancel)
            if len(session.tokens) + len(incoming) + max(map(len, admitted)) > self.limits.context:
                raise ValueError('scoring prompt and candidate exceed session context')
            base = self._capture_snapshot(session, session.tokens)
            prompt_snapshot = None
            primary = None
            restored = False
            try:
                logits = None
                for offset in range(0, len(incoming), self.limits.admitted_prefill_tokens):
                    chunk = incoming[offset:offset + self.limits.admitted_prefill_tokens]
                    logits = self._forward_model(session, list(chunk), cancellation)
                prompt_logits = np.array(logits[-1], copy=True)
                prompt_snapshot = self._capture_snapshot(session, session.tokens + incoming)
                raw_scores = []
                for candidate in admitted:
                    self._cancelled(cancellation)
                    session.model.restore(prompt_snapshot)
                    self._model_position(session, session.tokens + incoming)
                    row_logits = prompt_logits
                    terms = []
                    for index, token in enumerate(candidate):
                        self._cancelled(cancellation)
                        row = np.asarray(row_logits, dtype=np.float64)
                        if row.shape != (self.limits.vocabulary,) or not np.isfinite(row).all():
                            raise ValueError('executor returned invalid scoring logits')
                        log_normalizer = float(np.logaddexp.reduce(row))
                        if not math.isfinite(log_normalizer):
                            raise ValueError('executor returned invalid scoring normalization')
                        terms.append(min(0.0, float(row[token]) - log_normalizer))
                        if index + 1 < len(candidate):
                            row_logits = self._forward_model(session, [token], cancellation)[-1]
                    total = math.fsum(terms)
                    raw_scores.append(total / len(terms) if normalize == 'mean' else total)

                score_array = np.asarray(raw_scores, dtype=np.float64)
                if not np.isfinite(score_array).all():
                    raise ValueError('closed-set scores are nonfinite')
                shifted = score_array - score_array.max()
                weights = np.exp(shifted)
                denominator = float(weights.sum())
                if not math.isfinite(denominator) or denominator <= 0:
                    raise ValueError('closed-set probability normalization failed')
                probabilities = weights / denominator
                entropy = -math.fsum(float(p) * math.log(float(p))
                                     for p in probabilities if p > 0)
                choices = tuple({
                    'candidate_index': index,
                    'logprob': raw_scores[index] if normalize == 'sum' else
                        raw_scores[index] * len(admitted[index]),
                    'mean_logprob': raw_scores[index] / len(admitted[index])
                        if normalize == 'sum' else raw_scores[index],
                    'normalized_score': raw_scores[index],
                    'tokens': len(admitted[index]),
                    'probability': float(probabilities[index]),
                } for index in range(len(admitted)))
                best = max(range(len(choices)), key=lambda index: (choices[index]['probability'], -index))
                result = {
                    'answer_index': best,
                    'normalize': normalize,
                    'entropy': entropy,
                    'choices': choices,
                    'usage': {'prompt_tokens': len(incoming), 'completion_tokens': 0,
                              'read_tokens': total_read},
                }
                self._cancelled(cancellation)
                session.model.restore(base)
                self._model_position(session, session.tokens)
                restored = True
                return result
            except BaseException as error:
                primary = error
                if not restored:
                    try:
                        session.model.restore(base)
                        self._model_position(session, session.tokens)
                        restored = True
                    except BaseException as recovery:
                        session.closed = True
                        add_failure_note(error, 'scoring rollback failed; session quarantined: ' + failure_summary(recovery))
                raise
            finally:
                cleanup = None
                for snapshot in (prompt_snapshot, base):
                    if snapshot is None:
                        continue
                    try:
                        self._release_transaction_snapshot(session, snapshot, primary or cleanup)
                    except BaseException as error:
                        if cleanup is None:
                            cleanup = error
                        else:
                            add_failure_note(cleanup, 'additional scoring snapshot cleanup failed: ' + failure_summary(error))
                if primary is None and cleanup is not None:
                    raise cleanup

    def score_continuation_groups(self, session_id, prompt_groups, continuation_groups, *,
                                  normalizations=None, cancel=None):
        """Score multiple closed sets while sharing one immutable prompt prefix."""
        session = self._session(session_id)
        if (type(prompt_groups) not in (tuple, list) or not prompt_groups or
                len(prompt_groups) > 64 or type(continuation_groups) not in (tuple, list) or
                len(continuation_groups) != len(prompt_groups)):
            raise ValueError('bounded matching prompt and continuation groups required')
        if normalizations is None:
            normalizations = ('mean',) * len(prompt_groups)
        if (type(normalizations) not in (tuple, list) or
                len(normalizations) != len(prompt_groups) or
                any(value not in ('mean', 'sum') for value in normalizations)):
            raise ValueError('one mean/sum normalization required per scoring group')
        prompts = [self._tokens(prompt, self.limits.context) for prompt in prompt_groups]
        candidates = []
        total_read = 0
        for group in continuation_groups:
            if type(group) not in (tuple, list) or not 2 <= len(group) <= 255:
                raise ValueError('each closed set requires 2 to 255 candidates')
            admitted_group = []
            for candidate in group:
                if (type(candidate) not in (tuple, list) or not candidate or
                        len(candidate) > self.limits.max_new_tokens):
                    raise ValueError('each scoring candidate must be a bounded nonempty token sequence')
                tokens = tuple(candidate)
                if any(type(token) is not int or not 0 <= token < self.limits.vocabulary
                       for token in tokens):
                    raise ValueError('scoring candidate contains an invalid token ID')
                admitted_group.append(tokens)
                total_read += len(tokens)
            candidates.append(admitted_group)
        if total_read > 65536:
            raise MemoryError('multi-question scoring exceeds the bounded option/token budget')
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('cancellation must provide is_set')

        common = list(prompts[0])
        for prompt in prompts[1:]:
            limit = min(len(common), len(prompt))
            index = 0
            while index < limit and common[index] == prompt[index]:
                index += 1
            del common[index:]
        if not common:
            raise ValueError('scoring prompt groups require a nonempty shared token prefix')
        if sum(len(prompt) - len(common) for prompt in prompts) > 65536:
            raise MemoryError('multi-question prompt suffix token budget exceeded')

        with self._generation_slot(session, cancel), session.transaction():
            cancellation = _SessionCancellation(session, cancel)
            if any(len(session.tokens) + len(prompt) + max(map(len, group)) > self.limits.context
                   for prompt, group in zip(prompts, candidates)):
                raise ValueError('scoring prompt and candidate exceed session context')
            base = self._capture_snapshot(session, session.tokens)
            shared_snapshot = None
            primary = None
            restored = False
            try:
                common_logits = None
                for offset in range(0, len(common), self.limits.admitted_prefill_tokens):
                    chunk = common[offset:offset + self.limits.admitted_prefill_tokens]
                    common_logits = self._forward_model(session, list(chunk), cancellation)
                common_last = np.array(common_logits[-1], copy=True)
                shared_snapshot = self._capture_snapshot(session, session.tokens + tuple(common))
                results = []
                for prompt, group, normalize in zip(prompts, candidates, normalizations):
                    self._cancelled(cancellation)
                    session.model.restore(shared_snapshot)
                    prefix_history = session.tokens + tuple(common)
                    self._model_position(session, prefix_history)
                    suffix = prompt[len(common):]
                    prompt_logits = common_last
                    for offset in range(0, len(suffix), self.limits.admitted_prefill_tokens):
                        chunk = suffix[offset:offset + self.limits.admitted_prefill_tokens]
                        prompt_logits = self._forward_model(session, list(chunk), cancellation)[-1]
                    prompt_logits = np.array(prompt_logits, copy=True)
                    prompt_snapshot = self._capture_snapshot(session, session.tokens + prompt)
                    group_primary = None
                    try:
                        raw_scores = []
                        for candidate in group:
                            self._cancelled(cancellation)
                            session.model.restore(prompt_snapshot)
                            self._model_position(session, session.tokens + prompt)
                            row_logits = prompt_logits
                            terms = []
                            for index, token in enumerate(candidate):
                                self._cancelled(cancellation)
                                row = np.asarray(row_logits, dtype=np.float64)
                                if row.shape != (self.limits.vocabulary,) or not np.isfinite(row).all():
                                    raise ValueError('executor returned invalid scoring logits')
                                log_normalizer = float(np.logaddexp.reduce(row))
                                if not math.isfinite(log_normalizer):
                                    raise ValueError('executor returned invalid scoring normalization')
                                terms.append(min(0.0, float(row[token]) - log_normalizer))
                                if index + 1 < len(candidate):
                                    row_logits = self._forward_model(session, [token], cancellation)[-1]
                            total = math.fsum(terms)
                            raw_scores.append(total / len(terms) if normalize == 'mean' else total)
                        score_array = np.asarray(raw_scores, dtype=np.float64)
                        if not np.isfinite(score_array).all():
                            raise ValueError('closed-set scores are nonfinite')
                        weights = np.exp(score_array - score_array.max())
                        denominator = float(weights.sum())
                        if not math.isfinite(denominator) or denominator <= 0:
                            raise ValueError('closed-set probability normalization failed')
                        probabilities = weights / denominator
                        entropy = -math.fsum(float(p) * math.log(float(p))
                                             for p in probabilities if p > 0)
                        choices = tuple({
                            'candidate_index': index,
                            'logprob': raw_scores[index] if normalize == 'sum' else
                                raw_scores[index] * len(group[index]),
                            'mean_logprob': raw_scores[index] / len(group[index])
                                if normalize == 'sum' else raw_scores[index],
                            'normalized_score': raw_scores[index],
                            'tokens': len(group[index]),
                            'probability': float(probabilities[index]),
                        } for index in range(len(group)))
                        best = max(range(len(choices)), key=lambda index:
                                   (choices[index]['probability'], -index))
                        results.append({
                            'answer_index': best, 'normalize': normalize,
                            'entropy': entropy, 'choices': choices,
                            'usage': {'prompt_tokens': len(prompt), 'completion_tokens': 0,
                                      'read_tokens': sum(map(len, group))},
                        })
                    except BaseException as error:
                        group_primary = error
                        raise
                    finally:
                        self._release_transaction_snapshot(session, prompt_snapshot, group_primary)

                self._cancelled(cancellation)
                session.model.restore(base)
                self._model_position(session, session.tokens)
                restored = True
                return tuple(results)
            except BaseException as error:
                primary = error
                if not restored:
                    try:
                        session.model.restore(base)
                        self._model_position(session, session.tokens)
                        restored = True
                    except BaseException as recovery:
                        session.closed = True
                        add_failure_note(error, 'group scoring rollback failed; session quarantined: ' + failure_summary(recovery))
                raise
            finally:
                cleanup = None
                for snapshot in (shared_snapshot, base):
                    if snapshot is None:
                        continue
                    try:
                        self._release_transaction_snapshot(session, snapshot, primary or cleanup)
                    except BaseException as error:
                        if cleanup is None:
                            cleanup = error
                        else:
                            add_failure_note(cleanup, 'additional group scoring snapshot cleanup failed: ' + failure_summary(error))
                if primary is None and cleanup is not None:
                    raise cleanup

    @staticmethod
    def _release_snapshot(session, snapshot):
        try:
            release = getattr(session.model, 'release_snapshot', None)
            if callable(release):
                release(snapshot)
        except BaseException as error:
            session.cleanup_snapshots[id(snapshot)] = snapshot
            session.closed = True
            raise RuntimeError('snapshot cleanup failed; session quarantined') from error
        else:
            session.cleanup_snapshots.pop(id(snapshot), None)

    def _release_transaction_snapshot(self,session,snapshot,primary):
        try:self._release_snapshot(session,snapshot)
        except BaseException as cleanup:
            if primary is None:raise
            add_failure_note(primary, 'transaction snapshot cleanup retained; session quarantined: '+failure_summary(cleanup))

    def generate(self, session_id, prompt_ids, *, max_new_tokens=None,
                 stop_ids=(), stop_prefixes=(), cancel=None, on_event=None, sampling=None, media=None,
                 resume_boundary=None,checkpoint_sink=None,max_checkpoint_bytes=0,application_identity=''):
        from .sampling import SamplingPolicy, TokenSampler
        session = self._session(session_id)
        with self._generation_slot(session, cancel), session.transaction():
            count = self.limits.max_new_tokens if max_new_tokens is None else max_new_tokens
            if type(count) is not int or not 1 <= count <= self.limits.max_new_tokens:
                raise ValueError('invalid generation length')
            stop = frozenset(self._tokens(stop_ids, self.limits.vocabulary)) if stop_ids else frozenset()
            prefixes = _admit_stop_prefixes(stop_prefixes, self.limits.vocabulary)
            if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
                raise ValueError('cancellation must provide is_set')
            cancel = _SessionCancellation(session, cancel)
            if on_event is not None and not callable(on_event):
                raise ValueError('event handler must be callable')
            from .service_generation_checkpoint import ServiceGenerationBoundary,write_boundary
            _admit_checkpoint_publication(checkpoint_sink,max_checkpoint_bytes)
            if (type(application_identity) is not str or len(application_identity)>8192 or
                    '\x00' in application_identity or
                    ((checkpoint_sink is not None or resume_boundary is not None) and
                     not application_identity)):
                raise ValueError('bounded generation application identity required')
            if resume_boundary is None:
                incoming = self._tokens(prompt_ids, self.limits.context - len(session.tokens))
            else:
                if type(resume_boundary) is not ServiceGenerationBoundary or tuple(islice(iter(prompt_ids),1)) or media is not None:
                    raise ValueError('recovered generation resumes without prompt replay or media')
                resume_boundary.admit(self.limits,self.limits.max_checkpoint_buffer_bytes)
                if resume_boundary.identity!=session.identity or resume_boundary.tokens!=session.tokens:
                    raise ValueError('recovered generation differs from live session')
                self._model_position(session,session.tokens,required=True)
                sampling=resume_boundary.policy;stop=frozenset(resume_boundary.stops)
                # A recovered boundary owns the stop policy that governed its
                # next-token state.  Permit an omitted request policy, but do
                # not silently replace it with a conflicting caller policy.
                if prefixes and prefixes != resume_boundary.stop_prefixes:
                    raise ValueError('recovered generation stop-prefix policy differs')
                prefixes=resume_boundary.stop_prefixes;incoming=()
                if application_identity!=resume_boundary.application_identity:
                    raise ValueError('generation application identity differs')
                if resume_boundary.stopped:
                    self._cancelled(cancel)
                    return GenerationResult(session_id,(),len(session.tokens),'stop')
            if media is not None:
                if type(media) is not dict or not media:
                    raise ValueError('explicit nonempty processor media inventory required')
                if not callable(getattr(session.model,'forward_media',None)):
                    raise ValueError('session executor does not admit media inputs')
                # This service has no media-span planner: forwarding the same
                # processor tensors once per text chunk would duplicate media
                # and misalign placeholders/positions. Admit media only as one
                # complete, nonempty prompt operation until a span planner is
                # part of the public request contract.
                if not incoming:
                    raise ValueError('media execution requires a nonempty prompt in the same request')
                if len(incoming)>self.limits.admitted_prefill_tokens:
                    raise MemoryError('media prompt requires a single admitted prefill operation')
            if sampling is not None and type(sampling) is not SamplingPolicy:
                raise TypeError('SamplingPolicy required')
            if sampling is not None:
                if sampling.top_k > self.limits.vocabulary:
                    raise ValueError('top-k exceeds vocabulary')
                if self.limits.vocabulary*64 > self.limits.max_sampling_buffer_bytes:
                    raise MemoryError('sampling buffers exceed vocabulary admission')
            sampler = None if sampling is None else TokenSampler(sampling,
                max_buffer_bytes=self.limits.max_sampling_buffer_bytes,
                max_history_tokens=self.limits.context)
            if resume_boundary is not None and sampler is not None:sampler.restore(resume_boundary.sampler)
            def checkpoint(tokens,logits,stopped=False):
                if checkpoint_sink is None:return
                self._cancelled(cancel)
                boundary=ServiceGenerationBoundary.capture(session,tokens,logits[-1],sampler,stop,self.limits,
                    stopped,application_identity,prefixes)
                boundary.admit(self.limits,max_checkpoint_bytes)
                active=True;used=False;written=False;thread=threading.get_ident()
                def write(output):
                    nonlocal used,written
                    if not active or used or threading.get_ident()!=thread:
                        raise RuntimeError('generation checkpoint writer is expired or already used')
                    used=True
                    self._cancelled(cancel)
                    service=self
                    class CancelledWriter:
                        def write(self,value):
                            service._cancelled(cancel);return output.write(value)
                    size=write_boundary(session.model,boundary,CancelledWriter(),limits=self.limits,max_bytes=max_checkpoint_bytes)
                    written=True
                    return size
                try:checkpoint_sink(boundary,write)
                finally:active=False
                if not written:
                    raise RuntimeError('checkpoint sink did not consume the complete generation frame')
                self._cancelled(cancel)
            # Every emitted token is fed into the model, so continuation caches
            # and the committed token history describe the same prefix.
            if len(session.tokens) + len(incoming) + count > self.limits.context:
                raise ValueError('generation exceeds session context capacity')
            generated = []
            reason = 'length'
            snapshot = (self._capture_snapshot(session, session.tokens)
                        if session.identity is not None else session.model.snapshot())
            primary=None
            try:
                self._cancelled(cancel)
                if resume_boundary is not None:logits=resume_boundary.row()[None,:]
                prefill_chunk = self.limits.admitted_prefill_tokens
                for offset in range(0, len(incoming), prefill_chunk):
                    self._cancelled(cancel)
                    chunk = incoming[offset:offset + prefill_chunk]
                    if media is not None:
                        logits = self._logits(session.model.forward_media(list(chunk),media,cancel),len(chunk))
                        self._cancelled(cancel)
                    else:
                        logits = self._forward_model(session, list(chunk), cancel)
                    if on_event is not None:
                        on_event({'event': 'generation.prefill',
                                  'processed_tokens': offset + len(chunk),
                                  'total_tokens': len(incoming), 'committed': False})
                checkpoint(session.tokens+incoming,logits)
                for index in range(count):
                    self._cancelled(cancel)
                    token = (int(np.argmax(logits[-1])) if sampler is None else
                             sampler.sample(logits[-1], history=(session.tokens+incoming+tuple(generated)
                                 if sampler.needs_history else ())))
                    generated.append(token)
                    logits = self._forward_model(session, [token], cancel)
                    # API stop prefixes describe generated output, not the
                    # prompt/cache boundary. Matching against the full session
                    # history lets a prompt ending in a stop-prefix fragment
                    # truncate the first generated token unexpectedly.
                    stopped = token in stop or _matches_stop_prefix(tuple(generated), prefixes)
                    checkpoint(session.tokens+incoming+tuple(generated),logits,stopped)
                    if on_event is not None:
                        on_event({'event': 'generation.token', 'index': index,
                                  'token_id': token, 'committed': False})
                    self._cancelled(cancel)
                    if stopped:
                        reason = 'stop'
                        break
                committed = session.tokens + incoming + tuple(generated)
                self._history(committed)
                self._model_position(session, committed)
                # Keep this final cancellation boundary inside the rollback
                # protected block.  A cancellation raised from the ``else``
                # clause would bypass the exception handler and leave the
                # physical executor advanced while logical history stayed old.
                self._cancelled(cancel)
                result = GenerationResult(session_id, tuple(generated), len(committed), reason)
            except BaseException as error:
                primary=error
                try:
                    session.model.restore(snapshot)
                    self._model_position(session, session.tokens)
                except BaseException as recovery:
                    session.closed = True
                    add_failure_note(error, 'executor rollback failed; session quarantined: '+failure_summary(recovery))
                raise
            else:
                session.tokens = committed
            finally:
                self._release_transaction_snapshot(session,snapshot,primary)
            return result

    def restore_generation_checkpoint(self,session_id,source,*,max_bytes,cancel=None,application_identity=''):
        """Restore full native continuation and its sampler boundary atomically."""
        from .service_generation_checkpoint import read_boundary
        session=self._session(session_id)
        with self._generation_slot(session,cancel),session.transaction():
            cancel=_SessionCancellation(session,cancel);self._cancelled(cancel)
            reader=getattr(session.model,'restore_cache',None)
            if not callable(reader):raise ValueError('executor has no complete generation cache transport')
            service=self
            class CancelledReader:
                def read(self,count):
                    service._cancelled(cancel);return source.read(count)
            boundary,cache,size=read_boundary(CancelledReader(),limits=self.limits,max_bytes=max_bytes)
            rollback=None;primary=None;previous=session.tokens;cache_closed=False
            try:
                if boundary.identity!=session.identity or boundary.application_identity!=application_identity:
                    raise ValueError('generation checkpoint execution or application identity differs')
                self._history(boundary.tokens);self._cancelled(cancel)
                rollback=self._capture_snapshot(session,previous)
                class CancelledCache:
                    def read(self,count):
                        service._cancelled(cancel);return cache.read(count)
                reader(CancelledCache(),max_bytes=size)
                if cache.tell()!=size:raise ValueError('generation cache reader did not consume the complete staged frame')
                self._model_position(session,boundary.tokens,required=True);self._cancelled(cancel)
                # Close staging before publishing service history.  A close
                # failure is still a failed restore and must take the rollback
                # path rather than leaving a committed state behind an error.
                cache.close();cache_closed=True
                session.tokens=boundary.tokens
                return boundary
            except BaseException as error:
                primary=error
                if rollback is not None:
                    try:
                        session.model.restore(rollback);self._model_position(session,previous)
                        session.tokens=previous
                    except BaseException as recovery:
                        session.closed=True
                        add_failure_note(error, 'generation recovery rollback failed; session quarantined: '+failure_summary(recovery))
                raise
            finally:
                failed=primary is not None
                # Do not let temporary-frame cleanup replace the actual
                # restore/identity/cancellation failure.  Still attempt both
                # cleanup actions, since either owner may retain physical
                # state after a failed close.
                if not cache_closed:
                    try:
                        cache.close()
                    except BaseException as cleanup:
                        if primary is None:
                            primary=cleanup
                        else:
                            add_failure_note(primary, 'generation recovery staged-cache cleanup failed: '+failure_summary(cleanup))
                if rollback is not None:
                    try:self._release_snapshot(session,rollback)
                    except BaseException as cleanup:
                        if primary is None:
                            primary=cleanup
                        else:
                            add_failure_note(primary, 'generation recovery receipt cleanup retained: '+failure_summary(cleanup))
                if primary is not None and not failed:
                    raise primary

    def resume_generation(self,session_id,boundary,*,max_new_tokens=None,stop_prefixes=(),cancel=None,on_event=None,
                          checkpoint_sink=None,max_checkpoint_bytes=0,application_identity=''):
        return self.generate(session_id,(),max_new_tokens=max_new_tokens,stop_prefixes=stop_prefixes,
            cancel=cancel,on_event=on_event,
            resume_boundary=boundary,checkpoint_sink=checkpoint_sink,max_checkpoint_bytes=max_checkpoint_bytes,
            application_identity=application_identity)

    def reset(self, session_id):
        session = self._session(session_id)
        with session.transaction():
            try:
                session.model.reset()
                self._model_position(session, ())
            except BaseException:
                # A failed reset can leave a partially cleared native cache.
                session.closed = True
                raise
            session.tokens = ()

    def write_session_cache(self, session_id, output, *, max_bytes):
        """Write service history plus a complete executor-owned cache frame.

        The caller owns atomic file publication. Executors without an explicit
        complete cache codec fail; this method never substitutes token replay.
        """
        import hashlib
        import json
        import struct
        from .segment_snapshot import _write
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive session cache byte budget required')
        session = self._session(session_id)
        with session.transaction():
            writer = getattr(session.model, 'write_cache', None)
            if not callable(writer):
                raise ValueError('executor has no complete cache transport')
            self._model_position(session, session.tokens, required=True)
            identity = getattr(session.model, 'identity', None)
            if type(identity) is not str or not identity:
                raise ValueError('executor has no cache execution identity')
            header = json.dumps(dict(version=1, identity=identity,
                vocabulary=self.limits.vocabulary, context=self.limits.context,
                tokens=session.tokens), sort_keys=True, separators=(',', ':')).encode('ascii')
            overhead = 44+len(header)
            if overhead+44 > max_bytes or len(header)>self.limits.context*12+8192:
                raise MemoryError('session cache metadata exceeds admission')
            prefix = b'OASVKV01'+struct.pack('<I',len(header))
            _write(output,prefix);_write(output,header)
            _write(output,hashlib.sha256(prefix+header).digest())
            size = writer(output,max_bytes=max_bytes-overhead)
            if type(size) is not int or not 0<size<=max_bytes-overhead:
                raise ValueError('executor cache writer returned invalid extent')
            self._model_position(session, session.tokens, required=True)
            session.identity = identity
            return overhead+size

    def restore_session_cache(self, session_id, source, *, max_bytes,history_receipt=None):
        """Restore native state and service history as one session transaction."""
        import hashlib
        import hmac
        import json
        import struct
        from .segment_snapshot import _read, _pairs
        if type(max_bytes) is not int or max_bytes < 88:
            raise ValueError('positive session cache byte budget required')
        session = self._session(session_id)
        with self._generation_slot(session, None), session.transaction():
            reader = getattr(session.model, 'restore_cache', None)
            if not callable(reader):
                raise ValueError('executor has no complete cache transport')
            prefix = _read(source,12)
            if prefix[:8]!=b'OASVKV01':
                raise ValueError('invalid service cache magic')
            length = struct.unpack('<I',prefix[8:])[0]
            if not 0<length<=min(self.limits.context*12+8192,max_bytes-88):
                raise MemoryError('session cache metadata exceeds admission')
            encoded = _read(source,length)
            if not hmac.compare_digest(bytes(_read(source,32)),hashlib.sha256(prefix+encoded).digest()):
                raise ValueError('session cache metadata checksum differs')
            header = json.loads(encoded,object_pairs_hook=_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if type(header) is not dict or set(header)!={'version','identity','vocabulary','context','tokens'}:
                raise ValueError('invalid session cache metadata')
            if (type(header['version']) is not int or header['version']!=1 or
                    type(header['vocabulary']) is not int or header['vocabulary']!=self.limits.vocabulary or
                    type(header['context']) is not int or header['context']!=self.limits.context or
                    type(header['tokens']) is not list):
                raise ValueError('session cache geometry differs')
            tokens = self._history(tuple(header['tokens']))
            if history_receipt is not None:
                expected_count,expected_digest=history_receipt
                actual=hashlib.sha256(json.dumps(list(tokens),separators=(',',':')).encode()).hexdigest()
                if (type(expected_count) is not int or expected_count!=len(tokens) or
                        type(expected_digest) is not str or not hmac.compare_digest(actual,expected_digest)):
                    raise ValueError('continuation history differs from durable catalog')
            if header['identity']!=session.identity:
                raise ValueError('session cache identity differs')
            rollback = self._capture_snapshot(session,session.tokens);primary=None
            try:
                reader(source,max_bytes=max_bytes-44-length)
                # The outer file helper performs the same check, but direct
                # service callers must also reject a valid frame followed by
                # unowned bytes before publishing the new history.
                trailing = source.read(1)
                if not isinstance(trailing, bytes):
                    raise ValueError('session cache EOF probe returned invalid bytes')
                if trailing:
                    raise ValueError('session cache contains trailing bytes')
                self._model_position(session,tokens,required=True)
                # Stable file-backed readers can perform a final descriptor /
                # path identity check. Keep that check inside this transaction:
                # the outer file adapter may repeat it after return, but a
                # failure there must not report restore failure after history
                # and physical cache have already committed.
                require_complete = getattr(source, 'require_complete', None)
                if callable(require_complete):
                    require_complete()
            except BaseException as error:
                primary=error
                try:
                    session.model.restore(rollback)
                    self._model_position(session,session.tokens)
                except BaseException as recovery:
                    session.closed = True
                    add_failure_note(error, 'cache recovery rollback failed; session quarantined: '+failure_summary(recovery))
                raise
            else:
                session.tokens = tokens
            finally:
                self._release_transaction_snapshot(session,rollback,primary)
            return len(tokens)

    def save_session_cache(self, session_id, destination, *, max_bytes):
        from .cache_file_recovery import save_service_cache
        return save_service_cache(self, session_id, destination, max_bytes=max_bytes)

    def load_session_cache(self, session_id, source, *, max_bytes,history_receipt=None,
                           expected_file_receipt=None):
        from .cache_file_recovery import load_service_cache
        return load_service_cache(self, session_id, source, max_bytes=max_bytes,
                                  history_receipt=history_receipt,
                                  expected_file_receipt=expected_file_receipt)

    def recover_cached_session(self, source, *, max_bytes, expected_file_receipt=None):
        session_id = self.open_session()
        try:
            self.load_session_cache(session_id, source, max_bytes=max_bytes,
                                    expected_file_receipt=expected_file_receipt)
            return session_id
        except BaseException as error:
            try:
                self.close_session(session_id)
            except BaseException as cleanup:
                add_failure_note(error, 'cache recovery cleanup pending for '+session_id+': '+failure_summary(cleanup))
            raise

    def evict_cached_session(self, session_id, destination, *, max_bytes):
        """Publish and read back complete state before closing an idle model."""
        from .cache_file_recovery import verify_service_cache_file
        session = self._session(session_id)
        if not session.lock.acquire(blocking=False):
            raise RuntimeError('cannot evict a busy execution session')
        try:
            if session.active or session.closed or session.shutdown_requested.is_set():
                raise RuntimeError('cannot evict an active or quarantined session')
            receipt = self.save_session_cache(session_id,destination,max_bytes=max_bytes)
            verify_service_cache_file(destination,receipt=receipt,max_bytes=max_bytes)
            receipt.update(session_id=session_id,identity=session.identity,
                           context_tokens=len(session.tokens),cache_bytes_preserved=True)
            try:
                self.close_session(session_id)
            except BaseException as error:
                error.add_note('durable complete-cache checkpoint retained at '+receipt['path'])
                raise
            return receipt
        finally:
            session.lock.release()

    def export_replay_checkpoint(self, session_id):
        """Capture model identity and committed tokens, not native cache bytes."""
        from .session_recovery import ReplayCheckpoint
        session = self._session(session_id)
        with session.transaction():
            snapshot = self._capture_snapshot(session, session.tokens)
            try:
                return ReplayCheckpoint(session.identity, self.limits.vocabulary,
                    self.limits.context, session.tokens).validate(max_tokens=self.limits.context)
            finally:
                self._release_snapshot(session, snapshot)

    def restore_replay_checkpoint(self, session_id, checkpoint, *, cancel=None, on_event=None):
        """Rebuild cache from tokens; failures restore the prior session snapshot.

        Replay uses configured prefill chunks and does not promise bit-identical
        cache state relative to a different original chunking schedule.
        """
        from .session_recovery import ReplayCheckpoint
        if type(checkpoint) is not ReplayCheckpoint:
            raise TypeError('ReplayCheckpoint required')
        checkpoint.validate(max_tokens=self.limits.context)
        if (checkpoint.vocabulary != self.limits.vocabulary or
                checkpoint.context != self.limits.context):
            raise ValueError('replay checkpoint model geometry mismatch')
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('cancellation must provide is_set')
        if on_event is not None and not callable(on_event):
            raise ValueError('event handler must be callable')
        session = self._session(session_id)
        with self._generation_slot(session, cancel), session.transaction():
            cancel = _SessionCancellation(session, cancel)
            rollback = self._capture_snapshot(session, session.tokens);primary=None
            try:
                if checkpoint.identity != session.identity:
                    raise ValueError('replay checkpoint model identity mismatch')
                self._cancelled(cancel)
                session.model.reset()
                self._model_position(session, ())
                chunk = self.limits.admitted_prefill_tokens
                for offset in range(0, len(checkpoint.token_ids), chunk):
                    self._cancelled(cancel)
                    incoming = checkpoint.token_ids[offset:offset+chunk]
                    self._forward_model(session, list(incoming), cancel)
                    if on_event is not None:
                        on_event(dict(event='recovery.prefill', processed_tokens=offset+len(incoming),
                                      total_tokens=len(checkpoint.token_ids), committed=False))
                self._cancelled(cancel)
                verification = self._capture_snapshot(session, checkpoint.token_ids)
                self._release_snapshot(session, verification)
            except BaseException as error:
                primary=error
                try:
                    session.model.restore(rollback)
                    self._model_position(session, session.tokens)
                except BaseException as recovery:
                    session.closed = True
                    add_failure_note(error, 'replay rollback failed; session quarantined: '+failure_summary(recovery))
                raise
            else:
                session.tokens = checkpoint.token_ids
            finally:
                self._release_transaction_snapshot(session,rollback,primary)
            return len(session.tokens)

    def evict_session(self, session_id, destination):
        """Persist committed token history before releasing an idle executor.

        Recovery replays tokens, not native cache bytes. A failed write or
        readback leaves the live session registered. A cleanup failure leaves
        the durable checkpoint available and the executor quarantined.
        """
        from pathlib import Path
        from .session_recovery import save_replay_checkpoint, load_replay_checkpoint
        session = self._session(session_id)
        if not session.lock.acquire(blocking=False):
            raise RuntimeError('cannot evict a busy execution session')
        try:
            if session.active or session.closed or session.shutdown_requested.is_set():
                raise RuntimeError('cannot evict an active or quarantined session')
            checkpoint = self.export_replay_checkpoint(session_id)
            save_replay_checkpoint(checkpoint, destination, max_tokens=self.limits.context)
            observed = load_replay_checkpoint(destination, max_tokens=self.limits.context)
            if observed != checkpoint:
                raise ValueError('eviction checkpoint changed during readback')
            receipt = {'session_id': session_id, 'identity': checkpoint.identity,
                       'checkpoint': str(Path(destination).absolute()),
                       'context_tokens': len(checkpoint.token_ids),
                       'recovery': 'token-replay', 'cache_bytes_preserved': False}
            try:
                self.close_session(session_id)
            except BaseException as error:
                error.add_note('durable eviction checkpoint retained at '+receipt['checkpoint'])
                raise
            return receipt
        finally:
            session.lock.release()

    def open_recovered_session(self, source, *, cancel=None, on_event=None):
        """Open and replay a durable session, retaining cleanup failures."""
        from .session_recovery import load_replay_checkpoint
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('cancellation must provide is_set')
        self._cancelled(cancel)
        checkpoint = load_replay_checkpoint(source, max_tokens=self.limits.context)
        if (checkpoint.vocabulary != self.limits.vocabulary or
                checkpoint.context != self.limits.context):
            raise ValueError('recovery checkpoint model geometry mismatch')
        self._cancelled(cancel)
        session_id = self.open_session()
        try:
            self.restore_replay_checkpoint(session_id, checkpoint, cancel=cancel, on_event=on_event)
            return session_id
        except BaseException as error:
            try:
                self.close_session(session_id)
            except BaseException as cleanup:
                add_failure_note(error, 'recovery executor cleanup pending for '+session_id+': '+failure_summary(cleanup))
            raise

    def close_session(self, session_id):
        session = self._session(session_id)
        session.shutdown_requested.set()
        with session.lock:
            if session.active:
                raise RuntimeError('cannot close an active session')
            session.active = True
            session.closed = True
            try:
                failures = []
                for pending in tuple(session.cleanup_snapshots.values()):
                    try:
                        self._release_snapshot(session, pending)
                    except BaseException as error:
                        failures.append(error)
                while session.snapshots:
                    _, owned = session.snapshots.popitem()
                    try:
                        self._release_snapshot(session, owned[1])
                    except BaseException as error:
                        failures.append(error)
                if (not session.cleanup_snapshots and not session.released and
                        callable(getattr(session.model, 'close', None))):
                    try:
                        session.model.close()
                    except BaseException as error:
                        failures.append(error)
                    else:
                        session.released = True
                if failures:
                    raise RuntimeError('execution session cleanup failed; session quarantined') from failures[0]
            finally:
                session.active = False
            session.released = True
            session.closed = True
            with self._lock:
                self._sessions.pop(session_id, None)

    def history(self, session_id):
        session = self._session(session_id)
        with session.lock:
            if session.closed or session.shutdown_requested.is_set():
                raise RuntimeError('execution session is closed')
            return session.tokens

    def close(self):
        with self._cleanup_lock:
            if self._closing:
                raise RuntimeError('reentrant execution service cleanup')
            self._closing = True
            try:
                self._close()
            finally:
                self._closing = False

    def _close(self):
        """Stop admitting sessions and release idle executors without deadlocks.

        Running sessions are not interrupted inside native code. They remain
        registered for cancellation and a later close attempt. A close racing
        with factory creation prevents publication of the new executor.
        """
        with self._lock:
            self._closed = True
            sessions = tuple(self._sessions.items())
            opening = self._opening
            orphaned = tuple(self._orphaned.items())
        self._generation_gate.close()
        failures = []
        for session_id, session in sessions:
            session.shutdown_requested.set()
            if not session.lock.acquire(blocking=False):
                failures.append((session_id, 'request is active'))
                continue
            try:
                if session.active:
                    failures.append((session_id, 'request is active'))
                    continue
                try:
                    self.close_session(session_id)
                except ValueError:
                    # Another close may already have removed this session.
                    with self._lock:
                        if session_id in self._sessions:
                            failures.append((session_id, 'session lookup failed'))
                except Exception as error:
                    failures.append((session_id, type(error).__name__))
            finally:
                session.lock.release()
        if opening:
            failures.append(('factory', 'executor creation is finishing'))
        for key, model in orphaned:
            try:
                model.close()
            except Exception as error:
                failures.append(('orphaned-'+str(key), type(error).__name__))
            else:
                with self._lock:
                    self._orphaned.pop(key, None)
        if not failures and self._factory_cleanup is not None:
            try:
                self._factory_cleanup()
            except Exception as error:
                failures.append(('factory-cleanup', type(error).__name__))
        if failures:
            raise RuntimeError('execution service cleanup is incomplete: ' + repr(failures))
