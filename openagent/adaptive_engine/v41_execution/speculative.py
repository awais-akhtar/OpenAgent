"""Transactional speculative sampling against a stateful target decoder."""
import copy
import sys
from contextlib import contextmanager,ExitStack
import threading
from dataclasses import dataclass

import numpy as np


def _stop_prefixes(value, vocabulary):
    if value is None:
        return ()
    if type(value) not in (tuple, list) or len(value) > 64:
        raise ValueError('bounded stop-prefix inventory required')
    result = []
    for prefix in value:
        if (type(prefix) not in (tuple, list) or not prefix or len(prefix) > 128 or
                any(type(token) is not int or not 0 <= token < vocabulary
                    for token in prefix)):
            raise ValueError('finite valid stop-token prefixes required')
        result.append(tuple(prefix))
    if len(set(result)) != len(result):
        raise ValueError('duplicate stop-token prefix')
    return tuple(result)


def _matches_stop_prefix(history, prefixes):
    return any(len(history) >= len(prefix) and
               tuple(history[-len(prefix):]) == prefix for prefix in prefixes)


@dataclass(frozen=True)
class DraftProposal:
    tokens: tuple
    probabilities: object


@dataclass(frozen=True)
class SpeculativeStep:
    tokens: tuple
    accepted: int
    proposed: int
    rejected: bool


@dataclass(frozen=True)
class SpeculativeSnapshot:
    target: object
    draft: object
    history: tuple
    logits: bytes
    rng_state: dict
    policy: tuple
    proposal_temperature: object = None
    stopped: bool = False
    proposal_rng_state: object = None
    stop_prefixes: tuple = ()


class AutoregressiveDraftProposer:
    """Use a separate stateful decoder as a conditional proposal model."""

    def __init__(self, model, *, vocabulary, context, temperature=1.0,
                 max_bytes=64*1024**2, rng=None):
        for name in ('forward', 'snapshot', 'restore', 'reset'):
            if not callable(getattr(model, name, None)):
                raise ValueError('draft decoder must implement '+name)
        if (type(vocabulary) is not int or vocabulary <= 0 or
                type(context) is not int or context <= 0 or
                type(max_bytes) is not int or max_bytes <= 0 or
                not np.isfinite(temperature) or temperature < 0):
            raise ValueError('invalid draft geometry, budget or temperature')
        self.model, self.vocabulary, self.context = model, vocabulary, context
        self.temperature, self.max_bytes = float(temperature), max_bytes
        self.rng = np.random.default_rng() if rng is None else rng
        self.history, self.logits = (), None
        self._lock = threading.RLock()
        self._pending_release = []

        self._failed=False
        self._closed=False

    def _check_open(self):
        if self._closed:
            raise RuntimeError('draft proposer is closed')

    def release_snapshot(self, state):
        release = getattr(self.model, 'release_snapshot', None)
        if callable(release):
            release(state[0])

    def retry_cleanup(self):
        with self._lock:
            self._check_open()
            pending,self._pending_release = self._pending_release,[]
            errors = []
            for state in pending:
                try:
                    self.release_snapshot(state)
                except BaseException as error:
                    self._pending_release.append(state)
                    errors.append(error)
            if errors:
                raise RuntimeError('draft snapshot cleanup remains pending') from errors[0]

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.retry_cleanup()
            close = getattr(self.model, 'close', None)
            if callable(close):
                close()
            self._closed = True

    def snapshot(self):
        with self._lock:
            self._check_open()
            if self._failed:raise RuntimeError('draft owner requires reset or successful restore')
            if self._pending_release:raise RuntimeError('draft snapshot cleanup remains pending')
            if 16*self.vocabulary+128*len(self.history)>self.max_bytes:
                raise MemoryError('complete draft snapshot exceeds admission')
            state=self.model.snapshot()
            try:
                return (state,self.history,None if self.logits is None else self.logits.copy(),
                    copy.deepcopy(self.rng.bit_generator.state))
            except BaseException as primary:
                try:self.release_snapshot((state,))
                except BaseException as cleanup:
                    self._pending_release.append((state,));primary.add_note('draft snapshot cleanup retained: '+repr(cleanup))
                raise

    def reset(self):
        with self._lock:
            self._check_open()
            self.retry_cleanup()
            self.model.reset()
            self.history, self.logits = (), None
            self._failed=False

    def restore(self, state):
        with self._lock:
            self._check_open()
            model_state, history, logits, rng_state = state
            if (type(history) is not tuple or len(history)>self.context or
                    any(type(token) is not int or not 0<=token<self.vocabulary for token in history) or
                    (logits is None)!= (not history)):
                raise ValueError('complete draft history and next logits required')
            if logits is not None and (not isinstance(logits,np.ndarray) or
                    logits.shape!=(self.vocabulary,) or logits.dtype!=np.float64 or not np.isfinite(logits).all()):
                raise ValueError('finite full-vocabulary draft logits required')
            if 16*self.vocabulary+128*len(history)>self.max_bytes:
                raise MemoryError('restored draft boundary exceeds admission')
            pending_logits=None if logits is None else logits.copy()
            pending_rng=copy.deepcopy(rng_state)
            old_rng = copy.deepcopy(self.rng.bit_generator.state)
            try:
                self.rng.bit_generator.state = pending_rng
                self.model.restore(model_state)
            except BaseException as primary:
                self._failed=True
                try:self.rng.bit_generator.state = old_rng
                except BaseException as rollback:
                    primary.add_note('draft RNG rollback failed; owner quarantined: '+repr(rollback))
                raise
            self.history = history
            self.logits = pending_logits
            self._failed=False

    def _forward(self, tokens):
        result = np.asarray(self.model.forward(list(tokens)), dtype=np.float64)
        if result.shape != (len(tokens), self.vocabulary) or not np.isfinite(result).all():
            raise ValueError('draft model returned invalid logits')
        self.history += tuple(tokens)
        self.logits = result[-1].copy()

    def __call__(self, history, count):
        with self._lock:
            self._check_open()
            if self._failed:raise RuntimeError('draft owner requires reset or successful restore')
            if self._pending_release:
                raise RuntimeError('draft snapshot cleanup pending')
            if (type(history) is not tuple or not history or
                    any(type(t) is not int or not 0 <= t < self.vocabulary for t in history) or
                    type(count) is not int or count <= 0 or len(history)+count > self.context):
                raise ValueError('valid draft history and available context required')
            if 8*self.vocabulary*(count+len(history)+3) > self.max_bytes:
                raise MemoryError('draft logits and distribution workspace exceeds budget')
            saved = self.snapshot()
            primary=None
            try:
                if history[:len(self.history)] != self.history:
                    self.model.reset()
                    self.history, self.logits = (), None
                if len(self.history) < len(history):
                    self._forward(history[len(self.history):])
                tokens, probabilities = [], []
                for _ in range(count):
                    if self.temperature == 0:
                        q = np.zeros(self.vocabulary, dtype=np.float64)
                        q[int(np.argmax(self.logits))] = 1
                    else:
                        q = np.exp((self.logits-self.logits.max())/self.temperature)
                        q /= q.sum()
                    token = int(self.rng.choice(self.vocabulary, p=q))
                    tokens.append(token)
                    probabilities.append(q)
                    self._forward((token,))
                return DraftProposal(tuple(tokens), np.stack(probabilities))
            except BaseException as failure:
                primary=failure
                try:self.restore(saved)
                except BaseException as rollback:
                    self._failed=True;primary.add_note('draft proposal rollback failed: '+repr(rollback))
                raise
            finally:
                try:
                    self.release_snapshot(saved)
                except BaseException as cleanup:
                    self._pending_release.append(saved)
                    if primary is None:raise
                    primary.add_note('draft proposal snapshot cleanup retained: '+repr(cleanup))


class SpeculativeDecoder:
    """Keep target sampling authoritative; never accept unverified draft tokens.

    The proposer supplies conditional distributions for each proposed token.
    A proposer reconciles its own cache with the supplied committed history.
    Snapshot-capable proposers are rolled back when a transaction fails.
    """

    @staticmethod
    def _cancelled(cancelled):
        if cancelled is not None and cancelled():
            from ..execution_service import ExecutionCancelled
            raise ExecutionCancelled('speculative transaction cancelled')

    @contextmanager
    def _cancel_scope(self,cancelled):
        if cancelled is not None and not callable(cancelled):raise ValueError('callable speculation cancellation required')
        class Event:
            def is_set(self):return cancelled is not None and bool(cancelled())
        with ExitStack() as scopes:
            seen=set()
            if cancelled is not None:
                for owner in (self.target,self.proposer):
                    if id(owner) in seen:continue
                    seen.add(id(owner));scope=getattr(owner,'cancellation_scope',None)
                    if callable(scope):scopes.enter_context(scope(Event()))
            yield

    def __init__(self, target, proposer, *, vocabulary, context, temperature=1.0,
                 draft_length=5, max_bytes=64*1024**2, rng=None):
        for name in ('forward', 'snapshot', 'restore', 'reset'):
            if not callable(getattr(target, name, None)):
                raise ValueError('target must implement '+name)
        if not callable(proposer):
            raise ValueError('conditional draft proposer required')
        for value in (vocabulary, context, draft_length, max_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError('positive integer geometry and budget required')
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError('finite nonnegative temperature required')
        if 8*vocabulary*(3*draft_length+4) > max_bytes:
            raise MemoryError('speculative distribution workspace exceeds budget')
        has_snapshot = callable(getattr(proposer, 'snapshot', None))
        has_restore = callable(getattr(proposer, 'restore', None))
        if not has_snapshot or not has_restore:
            raise ValueError(
                'transactional speculative execution requires a snapshot-capable '
                'proposer with paired restore; unsnapshotted proposal state cannot '
                'be rolled back after rejection, cancellation, or target failure')
        self.target, self.proposer = target, proposer
        self.vocabulary, self.context = vocabulary, context
        self.temperature, self.draft_length = float(temperature), draft_length
        self.max_bytes = max_bytes
        self.rng = np.random.default_rng() if rng is None else rng
        self.history, self.next_logits = (), None
        self._lock = threading.RLock()
        self._busy = False
        self._quarantined = False
        self._pending_release = []
        self._stopped = False
        self._stop_prefixes = ()
        self._closed = False
        self._closing = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError('speculative decoder is closed')
        if self._closing:
            raise RuntimeError('speculative decoder is closing')

    def _release_states(self, target=None, draft=None, *, primary=None):
        errors = []
        for owner,state in ((self.target,target),(self.proposer,draft)):
            release = getattr(owner,'release_snapshot',None)
            if state is None or not callable(release):
                continue
            try:
                release(state)
            except BaseException as failure:
                self._pending_release.append((owner,state))
                errors.append(failure)
        if errors:
            self._quarantined = True
            if primary is not None:
                for error in errors:primary.add_note('speculative snapshot cleanup retained: '+repr(error))
                return
            raise RuntimeError('speculative snapshot cleanup retained for retry') from errors[0]

    def retry_cleanup(self):
        with self._lock:
            self._check_open()
            if self._busy:
                raise RuntimeError('cannot retry active speculative cleanup')
            pending,self._pending_release = self._pending_release,[]
            errors = []
            for owner,state in pending:
                try:
                    owner.release_snapshot(state)
                except BaseException as failure:
                    self._pending_release.append((owner,state))
                    errors.append(failure)
            if errors:
                raise RuntimeError('speculative snapshot cleanup remains pending') from errors[0]
            retry = getattr(self.proposer,'retry_cleanup',None)
            if callable(retry):
                retry()

    def close(self):
        """Close target and proposer resources with retryable failure semantics."""
        with self._lock:
            if self._closed:
                return
            if self._busy:
                raise RuntimeError('cannot close active speculative generation')
            errors = []
            try:
                try:
                    self.retry_cleanup()
                except BaseException as error:
                    errors.append(error)
                self._closing = True
                for owner in (self.proposer, self.target):
                    close = getattr(owner, 'close', None)
                    if callable(close):
                        try:
                            close()
                        except BaseException as error:
                            errors.append(error)
                if errors:
                    raise RuntimeError('speculative decoder cleanup remains pending') from errors[0]
                self._closed = True
            finally:
                self._closing = False

    def release_snapshot(self, snapshot):
        with self._lock:
            self._check_open()
            if type(snapshot) is not SpeculativeSnapshot or self._busy:
                raise ValueError('idle speculative snapshot required')
            self._release_states(snapshot.target,snapshot.draft)

    def write_cache(self, output, *, max_bytes):
        from .speculative_cache import transport_speculative_cache
        with self._lock:
            self._check_open()
        return transport_speculative_cache(self, output, max_bytes=max_bytes)

    def restore_cache(self, source, *, max_bytes):
        from .speculative_cache import transport_speculative_cache
        with self._lock:
            self._check_open()
        return transport_speculative_cache(self, source, max_bytes=max_bytes, restoring=True)

    def save_cache(self, destination, *, max_bytes):
        from ..cache_file_recovery import save_executor_cache
        with self._lock:
            self._check_open()
        return save_executor_cache(self, destination, max_bytes=max_bytes)

    def load_cache(self, source, *, max_bytes, cancelled=None):
        from ..cache_file_recovery import load_executor_cache
        with self._lock:
            self._check_open()
        return load_executor_cache(self, source, max_bytes=max_bytes, cancelled=cancelled)

    def _rollback(self, target, draft, rng_state, failure, proposal_rng_state=None):
        """Try every independent owner even if an earlier restore fails."""
        errors = []
        actions = [('target', lambda: self.target.restore(target)),
                   ('random generator', lambda: setattr(self.rng.bit_generator,
                                                        'state', copy.deepcopy(rng_state)))]
        if draft is not None:
            actions.append(('draft', lambda: self.proposer.restore(draft)))
        proposal_rng = getattr(self.proposer, 'rng', None)
        if proposal_rng is not None and hasattr(proposal_rng, 'bit_generator'):
            proposal_state = proposal_rng_state
            if proposal_state is not None:
                actions.append(('proposal random generator',
                                lambda: setattr(proposal_rng.bit_generator,
                                                'state', copy.deepcopy(proposal_state))))
        for name, action in actions:
            try:
                action()
            except BaseException as error:
                errors.append((name, error))
        if errors:
            self._quarantined = True
            for name, error in errors:
                failure.add_note('speculative '+name+' rollback failed: '+str(error))

    def snapshot(self):
        with self._lock:
            self._check_open()
            if self._busy or self._quarantined or self.next_logits is None:
                raise RuntimeError('idle prefilled generation required for snapshot')
            self._busy = True
            target,draft = None,None
            try:
                target = self.target.snapshot()
                draft = self.proposer.snapshot() if callable(getattr(self.proposer, 'snapshot', None)) else None
                proposal_rng = getattr(self.proposer, 'rng', None)
                proposal_rng_state = (copy.deepcopy(proposal_rng.bit_generator.state)
                    if proposal_rng is not None and hasattr(proposal_rng, 'bit_generator') else None)
                return SpeculativeSnapshot(target, draft,
                    self.history, self.next_logits.astype('<f8', copy=False).tobytes(),
                    copy.deepcopy(self.rng.bit_generator.state),
                    (self.vocabulary, self.context, self.temperature, self.draft_length),
                    getattr(self.proposer, 'temperature', None), self._stopped,
                    proposal_rng_state, self._stop_prefixes)
            except BaseException:
                self._release_states(target,draft,primary=sys.exc_info()[1])
                raise
            finally:
                self._busy = False

    def restore(self, snapshot):
        with self._lock:
            self._check_open()
            if self._busy or self._quarantined:
                raise RuntimeError('cannot restore active generation')
            if (type(snapshot) is not SpeculativeSnapshot or snapshot.policy !=
                    (self.vocabulary, self.context, self.temperature, self.draft_length) or
                    type(snapshot.history) is not tuple or not snapshot.history or
                    len(snapshot.history) > self.context or
                    any(type(t) is not int or not 0 <= t < self.vocabulary for t in snapshot.history) or
                    type(snapshot.logits) is not bytes or len(snapshot.logits) != 8*self.vocabulary):
                raise ValueError('generation snapshot policy or extent mismatch')
            if snapshot.proposal_temperature != getattr(self.proposer, 'temperature', None):
                raise ValueError('generation snapshot proposal temperature mismatch')
            if type(snapshot.stopped) is not bool:
                raise ValueError('explicit saved speculative stop boundary required')
            if (type(snapshot.stop_prefixes) is not tuple or len(snapshot.stop_prefixes) > 64 or
                    any(type(prefix) is not tuple or not prefix or len(prefix) > 128 or
                        any(type(token) is not int or not 0 <= token < self.vocabulary
                            for token in prefix) for prefix in snapshot.stop_prefixes) or
                    len(set(snapshot.stop_prefixes)) != len(snapshot.stop_prefixes)):
                raise ValueError('saved speculative stop-prefix policy differs')
            proposal_rng = getattr(self.proposer, 'rng', None)
            if snapshot.proposal_rng_state is not None:
                if proposal_rng is None or not hasattr(proposal_rng, 'bit_generator'):
                    raise ValueError('saved proposal RNG has no owned proposer RNG')
                expected = proposal_rng.bit_generator.state.get('bit_generator')
                if (type(snapshot.proposal_rng_state) is not dict or
                        snapshot.proposal_rng_state.get('bit_generator') != expected):
                    raise ValueError('saved proposal RNG identity differs from proposer')
            logits = np.frombuffer(snapshot.logits, dtype='<f8').copy()
            if not np.isfinite(logits).all():
                raise ValueError('invalid saved generation logits')
            has_draft = callable(getattr(self.proposer, 'snapshot', None))
            if has_draft != (snapshot.draft is not None):
                raise ValueError('generation snapshot draft state mismatch')
            self._busy = True
            old_target,old_draft = None,None
            try:
                old_target = self.target.snapshot()
                old_draft = self.proposer.snapshot() if has_draft else None
                old_rng = copy.deepcopy(self.rng.bit_generator.state)
                old_proposal_rng = (copy.deepcopy(proposal_rng.bit_generator.state)
                    if proposal_rng is not None and hasattr(proposal_rng, 'bit_generator') else None)
                try:
                    self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)
                    self.target.restore(snapshot.target)
                    if has_draft:
                        self.proposer.restore(snapshot.draft)
                    if snapshot.proposal_rng_state is not None:
                        proposal_rng.bit_generator.state = copy.deepcopy(snapshot.proposal_rng_state)
                    self._check_target_state(snapshot.history, logits)
                except BaseException as failure:
                    self._rollback(old_target, old_draft, old_rng, failure, old_proposal_rng)
                    raise
                self.history, self.next_logits = snapshot.history, logits
                self._stopped = snapshot.stopped
                self._stop_prefixes = snapshot.stop_prefixes
            finally:
                try:
                    self._release_states(old_target,old_draft,primary=sys.exc_info()[1])
                finally:
                    self._busy = False

    def _check_target_state(self, history, logits):
        """Check public retained evidence when the target exposes it."""
        target_history = getattr(self.target, 'history', None)
        if target_history is not None and not callable(target_history) and tuple(target_history) != history:
            raise ValueError('speculative history differs from target state')
        target_logits = getattr(self.target, 'logits', None)
        if target_logits is not None and not callable(target_logits):
            observed = np.asarray(target_logits, dtype=np.float64)
            if observed.shape != (self.vocabulary,) or not np.array_equal(observed, logits):
                raise ValueError('speculative verifier logits differ from target evidence')

    def generate(self, max_new_tokens, *, stop_tokens=(), stop_prefixes=(), cancelled=None):
        """Yield committed steps; cancellation is checked between transactions."""
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('positive generation limit required')
        if cancelled is not None and not callable(cancelled):
            raise ValueError('cancellation predicate must be callable')
        if (type(stop_tokens) not in (tuple, list, set, frozenset) or
                len(stop_tokens) > self.vocabulary):
            raise ValueError('bounded explicit stop-token inventory required')
        if any(type(t) is not int or not 0 <= t < self.vocabulary for t in stop_tokens):
            raise ValueError('invalid stop token')
        stops = frozenset(stop_tokens)
        prefixes = _stop_prefixes(stop_prefixes, self.vocabulary)
        emitted = 0
        while emitted < max_new_tokens:
            # Cancellation between yielded transactions is still a cancelled
            # generation, not a successful short stream.  Raise the same
            # domain exception used inside step() so direct iterator consumers
            # cannot mistake a partial result for completion.
            self._cancelled(cancelled)
            with self._lock:
                if self._stopped:
                    return
                remaining = self.context-len(self.history)
            if remaining <= 0:
                return
            result = self.step(max_tokens=min(max_new_tokens-emitted, remaining),
                               stop_tokens=stops, stop_prefixes=prefixes, cancelled=cancelled)
            emitted += len(result.tokens)
            yield result
            if self._stopped or result.tokens[-1] in stops:
                return

    def _logits(self, output, count):
        rows = np.asarray(output, dtype=np.float64)
        if rows.shape != (count, self.vocabulary) or not np.isfinite(rows).all():
            raise ValueError('target returned invalid logits')
        return rows

    def _probabilities(self, logits):
        if self.temperature == 0:
            row = np.zeros(self.vocabulary, dtype=np.float64)
            row[int(np.argmax(logits))] = 1
            return row
        scaled = (logits-np.max(logits))/self.temperature
        row = np.exp(scaled)
        return row/row.sum()

    def _sample(self, probabilities):
        return int(self.rng.choice(self.vocabulary, p=probabilities))

    def prefill(self, tokens, *,cancelled=None,**forward_options):
        with self._lock,self._cancel_scope(cancelled):
            self._check_open()
            self._cancelled(cancelled)
            if self._busy or self._quarantined or self.history:
                raise RuntimeError('prefill requires an idle empty session')
            ids = tuple(tokens)
            if (not ids or len(ids) >= self.context or
                    any(type(t) is not int or not 0 <= t < self.vocabulary for t in ids)):
                raise ValueError('valid prompt with remaining context required')
            self._busy = True
            saved = None
            try:
                saved = self.target.snapshot()
                logits = self._logits(self.target.forward(list(ids), **forward_options), len(ids))
                self._cancelled(cancelled)
                self.history, self.next_logits = ids, logits[-1].copy()
                self._stopped = False
                self._stop_prefixes = ()
            except BaseException as failure:
                if saved is not None:
                    try:
                        self.target.restore(saved)
                    except BaseException as rollback:
                        self._quarantined = True
                        failure.add_note('speculative prefill rollback failed: '+str(rollback))
                raise
            finally:
                try:
                    self._release_states(saved,primary=sys.exc_info()[1])
                finally:
                    self._busy = False

    def step(self, *, max_tokens=None, stop_tokens=(), stop_prefixes=(), cancelled=None):
        with self._lock,self._cancel_scope(cancelled):
            self._check_open()
            self._cancelled(cancelled)
            if self._busy or self._quarantined or self.next_logits is None:
                raise RuntimeError('idle prefilled session required')
            remaining = self.context-len(self.history)
            limit = remaining if max_tokens is None else max_tokens
            if type(limit) is not int or limit <= 0 or remaining <= 0:
                raise ValueError('positive generation budget and remaining context required')
            limit = min(limit, remaining)
            if (type(stop_tokens) not in (tuple, list, set, frozenset) or
                    len(stop_tokens) > self.vocabulary):
                raise ValueError('bounded explicit stop-token inventory required')
            if any(type(t) is not int or not 0 <= t < self.vocabulary for t in stop_tokens):
                raise ValueError('invalid stop token')
            stops = frozenset(stop_tokens)
            prefixes = _stop_prefixes(stop_prefixes, self.vocabulary)
            old_stop_prefixes = self._stop_prefixes
            if self._stopped:
                raise RuntimeError('generation is already at a stop boundary')
            self._stop_prefixes = prefixes
            self._busy = True
            saved,draft_state = None,None
            try:
                saved = self.target.snapshot()
                rng_state = copy.deepcopy(self.rng.bit_generator.state)
                draft_state = (self.proposer.snapshot() if
                    callable(getattr(self.proposer, 'snapshot', None)) else None)
                proposal_rng = getattr(self.proposer, 'rng', None)
                proposal_rng_state = (copy.deepcopy(proposal_rng.bit_generator.state)
                    if proposal_rng is not None and hasattr(proposal_rng, 'bit_generator') else None)
            except BaseException as failure:
                # Stop-prefix policy is part of the decoder boundary. Snapshot
                # acquisition can fail before the main transaction handler is
                # entered, so restore that policy here as well.
                self._stop_prefixes = old_stop_prefixes
                self._release_states(saved,draft_state,primary=failure)
                raise
            finally:
                self._busy = False
            self._busy = True
            try:
                proposal = self.proposer(self.history, min(self.draft_length, limit))
                self._cancelled(cancelled)
                if type(proposal) is not DraftProposal:
                    raise ValueError('DraftProposal required')
                ids = proposal.tokens
                if (type(ids) is not tuple or not ids or len(ids) > min(self.draft_length, limit) or
                        any(type(t) is not int or not 0 <= t < self.vocabulary for t in ids)):
                    raise ValueError('nonempty valid proposed token inventory required')
                q = np.asarray(proposal.probabilities, dtype=np.float64)
                if (q.shape != (len(ids), self.vocabulary) or not np.isfinite(q).all() or
                        np.any(q < 0) or (len(ids) and
                        not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-8))):
                    raise ValueError('normalized conditional draft probabilities required')
                if any(q[i, t] <= 0 for i, t in enumerate(ids)):
                    raise ValueError('proposed tokens must have positive draft probability')
                rows = (self._logits(self.target.forward(list(ids)), len(ids)) if ids else
                    np.empty((0, self.vocabulary), dtype=np.float64))
                self._cancelled(cancelled)
                output, accepted, rejected = [], 0, False
                for i, token in enumerate(ids):
                    self._cancelled(cancelled)
                    p = self._probabilities(self.next_logits if i == 0 else rows[i-1])
                    accept = (token == int(np.argmax(p)) if self.temperature == 0 else
                        self.rng.random() < min(1.0, p[token]/q[i, token]))
                    if not accept:
                        residual = np.maximum(p-q[i], 0)
                        total = residual.sum()
                        if self.temperature == 0:
                            replacement = int(np.argmax(p))
                        elif total > 0:
                            replacement = self._sample(residual/total)
                        else:
                            raise ArithmeticError('rejected proposal has no residual probability')
                        output.append(replacement)
                        rejected = True
                        break
                    output.append(token)
                    accepted += 1
                    if token in stops or _matches_stop_prefix(self.history + tuple(output), prefixes):
                        break
                if (not rejected and len(output) == len(ids) and len(output) < limit and
                        (not output or output[-1] not in stops) and
                        not _matches_stop_prefix(self.history + tuple(output), prefixes)):
                    p = self._probabilities(rows[-1] if ids else self.next_logits)
                    output.append(self._sample(p))
                # Keep verified cache pages when the entire proposal was accepted.
                if not rejected and accepted == len(ids):
                    if len(output) > len(ids):
                        committed = self._logits(self.target.forward(output[len(ids):]),
                            len(output)-len(ids))
                    else:
                        committed = rows
                else:
                    # The target proposal path is discarded on rejection or
                    # an early stop.  The proposer has also consumed the
                    # complete speculative draft, so it must be returned to
                    # the same pre-proposal boundary; otherwise its retained
                    # suffix can be mistaken for the committed history on the
                    # next transaction.
                    self.target.restore(saved)
                    if draft_state is not None:
                        self.proposer.restore(draft_state)
                    committed = self._logits(self.target.forward(output), len(output))
                emitted = tuple(output)
                new_history = self.history+emitted
                new_logits = committed[-1].copy()
                result = SpeculativeStep(emitted, accepted, len(ids), rejected)
                self._cancelled(cancelled)
                self.history, self.next_logits = new_history, new_logits
                self._stopped = bool(emitted and
                                     (emitted[-1] in stops or
                                      _matches_stop_prefix(self.history, prefixes)))
                return result
            except BaseException as failure:
                self._rollback(saved, draft_state, rng_state, failure, proposal_rng_state)
                # Stop policy is part of the transactional continuation state;
                # a failed step must not leave the next retry governed by the
                # caller's uncommitted policy.
                self._stop_prefixes = old_stop_prefixes
                raise
            finally:
                try:
                    self._release_states(saved,draft_state,primary=sys.exc_info()[1])
                finally:
                    self._busy = False

    def reset(self):
        with self._lock:
            self._check_open()
            if self._busy:
                raise RuntimeError('cannot reset an active generation step')
            self.retry_cleanup()
            recovering = self._quarantined
            self._quarantined = True
            self._busy = True
            try:
                self.target.reset()
                reset_draft = getattr(self.proposer, 'reset', None)
                if callable(reset_draft):
                    reset_draft()
                elif recovering and callable(getattr(self.proposer, 'snapshot', None)):
                    raise RuntimeError('stateful proposer requires reset to recover generation')
                self.history, self.next_logits = (), None
                self._stopped = False
                self._stop_prefixes = ()
                self._quarantined = False
            finally:
                self._busy = False
