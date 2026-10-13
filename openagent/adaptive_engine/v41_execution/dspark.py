"""Backbone evidence capture and DSpark proposals for speculative sampling."""
import copy
import threading
from contextlib import contextmanager,ExitStack
from dataclasses import dataclass

import numpy as np

from .publisher import PublisherV41Execution
from .publisher_mtp import PublisherV41MTP
from .speculative import DraftProposal, SpeculativeDecoder


@dataclass(frozen=True)
class DSparkBackboneSnapshot:
    backbone: object
    hidden: tuple
    logits: bytes
    layers: tuple


class DSparkBackbone:
    """Retain target attention inputs alongside transactional backbone caches."""

    def __init__(self, model, target_layers, *, max_bytes=64*1024**2):
        if not isinstance(model, PublisherV41Execution):
            raise ValueError('publisher V4.1 backbone required')
        layers = tuple(target_layers)
        if (not layers or len(set(layers)) != len(layers) or
                any(type(i) is not int or not 0 <= i < len(model.profile.layers) for i in layers)):
            raise ValueError('unique valid DSpark target layers required')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive hidden-state retention budget required')
        self.model, self.layers, self.max_bytes = model, layers, max_bytes
        self.hidden, self.logits = (), None
        self._lock = threading.RLock()
        self._cache_transport = False
        self._quarantined = False
        self._closed = False

    def _check_transport(self):
        if self._closed:
            raise RuntimeError('DSpark evidence owner is closed')
        if self._cache_transport or self._quarantined:
            raise RuntimeError('DSpark evidence owner is unavailable')

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._cache_transport:
                raise RuntimeError('cannot close active DSpark cache transport')
            self._closed = True
            self.hidden, self.logits = (), None

    @contextmanager
    def cancellation_scope(self,cancel):
        with ExitStack() as scopes:
            scopes.enter_context(self.model.cancellation_scope(cancel))
            arithmetic=getattr(self.model.provider,'arithmetic',self.model.provider)
            scope=getattr(arithmetic,'cancellation_scope',None)
            if callable(scope):scopes.enter_context(scope(cancel))
            yield self

    def write_cache(self, output, *, max_bytes):
        from .dspark_cache import transport_backbone_cache
        return transport_backbone_cache(self, output, max_bytes=max_bytes)

    def restore_cache(self, source, *, max_bytes):
        from .dspark_cache import transport_backbone_cache
        return transport_backbone_cache(self, source, max_bytes=max_bytes, restoring=True)

    @property
    def history(self):
        with self._lock:
            self._check_transport()
            snapshot = self.model.snapshot()
            try:
                return snapshot.tokens
            finally:
                self.model.release_snapshot(snapshot)

    def forward(self, tokens, *, images=None, vision=None, token_types=None):
        with self._lock:
            self._check_transport()
            ids = tuple(tokens)
            width = len(self.layers)*self.model.profile.hidden_size
            if 4*(len(self.hidden)+3*len(ids))*width+8*self.model.profile.vocabulary > self.max_bytes:
                raise MemoryError('DSpark retained hidden state exceeds admission budget')
            old = self.model.snapshot()
            primary_failure = None
            try:
                logits, hidden = self.model.forward(list(ids), target_layers=self.layers,
                    images=images, vision=vision, token_types=token_types)
                if (hidden.shape != (len(ids), width) or
                        logits.shape != (len(ids), self.model.profile.vocabulary) or
                        not np.isfinite(hidden).all() or not np.isfinite(logits).all()):
                    raise ValueError('invalid backbone DSpark evidence')
                rows = tuple(np.asarray(row, dtype='<f4').tobytes() for row in hidden)
                next_logits = logits[-1].copy() if ids else self.logits
                next_hidden = self.hidden+rows
            except BaseException as failure:
                primary_failure = failure
                try:
                    self.model.restore(old)
                except BaseException as rollback:
                    self._quarantined = True
                    failure.add_note('DSpark evidence forward rollback failed: '+str(rollback))
                raise
            finally:
                try:
                    self.model.release_snapshot(old)
                except BaseException as cleanup:
                    self._quarantined = True
                    if primary_failure is not None:
                        primary_failure.add_note('DSpark evidence snapshot cleanup failed: '+str(cleanup))
                    else:
                        raise RuntimeError('DSpark evidence snapshot cleanup failed') from cleanup
            self.hidden, self.logits = next_hidden, next_logits
            return logits

    def snapshot(self):
        with self._lock:
            self._check_transport()
            return DSparkBackboneSnapshot(self.model.snapshot(), self.hidden,
                b'' if self.logits is None else np.asarray(self.logits, dtype='<f4').tobytes(),
                self.layers)

    def release_snapshot(self, snapshot):
        """Release the native target snapshot retained by a backbone snapshot."""
        with self._lock:
            if type(snapshot) is not DSparkBackboneSnapshot:
                raise ValueError('DSpark backbone snapshot required')
            self.model.release_snapshot(snapshot.backbone)

    def restore(self, snapshot):
        with self._lock:
            self._check_transport()
            p = self.model.profile
            width = len(self.layers)*p.hidden_size
            if (type(snapshot) is not DSparkBackboneSnapshot or snapshot.layers != self.layers or
                    type(snapshot.hidden) is not tuple or
                    len(snapshot.hidden) != len(snapshot.backbone.tokens) or
                    any(type(row) is not bytes or len(row) != 4*width for row in snapshot.hidden) or
                    type(snapshot.logits) is not bytes or
                    len(snapshot.logits) != (4*p.vocabulary if snapshot.hidden else 0)):
                raise ValueError('DSpark snapshot evidence inventory mismatch')
            if 4*len(snapshot.hidden)*width+len(snapshot.logits) > self.max_bytes:
                raise MemoryError('DSpark snapshot exceeds retention budget')
            if any(not np.isfinite(np.frombuffer(row, dtype='<f4')).all() for row in snapshot.hidden):
                raise ValueError('nonfinite DSpark hidden state')
            logits = np.frombuffer(snapshot.logits, dtype='<f4').copy() if snapshot.logits else None
            if logits is not None and not np.isfinite(logits).all():
                raise ValueError('nonfinite DSpark target logits')
            self.model.restore(snapshot.backbone)
            self.hidden, self.logits = snapshot.hidden, logits

    def reset(self):
        with self._lock:
            self._check_transport()
            if self._cache_transport:
                raise RuntimeError('DSpark cache transport is active')
            self._quarantined = True
            self.model.reset()
            self.hidden, self.logits = (), None
            self._quarantined = False


class DSparkProposer:
    """Seed draft windows from observed backbone inputs, then propose a block.

    Reseeds only the occupied sliding-window tail at absolute rotary positions.
    Proposal snapshots retain rollback consistency across target rejection.
    """

    @contextmanager
    def cancellation_scope(self,cancel):
        arithmetic=getattr(self.draft.provider,'arithmetic',self.draft.provider)
        scope=getattr(arithmetic,'cancellation_scope',None)
        with ExitStack() as scopes:
            if callable(scope):scopes.enter_context(scope(cancel))
            yield self

    def __init__(self, backbone, draft, *, temperature=1.0, rng=None):
        if not isinstance(backbone, DSparkBackbone) or not isinstance(draft, PublisherV41MTP):
            raise ValueError('observed publisher backbone and DSpark draft required')
        if (tuple(draft.targets) != backbone.layers or
                draft.vocab != backbone.model.profile.vocabulary or
                draft.h != backbone.model.profile.hidden_size):
            raise ValueError('DSpark draft and backbone geometry mismatch')
        target_arithmetic = getattr(backbone.model.provider, 'arithmetic', None)
        if target_arithmetic is None or target_arithmetic is not getattr(draft.provider, 'arithmetic', None):
            raise ValueError('draft and backbone must share publisher arithmetic')
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError('finite nonnegative sampling temperature required')
        self.backbone, self.draft = backbone, draft
        self.temperature = float(temperature)
        self.rng = np.random.default_rng() if rng is None else rng
        self._lock = threading.RLock()
        self._quarantined = False
        self._closed = False

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._quarantined:
                raise RuntimeError('cannot close quarantined DSpark proposer')
            self.draft.close()
            self._closed = True

    def snapshot(self):
        with self._lock:
            if self._closed or self._quarantined:
                raise RuntimeError('DSpark proposer rollback requires recovery')
            return self.draft.snapshot(), copy.deepcopy(self.rng.bit_generator.state)

    def reset(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('DSpark proposer is closed')
            self._quarantined = True
            self.draft.reset()
            self._quarantined = False

    def restore(self, state):
        with self._lock:
            if self._closed:
                raise RuntimeError('DSpark proposer is closed')
            draft_state, rng_state = state
            old_rng = copy.deepcopy(self.rng.bit_generator.state)
            try:
                self.rng.bit_generator.state = copy.deepcopy(rng_state)
                self.draft.restore(draft_state)
            except BaseException as failure:
                self._quarantined = True
                try:self.rng.bit_generator.state = old_rng
                except BaseException as rollback:
                    self._quarantined = True
                    failure.add_note('DSpark proposal RNG rollback failed: '+repr(rollback))
                raise
            self._quarantined = False

    def _distribution(self, logits, *, draft=False):
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (self.draft.vocab,) or not np.isfinite(logits).all():
            raise ValueError('finite vocabulary logits required')
        if self.temperature == 0:
            result = np.zeros(self.draft.vocab, dtype=np.float64)
            result[int(np.argmax(logits))] = 1
            return result
        temperature = max(self.temperature, 1e-5) if draft else self.temperature
        result = np.exp((logits-logits.max())/temperature)
        return result/result.sum()

    def __call__(self, history, count):
        with self._lock, self.backbone._lock:
            if self._closed or self._quarantined:
                raise RuntimeError('DSpark proposer rollback requires recovery')
            if (type(count) is not int or count <= 0 or history != self.backbone.history or
                    not history or self.backbone.logits is None):
                raise ValueError('committed backbone history and positive proposal count required')
            saved = self.snapshot()
            try:
                first_q = self._distribution(self.backbone.logits)
                first = int(self.rng.choice(self.draft.vocab, p=first_q))
                # A one-token prompt cannot enter the publisher's incremental MTP
                # path. Near the context boundary, avoid out-of-range draft RoPE.
                if (count == 1 or len(history) == 1 or
                        len(history)+self.draft.size > self.backbone.model.profile.context):
                    return DraftProposal((first,), first_q[None, :])
                width = self.draft.h*len(self.draft.targets)
                occupied = min(len(history)-1, self.draft.config['sliding_window'])
                begin = len(history)-1-occupied
                if 4*width*(len(self.backbone.hidden)+2*occupied+1) > self.backbone.max_bytes:
                    raise MemoryError('DSpark reseeding scratch exceeds evidence budget')
                prefix = np.stack([np.frombuffer(self.backbone.hidden[index], dtype='<f4')
                                   for index in range(begin, len(history)-1)])
                last = np.frombuffer(self.backbone.hidden[-1], dtype='<f4').reshape(1, width)
                self.draft.seed_window(prefix, start_pos=begin)
                ids, logits, _ = self.draft.forward_spec(first, last, len(history)-1,
                    temperature=self.temperature)
                take = min(count-1, self.draft.size)
                probabilities = [first_q]
                probabilities.extend(self._distribution(row, draft=True) for row in logits[:take])
                return DraftProposal(tuple(int(t) for t in ids[:take+1]), np.stack(probabilities))
            except BaseException as failure:
                try:self.restore(saved)
                except BaseException as rollback:
                    self._quarantined = True
                    failure.add_note('DSpark proposal rollback failed: '+repr(rollback))
                raise


def create_dspark_decoder(model, draft, *, temperature=1.0,
                          max_bytes=64*1024**2, evidence_max_bytes=64*1024**2,
                          target_rng=None, proposal_rng=None):
    backbone = DSparkBackbone(model, draft.targets, max_bytes=evidence_max_bytes)
    proposer = DSparkProposer(backbone, draft, temperature=temperature, rng=proposal_rng)
    return SpeculativeDecoder(backbone, proposer, vocabulary=model.profile.vocabulary,
        context=model.profile.context, temperature=temperature, draft_length=draft.size+1,
        max_bytes=max_bytes, rng=target_rng)
