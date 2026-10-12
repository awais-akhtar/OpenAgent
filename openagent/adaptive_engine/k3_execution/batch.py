"""Independent K3 sequence slots with atomic padded batch dispatch."""
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from uuid import uuid4

import numpy as np

from .executor import K3Execution, K3Snapshot
from ..executor_budget import _ReservedExecutor,_ConstructionCleanup


@dataclass(frozen=True)
class K3BatchResult:
    session_ids: tuple
    logits: tuple
    occupied_indices: tuple
    positions: tuple


@dataclass(frozen=True)
class K3BatchSnapshot:
    owner: str
    sessions: tuple


class K3BatchExecution:
    """Own factory-created executors; commit or roll back every selected slot.

    This coordinator batches independent conversations with exact padding
    removal. Each executor retains the model's native/FLA arithmetic path.
    It does not claim a fused device batch or replace recurrent state with KV.
    """

    published_checkpoint_supported = False

    def __init__(self, factory, *, max_sessions, max_model_bytes,
                 max_batch_bytes, max_snapshot_bytes):
        if not callable(factory):
            raise ValueError('fresh K3 executor factory required')
        for value in (max_sessions, max_model_bytes, max_batch_bytes, max_snapshot_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError('positive batch, model and snapshot admission limits required')
        self.factory, self.max_sessions = factory, max_sessions
        self.max_model_bytes, self.max_batch_bytes = max_model_bytes, max_batch_bytes
        self.max_snapshot_bytes = max_snapshot_bytes
        self._owner = uuid4().hex
        self._models, self._reserved, self._pending = {}, 0, []
        self._config, self._identity = None, None
        self._lock, self._busy, self._closed = threading.RLock(), False, False

    @contextmanager
    def _operation(self, *, closing=False):
        with self._lock:
            if self._busy or (self._closed and not closing):
                raise RuntimeError('batch executor is closed or has an active operation')
            self._busy = True
            try:
                yield
            finally:
                self._busy = False

    def open_session(self):
        with self._operation():
            if len(self._models) >= self.max_sessions or self._pending:
                raise RuntimeError('batch slot admission exhausted or cleanup is pending')
            declared=getattr(self.factory,'executor_bytes',None)
            if declared is not None and (type(declared) is not int or declared<=0 or
                    declared>self.max_model_bytes-self._reserved):
                raise MemoryError('next declared K3 slot exceeds aggregate model admission')
            try:model = self.factory()
            except BaseException as primary:
                cleanup=getattr(primary,'cleanup',None)
                if callable(cleanup):
                    self._pending.append(_ConstructionCleanup(cleanup))
                    primary.add_note('K3 batch retains failed slot construction cleanup')
                raise
            concrete = model._executor if type(model) is _ReservedExecutor else model
            if any(concrete is (item._executor if type(item) is _ReservedExecutor else item)
                   for item in self._models.values()):
                raise ValueError('factory reused a mutable K3 sequence executor')
            try:
                if type(concrete) is not K3Execution or model.position != 0:
                    raise ValueError('factory must create an empty exact K3Execution')
                if self._config is not None and (model.config != self._config or
                                                  model._identity != self._identity):
                    raise ValueError('batch slots require the same model and arithmetic identity')
                if model.max_bytes > self.max_model_bytes-self._reserved:
                    raise MemoryError('aggregate declared model budgets exceed batch admission')
                session_id = uuid4().hex
                self._models[session_id] = model
                self._reserved += model.max_bytes
                self._config, self._identity = model.config, model._identity
                return session_id
            except BaseException as primary:
                close = getattr(model, 'close', None)
                if callable(close):
                    try:
                        close()
                    except BaseException as cleanup:
                        self._pending.append(model)
                        primary.add_note('K3 slot admission cleanup retained: '+repr(cleanup))
                raise

    def _selected(self, session_ids):
        if (type(session_ids) not in (tuple, list) or not session_ids or
                len(session_ids) > self.max_sessions or
                any(type(value) is not str for value in session_ids) or
                len(set(session_ids)) != len(session_ids)):
            raise ValueError('unique existing sequence slot IDs required')
        ids = tuple(session_ids)
        if any(value not in self._models for value in ids):
            raise ValueError('unknown K3 batch slot')
        return ids, tuple(self._models[value] for value in ids)

    @staticmethod
    def _snapshot_bytes(snapshot):
        def charge(value):
            if isinstance(value, np.ndarray):
                return value.nbytes
            if isinstance(value, bytes):
                return len(value)
            if isinstance(value, (tuple, list)):
                return sum(charge(item) for item in value)
            if hasattr(value, '__dataclass_fields__'):
                return sum(charge(getattr(value, name)) for name in value.__dataclass_fields__)
            if isinstance(value, dict):
                return sum(charge(item) for item in value.values())
            return 64
        return charge(snapshot)

    def _capture(self, models):
        # Reserve before snapshot copies: include every occupied latent row,
        # recurrent matrix and complete convolution state plus descriptor room.
        estimate = 0
        for model in models:
            c = model.config
            mla = sum(pages.nbytes for pages in model._mla.values())
            recurrent = c.layer_types.count('kda')*4*(c.kda_heads*c.kda_dim**2+
                         3*c.kda_heads*c.kda_dim*c.kernel)
            estimate += 2*(mla+recurrent)+4096*(len(c.layer_types)+1)
        if estimate > self.max_snapshot_bytes:
            raise MemoryError('batch transaction snapshots exceed admission')
        snapshots, charged = [], 0
        for model in models:
            state = model.snapshot()
            charged += self._snapshot_bytes(state)
            if charged > self.max_snapshot_bytes:
                raise MemoryError('actual batch snapshots exceed admission')
            snapshots.append(state)
        return tuple(snapshots)

    def _restore_all(self, models, snapshots, failure):
        errors = []
        for model, snapshot in zip(models, snapshots):
            try:
                model.restore(snapshot)
            except BaseException as error:
                errors.append(str(error))
        if errors:
            self._closed = True
            failure.add_note('batch rollback failed; all slots quarantined: '+'; '.join(errors))

    def forward(self, session_ids, token_ids, *, attention_mask=None,
                position_ids=None, cancelled=None):
        """Return occupied logits per row, ignoring left/right/interior padding.

        Position IDs, when supplied, match the physical padded input shape and
        are selected using the same mask. Cache positions count occupied tokens.
        This text-batch entry point does not invent media expansion positions.
        """
        with self._operation():
            if cancelled is not None and not callable(cancelled):
                raise ValueError('cancellation must be callable')
            ids, models = self._selected(session_ids)
            raw = np.asarray(token_ids)
            if (raw.ndim != 2 or raw.shape[0] != len(ids) or
                    not 1 <= raw.shape[1] <= 4096 or raw.dtype.kind not in 'iu'):
                raise ValueError('integer [selected slots, padded tokens] input required')
            input_charge = raw.nbytes+raw.size*64
            if input_charge > self.max_batch_bytes:
                raise MemoryError('padded batch inputs exceed admission')
            tokens = np.array(raw, copy=True)
            if attention_mask is None:
                mask = np.ones(tokens.shape, dtype=bool)
            else:
                observed = np.asarray(attention_mask)
                if (observed.shape != tokens.shape or observed.dtype.kind not in 'biu' or
                        np.any((observed != 0) & (observed != 1))):
                    raise ValueError('binary padding mask must match the physical batch shape')
                mask = np.array(observed, dtype=bool, copy=True)
            counts = mask.sum(axis=1)
            output_charge = int(counts.sum())*models[0].config.vocab*4
            if input_charge+output_charge > self.max_batch_bytes:
                raise MemoryError('occupied batch logits exceed output admission')
            positions = None
            if position_ids is not None:
                observed = np.asarray(position_ids)
                if observed.shape != tokens.shape or observed.dtype.kind not in 'iu':
                    raise ValueError('integer rotary positions must match padded batch shape')
                positions = np.array(observed, copy=True)
            for index, model in enumerate(models):
                occupied = tokens[index, mask[index]]
                if np.any(occupied < 0) or np.any(occupied >= model.config.vocab):
                    raise ValueError('occupied batch token outside vocabulary')
                if model.position+int(counts[index]) > model.config.capacity:
                    raise ValueError('one batch slot exceeds its context capacity')
                if positions is not None:
                    selected = positions[index, mask[index]]
                    if np.any(selected < 0) or np.any(selected >= model.config.capacity):
                        raise ValueError('occupied rotary position outside context capacity')
            saved = self._capture(models)
            logits, indices, lengths = [], [], []
            try:
                for index, model in enumerate(models):
                    if cancelled is not None and cancelled():
                        from ..execution_service import ExecutionCancelled
                        raise ExecutionCancelled('K3 batch cancelled before slot dispatch')
                    class Cancellation:
                        def is_set(self):
                            return cancelled is not None and bool(cancelled())
                    with model.cancellation_scope(Cancellation()):
                        result = model.forward(tokens[index], attention_mask=mask[index],
                            position_ids=None if positions is None else positions[index])
                    if (result.dtype != np.float32 or
                            result.shape != (int(counts[index]), model.config.vocab) or
                            not np.isfinite(result).all()):
                        raise ArithmeticError('invalid occupied K3 batch logits')
                    logits.append(result)
                    indices.append(tuple(int(v) for v in np.flatnonzero(mask[index])))
                    lengths.append(model.position)
                if cancelled is not None and cancelled():
                    from ..execution_service import ExecutionCancelled
                    raise ExecutionCancelled('K3 batch cancelled before commit')
                return K3BatchResult(ids, tuple(logits), tuple(indices), tuple(lengths))
            except BaseException as failure:
                self._restore_all(models, saved, failure)
                raise

    def snapshot(self, session_ids):
        with self._operation():
            ids, models = self._selected(session_ids)
            return K3BatchSnapshot(self._owner, tuple(zip(ids, self._capture(models))))

    def restore(self, snapshot):
        with self._operation():
            if type(snapshot) is not K3BatchSnapshot or snapshot.owner != self._owner:
                raise ValueError('foreign K3 batch snapshot')
            if (type(snapshot.sessions) is not tuple or not snapshot.sessions or
                    any(type(item) is not tuple or len(item) != 2 or
                        type(item[1]) is not K3Snapshot for item in snapshot.sessions)):
                raise ValueError('explicit complete batch snapshot entries required')
            ids, models = self._selected(tuple(item[0] for item in snapshot.sessions))
            targets = tuple(item[1] for item in snapshot.sessions)
            if sum(self._snapshot_bytes(state) for state in targets) > self.max_snapshot_bytes:
                raise MemoryError('batch restore snapshot exceeds admission')
            saved = self._capture(models)
            try:
                for model, target in zip(models, targets):
                    model.restore(target)
            except BaseException as failure:
                self._restore_all(models, saved, failure)
                raise

    def reset(self, session_ids):
        with self._operation():
            _, models = self._selected(session_ids)
            saved = self._capture(models)
            try:
                for model in models:
                    model.reset()
            except BaseException as failure:
                self._restore_all(models, saved, failure)
                raise

    def close_session(self, session_id):
        with self._operation():
            _, (model,) = self._selected((session_id,))
            reserved = model.max_bytes
            model.close()
            self._reserved -= reserved
            del self._models[session_id]

    def close(self):
        with self._operation(closing=True):
            self._closed = True
            errors = []
            for session_id, model in tuple(self._models.items()):
                reserved = model.max_bytes
                try:
                    model.close()
                except BaseException as error:
                    errors.append(str(error))
                else:
                    self._reserved -= reserved
                    del self._models[session_id]
            for model in tuple(self._pending):
                try:
                    model.close()
                except BaseException as error:
                    errors.append(str(error))
                else:
                    self._pending.remove(model)
            cleanup = getattr(self.factory, 'retry_cleanup', None)
            if callable(cleanup):
                try:
                    cleanup()
                except BaseException as error:
                    errors.append(str(error))
            if errors:
                raise RuntimeError('K3 batch cleanup remains pending: '+'; '.join(errors))
