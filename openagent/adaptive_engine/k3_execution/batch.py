"""Independent K3 sequence slots with atomic padded batch dispatch."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
from uuid import uuid4
import weakref

import numpy as np

from .executor import K3Execution, K3Snapshot
from ..executor_budget import _ReservedExecutor,_ConstructionCleanup
from ..failure_reporting import add_failure_note, failure_summary


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
    _lifetime: object | None=field(default=None, repr=False, compare=False)


class _K3BatchSnapshotLifetime:
    __slots__ = ('released', 'released_indices')

    def __init__(self):
        self.released = False
        self.released_indices = set()


class K3BatchExecution:
    """Own factory-created executors; commit or roll back every selected slot.

    This coordinator batches independent conversations with exact padding
    removal. Each executor retains the model's native/FLA arithmetic path.
    It does not claim a fused device batch or replace recurrent state with KV.
    """

    published_checkpoint_supported = False

    def __init__(self, factory, *, max_sessions, max_model_bytes,
                 max_batch_bytes, max_snapshot_bytes, owner_service=None):
        if not callable(factory):
            raise ValueError('fresh K3 executor factory required')
        for value in (max_sessions, max_model_bytes, max_batch_bytes, max_snapshot_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError('positive batch, model and snapshot admission limits required')
        self.factory, self.max_sessions = factory, max_sessions
        if owner_service is not None and not callable(getattr(owner_service, 'close', None)):
            raise ValueError('batch owner service must expose close')
        self._owner_service = owner_service
        self.max_model_bytes, self.max_batch_bytes = max_model_bytes, max_batch_bytes
        self.max_snapshot_bytes = max_snapshot_bytes
        self._owner = uuid4().hex
        self._models, self._reserved, self._pending = {}, 0, []
        # Batch snapshots own detached recurrent state outside the live slots.
        # Track them weakly so retained receipts are charged without keeping
        # otherwise-discarded snapshots alive forever.
        self._snapshot_receipts = []
        self._config, self._identity = None, None
        self._lock, self._busy, self._closed = threading.RLock(), False, False

    @contextmanager
    def _operation(self, *, closing=False):
        with self._lock:
            if (self._busy or (self._closed and not closing) or
                    (self._pending and not closing)):
                raise RuntimeError('batch executor is closed, has pending cleanup or is active')
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
                    add_failure_note(primary,'K3 batch retains failed slot construction cleanup')
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
                        add_failure_note(primary,'K3 slot admission cleanup retained: '+failure_summary(cleanup))
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
            if type(value).__module__.split('.')[0]=='torch':
                elements=value.numel()
                width=value.element_size()
                if type(elements) is not int or elements<0 or type(width) is not int or width<=0:
                    raise ValueError('invalid physical device snapshot extent')
                return elements*width
            if isinstance(value, (tuple, list)):
                return sum(charge(item) for item in value)
            if hasattr(value, '__dataclass_fields__'):
                return sum(charge(getattr(value, name)) for name in value.__dataclass_fields__)
            if isinstance(value, dict):
                return sum(charge(item) for item in value.values())
            return 64
        return charge(snapshot)

    def _retained_snapshot_bytes(self):
        live=[];total=0
        for receipt,size in self._snapshot_receipts:
            if receipt() is not None:
                live.append((receipt,size));total+=size
        self._snapshot_receipts=live
        return total

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
        retained=self._retained_snapshot_bytes()
        if estimate+retained > self.max_snapshot_bytes:
            raise MemoryError('batch transaction snapshots exceed admission')
        snapshots, charged = [], 0
        try:
            for model in models:
                state = model.snapshot()
                snapshots.append(state)
                charged += self._snapshot_bytes(state)
                if charged+retained > self.max_snapshot_bytes:
                    raise MemoryError('actual batch snapshots exceed admission')
        except BaseException as primary:
            cleanup_errors = []
            for model, state in reversed(tuple(zip(models, snapshots))):
                try:
                    model.release_snapshot(state)
                except BaseException as cleanup:
                    cleanup_errors.append(cleanup)
                    # Keep a retryable owner reference. A failed release is
                    # not safe to downgrade to a note: the executor still
                    # owns the detached physical snapshot and closing that
                    # executor first would make the receipt unreleasable.
                    self._pending.append(_ConstructionCleanup(
                        lambda model=model, state=state: model.release_snapshot(state)))
            if cleanup_errors:
                add_failure_note(primary,'partial K3 batch snapshot cleanup failed: '+
                                 '; '.join(failure_summary(error) for error in cleanup_errors))
            raise
        return tuple(snapshots)

    def _restore_all(self, models, snapshots, failure):
        errors = []
        for model, snapshot in zip(models, snapshots):
            try:
                model.restore(snapshot)
            except BaseException as error:
                errors.append(failure_summary(error))
        if errors:
            self._closed = True
            add_failure_note(failure,'batch rollback failed; all slots quarantined: '+'; '.join(errors))

    def _release_captured(self, models, snapshots, primary=None):
        errors=[]
        for model, snapshot in reversed(tuple(zip(models, snapshots))):
            try:
                model.release_snapshot(snapshot)
            except BaseException as error:
                errors.append(error)
                self._pending.append(_ConstructionCleanup(
                    lambda model=model, snapshot=snapshot:
                        model.release_snapshot(snapshot)))
        if errors:
            self._closed=True
            message='batch transaction snapshot cleanup failed: '+'; '.join(failure_summary(error) for error in errors)
            if primary is not None:
                add_failure_note(primary,message)
            else:
                raise RuntimeError(message) from errors[0]

    def forward(self, session_ids, token_ids, *, attention_mask=None,
                position_ids=None, cancelled=None, media=None):
        """Return occupied logits per row, ignoring left/right/interior padding.

        Position IDs, when supplied, match the physical padded input shape and
        are selected using the same mask. Cache positions count occupied tokens.
        This text-batch entry point does not invent media expansion positions.
        """
        with self._operation():
            if media is not None:
                raise ValueError('K3 batch execution is text-only; media expansion requires an individual K3Execution slot')
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
            if np.any(counts <= 0):
                raise ValueError('every selected K3 batch slot requires an occupied token')
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
                    expected = np.arange(model.position,
                                         model.position + int(counts[index]),
                                         dtype=selected.dtype)
                    if not np.array_equal(selected, expected):
                        raise ValueError('occupied rotary positions must extend the slot cache contiguously')
            saved = self._capture(models)
            logits, indices, lengths = [], [], []
            primary = None
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
                    # Do not publish a backend-owned view: another slot
                    # dispatch or recurrent update must not be able to mutate
                    # a completed batch result after the transaction commits.
                    detached = np.array(result, dtype=np.float32, order='C', copy=True)
                    detached.flags.writeable = False
                    logits.append(detached)
                    indices.append(tuple(int(v) for v in np.flatnonzero(mask[index])))
                    lengths.append(model.position)
                if cancelled is not None and cancelled():
                    from ..execution_service import ExecutionCancelled
                    raise ExecutionCancelled('K3 batch cancelled before commit')
                return K3BatchResult(ids, tuple(logits), tuple(indices), tuple(lengths))
            except BaseException as failure:
                primary = failure
                self._restore_all(models, saved, failure)
                raise
            finally:
                self._release_captured(models, saved, primary)

    def snapshot(self, session_ids):
        with self._operation():
            ids, models = self._selected(session_ids)
            states=self._capture(models)
            receipt=K3BatchSnapshot(self._owner, tuple(zip(ids, states)),
                                    _K3BatchSnapshotLifetime())
            self._snapshot_receipts.append((weakref.ref(receipt),self._snapshot_bytes(states)))
            return receipt

    def release_snapshot(self, snapshot):
        """Release the detached physical states owned by a batch receipt."""
        with self._operation():
            return self._release_snapshot_locked(snapshot)

    def _release_snapshot_locked(self, snapshot):
        lifetime = getattr(snapshot, '_lifetime', None)
        if (type(snapshot) is not K3BatchSnapshot or snapshot.owner != self._owner
                or type(lifetime) is not _K3BatchSnapshotLifetime):
            raise ValueError('foreign or non-owned K3 batch snapshot release')
        if lifetime.released:
            raise ValueError('K3 batch snapshot was already released')
        ids = tuple(item[0] for item in snapshot.sessions)
        _, models = self._selected(ids)
        errors = []
        for index, (model, item) in enumerate(zip(models, snapshot.sessions)):
            if index in lifetime.released_indices:
                continue
            try:
                model.release_snapshot(item[1])
            except BaseException as error:
                errors.append(error)
            else:
                lifetime.released_indices.add(index)
        if errors:
            raise RuntimeError('K3 batch snapshot release remains pending: '+
                               '; '.join(failure_summary(error) for error in errors)) from errors[0]
        lifetime.released = True
        object.__setattr__(snapshot, 'sessions', ())
        self._snapshot_receipts = [
            (receipt, 0 if receipt() is snapshot else size)
            for receipt, size in self._snapshot_receipts
        ]

    def restore(self, snapshot, *, cancelled=None):
        with self._operation():
            if cancelled is not None and not callable(cancelled):
                raise ValueError('cancellation must be callable')

            def check_cancelled():
                if cancelled is not None and cancelled():
                    from ..execution_service import ExecutionCancelled
                    raise ExecutionCancelled('K3 batch restore cancelled before commit')

            if (type(snapshot) is not K3BatchSnapshot or snapshot.owner != self._owner
                    or type(snapshot._lifetime) is not _K3BatchSnapshotLifetime
                    or snapshot._lifetime.released or snapshot._lifetime.released_indices):
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
            primary = None
            try:
                for model, target in zip(models, targets):
                    check_cancelled()
                    # Restore owns potentially large detached recurrent state.
                    # Use the executor's cancellation-aware staging path while
                    # holding its normal operation lease; cancellation never
                    # interrupts the subsequent rollback to `saved`.
                    with model._operation():
                        model._restore_locked(target, cancelled=check_cancelled)
                check_cancelled()
            except BaseException as failure:
                primary = failure
                self._restore_all(models, saved, failure)
                raise
            finally:
                self._release_captured(models, saved, primary)

    def reset(self, session_ids, *, cancelled=None):
        with self._operation():
            if cancelled is not None and not callable(cancelled):
                raise ValueError('cancellation must be callable')

            def check_cancelled():
                if cancelled is not None and cancelled():
                    from ..execution_service import ExecutionCancelled
                    raise ExecutionCancelled('K3 batch reset cancelled before commit')

            _, models = self._selected(session_ids)
            check_cancelled()
            saved = self._capture(models)
            primary = None
            try:
                for model in models:
                    # A single slot reset is an atomic executor operation.  Do
                    # not interrupt it midway; cancellation is observed at
                    # slot boundaries and the saved batch is restored before
                    # cancellation escapes.
                    check_cancelled()
                    model.reset()
                check_cancelled()
            except BaseException as failure:
                primary = failure
                self._restore_all(models, saved, failure)
                raise
            finally:
                self._release_captured(models, saved, primary)

    def close_session(self, session_id):
        with self._operation():
            _, (model,) = self._selected((session_id,))
            self._retained_snapshot_bytes()
            if any(receipt() is not None and any(
                    type(item) is tuple and len(item) == 2 and item[0] == session_id
                    for item in getattr(receipt(), 'sessions', ()))
                   for receipt, _ in self._snapshot_receipts):
                raise RuntimeError('release batch snapshots before closing a referenced session')
            reserved = model.max_bytes
            model.close()
            self._reserved -= reserved
            del self._models[session_id]

    def close(self):
        with self._operation(closing=True):
            self._closed = True
            errors = []
            # Snapshots contain copied host/device state.  Release them before
            # closing their executors so close is a complete ownership boundary.
            for receipt_ref, _ in tuple(self._snapshot_receipts):
                receipt = receipt_ref()
                if receipt is None or getattr(getattr(receipt, '_lifetime', None), 'released', True):
                    continue
                try:
                    self._release_snapshot_locked(receipt)
                except BaseException as error:
                    errors.append(failure_summary(error))
            if errors:
                raise RuntimeError('K3 batch snapshot cleanup remains pending: '+'; '.join(errors))
            # Retry private rollback/partial-capture receipts before closing
            # their model owners. K3Execution rejects operations after close,
            # so closing a slot first would turn a recoverable release failure
            # into a permanently stranded snapshot.
            pending_errors = []
            for pending in tuple(self._pending):
                try:
                    pending.close()
                except BaseException as error:
                    pending_errors.append(failure_summary(error))
                else:
                    self._pending.remove(pending)
            if pending_errors:
                raise RuntimeError('K3 batch private snapshot cleanup remains pending: '+
                                   '; '.join(pending_errors))
            for session_id, model in tuple(self._models.items()):
                reserved = model.max_bytes
                try:
                    model.close()
                except BaseException as error:
                    errors.append(failure_summary(error))
                else:
                    self._reserved -= reserved
                    del self._models[session_id]
            # Do not tear down the shared service while a failed slot still
            # owns native state.  A later close retry must release slots first
            # and only then destroy the service they borrow.
            if self._owner_service is not None and not self._models and not errors:
                try:
                    self._owner_service.close()
                except BaseException as error:
                        errors.append(failure_summary(error))
            elif self._owner_service is None and not self._models and not errors:
                cleanup = getattr(self.factory, 'retry_cleanup', None)
                if callable(cleanup):
                    try:
                        cleanup()
                    except BaseException as error:
                        errors.append(failure_summary(error))
            if errors:
                raise RuntimeError('K3 batch cleanup remains pending: '+'; '.join(errors))
