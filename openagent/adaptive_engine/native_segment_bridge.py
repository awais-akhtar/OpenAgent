"""Native OLMoE execution behind OpenAgent's lifecycle and transfer ABI."""
from contextlib import ExitStack
import numpy as np

from .segment_abi import SegmentBridge
from .native_olmoe_segments import (
    NativeOlmoeSegmentEngine, NativeOlmoeSegmentSession, NativeOlmoeEdge,
)
from .native_segment_snapshot import write_native_snapshot, restore_native_snapshot


class NativeSegmentBridge(SegmentBridge):
    """Transfer handles to C-backed ranges and edges, with owned child cleanup.

    Python handles transport and checkpoints; numerical range/edge operations
    use the backend's native library. This is not the reference project's ABI.
    Leased packed engines inherit the same range and session interfaces.
    """

    native_range_execution = True
    upstream_abi_compatible = False
    engine_types = (NativeOlmoeSegmentEngine,)
    session_types = (NativeOlmoeSegmentSession,)
    edge_types = (NativeOlmoeEdge,)

    def _check_backend(self, backend):
        if isinstance(backend, self.edge_types):
            with backend._lock:
                backend._check('head' if backend.mode == 'head' else 'embedding')
        else:
            backend._check()

    def _write_snapshot(self, backend, output):
        write_native_snapshot(backend, output, max_bytes=self.max_bytes)

    def _restore_snapshot(self, backend, source):
        restore_native_snapshot(backend, source, max_bytes=self.max_bytes)

    def token_logits(self, edge, session, token_ids, *, cancelled=None):
        """Run a complete admitted range and edge, restoring KV on failure."""
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or not ids.size or ids.dtype.kind not in 'iu':
            raise ValueError('nonempty integer token vector required')
        with self.lock, ExitStack() as owned:
            boundary = self._record(edge)[0]
            target = self._record(session)[0]
            if (not isinstance(boundary, self.edge_types) or
                    not isinstance(target, self.session_types)):
                raise TypeError('native edge and range session required')
            engine = target._engine
            owned.enter_context(engine._lock)
            owned.enter_context(boundary._lock)
            owned.enter_context(target._lock)
            target._check()
            boundary._check('both')
            if (engine.begin != 0 or engine.end != engine.config.num_hidden_layers or
                    engine.config != boundary.config or engine.root != boundary.root):
                raise ValueError('matching complete decoder range and edge required')
            if len(ids)+target.position > target.context_tokens:
                raise MemoryError('token request exceeds session context')
            if ids.dtype.kind == 'u' and np.any(ids > np.iinfo(np.int64).max):
                raise ValueError('token exceeds signed vocabulary representation')
            if np.any(ids < 0) or np.any(ids >= engine.config.vocab_size):
                raise ValueError('token outside checkpoint vocabulary')
            staged = len(ids)*(8+engine.config.hidden_size*8+engine.config.vocab_size*8)
            if staged > self.max_bytes:
                raise MemoryError('token-to-logit transport exceeds byte budget')
            if cancelled is not None and cancelled():
                raise InterruptedError('native token request cancelled')
            snapshot = self.snapshot(session)
            if len(snapshot)+staged > self.max_bytes:
                raise MemoryError('transactional snapshot and output exceed byte budget')
            try:
                embedded = self.embeddings(edge, ids)
                hidden = self.hidden(session, embedded, cancelled=cancelled)
                logits = self.hidden(edge, hidden, final_head=True)
                if cancelled is not None and cancelled():
                    raise InterruptedError('native token request cancelled')
                return logits
            except BaseException as error:
                try:
                    self.restore(session, snapshot)
                except BaseException as restore_error:
                    error.add_note('native cache rollback failed: '+str(restore_error))
                    try:
                        self.release(session)
                    except BaseException as close_error:
                        error.add_note('damaged session release failed: '+str(close_error))
                raise

    def status(self):
        with self.lock:
            return dict(
                abi='openagent.segment-transfer.v1',
                native_range_execution=True,
                upstream_abi_compatible=False,
                closed=self._closed,
                handles={handle: backend.status() if hasattr(backend, 'status')
                         else backend.snapshot_metadata()
                         for handle, (backend, _, _) in self._objects.items()},
            )
