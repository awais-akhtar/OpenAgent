"""Clean-room native lifecycle/transfer ABI with pluggable execution backends.

Load an explicitly built library. No automatic compilation, downloads, or
fallback pretending to be native. All adapters for a library share one lock.
The base bridge uses Python/NumPy models; NativeSegmentBridge uses C execution.
"""
import ctypes as C
import io
from pathlib import Path
import threading

import numpy as np

from .olmoe_segments import OlmoeEdge, OlmoeSegmentEngine, OlmoeSegmentSession
from .segment_snapshot import restore_segment_snapshot, write_segment_snapshot

VERSION = 1
OK, INVALID, BAD_VERSION, BAD_HANDLE, BUSY, LIMIT, CALLBACK_ERROR, CANCELLED = range(8)
ENGINE, SESSION, EDGE = 1, 2, 3
CLOSE, FORWARD, EMBED, HEAD, RESET, SNAPSHOT, RESTORE = range(1, 8)
U64 = C.c_uint64
CALLBACK = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.c_uint32, U64,
                      C.c_void_p, U64, C.c_void_p, U64, C.POINTER(U64))
_locks = {}
_locks_guard = threading.Lock()
_owners = set()  # Pin callbacks until every native handle has been released.
_backends = set()  # Backend identities with transferred close ownership.


class SegmentABIError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f'Segment ABI status {status}')


class SegmentABICloseError(SegmentABIError):
    """Cleanup failures, including parents still blocked by live children."""
    def __init__(self, failures, remaining_handles):
        self.failures = tuple(failures)
        self.remaining_handles = tuple(remaining_handles)
        super().__init__(self.failures[0][1].status)
        self.args = (f'Segment bridge cleanup failed; live handles: '
                     f'{self.remaining_handles}; failures: ' + ', '.join(
                         f'{handle}: {error}' for handle, error in self.failures),)


def _uint(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= (1 << 64) - 1:
        raise ValueError(f'invalid {name}')
    return value


class _BoundedOutput(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise OverflowError('snapshot exceeds transfer capacity')
        return super().write(data)


class NativeSegmentMath:
    """Independent native numerical primitives, not a complete native decoder.

    Requires a library built with segment_math.c; never uses Python fallback.
    """
    def __init__(self, library, *, max_bytes=64 * 1024 * 1024):
        self.max_bytes = _uint(max_bytes, 'max_bytes', 4)
        if self.max_bytes > np.iinfo(np.intp).max:
            raise ValueError('max_bytes exceeds host address size')
        self.lib = C.CDLL(str(Path(library).resolve(strict=True)))
        self.lib.oa_segment_version.argtypes = []
        self.lib.oa_segment_version.restype = C.c_uint32
        if self.lib.oa_segment_version() != VERSION:
            raise SegmentABIError(BAD_VERSION)
        self._rmsnorm = self.lib.oa_segment_rmsnorm_f32
        self._rmsnorm.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
                                 U64, U64, C.c_double, U64]
        self._rmsnorm.restype = C.c_int32

    def rmsnorm(self, hidden, weight, *, epsilon=1e-5):
        hidden, weight = np.asarray(hidden), np.asarray(weight)
        if (hidden.dtype != np.dtype('float32') or weight.dtype != np.dtype('float32')
                or hidden.ndim != 2 or weight.ndim != 1
                or not hidden.shape[0] or not hidden.shape[1]
                or weight.shape[0] != hidden.shape[1]):
            raise ValueError('expected FP32 matrix and matching FP32 weight vector')
        if hidden.nbytes > self.max_bytes or weight.nbytes > self.max_bytes:
            raise SegmentABIError(LIMIT)
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('expected positive finite epsilon')
        hidden = np.ascontiguousarray(hidden)
        weight = np.ascontiguousarray(weight)
        output = np.empty(hidden.shape, dtype=np.float32)
        status = self._rmsnorm(hidden.ctypes.data, weight.ctypes.data,
            output.ctypes.data, hidden.shape[0], hidden.shape[1], epsilon, self.max_bytes)
        if status != OK:
            raise SegmentABIError(status)
        return output


class SegmentBridge:
    """Own registered backends until successful release; close children first.

    Backends must not be mutated outside this adapter while registered.
    `lib` is exposed for C contract tests; raw callers must use `lock` too.
    """
    native_range_execution = False
    upstream_abi_compatible = False
    engine_types = (OlmoeSegmentEngine,)
    session_types = (OlmoeSegmentSession,)
    edge_types = (OlmoeEdge,)

    def _check_backend(self, backend):
        backend._check()

    def _write_snapshot(self, backend, output):
        write_segment_snapshot(backend, output)

    def _restore_snapshot(self, backend, source):
        restore_segment_snapshot(backend, source, max_bytes=self.max_bytes)

    def __init__(self, library, *, max_bytes=64 * 1024 * 1024):
        self.max_bytes = _uint(max_bytes, 'max_bytes', 44)
        if self.max_bytes > np.iinfo(np.intp).max:
            raise ValueError('max_bytes exceeds host address size')
        path = str(Path(library).resolve(strict=True))
        self.lib = C.CDLL(path)
        with _locks_guard:
            self.lock = _locks.setdefault(self.lib._handle, threading.RLock())
        self.lib.oa_segment_version.argtypes = []
        self.lib.oa_segment_version.restype = C.c_uint32
        self.lib.oa_segment_register.argtypes = [C.c_uint32, C.c_uint32, U64,
            U64, CALLBACK, C.c_void_p, C.POINTER(U64)]
        self.lib.oa_segment_register.restype = C.c_int32
        self.lib.oa_segment_call.argtypes = [U64, C.c_uint32, U64, C.c_void_p,
            U64, C.c_void_p, U64, C.POINTER(U64)]
        self.lib.oa_segment_call.restype = C.c_int32
        self.lib.oa_segment_release.argtypes = [U64]
        self.lib.oa_segment_release.restype = C.c_int32
        if self.lib.oa_segment_version() != VERSION:
            raise SegmentABIError(BAD_VERSION)
        self._objects = {}
        self._active = set()
        self._quarantined = set()
        self._cancelled = None
        self._closed = False

    @staticmethod
    def _status(status, error=None):
        if status != OK:
            raise SegmentABIError(status) from error

    def register(self, backend, *, parent=0):
        """Transfer close responsibility on success, not on registration error."""
        _uint(parent, 'parent')
        with self.lock:
            if self._closed:
                raise RuntimeError('segment bridge is closed')
            if self._active:
                raise SegmentABIError(BUSY)
            with _locks_guard:
                if id(backend) in _backends:
                    raise ValueError('backend already registered')
            if isinstance(backend, self.session_types):
                kind = SESSION
                if parent not in self._objects or self._objects[parent][0] is not backend._engine:
                    raise ValueError('session parent mismatch')
            elif isinstance(backend, self.engine_types):
                kind = ENGINE
            elif isinstance(backend, self.edge_types):
                kind = EDGE
            else:
                raise TypeError('expected OLMoE segment engine, session, or edge')
            self._check_backend(backend)
            errors = []
            callback = CALLBACK(lambda *args: self._dispatch(backend, errors, *args))
            handle = U64()
            with _locks_guard:
                if id(backend) in _backends:
                    raise ValueError('backend already registered')
                self._status(self.lib.oa_segment_register(VERSION, kind, parent,
                    self.max_bytes, callback, None, C.byref(handle)))
                self._objects[handle.value] = (backend, callback, errors)
                _backends.add(id(backend))
                _owners.add(self)
            return handle.value

    def create_session(self, engine, *, context_tokens=None):
        with self.lock:
            record = self._record(engine)
            if engine in self._active:
                raise SegmentABIError(BUSY)
            if not isinstance(record[0], self.engine_types):
                raise ValueError('expected engine handle')
            session = record[0].create_session(context_tokens=context_tokens)
            try:
                return self.register(session, parent=engine)
            except BaseException:
                session.close()
                raise

    def _record(self, handle, *, allow_quarantined=False):
        _uint(handle, 'handle', 1)
        if handle not in self._objects:
            raise SegmentABIError(BAD_HANDLE)
        if handle in self._quarantined and not allow_quarantined:
            raise SegmentABIError(CALLBACK_ERROR)
        return self._objects[handle]

    def _dispatch(self, backend, errors, context, op, rows, src, size, dst, capacity, written):
        written[0] = 0
        try:
            # Native callbacks may provide malformed pointers even when the
            # byte counts look valid. Reject those records before ctypes reads
            # or writes through them; an escaping access violation cannot be
            # converted into an ABI status by Python.
            if size and not src:
                raise ValueError('nonzero input requires a source buffer')
            if capacity and not dst:
                raise ValueError('nonzero output capacity requires a destination buffer')
            if op == CLOSE:
                backend.close()
                return OK
            if op in (FORWARD, HEAD, EMBED):
                if (self._cancelled is not None and
                        (self._cancelled() if callable(self._cancelled) else self._cancelled.is_set())):
                    raise InterruptedError('native ABI operation cancelled before dispatch')
                width = backend._engine.config.hidden_size if op == FORWARD else backend.config.hidden_size
                out_width = backend.config.vocab_size if op == HEAD else width
                expected = rows * (8 if op == EMBED else width * 4)
                if not rows or size != expected:
                    raise ValueError('invalid tensor byte length')
                if rows * out_width * 4 > capacity:
                    raise OverflowError('output exceeds transfer capacity')
                data = C.string_at(src, size)
                if op == EMBED:
                    result = backend.embeddings(np.frombuffer(data, dtype='<i8'))
                else:
                    hidden = np.frombuffer(data, dtype='<f4').astype(np.float32).reshape(rows, width)
                    if op == FORWARD:
                        result = backend.forward(hidden, cancelled=self._cancelled)
                    else:
                        result = backend.final_head(hidden)
                if (self._cancelled is not None and
                        (self._cancelled() if callable(self._cancelled) else self._cancelled.is_set())):
                    raise InterruptedError('native ABI operation cancelled after dispatch')
                data = result.astype('<f4', copy=False).tobytes(order='C')
            elif op == RESET:
                if rows or size or capacity:
                    raise ValueError('reset takes no buffers')
                backend.reset()
                return OK
            elif op == SNAPSHOT:
                if rows or size:
                    raise ValueError('snapshot takes no input')
                output = _BoundedOutput(capacity)
                self._write_snapshot(backend, output)
                data = output.getvalue()
            elif op == RESTORE:
                if rows or capacity:
                    raise ValueError('restore takes no output')
                self._restore_snapshot(backend, io.BytesIO(C.string_at(src, size)))
                return OK
            else:
                raise ValueError('unknown operation')
            if len(data) > capacity:
                raise OverflowError('output exceeds transfer capacity')
            if data:
                C.memmove(dst, data, len(data))
            written[0] = len(data)
            return OK
        except BaseException as exc:
            # ctypes must never receive an escaping Python exception.
            errors[:] = [exc]
            if isinstance(exc, InterruptedError):
                return CANCELLED
            if isinstance(exc, (OverflowError, MemoryError)):
                return LIMIT
            if isinstance(exc, (ValueError, TypeError)):
                return INVALID
            return CALLBACK_ERROR

    def call(self, handle, operation, data=b'', *, rows=0, capacity=0, cancelled=None):
        _uint(rows, 'rows')
        _uint(operation, 'operation', 1)
        _uint(capacity, 'capacity')
        if operation > (1 << 32) - 1:
            raise ValueError('operation exceeds uint32')
        if not isinstance(data, bytes):
            raise TypeError('data must be bytes')
        if (cancelled is not None and not callable(cancelled) and
                not callable(getattr(cancelled, 'is_set', None))):
            raise TypeError('ABI cancellation owner required')
        if cancelled is not None and operation not in (FORWARD, EMBED, HEAD):
            raise ValueError('cancellation is supported only for execution operations')
        if len(data) > self.max_bytes or capacity > self.max_bytes:
            raise SegmentABIError(LIMIT)
        with self.lock:
            record = self._record(handle)
            if self._active:
                raise SegmentABIError(BUSY)
            # A cancelled session forward is a failed transaction, not a
            # partially committed continuation. Capture only when a
            # cancellation owner was supplied; non-cancellable ABI calls keep
            # their historical low-overhead path.
            rollback = None
            if (cancelled is not None and operation == FORWARD and
                    isinstance(record[0], self.session_types)):
                output = _BoundedOutput(self.max_bytes)
                self._write_snapshot(record[0], output)
                rollback = output.getvalue()
            src = C.create_string_buffer(data) if data else None
            dst = C.create_string_buffer(capacity) if capacity else None
            written = U64()
            record[2].clear()
            self._active.add(handle)
            self._cancelled = cancelled
            try:
                status = self.lib.oa_segment_call(handle, operation, rows, src,
                    len(data), dst, capacity, C.byref(written))
                if status == CANCELLED and rollback is not None:
                    try:
                        self._restore_snapshot(record[0], io.BytesIO(rollback))
                    except BaseException as rollback_error:
                        # Do not permit continuation from state whose rollback
                        # could not be established. Release remains available
                        # so callers can still clean up the native handle.
                        self._quarantined.add(handle)
                        raise SegmentABIError(CALLBACK_ERROR) from rollback_error
                self._status(status, record[2][0] if record[2] else None)
                if written.value > capacity:
                    raise SegmentABIError(LIMIT)
                return dst.raw[:written.value] if dst else b''
            finally:
                self._active.remove(handle)
                self._cancelled = None
                record[2].clear()

    def embeddings(self, edge, token_ids, *, cancelled=None):
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in 'iu':
            raise ValueError('expected integer token vector')
        if (cancelled is not None and not callable(cancelled) and
                not callable(getattr(cancelled, 'is_set', None))):
            raise TypeError('embedding cancellation owner required')
        if ids.dtype.kind == 'u' and np.any(ids > np.iinfo(np.int64).max):
            raise ValueError('token exceeds int64')
        with self.lock:
            width = self._record(edge)[0].config.hidden_size
            if ids.size * max(8, width * 4) > self.max_bytes:
                raise SegmentABIError(LIMIT)
            data = self.call(edge, EMBED, ids.astype('<i8').tobytes(), rows=len(ids),
                capacity=len(ids)*width*4, cancelled=cancelled)
            return np.frombuffer(data, '<f4').astype(np.float32).reshape(len(ids), width)

    def hidden(self, handle, value, *, final_head=False, cancelled=None):
        array = np.asarray(value)
        if array.ndim != 2 or array.dtype != np.dtype('float32') or not np.isfinite(array).all():
            raise ValueError('expected finite float32 matrix')
        with self.lock:
            backend = self._record(handle)[0]
            config = backend.config if final_head else backend._engine.config
            if array.shape[1] != config.hidden_size:
                raise ValueError('hidden width mismatch')
            width = config.vocab_size if final_head else config.hidden_size
            if array.size * 4 > self.max_bytes or len(array)*width*4 > self.max_bytes:
                raise SegmentABIError(LIMIT)
            data = self.call(handle, HEAD if final_head else FORWARD,
                array.astype('<f4', copy=False).tobytes(), rows=len(array),
                capacity=len(array)*width*4, cancelled=cancelled)
            return np.frombuffer(data, '<f4').astype(np.float32).reshape(len(array), width)

    def snapshot(self, session):
        return self.call(session, SNAPSHOT, capacity=self.max_bytes)

    def restore(self, session, data):
        self.call(session, RESTORE, data)

    def reset(self, session):
        self.call(session, RESET)

    def release(self, handle):
        with self.lock:
            record = self._record(handle, allow_quarantined=True)
            if self._active:
                raise SegmentABIError(BUSY)
            record[2].clear()
            self._active.add(handle)
            try:
                self._status(self.lib.oa_segment_release(handle), record[2][0] if record[2] else None)
                del self._objects[handle]
                self._quarantined.discard(handle)
                with _locks_guard:
                    _backends.discard(id(record[0]))
                    if not self._objects:
                        _owners.discard(self)
            finally:
                self._active.remove(handle)
                record[2].clear()

    def close(self):
        """Release children first, continuing independent cleanup after failures.

        Successful close is terminal and idempotent. A failed close retains
        live handles/callbacks and may be retried after fixing the cause.
        """
        with self.lock:
            if self._active:
                raise SegmentABIError(BUSY)
            handles = sorted(self._objects, key=lambda handle: not isinstance(
                self._objects[handle][0], self.session_types))
            failures = []
            for handle in handles:
                try:
                    self.release(handle)
                except SegmentABIError as error:
                    failures.append((handle, error))
            if failures:
                raise SegmentABICloseError(failures, self._objects) from failures[0][1]
            self._closed = True

    def __enter__(self):
        with self.lock:
            if self._closed:
                raise RuntimeError('segment bridge is closed')
            return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            BaseException.add_note(exc, f'SegmentBridge cleanup failed: {cleanup_error}; '
                                   'live callbacks are retained; retry bridge.close().')
        return False
