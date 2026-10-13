"""Explicit client for the published Segment v1 and Edge v2 native interfaces.

This module binds public data layouts and symbols, not the reference runtime
implementation. A separately built compatible library and registered model
adapters are required. It does not make OpenAgent's older bridge ABI compatible.
"""
from __future__ import annotations

import ctypes as C
import _ctypes
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
import sys
from pathlib import Path
import tempfile
import threading
import weakref

import numpy as np
from .native_library_identity import NativeLibraryIdentity
from .session_recovery import _path as _safe_path


_U32, _U64, _SIZE, _PTR = C.c_uint32, C.c_uint64, C.c_size_t, C.c_void_p
_CANCEL = C.CFUNCTYPE(C.c_int, _PTR)
_TRANSFER = C.CFUNCTYPE(C.c_int, _PTR, _PTR, _SIZE)
_ERROR_BYTES = 4096
_BACKENDS = sum(1 << bit for bit in range(8, 13))
_FAILED_OWNERS = set()
_FAILED_LIBRARIES = set()
_FAILED_LOCK = threading.Lock()
_LIBRARY_LOCKS = {}
_LIBRARY_LOCKS_GUARD = threading.Lock()


class _Record(C.Structure):
    def __init__(self, **values):
        super().__init__()
        self.size = C.sizeof(type(self))
        for name, value in values.items():
            setattr(self, name, value)


class _SegmentCaps(_Record):
    _fields_ = [("size", _U32), ("version", _U32), ("flags", _U64),
                ("engine", C.c_char * 64), ("schema", C.c_char * 128),
                ("numeric", C.c_char * 96), ("dtype", _U32), ("width", _U32),
                ("rows", _U32), ("context", _U32), ("layers", _U32),
                ("reserved32", _U32 * 7), ("reserved64", _U64 * 4)]


class _EdgeCaps(_Record):
    _fields_ = [("size", _U32), ("version", _U32), ("flags", _U64),
                ("engine", C.c_char * 64), ("schema", C.c_char * 128),
                ("numeric", C.c_char * 96), ("tokenizer", C.c_char * 64),
                ("dtype", _U32), ("width", _U32), ("vocab", _U32),
                ("rows", _U32), ("context", _U32), ("layers", _U32),
                ("bos", C.c_int32), ("eos", C.c_int32),
                ("reserved32", _U32 * 4), ("resident_bytes", _U64),
                ("reserved64", _U64 * 3)]


class _SegmentOptions(_Record):
    _fields_ = [("size", _U32), ("model", C.c_char_p), ("begin", _U32),
                ("end", _U32), ("context", _U32), ("reserved", _U32),
                ("memory", _U64), ("backend", _U64), ("plan", _PTR),
                ("plan_size", _SIZE)]


class _SessionOptions(_Record):
    _fields_ = [("size", _U32), ("context", _U32), ("memory", _U64),
                ("reserved", _U64 * 4)]


class _EdgeOptions(_Record):
    _fields_ = [("size", _U32), ("model", C.c_char_p), ("memory", _U64),
                ("backend", _U64), ("plan", _PTR), ("plan_size", _SIZE),
                ("reserved", _U64 * 3)]


class _Run(_Record):
    _fields_ = [("size", _U32), ("rows", _U32), ("position", _U64),
                ("tokens", _PTR), ("token_count", _SIZE), ("input", _PTR),
                ("input_bytes", _SIZE), ("output", _PTR), ("output_bytes", _SIZE),
                ("cancel", _CANCEL), ("cancel_data", _PTR), ("reserved", _U64 * 4)]


class _Embed(_Record):
    _fields_ = [("size", _U32), ("rows", _U32), ("tokens", _PTR),
                ("token_count", _SIZE), ("output", _PTR), ("output_bytes", _SIZE),
                ("cancel", _CANCEL), ("cancel_data", _PTR), ("reserved", _U64 * 3)]


class _Logits(_Record):
    _fields_ = [("size", _U32), ("rows", _U32), ("input", _PTR),
                ("input_bytes", _SIZE), ("output", _PTR), ("capacity", _SIZE),
                ("cancel", _CANCEL), ("cancel_data", _PTR), ("reserved", _U64 * 3)]


class _Select(_Record):
    _fields_ = [("size", _U32), ("rows", _U32), ("input", _PTR),
                ("input_bytes", _SIZE), ("tokens", _PTR), ("token_capacity", _SIZE),
                ("scores", _PTR), ("score_capacity", _SIZE), ("cancel", _CANCEL),
                ("cancel_data", _PTR), ("reserved", _U64 * 3)]


class SegmentEdgeError(RuntimeError):
    def __init__(self, operation, status, detail):
        self.operation, self.status = operation, status
        super().__init__(f"{operation} failed ({status}): {detail}")


def _integer(value, name, *, minimum=1, maximum=(1 << 32) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return value


def _identifier(value):
    if type(value) is not str or not value or "\0" in value:
        raise ValueError("adapter identifier must be a non-empty string without NUL")
    data = value.encode("utf-8")
    if len(data) >= 64:
        raise ValueError("adapter identifier exceeds the native field capacity")
    return data


def _path(value):
    path = _safe_path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("model directory required")
    data = str(path).encode("utf-8")
    if b"\0" in data:
        raise ValueError("model path contains NUL")
    return data


class _Functions:
    """Keep the library loaded while native handles and callbacks can use it."""

    def __init__(self, library):
        self.source=NativeLibraryIdentity.read(library)
        self.library = C.CDLL(str(self.source.path))
        release = _ctypes.FreeLibrary if os.name == "nt" else _ctypes.dlclose
        self._release = weakref.finalize(self, release, self.library._handle)
        with _LIBRARY_LOCKS_GUARD:
            self.lock = _LIBRARY_LOCKS.setdefault(self.library._handle, threading.RLock())
        p, s, u = _PTR, _SIZE, C.c_char_p
        specifications = {
            "segment_engine_open": [u, C.POINTER(_SegmentOptions), C.POINTER(p), p, s],
            "segment_engine_capabilities": [p, C.POINTER(_SegmentCaps), p, s],
            "segment_engine_close": [p, p, s],
            "segment_session_create": [p, C.POINTER(_SessionOptions), C.POINTER(p), p, s],
            "segment_session_destroy": [p],
            "segment_run": [p, C.POINTER(_Run), p, s],
            "segment_snapshot": [p, _TRANSFER, p, p, s],
            "segment_restore": [p, _TRANSFER, p, p, s],
            "edge_engine_open": [u, C.POINTER(_EdgeOptions), C.POINTER(p), p, s],
            "edge_engine_capabilities": [p, C.POINTER(_EdgeCaps), p, s],
            "edge_engine_close": [p],
            "edge_tokenize": [p, p, s, p, s, C.POINTER(s), p, s],
            "edge_detokenize": [p, p, s, p, s, C.POINTER(s), p, s],
            "edge_embed": [p, C.POINTER(_Embed), p, s],
            "edge_logits": [p, C.POINTER(_Logits), p, s],
            "edge_select": [p, C.POINTER(_Select), p, s],
        }
        try:
            self.source.assert_unchanged()
            initialize=getattr(self.library,'oa_segment_family_initialize',None)
            if initialize is not None:
                initialize.argtypes=[];initialize.restype=C.c_int
                with self.lock:
                    if initialize():
                        raise ValueError('native public family adapter initialization failed')
            self.check_layouts({1: _SegmentCaps, 2: _EdgeCaps, 3: _SegmentOptions,
                4: _SessionOptions, 5: _EdgeOptions, 6: _Run, 7: _Embed, 8: _Logits,
                11: _Select})
            for name, arguments in specifications.items():
                function = getattr(self.library, "coli_" + name)
                function.argtypes = arguments
                function.restype = None if name in {"segment_session_destroy", "edge_engine_close"} else C.c_int
                setattr(self, name, function)
            self.source.assert_unchanged()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                with _FAILED_LOCK:
                    _FAILED_LIBRARIES.add(self)
                error.add_note(f"native library cleanup pending: {cleanup}")
            raise

    def check_layouts(self, records):
        probe = getattr(self.library, "oa_segment_edge_layout", None)
        if probe is None:
            return False
        probe.argtypes = [_U32, C.POINTER(_U64), _SIZE]
        probe.restype = _SIZE
        for number, record in records.items():
            fields = record._fields_
            expected = [C.sizeof(record), len(fields)] + [getattr(record, name).offset for name, _ in fields]
            values = (_U64 * len(expected))()
            count = probe(number, values, len(values))
            if count != len(expected) or list(values) != expected:
                raise ValueError(f"native Segment/Edge record layout mismatch: {number}")
        return True

    def close(self):
        adapters = sys.modules.get(__package__ + ".segment_edge_adapters")
        cleanup = getattr(adapters, "retry_registered_adapter_cleanup", None)
        if callable(cleanup):
            cleanup(library_handle=self.library._handle)
        if self._release.alive:
            info = self._release.detach()
            try:
                info[1](*info[2], **info[3])
            except BaseException:
                self._release = weakref.finalize(self, info[1], *info[2], **info[3])
                raise
        with _FAILED_LOCK:
            _FAILED_LIBRARIES.discard(self)

    def call(self, name, *arguments):
        if name not in ('segment_engine_close','segment_session_destroy','edge_engine_close'):
            self.source.assert_unchanged()
        error = C.create_string_buffer(_ERROR_BYTES)
        status = getattr(self, name)(*arguments, error, len(error))
        if status:
            raise SegmentEdgeError(name, status, error.value.decode("utf-8", errors="replace"))


@dataclass(frozen=True)
class SegmentRange:
    adapter: str
    model_directory: str
    begin: int
    end: int

    def __post_init__(self):
        _identifier(self.adapter)
        if (not isinstance(self.model_directory, (str, Path)) or not str(self.model_directory)
                or "\0" in str(self.model_directory)):
            raise ValueError("segment model directory must be a bounded nonempty path")
        _integer(self.begin, "layer_begin", minimum=0)
        _integer(self.end, "layer_end", minimum=1)
        if self.end <= self.begin:
            raise ValueError("segment range must be nonempty and ordered")


@dataclass(frozen=True)
class SegmentEdgeSnapshot:
    owner: str
    number: int
    position: int
    identity: str


class _StateFile:
    def __init__(self, limit):
        self.file = tempfile.TemporaryFile(mode="w+b")
        self.limit = limit
        self.size = 0
        self.digest = None

    def close(self):
        self.file.close()

    def capture(self, native, session):
        if self.size or self.digest is not None:
            raise RuntimeError("native snapshot storage has already been captured")
        errors = []
        checksum = hashlib.sha256()

        def write(_, pointer, count):
            try:
                if count > self.limit - self.size or (count and not pointer):
                    raise MemoryError("native snapshot exceeds the admitted storage limit")
                offset = 0
                while offset < count:
                    chunk = C.string_at(pointer + offset, min(count - offset, 1024**2))
                    if self.file.write(chunk) != len(chunk):
                        raise OSError("short snapshot write")
                    checksum.update(chunk)
                    self.size += len(chunk)
                    offset += len(chunk)
                return 0
            except BaseException as error:
                errors.append(error)
                return 1

        callback = _TRANSFER(write)
        try:
            native.call("segment_snapshot", session, callback, None)
        except BaseException as error:
            if errors:
                raise errors[0] from error
            raise
        if errors:
            raise errors[0]
        self.file.flush()
        self.digest = checksum.digest()

    def restore(self, native, session):
        self.file.seek(0)
        checksum = hashlib.sha256()
        remaining = self.size
        while remaining:
            chunk = self.file.read(min(remaining, 1024**2))
            if not chunk:
                raise OSError("snapshot truncated")
            checksum.update(chunk)
            remaining -= len(chunk)
        if checksum.digest() != self.digest or self.file.read(1):
            raise ValueError("snapshot contents changed")
        self.file.seek(0)
        errors = []
        consumed = 0

        def read(_, pointer, count):
            nonlocal consumed
            try:
                if count > self.size - consumed or (count and not pointer):
                    raise ValueError("native restore requested bytes outside its snapshot")
                offset = 0
                while offset < count:
                    chunk = self.file.read(min(count - offset, 1024**2))
                    if not chunk:
                        raise OSError("short snapshot read")
                    C.memmove(pointer + offset, chunk, len(chunk))
                    offset += len(chunk)
                    consumed += len(chunk)
                return 0
            except BaseException as error:
                errors.append(error)
                return 1

        callback = _TRANSFER(read)
        try:
            native.call("segment_restore", session, callback, None)
        except BaseException as error:
            if errors:
                raise errors[0] from error
            raise
        if errors:
            raise errors[0]
        if consumed != self.size:
            raise ValueError("native restore did not consume the complete snapshot")


class SegmentEdgeExecution:
    """Local native edge/range composition with transactional sequence state.

    Supports the FP32 wire contract. Native adapters own transformer arithmetic,
    quantization and backend placement. The caller supplies those adapters; no
    missing implementation is replaced by Python model math. Snapshot storage is
    bounded separately from host activation memory and native adapter budgets.
    """

    def __init__(self, library, *, edge_adapter, model_directory, ranges,
                 checkpoint_identity, context_tokens, native_memory_bytes, max_transfer_bytes=256*1024**2,
                 max_snapshot_bytes=1024**3, max_snapshots=8, backend_mask=1 << 8):
        from itertools import islice
        from uuid import uuid4

        self._native = _Functions(library)
        self._lock = self._native.lock
        self._closed = False
        self._close_complete = False
        self._quarantined = False
        self._edge = _PTR()
        self._segments = []
        self._states = {}
        self._pending = []
        self._cancellation = threading.local()
        self._initial = None
        self._number = 0
        self._owner = uuid4().hex
        self._position = 0
        if (type(checkpoint_identity) is not str or len(checkpoint_identity) != 64
                or any(c not in "0123456789abcdef" for c in checkpoint_identity)):
            raise ValueError("checkpoint manifest SHA-256 identity required")
        self.context = _integer(context_tokens, "context_tokens")
        self._native_budget = _integer(native_memory_bytes, "native_memory_bytes", maximum=(1 << 63)-1)
        self._transfer_limit = _integer(max_transfer_bytes, "max_transfer_bytes", maximum=np.iinfo(np.intp).max)
        self._snapshot_limit = _integer(max_snapshot_bytes, "max_snapshot_bytes", maximum=(1 << 63)-1)
        self._snapshot_count = _integer(max_snapshots, "max_snapshots", maximum=64)
        _integer(backend_mask, "backend_mask", minimum=0, maximum=_BACKENDS)
        if backend_mask & ~_BACKENDS:
            raise ValueError("unknown backend mask bits")
        requested = tuple(islice(iter(ranges), 4097))
        if not requested or len(requested) > 4096:
            raise ValueError("one to 4096 contiguous layer ranges required")
        boundary = 0
        for item in requested:
            if type(item) is not SegmentRange or type(item.begin) is not int or item.begin != boundary:
                raise ValueError("ranges must partition the model from layer zero")
            _integer(item.end, "layer_end")
            if item.end <= item.begin:
                raise ValueError("empty or reversed layer range")
            boundary = item.end
        try:
            edge_options = _EdgeOptions(model=_path(model_directory), memory=self._native_budget,
                                        backend=backend_mask)
            self._native.call("edge_engine_open", _identifier(edge_adapter), C.byref(edge_options),
                              C.byref(self._edge))
            if not self._edge:
                raise ValueError("native edge open returned no handle")
            edge = _EdgeCaps()
            self._native.call("edge_engine_capabilities", self._edge, C.byref(edge))
            self._validate_caps(edge, 2, edge_adapter)
            if edge.layers != boundary or not edge.flags & (1 << 3):
                raise ValueError("edge must cover the requested model and expose logits")
            self.vocabulary = int(edge.vocab)
            _integer(self.vocabulary, "vocabulary", maximum=(1 << 31)-1)
            self._edge_caps = edge
            self._max_rows = min(int(edge.rows), self.context,
                self._transfer_limit // (4 * (2 * int(edge.width) + self.vocabulary + 1)))
            if not self._max_rows:
                raise MemoryError("one token's native transfer buffers exceed admission")
            for item in requested:
                engine, session = _PTR(), _PTR()
                entry = [engine, session, None]
                self._segments.append(entry)
                options = _SegmentOptions(model=_path(item.model_directory), begin=item.begin,
                    end=item.end, context=self.context, memory=self._native_budget, backend=backend_mask)
                self._native.call("segment_engine_open", _identifier(item.adapter), C.byref(options),
                                  C.byref(engine))
                if not engine:
                    raise ValueError("native segment open returned no handle")
                caps = _SegmentCaps()
                self._native.call("segment_engine_capabilities", engine, C.byref(caps))
                self._validate_caps(caps, 1, item.adapter)
                if (caps.schema != edge.schema or caps.numeric != edge.numeric
                        or caps.dtype != edge.dtype or caps.width != edge.width
                        or caps.layers != edge.layers or not caps.flags & (1 << 1)
                        or not caps.flags & (1 << 2)):
                    raise ValueError("segment boundary or snapshot/native-range capability mismatch")
                entry[2] = caps
                self._max_rows = min(self._max_rows, int(caps.rows))
                session_options = _SessionOptions(context=self.context, memory=self._native_budget)
                self._native.call("segment_session_create", engine, C.byref(session_options),
                                  C.byref(session))
                if not session:
                    raise ValueError("native session create returned no handle")
            self._native.source.assert_unchanged()
            self.identity = hashlib.sha256(repr((self._native.source.sha256, checkpoint_identity, edge_adapter,
                str(Path(model_directory).resolve()), requested, bytes(edge.schema),
                bytes(edge.numeric), self.context, backend_mask)).encode()).hexdigest()
            self._initial = self._capture()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                with _FAILED_LOCK:
                    _FAILED_OWNERS.add(self)
                error.add_note(f"native construction cleanup failed: {cleanup}")
            raise

    def _validate_caps(self, caps, version, adapter):
        if (caps.size != C.sizeof(type(caps)) or caps.version != version
                or caps.engine != _identifier(adapter) or caps.dtype != 1
                or not caps.width or not caps.rows or not caps.layers
                or caps.context < self.context or not caps.schema or not caps.numeric
                or not caps.flags & _BACKENDS):
            raise ValueError("native capability record does not match the required FP32 ABI")

    def _check(self):
        if getattr(self,'_cache_transport',False):
            raise RuntimeError('Segment/Edge cache transport is active')
        if self._closed:
            raise RuntimeError("Segment/Edge executor is closed")
        if self._quarantined:
            raise RuntimeError("Segment/Edge executor is quarantined after a failed restore")
        self._native.source.assert_unchanged()

    @property
    def position(self):
        with self._lock:
            self._check()
            return self._position

    def _capture(self):
        self._cleanup_pending()
        files = []
        total = sum(state.size for inventory in self._states.values() for state in inventory)
        if self._initial is not None:
            total += sum(state.size for state in self._initial)
        try:
            for _, session, _ in self._segments:
                state = _StateFile(self._snapshot_limit - total)
                files.append(state)
                state.capture(self._native, session)
                total += state.size
            return files
        except BaseException as error:
            try:
                self._dispose(files)
            except BaseException as cleanup:
                error.add_note(f"snapshot cleanup pending: {cleanup}")
            raise

    def _dispose(self, files):
        errors = []
        for state in files:
            try:
                state.close()
            except BaseException as error:
                if state not in self._pending:
                    self._pending.append(state)
                errors.append(error)
        if errors:
            raise errors[0]

    def _cleanup_pending(self):
        errors = []
        for state in tuple(self._pending):
            try:
                state.close()
            except BaseException as error:
                errors.append(error)
            else:
                self._pending.remove(state)
        if errors:
            raise errors[0]

    def _restore_files(self, files):
        try:
            for state, (_, session, _) in zip(files, self._segments, strict=True):
                state.restore(self._native, session)
        except BaseException:
            self._quarantined = True
            raise

    def write_cache(self, output, *, max_bytes):
        from .segment_edge_cache import write_segment_edge_cache
        return write_segment_edge_cache(self,output,max_bytes=max_bytes)

    def restore_cache(self, source, *, max_bytes):
        from .segment_edge_cache import restore_segment_edge_cache
        return restore_segment_edge_cache(self,source,max_bytes=max_bytes)

    def snapshot(self):
        with self._lock:
            self._check()
            if len(self._states) >= self._snapshot_count:
                raise MemoryError("public native snapshot inventory is full")
            files = self._capture()
            self._number += 1
            handle = SegmentEdgeSnapshot(self._owner, self._number, self._position, self.identity)
            self._states[handle] = files
            return handle

    def _snapshot_files(self, snapshot):
        """Return state only for the exact handle issued by this executor.

        ``SegmentEdgeSnapshot`` is a frozen dataclass, but its public fields can
        still be copied into a newly constructed value.  Dataclass equality is
        therefore not sufficient to establish ownership of the native cache
        files kept in ``_states``.
        """
        if type(snapshot) is not SegmentEdgeSnapshot:
            raise ValueError("snapshot is foreign or released")
        for issued, files in self._states.items():
            if issued is snapshot:
                return files
        raise ValueError("snapshot is foreign or released")

    def restore(self, snapshot):
        with self._lock:
            self._check()
            files = self._snapshot_files(snapshot)
            self._restore_files(files)
            self._position = snapshot.position

    def release_snapshot(self, snapshot):
        with self._lock:
            files = self._snapshot_files(snapshot)
            self._dispose(files)
            del self._states[snapshot]

    def reset(self):
        with self._lock:
            self._check()
            self._restore_files(self._initial)
            self._position = 0

    @contextmanager
    def cancellation_scope(self, cancel):
        previous = getattr(self._cancellation, "event", None)
        self._cancellation.event = cancel
        try:
            yield
        finally:
            self._cancellation.event = previous

    @property
    def max_prefill_tokens(self):
        return self._max_rows

    def embed_tokens(self,token_ids,*,cancel=None):
        """Return native Edge embeddings for publisher media token insertion."""
        from itertools import islice
        with self._lock:
            self._check()
            if cancel is None:cancel=getattr(self._cancellation,'event',None)
            ids=tuple(islice(iter(token_ids),self._max_rows+1))
            if not ids or len(ids)>self._max_rows or any(
                    type(token) is not int or not 0<=token<self.vocabulary for token in ids):
                raise ValueError('bounded in-vocabulary native embedding token rows required')
            rows,width=len(ids),int(self._edge_caps.width)
            if rows*(width*4+4)>self._transfer_limit:
                raise MemoryError('native embedding export exceeds transfer admission')
            failures=[]
            def cancelled(_):
                try:return int(cancel is not None and cancel.is_set())
                except BaseException as error:
                    failures.append(error);return 1
            callback=_CANCEL(cancelled)
            if cancelled(None):
                if failures:raise failures[0]
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('native embedding request cancelled')
            tokens=np.asarray(ids,dtype=np.int32)
            output=np.empty((rows,width),dtype=np.float32)
            request=_Embed(rows=rows,tokens=tokens.ctypes.data,token_count=rows,
                output=output.ctypes.data,output_bytes=output.nbytes,cancel=callback)
            self._native.call('edge_embed',self._edge,C.byref(request))
            if cancelled(None):
                if failures:raise failures[0]
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('native embedding request cancelled')
            if not np.isfinite(output).all():raise ArithmeticError('nonfinite native Edge embedding export')
            return output

    def forward(self, token_ids, *, cancel=None,inputs_embeds=None,return_hidden=False):
        from itertools import islice
        with self._lock:
            self._check()
            if type(return_hidden) is not bool:raise ValueError('boolean native hidden export policy required')
            if cancel is None:
                cancel = getattr(self._cancellation, "event", None)
            ids = tuple(islice(iter(token_ids), self._max_rows + 1))
            if not ids or len(ids) > self._max_rows or self._position + len(ids) > self.context:
                raise ValueError("token rows exceed native batch or context admission")
            if any(type(token) is not int or not 0 <= token < self.vocabulary for token in ids):
                raise ValueError("token IDs are outside the edge vocabulary")
            rows, width = len(ids), int(self._edge_caps.width)
            bytes_needed = rows * (2 * width + self.vocabulary + 1) * 4
            if inputs_embeds is not None:
                source=np.asarray(inputs_embeds)
                if source.shape!=(rows,width) or source.dtype!=np.float32:
                    raise ValueError('native Segment media rows must match Edge embedding geometry')
                bytes_needed+=source.nbytes
            if bytes_needed > self._transfer_limit:
                raise MemoryError("activation and logit buffers exceed host transfer admission")
            failures = []

            def cancelled(_):
                try:
                    return int(cancel is not None and cancel.is_set())
                except BaseException as error:
                    failures.append(error)
                    return 1

            callback = _CANCEL(cancelled)

            def check_cancel():
                if cancelled(None):
                    if failures:
                        raise failures[0]
                    from .execution_service import ExecutionCancelled
                    raise ExecutionCancelled("native Segment/Edge request cancelled")

            check_cancel()
            tokens = np.asarray(ids, dtype=np.int32)
            current = np.empty((rows, width), dtype=np.float32)
            scratch = np.empty_like(current)
            logits = np.empty((rows, self.vocabulary), dtype=np.float32)
            rollback = self._capture()
            primary_error = None
            try:
                if inputs_embeds is None:
                    embed = _Embed(rows=rows, tokens=tokens.ctypes.data, token_count=rows,
                        output=current.ctypes.data, output_bytes=current.nbytes, cancel=callback)
                    self._native.call("edge_embed", self._edge, C.byref(embed))
                else:
                    if not np.isfinite(source).all():raise ValueError('nonfinite native Segment media input')
                    np.copyto(current,source)
                if not np.isfinite(current).all():
                    raise ArithmeticError("native edge returned nonfinite embeddings")
                for _, session, caps in self._segments:
                    check_cancel()
                    request = _Run(rows=rows, position=self._position,
                        tokens=tokens.ctypes.data if caps.flags & 1 else None,
                        token_count=rows if caps.flags & 1 else 0,
                        input=current.ctypes.data, input_bytes=current.nbytes,
                        output=scratch.ctypes.data, output_bytes=scratch.nbytes, cancel=callback)
                    self._native.call("segment_run", session, C.byref(request))
                    if not np.isfinite(scratch).all():
                        raise ArithmeticError("native range returned nonfinite activations")
                    current, scratch = scratch, current
                check_cancel()
                request = _Logits(rows=rows, input=current.ctypes.data, input_bytes=current.nbytes,
                                  output=logits.ctypes.data, capacity=logits.size, cancel=callback)
                self._native.call("edge_logits", self._edge, C.byref(request))
                check_cancel()
                if not np.isfinite(logits).all():
                    raise ArithmeticError("native edge returned nonfinite logits")
                self._position += rows
                return (logits,current) if return_hidden else logits
            except BaseException as error:
                primary_error = error
                try:
                    self._restore_files(rollback)
                except BaseException as restore:
                    error.add_note(f"native rollback failed; executor quarantined: {restore}")
                raise
            finally:
                try:
                    self._dispose(rollback)
                except BaseException as cleanup:
                    if primary_error is not None:
                        primary_error.add_note(f"snapshot cleanup pending: {cleanup}")
                    else:
                        self._quarantined = True
                        raise

    def tokenize(self, text):
        with self._lock:
            self._check()
            if type(text) is not str or not self._edge_caps.flags & 1:
                raise ValueError("text and native tokenizer capability required")
            data = text.encode("utf-8")
            if len(data) > self._transfer_limit:
                raise MemoryError("text exceeds transfer admission")
            count = _SIZE()
            self._native.call("edge_tokenize", self._edge, data, len(data), None, 0, C.byref(count))
            if count.value > self.context or count.value * 4 > self._transfer_limit:
                raise MemoryError("tokenizer output exceeds context or transfer admission")
            if not count.value:
                return []
            tokens = np.empty(count.value, dtype=np.int32)
            capacity = count.value
            self._native.call("edge_tokenize", self._edge, data, len(data), tokens.ctypes.data,
                              capacity, C.byref(count))
            if count.value > capacity:
                raise ValueError("native tokenizer returned an invalid count")
            values = tokens[:count.value]
            if np.any(values < 0) or np.any(values >= self.vocabulary):
                raise ValueError("native tokenizer returned out-of-vocabulary IDs")
            return values.tolist()

    def select(self, hidden, *, cancel=None):
        """Apply the native final head and its deterministic greedy selection."""
        with self._lock:
            self._check()
            if not self._edge_caps.flags & 4:
                raise ValueError("native greedy selection capability required")
            vector = np.asarray(hidden)
            width = int(self._edge_caps.width)
            if (vector.dtype != np.float32 or vector.ndim != 2 or vector.shape[1] != width
                    or not 1 <= vector.shape[0] <= self._max_rows
                    or vector.nbytes*2+vector.shape[0]*8 > self._transfer_limit
                    or not np.isfinite(vector).all()):
                raise ValueError("greedy selection requires finite bounded FP32 activation rows")
            vector = np.ascontiguousarray(vector)
            tokens = np.empty(vector.shape[0], dtype=np.int32)
            scores = np.empty(vector.shape[0], dtype=np.float32)
            errors = []

            def cancelled(_):
                try:
                    return int(cancel is not None and cancel.is_set())
                except BaseException as error:
                    errors.append(error)
                    return 1

            callback = _CANCEL(cancelled)
            if cancelled(None):
                if errors:
                    raise errors[0]
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled("native greedy selection cancelled")
            request = _Select(rows=vector.shape[0], input=vector.ctypes.data, input_bytes=vector.nbytes,
                tokens=tokens.ctypes.data, token_capacity=tokens.size,
                scores=scores.ctypes.data, score_capacity=scores.size, cancel=callback)
            self._native.call("edge_select", self._edge, C.byref(request))
            if errors:
                raise errors[0]
            if cancelled(None):
                if errors:
                    raise errors[0]
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled("native greedy selection cancelled")
            if (np.any(tokens < 0) or np.any(tokens >= self.vocabulary)
                    or not np.isfinite(scores).all()):
                raise ArithmeticError("native greedy head returned invalid IDs or scores")
            return tokens, scores

    def detokenize(self, token_ids):
        from itertools import islice
        with self._lock:
            self._check()
            if not self._edge_caps.flags & 2:
                raise ValueError("native detokenizer capability required")
            values = tuple(islice(iter(token_ids), self.context + 1))
            if len(values) > self.context or any(type(t) is not int or not 0 <= t < self.vocabulary for t in values):
                raise ValueError("invalid detokenizer token inventory")
            if not values:
                return ""
            tokens = np.asarray(values, dtype=np.int32)
            length = _SIZE()
            self._native.call("edge_detokenize", self._edge, tokens.ctypes.data, tokens.size,
                              None, 0, C.byref(length))
            if length.value > self._transfer_limit:
                raise MemoryError("detokenized text exceeds transfer admission")
            if not length.value:
                return ""
            capacity = length.value
            output = C.create_string_buffer(max(1, capacity))
            self._native.call("edge_detokenize", self._edge, tokens.ctypes.data, tokens.size,
                              output, capacity, C.byref(length))
            if length.value > capacity:
                raise ValueError("native detokenizer returned an invalid byte count")
            return output.raw[:length.value].decode("utf-8", errors="strict")

    def status(self):
        with self._lock:
            return {"closed": self._closed, "quarantined": self._quarantined,
                    "position": self._position, "context": self.context,
                    "snapshots": len(self._states), "segment_abi": 1, "edge_abi": 2,
                    "pending_cleanup": len(self._pending),
                    "implementation": "public-native-adapters", "certified": False}

    def close(self):
        with self._lock:
            if getattr(self,'_cache_transport',False):
                raise RuntimeError('cannot close Segment/Edge during cache transport')
            if self._close_complete:
                return
            self._closed = True
            errors = []
            failed_sessions = set()
            for index, (_, session, _) in reversed(tuple(enumerate(self._segments))):
                if session:
                    try:
                        self._native.segment_session_destroy(session)
                    except BaseException as error:
                        errors.append(error)
                        failed_sessions.add(index)
                    else:
                        session.value = None
            for index, (engine, _, _) in reversed(tuple(enumerate(self._segments))):
                if engine:
                    if index in failed_sessions:
                        continue
                    try:
                        self._native.call("segment_engine_close", engine)
                    except BaseException as error:
                        errors.append(error)
                    else:
                        engine.value = None
            if self._edge:
                try:
                    self._native.edge_engine_close(self._edge)
                except BaseException as error:
                    errors.append(error)
                else:
                    self._edge.value = None
            inventories = list(self._states.values())
            if self._initial is not None:
                inventories.append(self._initial)
            for files in inventories:
                for state in files:
                    try:
                        state.close()
                    except BaseException as error:
                        errors.append(error)
            if errors:
                raise errors[0]
            self._cleanup_pending()
            self._states.clear()
            self._initial = None
            self._native.close()
            self._close_complete = True
            with _FAILED_LOCK:
                _FAILED_OWNERS.discard(self)


def cleanup_failed_segment_edge_executors():
    """Retry native owners retained after constructor cleanup failures."""
    with _FAILED_LOCK:
        owners = tuple(_FAILED_OWNERS)
        libraries = tuple(_FAILED_LIBRARIES)
    errors = []
    for owner in owners:
        try:
            owner.close()
        except BaseException as error:
            errors.append(error)
    for library in libraries:
        try:
            library.close()
        except BaseException as error:
            errors.append(error)
    if errors:
        raise RuntimeError("native Segment/Edge cleanup remains incomplete") from errors[0]


__all__ = ["SegmentRange", "SegmentEdgeSnapshot", "SegmentEdgeExecution", "SegmentEdgeError",
           "cleanup_failed_segment_edge_executors"]
