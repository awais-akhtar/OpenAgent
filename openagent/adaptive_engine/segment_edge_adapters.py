"""Register OpenAgent native OLMoE kernels through public Segment/Edge layouts.

Python handles admission, file leases and snapshot transport. Embedding,
attention, routing, expert products and final logits use the existing C kernels.
Registration callbacks are pinned for the loaded library's process lifetime,
because the public registry has no unregister operation.
"""
import ctypes as C
import copy
import json
from pathlib import Path
import threading
from contextlib import contextmanager

import numpy as np

from .segment_edge_runtime import (
    _Record, _PTR, _SIZE, _U32, _TRANSFER, _SegmentCaps, _EdgeCaps,
    _SegmentOptions, _SessionOptions, _EdgeOptions, _Run, _Embed, _Logits, _Select,
    _Functions, _identifier, _integer, _cancel_requested,
)


_VOID = C.CFUNCTYPE(None, _PTR)
_SOPEN = C.CFUNCTYPE(C.c_int, C.POINTER(_PTR), C.POINTER(_SegmentCaps),
                    C.POINTER(_SegmentOptions), _PTR, _SIZE)
_CREATE = C.CFUNCTYPE(C.c_int, _PTR, C.POINTER(_PTR), C.POINTER(_SessionOptions), _PTR, _SIZE)
_RUN = C.CFUNCTYPE(C.c_int, _PTR, C.POINTER(_Run), _PTR, _SIZE)
_SNAPSHOT = C.CFUNCTYPE(C.c_int, _PTR, _TRANSFER, _PTR, _PTR, _SIZE)
_EOPEN = C.CFUNCTYPE(C.c_int, C.POINTER(_PTR), C.POINTER(_EdgeCaps),
                    C.POINTER(_EdgeOptions), _PTR, _SIZE)
_TOKENIZE = C.CFUNCTYPE(C.c_int, _PTR, _PTR, _SIZE, _PTR, _SIZE, C.POINTER(_SIZE), _PTR, _SIZE)
_DETOKENIZE = _TOKENIZE
_EMBED = C.CFUNCTYPE(C.c_int, _PTR, C.POINTER(_Embed), _PTR, _SIZE)
_LOGITS = C.CFUNCTYPE(C.c_int, _PTR, C.POINTER(_Logits), _PTR, _SIZE)
_SELECT = C.CFUNCTYPE(C.c_int, _PTR, C.POINTER(_Select), _PTR, _SIZE)


class _SegmentAdapter(_Record):
    _fields_ = [("size", _U32), ("version", _U32), ("engine", C.c_char_p),
                ("open", _SOPEN), ("destroy", _VOID), ("create", _CREATE),
                ("destroy_session", _VOID), ("run", _RUN),
                ("snapshot", _SNAPSHOT), ("restore", _SNAPSHOT),
                ("reserved", _PTR * 8)]


class _EdgeAdapter(_Record):
    _fields_ = [("size", _U32), ("version", _U32), ("engine", C.c_char_p),
                ("open", _EOPEN), ("destroy", _VOID),
                ("tokenize", _TOKENIZE), ("detokenize", _DETOKENIZE),
                ("embed", _EMBED), ("select", _SELECT), ("logits", _LOGITS),
                ("reserved", _PTR * 7)]


_PINNED = set()
_PIN_LOCK = threading.Lock()
_WIRE_SCHEMA = b"openagent.olmoe.hidden.f32.v1"
_NUMERIC_CLASS = b"openagent.olmoe.native-fp32-boundary-double-dot.v1"


def _error(pointer, capacity, error):
    if pointer and capacity:
        message = (str(error) or type(error).__name__).encode("utf-8", errors="replace")[:capacity-1]
        C.memmove(pointer, message + b"\0", len(message)+1)
    return -1


class _WriteStream:
    def __init__(self, callback, user, limit):
        self.callback, self.user, self.limit = callback, user, limit
        self.count = 0
        self._owner_thread = None
        self._active = False

    def write(self, data):
        thread = threading.get_ident()
        if self._active or (self._owner_thread is not None and self._owner_thread != thread):
            raise RuntimeError("native snapshot writer callback reentry or thread handoff")
        self._owner_thread, self._active = thread, True
        try:
            return self._write(data)
        finally:
            self._active = False

    def _write(self, data):
        if len(data) > self.limit-self.count:
            raise MemoryError("snapshot exceeds adapter transport admission")
        raw = bytes(data)
        buffer = C.create_string_buffer(raw, max(1, len(raw)))
        if self.callback(self.user, buffer, len(raw)):
            raise OSError("native snapshot writer rejected data")
        self.count += len(raw)
        return len(raw)


class _ReadStream:
    """Read one framed OpenAgent snapshot through an exact-count callback."""
    def __init__(self, callback, user, limit):
        self.callback, self.user, self.limit = callback, user, limit
        self.count = 0
        self.extent = None
        self.header_bytes = None
        self._owner_thread = None
        self._active = False

    def read(self, count):
        thread = threading.get_ident()
        if self._active or (self._owner_thread is not None and self._owner_thread != thread):
            raise RuntimeError("native snapshot reader callback reentry or thread handoff")
        self._owner_thread, self._active = thread, True
        try:
            return self._read(count)
        finally:
            self._active = False

    def _read(self, count):
        if self.extent is not None and self.count == self.extent:
            return b""
        if type(count) is not int or count < 0 or count > self.limit-self.count:
            raise MemoryError("snapshot read exceeds adapter admission")
        if self.extent is not None and count > self.extent-self.count:
            raise ValueError("snapshot read exceeds its declared frame")
        buffer = C.create_string_buffer(max(1, count))
        if self.callback(self.user, buffer, count):
            raise OSError("native snapshot reader rejected data")
        raw = buffer.raw[:count]
        if self.count == 0:
            if count != 12 or raw[:8] != b"OANAT\0\1\0":
                raise ValueError("unexpected native snapshot frame")
            self.header_bytes = int.from_bytes(raw[8:], "little")
            if not 0 < self.header_bytes <= 65536:
                raise ValueError("native snapshot header extent exceeds admission")
        elif self.count == 12:
            if count != self.header_bytes:
                raise ValueError("snapshot header read differs from its declared length")
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate native snapshot header field")
                    result[key] = value
                return result
            document = json.loads(
                raw, object_pairs_hook=unique,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if (type(document) is not dict or
                    set(document) != {'identity', 'position', 'shapes'} or
                    type(document['identity']) is not dict or
                    type(document['position']) is not int or document['position'] < 0):
                raise ValueError("invalid native snapshot identity or boundary")
            shapes = document['shapes']
            if type(shapes) is not list or len(shapes) > 8192:
                raise ValueError("invalid native snapshot array inventory")
            payload = 0
            for shape in shapes:
                if type(shape) is not list or len(shape) != 3:
                    raise ValueError("invalid native snapshot shape")
                elements = 1
                for dimension in shape:
                    _integer(dimension, "snapshot dimension", minimum=0)
                    elements *= dimension
                    if elements > self.limit // 4:
                        raise MemoryError("snapshot array exceeds admission")
                payload += elements*4
            self.extent = 44+self.header_bytes+payload
            if self.extent > self.limit:
                raise MemoryError("snapshot frame exceeds admission")
        self.count += count
        return raw


class _PublicOlmoeSession:
    """Serialize public session operations and reject callback-stack reentry."""

    def __init__(self, model):
        self.model = model
        self._lock = threading.RLock()
        self._operation_owner = None

    @contextmanager
    def operation(self):
        with self._lock:
            owner = threading.get_ident()
            if self._operation_owner == owner:
                raise RuntimeError("reentrant public OLMoE session operation rejected")
            self._operation_owner = owner
            try:
                yield
            finally:
                self._operation_owner = None

    def close(self):
        with self.operation():
            self.model.close()


class NativeOlmoePublicAdapters:
    """Own callbacks and factory-created native resources for one model root."""
    wire_schema = _WIRE_SCHEMA
    numeric_class = _NUMERIC_CLASS

    def _create_edge(self, budget):
        from .native_olmoe_segments import NativeOlmoeEdge
        return NativeOlmoeEdge(self.root, self.kernel, budget, max_bytes=self.staging_bytes,
                               max_conversion_bytes=self.conversion_bytes)
    def __init__(self, runtime_library, *, kernel_library, model_directory, manifest,
                 segment_id="openagent-olmoe-streamed", edge_id="openagent-olmoe-edge",
                 expert_cache_bytes, max_staging_bytes, max_snapshot_bytes,
                 max_batch_rows=32, tokenizer_path=None, max_resource_bytes=None,
                 max_conversion_bytes=None):
        self.root = Path(model_directory).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("allowed checkpoint directory required")
        self.staging_bytes = _integer(max_staging_bytes, "max_staging_bytes", maximum=(1 << 63)-1)
        self.conversion_bytes = _integer(
            self.staging_bytes if max_conversion_bytes is None else max_conversion_bytes,
            "max_conversion_bytes", maximum=(1 << 63)-1)
        from .expert_runtime import SparseExpertManifest
        self.manifest = SparseExpertManifest.read(manifest) if isinstance(manifest, (str, Path)) else copy.deepcopy(manifest)
        if not isinstance(self.manifest, SparseExpertManifest):
            raise TypeError("explicit SparseExpertManifest required")
        self.manifest.validate()
        if self.manifest.architecture != 'olmoe':
            raise ValueError('public OLMoE adapter requires an OLMoE expert manifest')
        if not self.manifest.layers:
            raise ValueError('public OLMoE adapter requires a nonempty expert manifest')
        # The public ABI owner is a checkpoint-loading entrypoint too. Bind
        # complete source headers and the manifest's exact persisted format
        # before loading either native library; the CLI gate alone is
        # bypassable by callers that construct this adapter directly.
        from .source_scope import require_checkpoint_inventory_admission
        inventory = require_checkpoint_inventory_admission(
            self.root, family='olmoe', model_type='olmoe',
            backend='public-range', arithmetic='fp32',
            max_metadata_bytes=self.staging_bytes,
            persisted_format=self.manifest.layers[0].dtype)
        from .olmoe import OlmoeConfig, olmoe_metadata_fingerprint, validate_olmoe_tensors
        config = OlmoeConfig.read(self.root)
        if self.manifest.fingerprint != olmoe_metadata_fingerprint(self.root):
            raise ValueError('public OLMoE expert manifest belongs to a different checkpoint')
        metadata = {name: (record.shard, {'shape': record.shape, 'dtype': record.dtype})
                    for name, record in inventory.tensors.items()}
        validate_olmoe_tensors(config, metadata)
        if len(self.manifest.layers) != config.num_hidden_layers:
            raise ValueError('public OLMoE expert inventory differs from checkpoint layers')
        payload_bytes = 3*config.hidden_size*config.intermediate_size + 4*(
            2*config.intermediate_size+config.hidden_size)
        for index, layer in enumerate(self.manifest.layers):
            if (layer.id != f'model.layers.{index}' or layer.dtype != 'int8-row' or
                    (layer.hidden_size, layer.intermediate_size, layer.experts_used) !=
                    (config.hidden_size, config.intermediate_size, config.num_experts_per_tok) or
                    set(layer.regions) != {str(expert) for expert in range(config.num_experts)} or
                    any(region.size != payload_bytes or not region.sha256
                        for region in layer.regions.values())):
                raise ValueError('public OLMoE INT8 expert manifest differs from checkpoint geometry')
        inventory.assert_unchanged()
        self.kernel = str(Path(kernel_library).resolve(strict=True))
        if not Path(self.kernel).is_file():
            raise ValueError("explicit native kernel library required")
        self.native = _Functions(runtime_library)
        self.cache_bytes = _integer(expert_cache_bytes, "expert_cache_bytes", maximum=(1 << 63)-1)
        self.snapshot_bytes = _integer(max_snapshot_bytes, "max_snapshot_bytes", maximum=(1 << 63)-1)
        self.rows = _integer(max_batch_rows, "max_batch_rows", maximum=4096)
        self.segment_id, self.edge_id = _identifier(segment_id), _identifier(edge_id)
        self.lock = threading.RLock()
        self.objects = {}
        self.pending = {}
        self._destroying = set()
        self.next_id = 1
        self._initialize_resource_ledger(max_resource_bytes)
        self.registered_segment = self.registered_edge = False
        self.tokenizer = None
        if tokenizer_path is not None:
            from tokenizers import Tokenizer
            path = Path(tokenizer_path).resolve(strict=True)
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError("tokenizer must be a local file inside the allowed checkpoint")
            if path.stat().st_size > self.staging_bytes:
                raise MemoryError("tokenizer file exceeds metadata admission")
            self.tokenizer = Tokenizer.from_file(str(path))
        self._callbacks = []
        self.segment = self._segment_record()
        self.edge = self._edge_record()
        self.native.check_layouts({9: _SegmentAdapter, 10: _EdgeAdapter})

    def _put(self, resource, kind):
        with self.lock:
            if len(self.objects) >= 65536 or self.next_id >= (1 << (C.sizeof(_PTR)*8))-1:
                raise MemoryError("adapter resource handle inventory exhausted")
            number = self.next_id
            self.next_id += 1
            self.objects[number] = (kind, resource)
            return number

    def _initialize_resource_ledger(self, max_bytes):
        from .resource_ledger import ResourceLedger
        self.resource_ledger = ResourceLedger(
            max_bytes=(1 << 63)-1 if max_bytes is None else max_bytes)
        self._construction = threading.local()

    def _get(self, number, kind):
        with self.lock:
            entry = self.objects.get(number)
            if entry is None or entry[0] != kind or number in self.pending:
                raise ValueError("unknown or cleanup-pending adapter resource")
            return entry[1]

    def _allowed(self, options):
        """Admit only the exact current open-record ABI and this registered owner."""
        reserved = (getattr(options, name) for name in
                    ('reserved', 'reserved32', 'reserved64') if hasattr(options, name))
        if (int(options.size) != C.sizeof(type(options)) or
                any(bool(value) if isinstance(value, int) else any(value)
                    for value in reserved)):
            raise ValueError("native adapter open options size or reserved fields differ")
        if not options.model or Path(options.model.decode("utf-8")).resolve(strict=True) != self.root:
            raise ValueError("adapter may open only its registered checkpoint directory")
        if options.plan or options.plan_size:
            raise ValueError("this adapter does not accept an opaque resource plan")
        if options.backend and not options.backend & (1 << 8):
            raise ValueError("this registered adapter uses native CPU kernels")

    def _guard(self, callback_type, function):
        def guarded(*arguments):
            receipt = None
            frame = None
            capabilities = None
            status = 0
            try:
                if callback_type in (_SOPEN, _EOPEN, _CREATE):
                    # A failed constructor must never leave a caller-supplied
                    # stale handle looking like a successfully published
                    # resource. Clear the ABI out-parameter before admission
                    # or any operation that can fail.
                    output_index = 1 if callback_type is _CREATE else 0
                    output = arguments[output_index]
                    if not output:
                        raise ValueError("native adapter constructor requires an output handle")
                    output[0] = None
                    if callback_type in (_SOPEN, _EOPEN):
                        capability_output = arguments[1]
                        if not capability_output:
                            raise ValueError("native adapter open requires an output capability record")
                        capability_type = _SegmentCaps if callback_type is _SOPEN else _EdgeCaps
                        C.memset(capability_output, 0, C.sizeof(capability_type))
                        capabilities = (capability_output, capability_type)
                    else:
                        # Session creation is a separate versioned ABI record
                        # from engine-open options. Validate it before using
                        # its memory budget or reserving resources.
                        raw_options = arguments[2]
                        if not raw_options:
                            raise ValueError("native adapter session creation requires options")
                        options = raw_options.contents
                        if (int(options.size) != C.sizeof(_SessionOptions) or
                                any(options.reserved)):
                            raise ValueError("native adapter session options size or reserved fields differ")
                    if getattr(self._construction, 'frame', None) is not None:
                        raise RuntimeError('native adapter construction callback reentry')
                    kind = 'session' if callback_type is _CREATE else ('edge' if callback_type is _EOPEN else 'engine')
                    receipt = self.resource_ledger.reserve(kind, int(arguments[2].contents.memory))
                    frame = [receipt, False]
                    self._construction.frame = frame
                function(*arguments[:-2])
                status = 0
            except BaseException as error:
                # A family-specific open callback may have filled its
                # capabilities before a later handle-publication failure.
                # Re-clear that output on every error path so stale-looking
                # metadata can never accompany a failed open status.
                if capabilities is not None:
                    C.memset(capabilities[0], 0, C.sizeof(capabilities[1]))
                status = _error(arguments[-2], arguments[-1], error)
            finally:
                if frame is not None:
                    self._construction.frame = None
                    if not frame[1]:
                        try:
                            self.resource_ledger.abandon(receipt)
                        except BaseException as cleanup:
                            if status == 0:
                                status = _error(arguments[-2], arguments[-1], cleanup)

            return status
        callback = callback_type(guarded)
        self._callbacks.append(callback)
        return callback

    def _destroy(self, kind):
        def destroy(number):
            with self.lock:
                entry = self.objects.get(number)
                if entry is None or entry[0] != kind:
                    return
                # Native runtimes may repeat a destroy callback while an
                # earlier callback is still unwinding.  Claim the resource
                # before releasing the registry lock so only one owner calls
                # into its close path; failures remain registered for retry.
                if number in self._destroying:
                    return
                self._destroying.add(number)
            try:
                entry[1].close()
            except BaseException as error:
                with self.lock:
                    self.pending[number] = error
            else:
                with self.lock:
                    self.objects.pop(number, None)
                    self.pending.pop(number, None)
            finally:
                with self.lock:
                    self._destroying.discard(number)
        callback = _VOID(destroy)
        self._callbacks.append(callback)
        return callback

    def retry_cleanup(self):
        with self.lock:
            order = sorted((number for number in self.pending
                            if number not in self._destroying),
                           key=lambda n: self.objects[n][0] != "session")
        errors = []
        for number in order:
            with self.lock:
                if number in self._destroying:
                    continue
                entry = self.objects.get(number)
                if entry is not None:
                    self._destroying.add(number)
            if entry is None:
                continue
            try:
                entry[1].close()
            except BaseException as error:
                errors.append(error)
                with self.lock:
                    self.pending[number] = error
            else:
                with self.lock:
                    self.objects.pop(number, None)
                    self.pending.pop(number, None)
            finally:
                with self.lock:
                    self._destroying.discard(number)
        if errors:
            raise RuntimeError("registered native adapter cleanup is incomplete") from errors[0]

    def _publish(self, resource, kind, output):
        try:
            frame = getattr(self._construction, 'frame', None)
            if frame is not None:
                resource = self.resource_ledger.adopt(frame[0], resource)
                frame[1] = True
            output[0] = self._put(resource, kind)
        except BaseException as error:
            try:
                resource.close()
            except BaseException as cleanup:
                # Keep the resource reachable for explicit cleanup even when
                # its normal opaque handle inventory could not admit it.
                with self.lock:
                    orphan = -id(resource)
                    self.objects[orphan] = (kind, resource)
                    self.pending[orphan] = cleanup
                error.add_note("adapter construction cleanup pending: "+str(cleanup))
            raise

    def _segment_record(self):
        def open_engine(output, caps, raw_options):
            from .native_leased_olmoe_segments import NativeLeasedOlmoeSegmentEngine
            options = raw_options.contents
            self._allowed(options)
            budget = _integer(int(options.memory), "explicit native memory budget", maximum=(1 << 63)-1)
            engine = NativeLeasedOlmoeSegmentEngine(self.root, self.manifest, options.begin, options.end,
                self.kernel, budget, self.cache_bytes, options.context or None,
                max_bytes=self.staging_bytes, max_staging_bytes=self.staging_bytes,
                max_snapshot_bytes=self.snapshot_bytes)
            record = _SegmentCaps(version=1, flags=(1 << 8) | 2 | 4 | 8,
                engine=self.segment_id, schema=_WIRE_SCHEMA, numeric=_NUMERIC_CLASS,
                dtype=1, width=engine.config.hidden_size, rows=self.rows,
                context=engine.context_tokens, layers=engine.config.num_hidden_layers)
            C.memmove(caps, C.byref(record), C.sizeof(record))
            self._publish(engine, "engine", output)

        def create(engine_id, output, options):
            engine = self._get(engine_id, "engine")
            budget = int(options.contents.memory)
            c = engine.config
            context = int(options.contents.context) or engine.context_tokens
            cache_bytes = 2*(engine.end-engine.begin)*c.num_key_value_heads*context*(c.hidden_size//c.num_attention_heads)*4
            if not budget or cache_bytes > budget:
                raise MemoryError("session KV exceeds explicit native memory budget")
            self._publish(_PublicOlmoeSession(
                engine.create_session(context_tokens=context)), "session", output)

        def run(session_id, raw_request):
            session = self._get(session_id, "session")
            request = raw_request.contents
            with session.operation():
                model = session.model
                if (request.size != C.sizeof(_Run) or type(request.rows) is not int or
                        not 0 < request.rows <= self.rows or request.position < 0 or
                        request.token_count != 0 or any(request.reserved) or
                        not request.input or not request.output or
                        request.input_bytes != request.rows * model._engine.config.hidden_size * 4 or
                        request.output_bytes != request.input_bytes):
                    raise ValueError("invalid counted native segment request")
                if model.position != request.position:
                    raise ValueError("native segment position differs from the requested sequence")
                width = model._engine.config.hidden_size
                view = np.ctypeslib.as_array(C.cast(request.input, C.POINTER(C.c_float)),
                                            shape=(request.rows*width,)).reshape(request.rows, width)
                if not np.isfinite(view).all():
                    raise ValueError("native segment activation input must be finite")
                cancel = (lambda: _cancel_requested(request.cancel, request.cancel_data)) if request.cancel else None
                output = model.forward(view, cancelled=cancel)
                if output.shape != view.shape or output.dtype != np.float32 or not np.isfinite(output).all():
                    raise ArithmeticError("native segment returned an invalid activation buffer")
                C.memmove(request.output, output.ctypes.data, output.nbytes)

        def snapshot(session_id, writer, user):
            from .native_segment_snapshot import write_native_snapshot
            session = self._get(session_id, "session")
            stream = _WriteStream(writer, user, self.snapshot_bytes)
            with session.operation():
                write_native_snapshot(session.model, stream, max_bytes=self.snapshot_bytes)

        def restore(session_id, reader, user):
            from .native_segment_snapshot import restore_native_snapshot
            session = self._get(session_id, "session")
            stream = _ReadStream(reader, user, self.snapshot_bytes)
            with session.operation():
                restore_native_snapshot(session.model, stream, max_bytes=self.snapshot_bytes)

        return _SegmentAdapter(version=1, engine=self.segment_id,
            open=self._guard(_SOPEN, open_engine), destroy=self._destroy("engine"),
            create=self._guard(_CREATE, create), destroy_session=self._destroy("session"),
            run=self._guard(_RUN, run), snapshot=self._guard(_SNAPSHOT, snapshot),
            restore=self._guard(_SNAPSHOT, restore))

    def _edge_record(self):
        def open_engine(output, caps, raw_options):
            options = raw_options.contents
            self._allowed(options)
            budget = _integer(int(options.memory), "explicit edge memory budget", maximum=(1 << 63)-1)
            edge = self._create_edge(budget)
            c = edge.config
            flags = (1 << 8) | 8 | 4 | (3 if self.tokenizer is not None else 0)
            record = _EdgeCaps(version=2, flags=flags, engine=self.edge_id,
                schema=self.wire_schema, numeric=self.numeric_class,
                tokenizer=b"tokenizers-json" if self.tokenizer is not None else b"",
                dtype=1, width=c.hidden_size, vocab=c.vocab_size, rows=self.rows,
                context=c.max_position_embeddings, layers=c.num_hidden_layers,
                bos=-1, eos=-1, resident_bytes=edge.resident_weight_bytes)
            C.memmove(caps, C.byref(record), C.sizeof(record))
            self._publish(edge, "edge", output)

        def embed(edge_id, raw_request):
            edge = self._get(edge_id, "edge")
            request = raw_request.contents
            if (request.size != C.sizeof(_Embed) or not 0 < request.rows <= self.rows or
                    any(request.reserved) or
                    not request.tokens or not request.output or
                    request.token_count != request.rows or
                    request.output_bytes != request.rows * edge.config.hidden_size * 4):
                raise ValueError("invalid counted native embedding request")
            if _cancel_requested(request.cancel, request.cancel_data):
                raise RuntimeError("embedding cancelled")
            ids = np.ctypeslib.as_array(C.cast(request.tokens, C.POINTER(C.c_int32)), shape=(request.rows,))
            output = edge.embeddings(ids.astype(np.int64))
            if _cancel_requested(request.cancel, request.cancel_data):
                raise RuntimeError("embedding cancelled")
            C.memmove(request.output, output.ctypes.data, output.nbytes)

        def logits(edge_id, raw_request):
            edge = self._get(edge_id, "edge")
            request = raw_request.contents
            if (request.size != C.sizeof(_Logits) or not 0 < request.rows <= self.rows or
                    any(request.reserved) or
                    not request.input or not request.output or
                    request.input_bytes != request.rows * edge.config.hidden_size * 4 or
                    # The public ABI defines logits_capacity in float
                    # elements (unlike input_bytes); the C registry enforces
                    # the same unit before dispatching this callback.
                    request.capacity < request.rows * edge.config.vocab_size):
                raise ValueError("invalid counted native logits request")
            if _cancel_requested(request.cancel, request.cancel_data):
                raise RuntimeError("logit generation cancelled")
            view = np.ctypeslib.as_array(C.cast(request.input, C.POINTER(C.c_float)),
                shape=(request.rows*edge.config.hidden_size,)).reshape(request.rows, edge.config.hidden_size)
            output = edge.final_head(view)
            if _cancel_requested(request.cancel, request.cancel_data):
                raise RuntimeError("logit generation cancelled")
            C.memmove(request.output, output.ctypes.data, output.nbytes)

        def tokenize(edge_id, pointer, size, output, capacity, count):
            # This is an ABI out-parameter, not an input. Clear it before
            # handle lookup, admission, UTF-8 decoding, or tokenizer callbacks
            # so any failure cannot leave a caller's stale count looking like
            # a successful result. On a capacity error below, publish the
            # required count deliberately to support the sizing call pattern.
            if not count:
                raise ValueError("tokenizer requires an output count")
            count[0] = 0
            self._get(edge_id, "edge")
            if (self.tokenizer is None or size > self.staging_bytes or
                    (size and not pointer)):
                raise ValueError("tokenizer unavailable or text exceeds admission")
            text = C.string_at(pointer, size).decode("utf-8", errors="strict")
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids)*4 > self.staging_bytes:
                raise MemoryError("tokenizer output exceeds admission")
            count[0] = len(ids)
            if output:
                if capacity < len(ids):
                    raise ValueError("tokenizer output capacity is too small")
                value = np.asarray(ids, dtype=np.int32)
                C.memmove(output, value.ctypes.data, value.nbytes)

        def select(edge_id, raw_request):
            edge = self._get(edge_id, "edge")
            request = raw_request.contents
            if (request.size != C.sizeof(_Select) or not 0 < request.rows <= self.rows or
                    any(request.reserved) or
                    not request.input or not request.tokens or
                    request.input_bytes != request.rows * edge.config.hidden_size * 4 or
                    request.token_capacity < request.rows or
                    (request.scores and request.score_capacity < request.rows)):
                raise ValueError("invalid counted native selection request")
            if _cancel_requested(request.cancel, request.cancel_data):
                raise RuntimeError("greedy selection cancelled")
            view = np.ctypeslib.as_array(C.cast(request.input, C.POINTER(C.c_float)),
                shape=(request.rows*edge.config.hidden_size,)).reshape(request.rows, edge.config.hidden_size)
            output = edge.final_head(view)
            if not np.isfinite(output).all():
                raise ArithmeticError("native head returned nonfinite logits")
            ids = np.argmax(output, axis=1).astype(np.int32)
            scores = output[np.arange(request.rows), ids].astype(np.float32)
            if _cancel_requested(request.cancel, request.cancel_data):
                raise RuntimeError("greedy selection cancelled")
            C.memmove(request.tokens, ids.ctypes.data, ids.nbytes)
            if request.scores:
                C.memmove(request.scores, scores.ctypes.data, scores.nbytes)

        def detokenize(edge_id, pointer, count, output, capacity, size):
            # As with tokenize, never preserve an unrelated caller-provided
            # extent when handle validation or decoding fails.
            if not size:
                raise ValueError("detokenizer requires an output size")
            size[0] = 0
            self._get(edge_id, "edge")
            if (self.tokenizer is None or count*4 > self.staging_bytes or
                    (count and not pointer)):
                raise ValueError("detokenizer unavailable or input exceeds admission")
            ids = np.ctypeslib.as_array(C.cast(pointer, C.POINTER(C.c_int32)), shape=(count,))
            value = self.tokenizer.decode(ids.tolist(), skip_special_tokens=False).encode("utf-8")
            if len(value) > self.staging_bytes:
                raise MemoryError("detokenizer output exceeds admission")
            size[0] = len(value)
            if output:
                if capacity < len(value):
                    raise ValueError("detokenizer output capacity is too small")
                C.memmove(output, value, len(value))

        return _EdgeAdapter(version=2, engine=self.edge_id,
            open=self._guard(_EOPEN, open_engine), destroy=self._destroy("edge"),
            tokenize=self._guard(_TOKENIZE, tokenize), detokenize=self._guard(_DETOKENIZE, detokenize),
            embed=self._guard(_EMBED, embed), select=self._guard(_SELECT, select),
            logits=self._guard(_LOGITS, logits))

    def register(self):
        with self.native.lock:
            if self.registered_segment and self.registered_edge:
                return self
            segment_register = self.native.library.coli_segment_adapter_register
            segment_register.argtypes = [C.POINTER(_SegmentAdapter)]
            segment_register.restype = C.c_int
            edge_register = self.native.library.coli_edge_adapter_register
            edge_register.argtypes = [C.POINTER(_EdgeAdapter)]
            edge_register.restype = C.c_int
            if not self.registered_segment:
                if segment_register(C.byref(self.segment)):
                    raise RuntimeError("native segment registry rejected registration")
                self.registered_segment = True
            with _PIN_LOCK:
                _PINNED.add(self)
            if not self.registered_edge:
                if edge_register(C.byref(self.edge)):
                    raise RuntimeError("native edge registry rejected registration; segment remains pinned")
                self.registered_edge = True
        return self

    def status(self):
        with self.lock:
            return {"segment_registered": self.registered_segment,
                    "edge_registered": self.registered_edge,
                    "resources": len(self.objects), "pending_cleanup": len(self.pending),
                    "aggregate_memory": self.resource_ledger.status(),
                    "native_arithmetic": True, "certified": False}


def retry_registered_adapter_cleanup(*, library_handle=None):
    with _PIN_LOCK:
        owners = tuple(_PINNED)
    errors = []
    for owner in owners:
        if library_handle is not None and owner.native.library._handle != library_handle:
            continue
        try:
            owner.retry_cleanup()
        except BaseException as error:
            errors.append(error)
    if errors:
        raise RuntimeError("registered native adapter cleanup remains incomplete") from errors[0]


__all__ = ["NativeOlmoePublicAdapters", "retry_registered_adapter_cleanup"]
