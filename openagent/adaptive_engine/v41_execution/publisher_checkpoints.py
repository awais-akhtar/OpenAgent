"""Local disk recovery for the publisher's public packed session snapshots.

This stores packed cache bytes and FP32 pending pairs, not provider weights.
SHA-256 detects corruption, not malicious replacement. Recovery calls public restore
only after the entire file is admitted, decoded and checked. Direct codec calls
hold the executor lock and reject callback reentry throughout file processing.
No private executor schema or pickle deserialization is used.
Byte budgets account for payload buffers, not Python object overhead; separate
row/token and metadata limits bound the decoded inventory.
"""
from dataclasses import asdict
from functools import wraps
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import sys

import numpy as np

from .publisher import PublisherV41Execution, PublisherV41Snapshot
from .publisher_cache import PackedPublisherRow, packed_nbytes
from ..session_recovery import _path as _safe_path
from ..native_library_identity import _stamp
from ..generation_checkpoint_files import _remove_staging


_MAGIC = b'OAP41S03'
_HEADER_LIMIT = 65536
_PREFIX = struct.Struct('<8sIQ')
_CHUNK_BYTES = 65536


def _budget(value):
    if type(value) is not int or value <= 0:
        raise ValueError('positive publisher checkpoint payload budget required')
    return value


def _model(model):
    from .device_execution import DevicePublisherV41Execution
    if type(model) not in (PublisherV41Execution,DevicePublisherV41Execution):
        raise ValueError('publisher V4.1 executor required')
    return model.profile


def _checkpoint_operation(function):
    @wraps(function)
    def locked(model, *args, **kwargs):
        _model(model)
        with model._lock:
            if model._busy or model._closed:
                raise RuntimeError('executor unavailable for publisher checkpoint')
            model._busy = True
            try:
                model._cancel_rows()
                return function(model, *args, **kwargs)
            finally:
                model._busy = False
    return locked


class _CancelledReader:
    """Keep both verification passes and small metadata reads cancellable."""
    def __init__(self, stream, cancel):
        self._stream, self._cancel = stream, cancel

    def read(self, count=-1):
        if type(count) is not int or count < -1 or count > _CHUNK_BYTES:
            raise ValueError('bounded publisher checkpoint read required')
        self._cancel()
        value = self._stream.read(count)
        if type(value) is not bytes or (count >= 0 and len(value) > count):
            raise ValueError('publisher checkpoint reader returned invalid bytes')
        return value

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _profile_identity(profile):
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'),
                               ensure_ascii=True, allow_nan=False)
    for part in encoder.iterencode(asdict(profile)):
        digest.update(part.encode('ascii'))
    return digest.hexdigest()


def _header_bytes(identity, profile_identity, position, records, maximum):
    header = dict(version=3, identity=identity, profile_sha256=profile_identity,
                  position=position, records=records)
    encoded = bytearray()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'),
                               ensure_ascii=True, allow_nan=False)
    for part in encoder.iterencode(header):
        if len(part) > maximum - len(encoded):
            raise MemoryError('publisher checkpoint metadata exceeds write admission')
        encoded.extend(part.encode('ascii'))
    return bytes(encoded)


def _read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        requested = min(count-len(result), _CHUNK_BYTES)
        data = stream.read(requested)
        if type(data) is not bytes or not data or len(data) > requested:
            raise ValueError('truncated or oversized publisher checkpoint read')
        result.extend(data)
    return bytes(result)


def _write_all(output, data):
    view = memoryview(data).cast('B')
    while view:
        written = output.write(view)
        if type(written) is not int or not 0 < written <= len(view):
            raise OSError('publisher checkpoint write made no progress')
        view = view[written:]


def _pending_bytes(records):
    return sum(count * components * values_size
        for kind, _, count, _, components, _, values_size, _ in records if kind == 3)


def _packed_extents(width, encoding):
    total = packed_nbytes(width, encoding)
    # Public cache format: FP8 has one code byte; FP4 has two codes per byte.
    values = width if encoding == 'fp8-e4m3-g32-ue8m0' else width // 2
    return values, total - values


def _layout(profile, position, maximum):
    maximum = _budget(maximum)
    if (type(profile.context) is not int or profile.context <= 0 or
            type(profile.window) is not int or profile.window <= 0):
        raise ValueError('invalid publisher checkpoint context/window geometry')
    if type(position) is not int or not 0 <= position <= profile.context:
        raise ValueError('invalid publisher checkpoint position')
    if position > maximum // 9:
        raise MemoryError('publisher checkpoint tokens exceed payload admission')
    if any(
        type(layer.layer) is not int or layer.layer != index
        for index, layer in enumerate(profile.layers)
    ):
        raise ValueError('unsupported publisher checkpoint layer inventory')
    records = []
    size = position * 9
    metadata_size = 2
    for kind in range(4):
        for layer in profile.layers:
            ratio = layer.compression_ratio
            if type(ratio) is not int or ratio < 0:
                raise ValueError('invalid publisher checkpoint compression ratio')
            owned = bool(ratio and layer.kv_owner == layer.layer)
            count = min(position, profile.window) if kind == 0 else (
                position % ratio if kind == 3 and owned else
                position // ratio if owned else 0)
            width = profile.index_head_dim if kind == 2 else profile.head_dim
            components = 2 if kind == 3 else 1
            if (type(width) is not int or width <= 0 or
                    type(count) is not int or count < 0):
                raise ValueError('invalid publisher checkpoint geometry')
            if kind == 3:
                encoding, values_size, scales_size = 'f32-le', width * 4, 0
            else:
                encoding = (layer.window_encoding if kind == 0 else
                            layer.compressed_encoding if kind == 1 else layer.index_encoding)
                values_size, scales_size = _packed_extents(width, encoding)
            row_size = components * (values_size + scales_size)
            if count > (maximum - size) // row_size:
                raise MemoryError('publisher checkpoint rows exceed payload admission')
            record = [kind, layer.layer, count, width, components,
                      encoding, values_size, scales_size]
            metadata_size += len(json.dumps(record, separators=(',', ':'))) + bool(records)
            if metadata_size > _HEADER_LIMIT:
                raise MemoryError('publisher checkpoint records exceed metadata admission')
            records.append(record)
            size += count * row_size
    return records, size


def _validate_records(supplied, expected):
    if type(supplied) is not list or len(supplied) != len(expected):
        raise ValueError('publisher checkpoint record inventory mismatch')
    for actual, record in zip(supplied, expected):
        if type(actual) is not list or len(actual) != len(record):
            raise ValueError('publisher checkpoint record shape mismatch')
        for value, required in zip(actual, record):
            if type(value) is not type(required) or value != required:
                raise ValueError('publisher checkpoint cache layout mismatch')


def _resident_size(model, profile):
    # Public profile/length determine resident payloads without private state.
    existing_position = model.length
    _, existing_size = _layout(profile, existing_position,
                               _budget(model.max_bytes) + existing_position * 9)
    return existing_size


def _admit_metadata(model, profile, header_size):
    if _resident_size(model, profile) + 2 * header_size + _PREFIX.size + 64 > model.max_bytes:
        raise MemoryError('publisher checkpoint metadata exceeds recovery staging budget')


def _admit_restore(model, profile, records, size, header_size):
    existing_size = _resident_size(model, profile)
    pending_copy = _pending_bytes(records)
    # Packed bytes are shared by restore; pending rows are copied. Retain old
    # state until commit, plus the public restore's two-row conversion scratch.
    scratch = max(min(size, _CHUNK_BYTES), 8 * profile.head_dim)
    metadata = 2 * header_size + _PREFIX.size + 64
    if existing_size + size + pending_copy + scratch + metadata > model.max_bytes:
        raise MemoryError('publisher checkpoint exceeds executor restore staging budget')


def _admit_save(model, records, size, header_size):
    pending_copy = _pending_bytes(records)
    scratch = min(size, _CHUNK_BYTES) * 2
    if size + pending_copy + scratch + 2 * header_size + _PREFIX.size + 32 > _budget(model.max_bytes):
        raise MemoryError('publisher checkpoint exceeds executor write staging budget')


def _save_header_limit(model, records, size):
    available = (_budget(model.max_bytes) - size - _pending_bytes(records)
                 - min(size, _CHUNK_BYTES) * 2 - _PREFIX.size - 32)
    if available <= 0:
        raise MemoryError('publisher checkpoint exceeds executor write staging budget')
    return min(_HEADER_LIMIT, available // 2)


def _path(value, *, existing):
    path = _safe_path(value)
    if path.drive.startswith('\\\\'):
        raise ValueError('local publisher checkpoint path required')
    path = path.resolve(strict=existing)
    if path.drive.startswith('\\\\') or not path.parent.is_dir():
        raise ValueError('existing local publisher checkpoint directory required')
    return path


def _chunks(snapshot, records, profile):
    if type(snapshot.tokens) is not tuple or any(
        type(token) is not int or not 0 <= token < profile.vocabulary
        or token > 0xffffffffffffffff for token in snapshot.tokens
    ):
        raise ValueError('invalid publisher snapshot tokens')
    token_types = snapshot.token_types
    if (type(token_types) is not tuple or
            len(token_types) not in (0, len(snapshot.tokens)) or
            any(type(value) is not int or not -1 <= value <= 3 for value in token_types)):
        raise ValueError('invalid publisher snapshot token types')
    if (type(snapshot.state) is not tuple or len(snapshot.state) != 4 or
            any(type(kind) is not tuple or len(kind) != len(profile.layers)
                for kind in snapshot.state)):
        raise ValueError('invalid publisher snapshot inventory')
    for start in range(0, len(snapshot.tokens), 4096):
        part = snapshot.tokens[start:start + 4096]
        yield struct.pack('<' + 'Q' * len(part), *part)
    for start in range(0, len(snapshot.tokens), 4096):
        count = min(4096, len(snapshot.tokens) - start)
        part = token_types[start:start + count] if token_types else (-1,) * count
        yield struct.pack('<' + 'b' * count, *part)
    for kind, layer, count, width, components, encoding, values_size, scales_size in records:
        rows = snapshot.state[kind][layer]
        if type(rows) is not tuple or len(rows) != count:
            raise ValueError('publisher snapshot cache/history mismatch')
        for row in rows:
            if kind != 3:
                if (type(row) is not PackedPublisherRow or row.encoding != encoding
                        or row.size != width or type(row.size) is not int):
                    raise ValueError('publisher snapshot packed cache format mismatch')
                validated = PackedPublisherRow(row.encoding, row.size, row.values, row.scales)
                if len(validated.values) != values_size or len(validated.scales) != scales_size:
                    raise ValueError('publisher snapshot packed buffer extent mismatch')
                yield validated.values
                yield validated.scales
                continue
            values = row if components == 2 else (row,)
            if type(values) is not tuple or len(values) != components:
                raise ValueError('invalid publisher snapshot compression group')
            for value in values:
                if (type(value) is not np.ndarray or value.dtype != np.dtype('float32')
                        or value.shape != (width,)):
                    raise ValueError('invalid publisher snapshot FP32 row')
                for start in range(0, width, _CHUNK_BYTES // 4):
                    part = value[start:start + _CHUNK_BYTES // 4]
                    if not np.isfinite(part).all():
                        raise ValueError('nonfinite publisher snapshot pending row')
                    part = np.ascontiguousarray(part, dtype='<f4')
                    yield memoryview(part).cast('B')


@_checkpoint_operation
def save_publisher_v41_checkpoint(model, destination, *, max_payload_bytes):
    """Atomically replace a local file with one committed public snapshot.

    Both encoded payload bytes and decoded row/token inventory are bounded.
    The snapshot API shares immutable packed buffers and copies pending FP32
    pairs under the executor's own session policy.
    """
    profile = _model(model)
    maximum = _budget(max_payload_bytes)
    path = _path(destination, existing=False)
    # Reject over-budget current sessions before asking for a detached copy.
    position = model.length
    records, size = _layout(profile, position, maximum)
    header_limit = _save_header_limit(model, records, size)
    profile_identity = _profile_identity(profile)
    header = _header_bytes(model.identity, profile_identity, position, records, header_limit)
    _admit_save(model, records, size, len(header))
    del header, records
    snapshot = model._snapshot_locked()
    if type(snapshot) is not PublisherV41Snapshot or snapshot.identity != model.identity:
        raise ValueError('publisher snapshot identity mismatch')
    records, size = _layout(profile, len(snapshot.tokens), maximum)
    header_limit = _save_header_limit(model, records, size)
    header = _header_bytes(snapshot.identity, profile_identity, len(snapshot.tokens), records, header_limit)
    _admit_save(model, records, size, len(header))
    prefix = _PREFIX.pack(_MAGIC, len(header), size)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', buffering=0, dir=path.parent,
                                         prefix='.publisher-v41-', delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256(prefix)
            digest.update(header)
            _write_all(output, prefix)
            _write_all(output, header)
            written = 0
            for chunk in _chunks(snapshot, records, profile):
                written += len(chunk)
                if written > size:
                    raise ValueError('publisher snapshot payload extent mismatch')
                view = memoryview(chunk).cast('B')
                for start in range(0, len(view), _CHUNK_BYTES):
                    model._cancel_rows()
                    part = view[start:start + _CHUNK_BYTES]
                    digest.update(part)
                    _write_all(output, part)
            if written != size:
                raise ValueError('publisher snapshot payload extent mismatch')
            _write_all(output, digest.digest())
            output.flush()
            os.fsync(output.fileno())
        model._cancel_rows();_safe_path(path)
        os.replace(temporary, path)
        temporary = None
        if os.name != 'nt':
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        _remove_staging(temporary,sys.exc_info()[1])


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate publisher checkpoint metadata field')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite publisher checkpoint metadata constant')


def _verify_payload(stream, digest, size):
    remaining = size
    while remaining:
        data = _read_exact(stream, min(remaining, _CHUNK_BYTES))
        digest.update(data)
        remaining -= len(data)
    checksum = _read_exact(stream, 32)
    if not hmac.compare_digest(checksum, digest.digest()) or stream.read(1):
        raise ValueError('publisher checkpoint checksum mismatch or trailing bytes')
    return checksum


def _read_history(stream, digest, position, vocabulary, *, types=False):
    code, width = ('b', 1) if types else ('Q', 8)
    for start in range(0, position, 4096):
        count = min(4096, position - start)
        data = _read_exact(stream, count * width)
        digest.update(data)
        for (value,) in struct.iter_unpack('<' + code, data):
            if types:
                if not -1 <= value <= 3:
                    raise ValueError('invalid publisher checkpoint token type')
            elif value >= vocabulary:
                raise ValueError('publisher checkpoint token outside vocabulary')
            yield value


def _read_state(stream, digest, records):
    state = [[] for _ in range(4)]
    for kind, _, count, width, components, encoding, values_size, scales_size in records:
        rows = []
        for _ in range(count):
            if kind != 3:
                values = _read_exact(stream, values_size)
                scales = _read_exact(stream, scales_size)
                digest.update(values)
                digest.update(scales)
                rows.append(PackedPublisherRow(encoding, width, values, scales))
                continue
            values = []
            for _ in range(components):
                data = _read_exact(stream, values_size)
                digest.update(data)
                value = np.frombuffer(data, dtype='<f4')
                for start in range(0, width, _CHUNK_BYTES // 4):
                    if not np.isfinite(value[start:start + _CHUNK_BYTES // 4]).all():
                        raise ValueError('nonfinite publisher checkpoint cache')
                if not value.dtype.isnative:
                    value = np.frombuffer(value.astype(np.float32).tobytes(), dtype=np.float32)
                values.append(value)
            rows.append(tuple(values))
        state[kind].append(tuple(rows))
    return tuple(tuple(kind) for kind in state)


@_checkpoint_operation
def load_publisher_v41_checkpoint(model, source, *, max_payload_bytes):
    """Validate a bounded file completely, then commit via public restore.

    Failed decoding/checksum checks do not mutate the executor. The executor
    owns final cache validation and its independent restore staging budget.
    Packed rows are reconstructed without decoding or requantization. Version
    1 FP32 checkpoint files are deliberately rejected, not silently converted.
    """
    profile = _model(model)
    maximum = _budget(max_payload_bytes)
    path = _path(source, existing=True)
    identity = model.identity
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0)
                         | getattr(os, 'O_NONBLOCK', 0)|getattr(os,'O_NOFOLLOW',0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('regular publisher checkpoint file required')
        if _stamp(info)!=_stamp(_safe_path(path).stat()):
            raise ValueError('publisher checkpoint replaced during opening')
        if info.st_size > _PREFIX.size + _HEADER_LIMIT + maximum + 32:
            raise MemoryError('publisher checkpoint file exceeds admission')
        stream = os.fdopen(descriptor, 'rb', buffering=0)
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        stream = _CancelledReader(stream, model._cancel_rows)
        def read_exact(count):
            return _read_exact(stream, count)

        prefix = read_exact(_PREFIX.size)
        magic, header_size, payload_size = _PREFIX.unpack(prefix)
        if magic in (b'OAP41S01', b'OAP41S02'):
            raise ValueError('legacy publisher checkpoint rejected; version 3 required')
        if magic != _MAGIC:
            raise ValueError('unsupported publisher checkpoint format')
        if not 1 <= header_size <= _HEADER_LIMIT or payload_size > maximum:
            raise MemoryError('publisher checkpoint metadata/payload exceeds admission')
        if info.st_size != _PREFIX.size + header_size + payload_size + 32:
            raise ValueError('publisher checkpoint file extent mismatch')
        _admit_metadata(model, profile, header_size)
        header_bytes = read_exact(header_size)
        try:
            header = json.loads(header_bytes.decode('ascii'), object_pairs_hook=_unique_fields,
                                parse_constant=_reject_constant)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError('invalid publisher checkpoint metadata') from error
        if (type(header) is not dict or set(header) != {
                'version', 'identity', 'profile_sha256', 'position', 'records'} or
                header['identity'] != identity or
                header['profile_sha256'] != _profile_identity(profile)):
            raise ValueError('publisher checkpoint identity/profile mismatch')
        if type(header['version']) is not int or header['version'] != 3:
            raise ValueError('unsupported publisher checkpoint version; version 3 required')
        position = header['position']
        records, expected_size = _layout(profile, position, maximum)
        _validate_records(header['records'], records)
        if expected_size != payload_size:
            raise ValueError('publisher checkpoint payload extent mismatch')
        _admit_restore(model, profile, records, expected_size, header_size)
        digest = hashlib.sha256(prefix)
        digest.update(header_bytes)
        payload_digest = digest.copy()
        payload_offset = stream.tell()
        checksum = _verify_payload(stream, digest, payload_size)
        # Rehash decoding too: an in-place change between passes must not commit.
        stream.seek(payload_offset)
        del header, header_bytes, prefix
        tokens = tuple(_read_history(stream, payload_digest, position, profile.vocabulary))
        token_types = tuple(_read_history(stream, payload_digest, position,
                                          profile.vocabulary, types=True))
        state = _read_state(stream, payload_digest, records)
        decoded_checksum = read_exact(32)
        if (not hmac.compare_digest(decoded_checksum, checksum) or
                not hmac.compare_digest(decoded_checksum, payload_digest.digest()) or stream.read(1)):
            raise ValueError('publisher checkpoint changed during recovery')
        current = os.fstat(stream.fileno())
        if _stamp(current)!=_stamp(info) or _stamp(_safe_path(path).stat())!=_stamp(info):
            raise ValueError('publisher checkpoint changed during recovery')
    _admit_restore(model, profile, records, expected_size, header_size)
    staged = PublisherV41Snapshot(identity, tokens, state, token_types)
    model._cancel_rows()
    model._restore_locked(staged, cancelled=model._cancel_rows)
