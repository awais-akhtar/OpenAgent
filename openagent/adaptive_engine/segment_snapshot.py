"""Portable segment snapshots: magic, bounded JSON, LE FP32 data, SHA-256.

Callers serialize access to the session. Streams need not support seeking;
the input must end immediately after the checksum. max_bytes bounds both
the wire length and KV staging storage (including restore_state's copies).
"""

import hashlib
import hmac
import json
import struct
from functools import wraps

import numpy as np

from .olmoe_segments import NUMERIC_CLASS, OlmoeSegmentSession


_MAGIC = b'OASEG\x00\x02\x00'
_HEADER_LIMIT = 64 * 1024
_CHUNK = 1024 * 1024
_DEFAULT_LIMIT = 1024 * 1024 * 1024


def _session(session):
    if not isinstance(session, OlmoeSegmentSession):
        raise TypeError('expected OlmoeSegmentSession')
    session._check()


def _serialized(function):
    """Hold the session owner lease for the complete codec transaction."""
    @wraps(function)
    def call(session, *args, **kwargs):
        if not isinstance(session, OlmoeSegmentSession):
            raise TypeError('expected OlmoeSegmentSession')
        with session._lock:
            return function(session, *args, **kwargs)
    return call


def _identity(session):
    engine = session._engine
    return dict(format='openagent.olmoe-segment.v2',
                fingerprint=engine.manifest.fingerprint,
                execution_identity=engine.execution_identity,
                range=[engine.begin, engine.end], numeric_class=NUMERIC_CLASS,
                context_tokens=session.context_tokens)


def _write(output, data):
    view = memoryview(data).cast('B')
    while view:
        count = output.write(view[:_CHUNK])
        if type(count) is not int or not 0 < count <= min(len(view), _CHUNK):
            raise ValueError('invalid binary write result')
        view = view[count:]


def _read(source, size):
    result = bytearray()
    while len(result) < size:
        part = source.read(min(size - len(result), _CHUNK))
        if not isinstance(part, bytes) or not part or len(part) > size - len(result):
            raise ValueError('truncated or invalid binary input')
        result.extend(part)
    return result


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate header field')
        result[key] = value
    return result


def _chunks(array):
    # buffered iteration also bounds conversion of noncontiguous cache slices.
    iterator = np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                         op_flags=['readonly'], op_dtypes=['<f4'],
                         order='C', buffersize=_CHUNK // 4)
    for chunk in iterator:
        if not np.isfinite(chunk).all():
            raise ValueError('nonfinite KV array')
        yield chunk.tobytes(order='C')


@_serialized
def write_segment_snapshot(session, binary_output):
    """Write the active KV prefix without making a full snapshot copy."""
    _session(session)
    position = session.position
    if type(position) is not int or not 0 <= position <= session.context_tokens:
        raise ValueError('invalid session position')
    arrays = [a[:, :position] for a in session._k + session._v]
    header = dict(_identity(session), version=2, position=position,
                  shapes=[list(a.shape) for a in arrays],
                  lengths=[a.size * 4 for a in arrays])
    encoded = json.dumps(header, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(encoded) > _HEADER_LIMIT:
        raise ValueError('header too large')
    # Validate before emitting anything; the second pass streams the bytes.
    for array in arrays:
        if array.dtype != np.dtype('float32'):
            raise ValueError('invalid KV dtype')
        for _ in _chunks(array):
            pass
    digest = hashlib.sha256()
    for data in (_MAGIC, struct.pack('<I', len(encoded)), encoded):
        digest.update(data)
        _write(binary_output, data)
    for array in arrays:
        for data in _chunks(array):
            digest.update(data)
            _write(binary_output, data)
    _write(binary_output, digest.digest())


@_serialized
def restore_segment_snapshot(session, binary_input, max_bytes=_DEFAULT_LIMIT):
    """Validate an entire snapshot before committing it to the session."""
    _session(session)
    if type(max_bytes) is not int or max_bytes < 44:
        raise ValueError('invalid max_bytes')
    prefix = _read(binary_input, 12)
    if prefix[:8] != _MAGIC:
        raise ValueError('invalid snapshot magic or version')
    length = struct.unpack('<I', prefix[8:])[0]
    if not 0 < length <= min(_HEADER_LIMIT, max_bytes - 44):
        raise ValueError('invalid header length')
    encoded = _read(binary_input, length)
    try:
        header = json.loads(encoded, object_pairs_hook=_pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError('invalid snapshot JSON') from exc
    identity = _identity(session)
    if type(header) is not dict or set(header) != set(identity) | {'version', 'position', 'shapes', 'lengths'}:
        raise ValueError('invalid header fields')
    for field, expected in identity.items():
        if type(header[field]) is not type(expected) or header[field] != expected:
            raise ValueError(f'snapshot {field} mismatch')
    if any(type(i) is not int for i in header['range']):
        raise ValueError('invalid range')
    position = header['position']
    if type(header['version']) is not int or header['version'] != 2:
        raise ValueError('unsupported snapshot version')
    if type(position) is not int or not 0 <= position <= session.context_tokens:
        raise ValueError('invalid position')
    targets = session._k + session._v
    shapes = [[a.shape[0], position, a.shape[2]] for a in targets]
    lengths = [s[0] * s[1] * s[2] * 4 for s in shapes]
    if (type(header['shapes']) is not list or header['shapes'] != shapes
            or any(type(s) is not list or any(type(d) is not int for d in s)
                   for s in header['shapes'])
            or type(header['lengths']) is not list or header['lengths'] != lengths
            or any(type(n) is not int for n in header['lengths'])):
        raise ValueError('invalid KV dimensions or lengths')
    payload = sum(lengths)
    if (44 + length + payload > max_bytes
            or payload + sum(a.nbytes for a in targets) > max_bytes):
        raise ValueError('snapshot exceeds max_bytes')
    digest = hashlib.sha256(prefix + encoded)
    arrays = []
    for shape, size in zip(shapes, lengths):
        array = np.empty(shape, dtype=np.float32)
        flat = array.reshape(-1)
        for offset in range(0, size, _CHUNK):
            data = _read(binary_input, min(_CHUNK, size - offset))
            digest.update(data)
            values = np.frombuffer(data, dtype='<f4')
            if not np.isfinite(values).all():
                raise ValueError('nonfinite KV array')
            flat[offset // 4:offset // 4 + len(values)] = values
        arrays.append(array)
    if not hmac.compare_digest(digest.digest(), _read(binary_input, 32)):
        raise ValueError('snapshot checksum mismatch')
    if binary_input.read(1) != b'':
        raise ValueError('extra snapshot data')
    state = dict(identity, range=tuple(identity['range']), position=position,
                 keys=arrays[:len(session._k)], values=arrays[len(session._k):])
    session.restore_state(state)
