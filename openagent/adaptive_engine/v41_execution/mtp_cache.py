"""Bounded non-pickle recovery of DSpark windows and PCG sampling state."""
import hashlib
import hmac
import json
import struct

from .publisher_mtp import PublisherV41MTPSnapshot

_PREFIX = struct.Struct('<8sI')
_MAGIC = b'OAMTP001'


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate MTP cache metadata field')
        result[key] = value
    return result


def _rng(state, expected):
    if (type(state) is not dict or set(state) !=
            {'bit_generator', 'state', 'has_uint32', 'uinteger'} or
            state['bit_generator'] not in ('PCG64', 'PCG64DXSM') or
            state['bit_generator'] != expected or type(state['state']) is not dict or
            set(state['state']) != {'state', 'inc'}):
        raise ValueError('exact matching PCG sampling state required')
    for key in ('state', 'inc'):
        value = state['state'][key]
        if type(value) is not int or not 0 <= value < 2**128:
            raise ValueError('invalid PCG state word')
    if not state['state']['inc'] % 2:
        raise ValueError('PCG increment must be odd')
    if (type(state['has_uint32']) is not int or state['has_uint32'] not in (0, 1) or
            type(state['uinteger']) is not int or not 0 <= state['uinteger'] < 2**32):
        raise ValueError('invalid cached PCG integer')
    return state


def _budget(value):
    if type(value) is not int or value <= 0:
        raise ValueError('positive MTP cache frame budget required')


def _write(output, data):
    view = memoryview(data)
    while view:
        count = output.write(view)
        if type(count) is not int or not 0 < count <= len(view):
            raise OSError('MTP cache writer made invalid progress')
        view = view[count:]


def _read(source, count):
    result = bytearray()
    while len(result) < count:
        requested = min(65536, count-len(result))
        part = source.read(requested)
        if not isinstance(part, bytes) or not part or len(part) > requested:
            raise ValueError('truncated or oversized MTP cache read')
        result.extend(part)
    return bytes(result)


def transport_mtp_cache(model, stream, *, max_bytes, restoring=False):
    """Transport one complete frame while excluding reentrant state mutation."""
    _budget(max_bytes)
    with model._lock:
        model._check_transport()
        expected_rng = model.rng.bit_generator.state['bit_generator']
        extent = model.stages*model.config['sliding_window']*model.config['head_dim']*4
        if extent*3+262144 > model.max_bytes:
            raise MemoryError('MTP cache transport exceeds host workspace admission')
        model._cache_transport = True
        try:
            if restoring:
                prefix = _read(stream, _PREFIX.size)
                magic, size = _PREFIX.unpack(prefix)
                if magic != _MAGIC or not 0 < size <= 65536:
                    raise ValueError('invalid MTP cache header')
                if _PREFIX.size+size+extent+32 > max_bytes:
                    raise MemoryError('MTP cache frame exceeds transport admission')
                raw = _read(stream, size)
                header = json.loads(raw, object_pairs_hook=_unique,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite metadata')))
                if (type(header) is not dict or set(header) !=
                        {'version', 'identity', 'position', 'stages', 'window', 'head_dim', 'rng'} or
                        type(header['version']) is not int or header['version'] != 1 or
                        header['identity'] != model.provider.identity or
                        any(type(header[key]) is not int or header[key] != value for key, value in
                            (('stages', model.stages), ('window', model.config['sliding_window']),
                             ('head_dim', model.config['head_dim'])))):
                    raise ValueError('MTP cache identity or geometry mismatch')
                _rng(header['rng'], expected_rng)
                digest = hashlib.sha256(prefix+raw)
                windows = []
                for _ in range(model.stages):
                    window = _read(stream, extent//model.stages)
                    digest.update(window)
                    windows.append(window)
                if not hmac.compare_digest(_read(stream, 32), digest.digest()) or stream.read(1) != b'':
                    raise ValueError('MTP cache checksum or trailing bytes mismatch')
                saved = PublisherV41MTPSnapshot(header['identity'], header['position'],
                                                tuple(windows), header['rng'])
            else:
                # Snapshot is called only before external writer callbacks.
                model._cache_transport = False
                saved = model.snapshot()
                model._cache_transport = True
                _rng(saved.rng_state, expected_rng)
                raw = json.dumps(dict(version=1, identity=saved.identity, position=saved.position,
                    stages=model.stages, window=model.config['sliding_window'],
                    head_dim=model.config['head_dim'], rng=saved.rng_state),
                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
                if len(raw) > 65536 or _PREFIX.size+len(raw)+extent+32 > max_bytes:
                    raise MemoryError('MTP cache frame exceeds transport admission')
                prefix = _PREFIX.pack(_MAGIC, len(raw))
                digest = hashlib.sha256(prefix+raw)
                _write(stream, prefix)
                _write(stream, raw)
                for window in saved.windows:
                    digest.update(window)
                    for start in range(0, len(window), 65536):
                        _write(stream, memoryview(window)[start:start+65536])
                _write(stream, digest.digest())
        finally:
            model._cache_transport = False
        if restoring:
            model.restore(saved)
        return saved.position
