"""Complete DSpark target evidence plus its existing packed backbone codec."""
import hashlib
import hmac
import json
import struct
import tempfile

import numpy as np

from .mtp_cache import _read, _write, _unique, _budget
from ..checkpoint_stream import CheckpointStream

_PREFIX = struct.Struct('<8sI')
_MAGIC = b'OADSP001'


def transport_backbone_cache(backbone, stream, *, max_bytes, restoring=False):
    _budget(max_bytes)
    model = backbone.model
    stream = CheckpointStream(stream, model._cancel_rows)
    with backbone._lock, model._lock:
        backbone._check_transport()
        if model._busy or model._closed:
            raise RuntimeError('DSpark target is unavailable')
        width = len(backbone.layers)*model.profile.hidden_size
        backbone._cache_transport = True
        try:
            with tempfile.TemporaryFile() as packed:
                if restoring:
                    model._busy = True
                    try:
                        prefix = _read(stream, _PREFIX.size)
                        magic, size = _PREFIX.unpack(prefix)
                        if magic != _MAGIC or not 0 < size <= 65536:
                            raise ValueError('invalid DSpark cache header')
                        if _PREFIX.size+size+32 > max_bytes:
                            raise MemoryError('DSpark cache metadata exceeds frame admission')
                        raw = _read(stream, size)
                        header = json.loads(raw, object_pairs_hook=_unique,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite metadata')))
                        if (type(header) is not dict or set(header) !=
                                {'version', 'identity', 'layers', 'width', 'position', 'packed_bytes'} or
                                type(header['version']) is not int or header['version'] != 1 or
                                header['identity'] != model.identity or
                                type(header['layers']) is not list or
                                any(type(layer) is not int for layer in header['layers']) or
                                tuple(header['layers']) != backbone.layers or
                                type(header['width']) is not int or header['width'] != width or
                                type(header['position']) is not int or
                                not 0 <= header['position'] <= model.profile.context or
                                type(header['packed_bytes']) is not int or header['packed_bytes'] <= 0):
                            raise ValueError('DSpark cache identity or geometry mismatch')
                        count = header['position']
                        evidence = count*width*4+(4*model.profile.vocabulary if count else 0)
                        total = _PREFIX.size+size+header['packed_bytes']+evidence+32
                        if total > max_bytes:
                            raise MemoryError('DSpark cache frame exceeds admission')
                        retained = len(backbone.hidden)*width*4+(0 if backbone.logits is None else backbone.logits.nbytes)
                        if retained+2*evidence > backbone.max_bytes:
                            raise MemoryError('DSpark recovery evidence exceeds retention admission')
                        digest = hashlib.sha256(prefix+raw)
                        remaining = header['packed_bytes']
                        while remaining:
                            data = _read(stream, min(65536, remaining))
                            digest.update(data)
                            _write(packed, data)
                            remaining -= len(data)
                        hidden = []
                        for _ in range(count):
                            row = _read(stream, width*4)
                            if not np.isfinite(np.frombuffer(row, dtype='<f4')).all():
                                raise ValueError('nonfinite DSpark hidden evidence')
                            digest.update(row)
                            hidden.append(row)
                        raw_logits = _read(stream, 4*model.profile.vocabulary) if count else b''
                        digest.update(raw_logits)
                        logits = np.frombuffer(raw_logits, dtype='<f4').copy() if count else None
                        if logits is not None and not np.isfinite(logits).all():
                            raise ValueError('nonfinite DSpark saved logits')
                        if (not hmac.compare_digest(_read(stream, 32), digest.digest()) or
                                _read(stream, 1) != b''):
                            raise ValueError('DSpark cache checksum or trailing bytes mismatch')
                    finally:
                        model._busy = False
                    old = model.snapshot()
                    packed.seek(0)
                    primary_failure = None
                    try:
                        model.restore_cache(packed, max_bytes=header['packed_bytes'])
                        if model.length != count:
                            raise ValueError('DSpark evidence differs from packed target position')
                    except BaseException as failure:
                        primary_failure = failure
                        try:
                            model.restore(old)
                        except BaseException as rollback:
                            backbone._quarantined = True
                            failure.add_note('DSpark target recovery rollback failed: '+str(rollback))
                        raise
                    finally:
                        try:
                            model.release_snapshot(old)
                        except BaseException as cleanup:
                            backbone._quarantined = True
                            if primary_failure is not None:
                                primary_failure.add_note('DSpark recovery rollback snapshot cleanup failed: '+str(cleanup))
                            else:
                                raise RuntimeError('DSpark recovery rollback snapshot cleanup failed') from cleanup
                    backbone.hidden, backbone.logits = tuple(hidden), logits
                    return count
                packed_bytes = model.write_cache(packed, max_bytes=max_bytes)
                count = model.length
                if len(backbone.hidden) != count or (backbone.logits is None) != (count == 0):
                    raise ValueError('DSpark target evidence is inconsistent')
                raw = json.dumps(dict(version=1, identity=model.identity, layers=backbone.layers,
                    width=width, position=count, packed_bytes=packed_bytes), sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode('utf-8')
                evidence = count*width*4+(4*model.profile.vocabulary if count else 0)
                if len(raw) > 65536 or _PREFIX.size+len(raw)+packed_bytes+evidence+32 > max_bytes:
                    raise MemoryError('DSpark cache frame exceeds admission')
                prefix = _PREFIX.pack(_MAGIC, len(raw))
                digest = hashlib.sha256(prefix+raw)
                packed.seek(0)
                model._busy = True
                try:
                    _write(stream, prefix)
                    _write(stream, raw)
                    remaining = packed_bytes
                    while remaining:
                        data = _read(packed, min(65536, remaining))
                        digest.update(data)
                        _write(stream, data)
                        remaining -= len(data)
                    for row in backbone.hidden:
                        digest.update(row)
                        _write(stream, row)
                    if count:
                        data = np.asarray(backbone.logits, dtype='<f4').tobytes()
                        digest.update(data)
                        _write(stream, data)
                    _write(stream, digest.digest())
                finally:
                    model._busy = False
                return count
        finally:
            backbone._cache_transport = False
