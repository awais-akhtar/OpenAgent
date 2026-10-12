"""Checksummed, bounded transport for native Mixtral range continuations."""
import hashlib
import hmac
import json
import struct

from .native_mixtral_ranges import MixtralRangeSnapshot
from .segment_snapshot import _write, _read, _pairs

_MAGIC = b'OAMXR\x00\x01\x00'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def write_range_cache(model, output, *, max_bytes):
    with model._lock:
        model._check()
        position = model._position
        shape = [model.batch_size, position, model._shape[2]]
        payload = 8 * (model.end-model.begin) * model.batch_size * position * model._shape[2]
        payload += model.batch_size * position
        header = _json(dict(identity=model.identity, position=position, shape=shape,
                            layers=model.end-model.begin, payload=payload))
        if type(max_bytes) is not int or len(header) > 65536 or 44+len(header)+payload > max_bytes:
            raise MemoryError('Mixtral cache exceeds transport admission')
        if model._weight_bytes+2*model._state_bytes+payload+model._expert_bytes > model.max_bytes:
            raise MemoryError('Mixtral cache copy exceeds owner admission')
        state = model.snapshot()
        digest = hashlib.sha256()
        model._busy = True
        try:
            for data in (_MAGIC, struct.pack('<I',len(header)), header):
                digest.update(data)
                _write(output,data)
            for bank in state.caches:
                for data in bank:
                    digest.update(data)
                    _write(output,data)
            digest.update(state.mask)
            _write(output,state.mask)
            _write(output,digest.digest())
        finally:
            model._busy = False


def restore_range_cache(model, source, *, max_bytes):
    with model._lock:
        model._check()
        if type(max_bytes) is not int or max_bytes < 44:
            raise ValueError('explicit Mixtral cache transport budget required')
        model._busy = True
        try:
            prefix = _read(source,12)
            if prefix[:8] != _MAGIC:
                raise ValueError('invalid Mixtral cache magic/version')
            length = struct.unpack('<I',prefix[8:])[0]
            if not 0 < length <= min(65536,max_bytes-44):
                raise ValueError('invalid Mixtral cache header extent')
            encoded = _read(source,length)
            header = json.loads(encoded,object_pairs_hook=_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if type(header) is not dict or set(header) != {'identity','position','shape','layers','payload'}:
                raise ValueError('invalid Mixtral cache header')
            position = header['position']
            if type(position) is not int or not 0 <= position <= model.capacity:
                raise ValueError('invalid Mixtral cache position')
            shape = [model.batch_size,position,model._shape[2]]
            size = 4*model.batch_size*position*model._shape[2]
            payload = 2*(model.end-model.begin)*size+model.batch_size*position
            expected = dict(identity=model.identity,position=position,shape=shape,
                            layers=model.end-model.begin,payload=payload)
            if _json(header) != _json(expected):
                raise ValueError('Mixtral cache identity/geometry mismatch')
            if 44+length+payload > max_bytes:
                raise MemoryError('Mixtral cache exceeds transport admission')
            if model._weight_bytes+3*model._state_bytes+payload+model._expert_bytes > model.max_bytes:
                raise MemoryError('Mixtral restore staging exceeds owner admission')
            digest = hashlib.sha256(prefix+encoded)
            banks = []
            for _ in range(model.end-model.begin):
                bank = []
                for _ in range(2):
                    data = _read(source,size)
                    digest.update(data)
                    bank.append(data)
                banks.append(tuple(bank))
            mask = _read(source,model.batch_size*position)
            digest.update(mask)
            if not hmac.compare_digest(digest.digest(),_read(source,32)):
                raise ValueError('Mixtral cache checksum mismatch')
            if source.read(1) != b'':
                raise ValueError('extra Mixtral cache data')
            state = MixtralRangeSnapshot(model.identity,position,tuple(banks),mask)
        finally:
            model._busy = False
        model.restore(state)


__all__ = ['write_range_cache','restore_range_cache']
