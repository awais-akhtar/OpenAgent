"""Bounded, checksummed binary transport for owned native OLMoE KV state.

Serialization is Python-owned; model execution remains C-owned. This format
is independent of the callback bridge and upstream snapshot protocols.
"""
from contextlib import ExitStack
import hashlib
import hmac
import json
import struct

import numpy as np

from .native_olmoe_segments import NativeOlmoeSegmentSession, NativeOlmoeDecoderSession, _finite
from .segment_snapshot import _write, _read, _pairs, _chunks

_MAGIC = b'OANAT\x00\x01\x00'
_HEADER_LIMIT = 65536


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def _access(session,stack):
    if not isinstance(session,(NativeOlmoeSegmentSession,NativeOlmoeDecoderSession)):
        raise TypeError('expected native OLMoE segment or decoder session')
    stack.enter_context(session._engine._lock)
    if isinstance(session,NativeOlmoeDecoderSession):
        stack.enter_context(session._edge._lock)
        target = session._session
    else:
        target = session
    stack.enter_context(target._lock)
    session._check()
    return target


def write_native_snapshot(session,output,*,max_bytes=None):
    """Stream active KV without a full-cache snapshot copy, under session locks."""
    with ExitStack() as stack:
        target = _access(session,stack)
        limit = target._engine.max_snapshot_bytes if max_bytes is None else max_bytes
        if type(limit) is not int or limit < 44:
            raise ValueError('invalid max_bytes')
        position = target.position
        native = target._native
        arrays = [a[:,:position] for bank in (native._keys,native._values) for a in bank]
        header = dict(identity=session.snapshot_metadata(),position=position,
                      shapes=[list(a.shape) for a in arrays])
        encoded = _canonical(header)
        size = 44+len(encoded)+sum(a.nbytes for a in arrays)
        if len(encoded)>_HEADER_LIMIT or size>min(limit,target._engine.max_snapshot_bytes):
            raise MemoryError('native snapshot exceeds byte budget')
        for array in arrays:
            for _ in _chunks(array):
                pass
        digest = hashlib.sha256()
        for data in (_MAGIC,struct.pack('<I',len(encoded)),encoded):
            digest.update(data)
            _write(output,data)
        for array in arrays:
            for data in _chunks(array):
                digest.update(data)
                _write(output,data)
        _write(output,digest.digest())


def restore_native_snapshot(session,source,*,max_bytes=None):
    """Check identity, bounds, finite KV, checksum and EOF before publishing."""
    with ExitStack() as stack:
        target = _access(session,stack)
        limit = target._engine.max_snapshot_bytes if max_bytes is None else max_bytes
        if type(limit) is not int or limit < 44:
            raise ValueError('invalid max_bytes')
        limit = min(limit,target._engine.max_snapshot_bytes)
        prefix = _read(source,12)
        if prefix[:8]!=_MAGIC:
            raise ValueError('invalid native snapshot magic/version')
        length = struct.unpack('<I',prefix[8:])[0]
        if not 0<length<=min(_HEADER_LIMIT,limit-44):
            raise ValueError('invalid native snapshot header length')
        encoded = _read(source,length)
        try:
            header = json.loads(encoded,object_pairs_hook=_pairs,
                                parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        except (ValueError,UnicodeError,RecursionError) as error:
            raise ValueError('invalid native snapshot header') from error
        if type(header) is not dict or set(header)!={'identity','position','shapes'}:
            raise ValueError('invalid native snapshot fields')
        if _canonical(header['identity'])!=_canonical(session.snapshot_metadata()):
            raise ValueError('native snapshot identity mismatch')
        pos = header['position']
        if type(pos) is not int or not 0<=pos<=target.context_tokens:
            raise ValueError('invalid native snapshot position')
        native = target._native
        shapes = [[a.shape[0],pos,a.shape[2]] for bank in (native._keys,native._values) for a in bank]
        if _canonical(header['shapes'])!=_canonical(shapes):
            raise ValueError('invalid native snapshot dimensions')
        payload = sum(int(np.prod(s,dtype=object))*4 for s in shapes)
        if 44+length+payload>limit or payload+native.kv_bytes>limit:
            raise MemoryError('native restore staging exceeds byte budget')
        digest = hashlib.sha256(prefix+encoded)
        arrays = []
        for shape in shapes:
            size = int(np.prod(shape,dtype=object))*4
            data = _read(source,size)
            digest.update(data)
            array = np.frombuffer(data,dtype='<f4').reshape(shape)
            if not _finite(array):
                raise ValueError('nonfinite native snapshot KV')
            arrays.append(array)
        if not hmac.compare_digest(digest.digest(),_read(source,32)):
            raise ValueError('native snapshot checksum mismatch')
        if source.read(1)!=b'':
            raise ValueError('extra native snapshot data')
        state = dict(target.snapshot_metadata(),position=pos,
                     keys=arrays[:len(native._keys)],values=arrays[len(native._keys):])
        if isinstance(session,NativeOlmoeDecoderSession):
            state = dict(format='openagent.native-olmoe-decoder.v1',
                         edge_execution_identity=session._edge.execution_identity,range_state=state)
        session.restore_state(state)
