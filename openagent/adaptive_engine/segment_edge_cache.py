"""Checksummed transport of complete adapter-owned Segment continuation blobs."""
import hashlib
import hmac
import json
import struct
from contextlib import contextmanager

from .segment_snapshot import _read, _write, _pairs

_MAGIC=b'OASEKV01'
_CHUNK=1024*1024


@contextmanager
def _access(model, limit):
    from .segment_edge_runtime import SegmentEdgeExecution
    if type(model) is not SegmentEdgeExecution:
        raise TypeError('exact SegmentEdgeExecution required')
    if type(limit) is not int or limit<44:
        raise ValueError('explicit native cache transport limit required')
    with model._lock:
        model._check()
        model._cache_transport=True
        try:
            yield
        finally:
            model._cache_transport=False


def write_segment_edge_cache(model, output, *, max_bytes):
    with _access(model,max_bytes):
        files=model._capture()
        try:
            header=dict(version=1,identity=model.identity,context=model.context,position=model._position,
                        sizes=[state.size for state in files],digests=[state.digest.hex() for state in files])
            encoded=json.dumps(header,sort_keys=True,separators=(',',':')).encode('ascii')
            extent=44+len(encoded)+sum(state.size for state in files)
            if len(encoded)>65536 or extent>max_bytes:
                raise MemoryError('Segment/Edge frame exceeds admission')
            digest=hashlib.sha256()
            for data in (_MAGIC,struct.pack('<I',len(encoded)),encoded):
                digest.update(data);_write(output,data)
            for state in files:
                state.file.seek(0)
                remaining=state.size;check=hashlib.sha256()
                while remaining:
                    data=state.file.read(min(_CHUNK,remaining))
                    if not data:
                        raise ValueError('adapter snapshot scratch file truncated')
                    remaining-=len(data);check.update(data);digest.update(data);_write(output,data)
                if not hmac.compare_digest(check.digest(),state.digest):
                    raise ValueError('adapter snapshot scratch contents changed')
            _write(output,digest.digest())
            return extent
        finally:
            model._dispose(files)


def restore_segment_edge_cache(model, source, *, max_bytes):
    from .segment_edge_runtime import _StateFile
    with _access(model,max_bytes):
        prefix=_read(source,12)
        if prefix[:8]!=_MAGIC:
            raise ValueError('invalid Segment/Edge cache magic')
        length=struct.unpack('<I',prefix[8:])[0]
        if not 0<length<=min(65536,max_bytes-44):
            raise ValueError('Segment/Edge metadata exceeds admission')
        encoded=_read(source,length)
        header=json.loads(encoded,object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if type(header) is not dict or set(header)!={'version','identity','context','position','sizes','digests'}:
            raise ValueError('invalid Segment/Edge cache metadata')
        if (type(header['version']) is not int or header['version']!=1 or header['identity']!=model.identity or
                type(header['context']) is not int or header['context']!=model.context or
                type(header['position']) is not int or not 0<=header['position']<=model.context or
                type(header['sizes']) is not list or len(header['sizes'])!=len(model._segments) or
                any(type(size) is not int or size<0 for size in header['sizes']) or
                type(header['digests']) is not list or len(header['digests'])!=len(model._segments) or
                any(type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value)
                    for value in header['digests'])):
            raise ValueError('Segment/Edge cache identity or inventory differs')
        payload=sum(header['sizes'])
        if payload+44+length>max_bytes:
            raise MemoryError('Segment/Edge cache wire exceeds admission')
        rollback=model._capture()
        files=[]
        try:
            retained=sum(state.size for inventory in model._states.values() for state in inventory)
            retained+=sum(state.size for state in (model._initial or ()))
            if retained+sum(state.size for state in rollback)+payload>model._snapshot_limit:
                raise MemoryError('Segment/Edge restore scratch storage exceeds admission')
            digest=hashlib.sha256(prefix+encoded)
            for size,expected in zip(header['sizes'],header['digests'],strict=True):
                state=_StateFile(size);files.append(state);check=hashlib.sha256();remaining=size
                while remaining:
                    data=_read(source,min(_CHUNK,remaining));remaining-=len(data)
                    if state.file.write(data)!=len(data):
                        raise OSError('short Segment/Edge scratch write')
                    check.update(data);digest.update(data)
                if not hmac.compare_digest(check.hexdigest(),expected):
                    raise ValueError('Segment/Edge adapter blob checksum differs')
                state.size=size;state.digest=check.digest();state.file.flush()
            if not hmac.compare_digest(bytes(_read(source,32)),digest.digest()) or source.read(1)!=b'':
                raise ValueError('Segment/Edge frame checksum or EOF differs')
            try:
                model._restore_files(files)
            except BaseException as error:
                try:
                    model._restore_files(rollback)
                    model._quarantined=False
                except BaseException as recovery:
                    error.add_note('Segment/Edge cache rollback failed: '+str(recovery))
                raise
            model._position=header['position']
            return model._position
        finally:
            model._dispose(files+rollback)
