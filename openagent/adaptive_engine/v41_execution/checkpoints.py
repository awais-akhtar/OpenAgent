"""Bounded, atomic local checkpoints for the incremental equation executor."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from .execution import V41CacheSnapshot, V41Execution, V41Snapshot
from ..failure_reporting import add_failure_note, failure_summary


_MAGIC=b'OAV41S02'
_HEADER_LIMIT=65536


def _limit(value):
    if type(value) is not int or value<=0: raise ValueError('positive checkpoint payload budget required')
    return value


def _records(snapshot, config):
    yield 'tokens','u32',tuple(snapshot.tokens)
    state=snapshot.cache
    for layer in range(config.layers):
        yield f'{layer}.local','packed',tuple(row for pair in state.local[layer] for row in pair)
        for name,field,kind in (('pending',state.pending,'f32'),('gates',state.gates,'f32'),
                                ('compressed',state.compressed,'packed'),('index',state.index,'f32')):
            yield f'{layer}.{name}',kind,field[layer]
        yield f'{layer}.engram','u32',state.engram[layer][2]


def _descriptor(name,kind,rows,config):
    width=4 if kind=='u32' else config.hidden*4 if kind=='f32' else config.hidden//16*9
    return dict(name=name,kind=kind,rows=len(rows),width=width,size=len(rows)*width)


def _encoded_rows(kind,rows):
    if kind=='u32':
        for start in range(0,len(rows),4096):
            part=rows[start:start+4096]
            yield struct.pack('<'+'I'*len(part),*part)
    else:
        for row in rows:
            yield row if kind=='packed' else np.frombuffer(row,dtype=np.float32).astype('<f4').tobytes()


def save_v41_checkpoint(model,destination,*,max_payload_bytes):
    if type(model) is not V41Execution: raise ValueError('current V4.1 executor required')
    destination=Path(destination).resolve()
    with model._lock:
        if model._closed or model._active: raise RuntimeError('executor unavailable for checkpoint')
        return _save_v41_checkpoint(model,destination,max_payload_bytes=max_payload_bytes)


def _save_v41_checkpoint(model,destination,*,max_payload_bytes):
    """Save one committed snapshot; never includes fetched Engram table rows."""
    if type(model) is not V41Execution: raise ValueError('current V4.1 executor required')
    maximum=_limit(max_payload_bytes)
    snapshot=model.snapshot()
    model._validate_cache(snapshot.cache,snapshot.tokens)
    records=tuple(_records(snapshot,model.config))
    descriptors=[_descriptor(name,kind,rows,model.config) for name,kind,rows in records]
    size=sum(item['size'] for item in descriptors)
    if size>maximum: raise MemoryError('checkpoint payload exceeds byte budget')
    header=json.dumps(dict(version=2,identity=snapshot.identity,schema=model.config.schema,
        position=len(snapshot.tokens),records=descriptors,
        engram_kinds=[row[0] for row in snapshot.cache.engram]),
        sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
    if len(header)>_HEADER_LIMIT: raise MemoryError('checkpoint header exceeds 64 KiB')
    path=Path(destination).resolve()
    if not path.parent.is_dir() or path.drive.startswith('\\\\'):
        raise ValueError('existing local checkpoint directory required')
    prefix=_MAGIC+struct.pack('<IQ',len(header),size)+header
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix='.v41-',delete=False) as output:
            temporary=Path(output.name)
            digest=hashlib.sha256(prefix)
            output.write(prefix)
            for _,kind,rows in records:
                for chunk in _encoded_rows(kind,rows):
                    digest.update(chunk); output.write(chunk)
            output.write(digest.digest())
            output.flush(); os.fsync(output.fileno())
        os.replace(temporary,path)
        temporary=None
        if os.name!='nt':
            descriptor=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def load_v41_checkpoint(model,source,*,max_payload_bytes,cancelled=None):
    if type(model) is not V41Execution: raise ValueError('current V4.1 executor required')
    if cancelled is not None and not callable(cancelled):
        raise ValueError('checkpoint cancellation predicate must be callable')
    source=Path(source).resolve(strict=True)
    with model._lock:
        if model._closed or model._active: raise RuntimeError('executor unavailable for recovery')
        return _load_v41_checkpoint(model,source,max_payload_bytes=max_payload_bytes,
                                    cancelled=cancelled)


def _load_v41_checkpoint(model,source,*,max_payload_bytes,cancelled=None):
    """Hash and validate a complete staged file before mutating model state."""
    if type(model) is not V41Execution: raise ValueError('current V4.1 executor required')
    maximum=_limit(max_payload_bytes)
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError('V4.1 checkpoint restore cancelled')
    check_cancelled()
    path=Path(source).resolve(strict=True)
    if not path.is_file() or path.drive.startswith('\\\\'):
        raise ValueError('existing local checkpoint file required')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        identity = (before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns)
        if (not os.path.isfile(path) or
                (lambda info: (info.st_dev, info.st_ino, info.st_size,
                               info.st_mtime_ns, info.st_ctime_ns))(path.stat()) != identity):
            raise ValueError('V4.1 checkpoint changed during opening')
        def read_exact(count):
            check_cancelled()
            data=stream.read(count)
            if len(data)!=count: raise ValueError('truncated V4.1 checkpoint')
            return data
        prefix=read_exact(20)
        if prefix[:8]!=_MAGIC: raise ValueError('unsupported V4.1 checkpoint format')
        header_size,payload_size=struct.unpack_from('<IQ',prefix,8)
        if not 1<=header_size<=_HEADER_LIMIT or payload_size>maximum:
            raise MemoryError('checkpoint metadata/payload exceeds admission')
        header_bytes=read_exact(header_size)
        def unique_pairs(pairs):
            result={}
            for key,value in pairs:
                if key in result: raise ValueError('duplicate checkpoint metadata field')
                result[key]=value
            return result
        header=json.loads(header_bytes,object_pairs_hook=unique_pairs)
        if (type(header) is not dict or set(header)!= {'version','identity','schema','position','records','engram_kinds'}
            or type(header['version']) is not int or header['version']!=2
            or header['identity']!=model._identity or header['schema']!=model.config.schema):
            raise ValueError('foreign V4.1 checkpoint identity/schema')
        position=header['position']
        if type(position) is not int or not 0<=position<=model.config.capacity:
            raise ValueError('invalid checkpoint position')
        model._admit(position)
        kinds=['retrieved' if hasattr(value,'provider') else 'legacy' for value in model._engram]
        if header['engram_kinds']!=kinds:
            raise ValueError('checkpoint Engram layout differs from model')
        c=model.config
        expected=[dict(name='tokens',kind='u32',rows=position,width=4,size=position*4)]
        for layer,mode in enumerate(c.modes):
            counts=(('local','packed',2*min(position,c.window)),
                    ('pending','f32',position%c.ratio if mode=='full' else 0),
                    ('gates','f32',position%c.ratio if mode=='full' else 0),
                    ('compressed','packed',position//c.ratio if mode=='full' else 0),
                    ('index','f32',position//c.ratio if mode=='full' else 0),
                    ('engram','u32',min(position,3) if kinds[layer]=='retrieved' else position))
            for name,kind,count in counts:
                width=4 if kind=='u32' else c.hidden*4 if kind=='f32' else c.hidden//16*9
                expected.append(dict(name=f'{layer}.{name}',kind=kind,rows=count,width=width,size=count*width))
        if header['records']!=expected or sum(row['size'] for row in expected)!=payload_size:
            raise ValueError('checkpoint records differ from exact cache layout')
        digest=hashlib.sha256(prefix+header_bytes)
        records={}
        for record in expected:
            rows=[]
            for _ in range(record['rows']):
                check_cancelled()
                data=read_exact(record['width']); digest.update(data)
                if record['kind']=='u32': rows.append(struct.unpack('<I',data)[0])
                elif record['kind']=='f32': rows.append(np.frombuffer(data,dtype='<f4').astype(np.float32).tobytes())
                else: rows.append(data)
            records[record['name']]=tuple(rows)
        checksum = read_exact(32)
        check_cancelled()
        trailing = stream.read(1)
        if checksum != digest.digest() or trailing:
            raise ValueError('checkpoint hash mismatch or trailing bytes')
        current = os.fstat(stream.fileno())
        ondisk = path.stat()
        current_stamp = (current.st_dev, current.st_ino, current.st_size,
                         current.st_mtime_ns, current.st_ctime_ns)
        disk_stamp = (ondisk.st_dev, ondisk.st_ino, ondisk.st_size,
                      ondisk.st_mtime_ns, ondisk.st_ctime_ns)
        if current_stamp != identity or disk_stamp != identity:
            raise ValueError('V4.1 checkpoint changed during restoration')
    tokens=records['tokens']
    model._validate(tokens)
    local=[]
    for layer in range(c.layers):
        rows=records[f'{layer}.local']
        local.append(tuple(zip(rows[::2],rows[1::2])))
    state=V41CacheSnapshot(tuple(local),
        *(tuple(records[f'{layer}.{name}'] for layer in range(c.layers))
          for name in ('pending','gates','compressed','index')),
        tuple((kind,position,records[f'{layer}.engram']) for layer,kind in enumerate(kinds)))
    check_cancelled()
    previous=model.snapshot()
    try:
        model.restore(V41Snapshot(model._identity,tokens,state))
        # Cancellation can arrive during the final, atomic state publication.
        # Recheck before reporting success and restore the old boundary if it
        # did; restore() validates fully before its tuple assignment.
        check_cancelled()
    except BaseException as primary:
        try:
            model.restore(previous)
        except BaseException as recovery:
            add_failure_note(primary,'V4.1 checkpoint cancellation/restore rollback failed: '+failure_summary(recovery))
        raise
