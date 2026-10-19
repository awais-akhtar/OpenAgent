"""Bounded, checksummed transport for an atomic group of owned decoder caches."""
import hashlib
import hmac
import inspect
import json
import struct
import tempfile
from contextlib import ExitStack

_MAGIC=b'OACACHE2'

def _pairs(items):
    result={}
    for key,value in items:
        if key in result: raise ValueError('duplicate compound cache field')
        result[key]=value
    return result

def _read(source,size):
    chunks=[];remaining=size
    while remaining:
        value=source.read(min(remaining,1024**2))
        if not isinstance(value,bytes) or not value or len(value)>remaining:
            raise ValueError('truncated compound cache')
        chunks.append(value);remaining-=len(value)
    return b''.join(chunks)

def _write(output,value):
    remaining=memoryview(value)
    while remaining:
        count=output.write(remaining)
        if type(count) is not int or not 0<count<=len(remaining):
            raise OSError('compound cache output made no progress')
        remaining=remaining[count:]

class _BoundedWriter:
    def __init__(self,output,limit): self.output=output;self.limit=limit;self.size=0
    def write(self,value):
        if self.size+len(value)>self.limit: raise MemoryError('compound cache component exceeds transport admission')
        _write(self.output,value);self.size+=len(value);return len(value)

class _CancellableReader:
    """Expose a component frame while retaining the group cancel boundary."""
    def __init__(self, stream, boundary):
        self._stream = stream
        self._boundary = boundary

    def read(self, size=-1):
        self._boundary()
        value = self._stream.read(size)
        self._boundary()
        return value

    def readinto(self, buffer):
        self._boundary()
        count = self._stream.readinto(buffer)
        self._boundary()
        return count

    def seek(self, *args):
        self._boundary()
        return self._stream.seek(*args)

    def tell(self):
        self._boundary()
        return self._stream.tell()

    def __getattr__(self, name):
        # Preserve narrowly required binary-stream attributes (for example
        # ``fileno``) without bypassing the read/seek/tell guards above.
        return getattr(self._stream, name)

class _CancellationEvent:
    """Adapt a group cancellation callback to event-style owner scopes."""
    def __init__(self,cancelled): self._cancelled=cancelled
    def is_set(self): return self._cancelled is not None and self._cancelled()

def _accepts_keyword(function,name):
    try:
        parameters=inspect.signature(function).parameters.values()
        return any(parameter.name==name or parameter.kind is inspect.Parameter.VAR_KEYWORD
                   for parameter in parameters)
    except (TypeError,ValueError):
        return False

def write_components(output,components,*,identity,max_bytes,cancelled=None):
    """Spool component frames on disk, admitting their aggregate before publication."""
    if type(max_bytes) is not int or max_bytes<128 or not 1<=len(components)<=64:
        raise ValueError('bounded compound cache extent and component count required')
    def boundary():
        if cancelled is not None and cancelled(): raise InterruptedError('compound cache export cancelled')
    with ExitStack() as stack:
        frames=[];total=0
        for component in components:
            boundary();stream=stack.enter_context(tempfile.TemporaryFile(mode='w+b'))
            writer=_BoundedWriter(stream,max_bytes-total)
            write=component.write_cache
            scope=getattr(component,'cancellation_scope',None)
            with ExitStack() as component_scope:
                if callable(scope) and cancelled is not None:
                    component_scope.enter_context(scope(_CancellationEvent(cancelled)))
                if _accepts_keyword(write,'cancelled'):
                    write(writer,max_bytes=max_bytes-total,cancelled=cancelled)
                else:
                    write(writer,max_bytes=max_bytes-total)
            boundary()
            if not writer.size: raise ValueError('empty component cache frame')
            frames.append((stream,writer.size));total+=writer.size
        header=json.dumps(dict(identity=identity,lengths=[size for _,size in frames]),
                          sort_keys=True,separators=(',',':')).encode()
        extent=44+len(header)+total
        if len(header)>65536 or extent>max_bytes: raise MemoryError('compound cache aggregate exceeds admission')
        digest=hashlib.sha256()
        def publish(value): digest.update(value);_write(output,value)
        publish(_MAGIC+struct.pack('<I',len(header)));publish(header)
        for stream,size in frames:
            stream.seek(0)
            while size:
                boundary();chunk=_read(stream,min(size,1024**2));publish(chunk);size-=len(chunk)
        boundary();_write(output,digest.digest());return extent

def restore_components(source,components,*,identity,max_bytes,cancelled=None,validate=None):
    """Authenticate all frames before mutation, then restore with group rollback."""
    if type(max_bytes) is not int or max_bytes<128 or not 1<=len(components)<=64:
        raise ValueError('bounded compound cache extent and component count required')
    def boundary():
        if cancelled is not None and cancelled(): raise InterruptedError('compound cache recovery cancelled')
    boundary();prefix=_read(source,12);length=struct.unpack('<I',prefix[8:])[0]
    if prefix[:8]!=_MAGIC or not 0<length<=min(65536,max_bytes-44): raise ValueError('invalid compound cache header')
    raw=_read(source,length);header=json.loads(
        raw, object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if (type(header) is not dict or set(header) != {'identity','lengths'} or
            type(header['identity']) is not type(identity) or header['identity'] != identity):
        raise ValueError('compound cache execution identity differs')
    lengths=header['lengths']
    if type(lengths) is not list or len(lengths)!=len(components) or any(type(v) is not int or v<=0 for v in lengths):
        raise ValueError('compound cache component inventory differs')
    if 44+length+sum(lengths)>max_bytes: raise MemoryError('compound cache payload exceeds admission')
    with ExitStack() as stack:
        frames=[];digest=hashlib.sha256(prefix+raw)
        for size in lengths:
            stream=stack.enter_context(tempfile.TemporaryFile(mode='w+b'));frames.append(stream)
            while size:
                boundary();chunk=_read(source,min(size,1024**2));digest.update(chunk);_write(stream,chunk);size-=len(chunk)
            stream.seek(0)
        if not hmac.compare_digest(digest.digest(), _read(source, 32)):
            raise ValueError('compound cache checksum or EOF differs')
        boundary()
        trailing=source.read(1)
        boundary()
        if not isinstance(trailing,bytes) or trailing:
            raise ValueError('compound cache checksum or EOF differs')
        receipts=[]
        try:
            for component in components: boundary();receipts.append(component.snapshot())
            try:
                for component,stream,size in zip(components,frames,lengths):
                    boundary()
                    restore=component.restore_cache
                    reader=_CancellableReader(stream,boundary)
                    if _accepts_keyword(restore,'cancelled'):
                        restore(reader,max_bytes=size,cancelled=cancelled)
                    else:
                        restore(reader,max_bytes=size)
                    boundary()
                if validate is not None: validate()
                boundary()
            except BaseException as error:
                # Components may share dependent physical state. Undo in the
                # reverse order of publication, just like unwinding a staged
                # constructor, while still attempting every rollback.
                for component,receipt in reversed(tuple(zip(components,receipts))):
                    try: component.restore(receipt)
                    except BaseException as failure: error.add_note('compound cache rollback failed: '+str(failure))
                raise
        finally:
            failures=[]
            pending=[]
            for component,receipt in zip(components,receipts):
                try: component.release_snapshot(receipt)
                except BaseException as failure: failures.append(failure);pending.append((component,receipt))
            if failures:
                def cleanup():
                    while pending:
                        component,receipt=pending[-1];component.release_snapshot(receipt);pending.pop()
                error=RuntimeError('compound cache snapshot cleanup remains pending')
                error.cleanup=cleanup
                raise error from failures[0]

__all__=['write_components','restore_components']
