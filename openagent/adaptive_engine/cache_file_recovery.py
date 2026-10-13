"""Atomic bounded files for complete service/executor cache frames."""
import os
import stat
import tempfile
import hashlib
import hmac

from .session_recovery import _path


def _limit(value):
    if type(value) is not int or value < 88:
        raise ValueError('explicit service cache byte budget required')


def _stamp(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


class _DigestWriter:
    def __init__(self, stream):
        self.stream=stream
        self.digest=hashlib.sha256()

    def write(self, data):
        count=self.stream.write(data)
        if type(count) is not int or count != len(data):
            raise OSError('cache publication writer made short or invalid progress')
        self.digest.update(memoryview(data).cast('B'))
        return count


class _StableReader:
    def __init__(self, stream, path, stamp, cancelled=None):
        self.stream,self.path,self.stamp,self.cancelled=stream,path,stamp,cancelled

    def read(self, count):
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError('service cache restoration cancelled')
        if type(count) is not int or count < 0 or count > 1024*1024:
            raise ValueError('bounded cache read required')
        value=self.stream.read(count)
        if type(value) is not bytes or len(value) > count:
            raise ValueError('cache reader returned invalid bounded bytes')
        if value==b'':
            _path(self.path)
            if (_stamp(os.fstat(self.stream.fileno()))!=self.stamp or
                    _stamp(self.path.stat())!=self.stamp):
                raise ValueError('service cache file changed during restoration')
        return value

    def require_complete(self):
        """Require the codec to consume the exact admitted immutable frame."""
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError('service cache restoration cancelled')
        position = self.stream.tell()
        if type(position) is not int or position < 0:
            raise ValueError('service cache reader position is invalid')
        if position != self.stamp[2]:
            raise ValueError('service cache restore did not consume the complete frame')
        _path(self.path)
        current = _stamp(os.fstat(self.stream.fileno()))
        on_disk = _stamp(self.path.stat())
        if current != self.stamp or on_disk != self.stamp:
            raise ValueError('service cache file changed during restoration')


def save_service_cache(service, session_id, destination, *, max_bytes):
    """Publish one flushed frame atomically; containing directory must exist."""
    _limit(max_bytes)
    path=_path(destination)
    if path.exists() and not path.is_file():
        raise ValueError('regular service cache destination required')
    descriptor,temporary=tempfile.mkstemp(prefix='.session-cache-',dir=path.parent)
    try:
        try:
            output=os.fdopen(descriptor,'wb')
        except BaseException:
            os.close(descriptor)
            raise
        with output:
            writer=_DigestWriter(output)
            size=service.write_session_cache(session_id,writer,max_bytes=max_bytes)
            if type(size) is not int or not 88 <= size <= max_bytes:
                raise ValueError('service cache writer returned an invalid frame extent')
            output.flush()
            os.fsync(output.fileno())
            if os.fstat(output.fileno()).st_size!=size:
                raise ValueError('service cache encoded extent differs from writer receipt')
        _path(path)
        os.replace(temporary,path)
        if os.name!='nt':
            directory=os.open(path.parent,os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return dict(path=str(path),bytes=size,sha256=writer.digest.hexdigest(),
                    recovery='complete-executor-cache')
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_service_cache(service, session_id, source, *, max_bytes,history_receipt=None,cancelled=None):
    """Admit regular-file extent and detect replacement before state commit."""
    _limit(max_bytes)
    if cancelled is not None and not callable(cancelled):
        raise TypeError('cache restoration cancellation must be callable')
    path=_path(source)
    descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_BINARY',0)|
                       getattr(os,'O_NONBLOCK',0)|getattr(os,'O_NOFOLLOW',0))
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('regular service cache source required')
        if not 88<=info.st_size<=max_bytes:
            raise MemoryError('service cache file extent exceeds admission')
        stream=os.fdopen(descriptor,'rb')
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        options={} if history_receipt is None else dict(history_receipt=history_receipt)
        reader = _StableReader(stream, path, _stamp(info), cancelled)
        result = service.restore_session_cache(session_id, reader,
            max_bytes=max_bytes, **options)
        reader.require_complete()
        return result


class _ExecutorCacheAdapter:
    """Use the atomic file lifecycle without inventing a service session."""

    def __init__(self, executor):
        self.executor = executor

    def write_session_cache(self, session_id, output, *, max_bytes):
        class BoundedWriter:
            def __init__(self):
                self.count = 0

            def write(self, data):
                view = memoryview(data).cast('B')
                if len(view) > max_bytes-self.count:
                    raise MemoryError('executor cache writer exceeds frame admission')
                written = output.write(view)
                if type(written) is not int or not 0 < written <= len(view):
                    raise OSError('executor cache writer made invalid progress')
                self.count += written
                return written

        writer = BoundedWriter()
        self.executor.write_cache(writer, max_bytes=max_bytes)
        return writer.count

    def restore_session_cache(self, session_id, source, *, max_bytes):
        return self.executor.restore_cache(source, max_bytes=max_bytes)


def save_executor_cache(executor, destination, *, max_bytes):
    """Atomically publish a complete executor codec; owner manages serialization."""
    if not callable(getattr(executor, 'write_cache', None)):
        raise ValueError('complete executor cache writer required')
    return save_service_cache(_ExecutorCacheAdapter(executor), None, destination,
                              max_bytes=max_bytes)


def load_executor_cache(executor, source, *, max_bytes, cancelled=None):
    """Load a bounded stable regular file through the executor's own codec."""
    if not callable(getattr(executor, 'restore_cache', None)):
        raise ValueError('complete executor cache restore required')
    return load_service_cache(_ExecutorCacheAdapter(executor), None, source,
                              max_bytes=max_bytes, cancelled=cancelled)


def verify_service_cache_file(source, *, receipt, max_bytes):
    """Read back exact published bytes before releasing an evicted executor."""
    _limit(max_bytes)
    size,digest=receipt.get('bytes'),receipt.get('sha256')
    if (type(size) is not int or not 88<=size<=max_bytes or type(digest) is not str or
            len(digest)!=64 or any(value not in '0123456789abcdef' for value in digest)):
        raise ValueError('invalid service cache publication receipt')
    path=_path(source)
    with path.open('rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size!=size:
            raise ValueError('service cache readback extent differs')
        reader=_StableReader(stream,path,_stamp(info))
        actual=hashlib.sha256()
        total=0
        while True:
            data=reader.read(min(1024*1024,size-total+1))
            if not data:
                break
            total+=len(data)
            if total>size:
                raise ValueError('service cache grew during readback')
            actual.update(data)
        if total!=size or not hmac.compare_digest(actual.hexdigest(),digest):
            raise ValueError('service cache readback checksum differs')
    return receipt
