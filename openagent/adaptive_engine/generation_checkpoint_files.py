"""Atomic file publication and leased reads for gateway generation recovery.

This store is a synchronous checkpoint sink for PublisherSession and the
native publisher gateway. It publishes only a fully written generation frame;
an interrupted capture preserves the preceding committed checkpoint.
"""
from contextlib import contextmanager
import copy
import hashlib
import hmac
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading

from .cache_file_recovery import _DigestWriter,_StableReader,_stamp
from .checkpoint_recovery import _destination_lock,_sync_directory
from .session_recovery import _path


def _close_descriptor_after_failure(descriptor, primary):
    try:
        os.close(descriptor)
    except BaseException as cleanup:
        primary.add_note('generation checkpoint descriptor cleanup failed: '+repr(cleanup))


def _fdopen_owned(descriptor, *args, **kwargs):
    """Transfer a raw descriptor to a stream, closing it if wrapping fails."""
    try:
        return os.fdopen(descriptor, *args, **kwargs)
    except BaseException as primary:
        _close_descriptor_after_failure(descriptor,primary)
        raise


def _remove_staging(path,primary):
    if path is None:return
    try:Path(path).unlink(missing_ok=True)
    except BaseException as cleanup:
        if primary is None:
            cleanup.cleanup=lambda:Path(path).unlink(missing_ok=True)
            raise
        previous=getattr(primary,'cleanup',None)
        def retry():
            nonlocal previous
            if callable(previous):
                previous();previous=None
            Path(path).unlink(missing_ok=True)
        primary.cleanup=retry
        primary.add_note('generation checkpoint staging cleanup retained: '+repr(cleanup))


class AtomicGenerationCheckpoint:
    def __init__(self,destination,*,max_bytes):
        if type(max_bytes) is not int or not 128<=max_bytes<=2**63-1:
            raise ValueError('positive complete generation frame admission required')
        self.path=_path(destination);self.max_bytes=max_bytes
        self._lock=threading.Lock();self._closed=False;self._receipt=None

    @property
    def receipt(self):
        with self._lock:return copy.deepcopy(self._receipt)

    def close(self):
        with self._lock:self._closed=True

    def __call__(self,boundary,write_frame):
        """Publish exactly one bounded frame under a process and OS file lease."""
        if not callable(write_frame):raise ValueError('complete frame writer required')
        if not self._lock.acquire(blocking=False):raise RuntimeError('checkpoint publication already active')
        temporary=None
        try:
            if self._closed:raise RuntimeError('checkpoint store is closed')
            with _destination_lock(self.path):
                _path(self.path)
                if self.path.exists() and not self.path.is_file():raise ValueError('regular checkpoint destination required')
                descriptor,temporary=tempfile.mkstemp(prefix='.generation-',dir=self.path.parent)
                stream=_fdopen_owned(descriptor,'wb')
                with stream:
                    digest=_DigestWriter(stream)
                    maximum=self.max_bytes
                    class BoundedWriter:
                        count=0
                        def write(self,value):
                            view=memoryview(value).cast('B')
                            if len(view)>maximum-self.count:raise MemoryError('generation frame exceeds store admission')
                            count=digest.write(view)
                            if type(count) is not int or not 0<count<=len(view):raise OSError('invalid checkpoint write progress')
                            self.count+=count
                            return count
                    writer=BoundedWriter()
                    size=write_frame(writer)
                    stream.flush();os.fsync(stream.fileno())
                    if type(size) is not int or size!=writer.count or size!=os.fstat(stream.fileno()).st_size:
                        raise ValueError('generation frame writer receipt differs from staged extent')
                receipt=dict(schema='openagent.generation-file-receipt.v1',path=str(self.path),bytes=size,
                    sha256=digest.digest.hexdigest(),position=boundary.position)
                from .service_generation_checkpoint import ServiceGenerationBoundary
                from .v41_generation_recovery import V41GenerationBoundary
                if type(boundary) is ServiceGenerationBoundary:
                    receipt.update(execution_identity=boundary.identity,application_identity=boundary.application_identity)
                elif type(boundary) is V41GenerationBoundary:
                    receipt.update(execution_identity=boundary.identity,
                        tokenizer_sha256=boundary.tokenizer_sha256,dspark=boundary.dspark)
                else:
                    receipt.update(config_sha256=boundary.config_sha256,tokenizer_sha256=boundary.tokenizer_sha256)
                _path(self.path)
                os.replace(temporary,self.path);temporary=None
                _sync_directory(self.path.parent)
                self._receipt=receipt
                return copy.deepcopy(receipt)
        finally:
            try:
                _remove_staging(temporary,sys.exc_info()[1])
            finally:self._lock.release()

    @contextmanager
    def source(self,*,receipt=None):
        """Capture a leased frame, then release its publication lock before use.

        The detached scratch source permits the resumed request to publish its
        next boundary to this same store. Scratch lifetime belongs to this
        context and never to the provider's worker thread.
        """
        if not self._lock.acquire(blocking=False):raise RuntimeError('checkpoint store is busy')
        temporary=None
        try:
            if self._closed:raise RuntimeError('checkpoint store is closed')
            with _destination_lock(self.path):
                path=_path(self.path)
                descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_BINARY',0)|getattr(os,'O_NOFOLLOW',0)|
                    getattr(os,'O_NONBLOCK',0))
                try:
                    info=os.fstat(descriptor)
                    if not stat.S_ISREG(info.st_mode) or not 128<=info.st_size<=self.max_bytes:
                        raise ValueError('bounded regular generation checkpoint required')
                except BaseException as primary:
                    _close_descriptor_after_failure(descriptor,primary)
                    raise
                stream=_fdopen_owned(descriptor,'rb')
                with stream:
                    reader=_StableReader(stream,path,_stamp(info))
                    if receipt is not None:
                        digest = receipt.get('sha256') if type(receipt) is dict else None
                        if (type(receipt) is not dict or type(receipt.get('bytes')) is not int
                                or receipt.get('bytes')!=info.st_size
                                or receipt.get('path')!=str(path) or type(digest) is not str
                                or len(digest)!=64
                                or any(value not in '0123456789abcdef' for value in digest)):
                            raise ValueError('exact published generation receipt required')
                    descriptor,temporary=tempfile.mkstemp(prefix='.generation-read-',dir=path.parent)
                    captured=_fdopen_owned(descriptor,'wb')
                    digest=hashlib.sha256();total=0
                    with captured:
                        while True:
                            block=reader.read(min(1024*1024,info.st_size-total+1))
                            if not block:break
                            total+=len(block)
                            if total>info.st_size:raise ValueError('generation source grew during capture')
                            digest.update(block);captured.write(block)
                        captured.flush();os.fsync(captured.fileno())
                    if total!=info.st_size:raise ValueError('generation source truncated during capture')
                    if receipt is not None and not hmac.compare_digest(digest.hexdigest(),receipt['sha256']):
                        raise ValueError('published generation checksum differs')
        except BaseException as primary:
            _remove_staging(temporary,primary)
            raise
        finally:self._lock.release()
        try:
            scratch=Path(temporary)
            with scratch.open('rb') as stream:
                reader=_StableReader(stream,scratch,_stamp(os.fstat(stream.fileno())))
                try:yield reader
                finally:
                    primary=sys.exc_info()[1]
                    try:stream.seek(0,os.SEEK_END);reader.read(1)
                    except BaseException as failure:
                        if primary is None:raise
                        primary.add_note('generation source stability check failed: '+repr(failure))
        finally:_remove_staging(temporary,sys.exc_info()[1])


__all__=['AtomicGenerationCheckpoint']
