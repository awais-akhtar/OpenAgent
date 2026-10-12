"""Content identities for explicitly supplied prebuilt native libraries."""
from dataclasses import dataclass
import hashlib
import os
import stat
import weakref
from pathlib import Path
from .session_recovery import _path


def _stamp(status):
    return (status.st_dev,status.st_ino,status.st_size,status.st_mtime_ns,status.st_ctime_ns)


def release_finalizer(owner,attribute):
    """Keep failed native destruction retryable under the caller's owner lock."""
    finalizer=getattr(owner,attribute,None)
    if finalizer is None or not finalizer.alive:return
    info=finalizer.detach()
    try:info[1](*info[2],**info[3])
    except BaseException:
        setattr(owner,attribute,weakref.finalize(owner,info[1],*info[2],**info[3]))
        raise


@dataclass(frozen=True)
class NativeLibraryIdentity:
    path: Path
    sha256: str
    stamp: tuple

    @classmethod
    def read(cls,path,*,max_bytes=1024**3):
        if type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('positive prebuilt library extent admission required')
        resolved=_path(path).resolve(strict=True)
        if not resolved.is_file():raise ValueError('explicit prebuilt native library file required')
        digest=hashlib.sha256()
        expected=resolved.stat()
        descriptor=os.open(resolved,os.O_RDONLY|getattr(os,'O_BINARY',0)|
            getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError('regular opened native library required')
            stream=os.fdopen(descriptor,'rb')
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            before=os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or _stamp(before)!=_stamp(expected):
                raise ValueError('native library changed during identity opening')
            if not 0<before.st_size<=max_bytes:
                raise MemoryError('prebuilt native library exceeds source extent admission')
            count=0
            for block in iter(lambda:stream.read(1024**2),b''):
                count+=len(block)
                if count>before.st_size:raise ValueError('native library grew during identity read')
                digest.update(block)
            after=os.fstat(stream.fileno())
        if count!=before.st_size or _stamp(before)!=_stamp(after) or _stamp(_path(resolved).stat())!=_stamp(after):
            raise ValueError('native library changed during identity read')
        return cls(resolved,digest.hexdigest(),_stamp(after))

    def assert_unchanged(self):
        if _stamp(_path(self.path).stat())!=self.stamp:
            raise ValueError('prebuilt native library source changed: '+str(self.path))


__all__=['NativeLibraryIdentity']
