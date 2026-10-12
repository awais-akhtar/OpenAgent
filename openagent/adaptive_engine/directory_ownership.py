"""Process-lifetime advisory ownership for a dedicated durable cache directory."""
import os
from pathlib import Path
import stat
import threading
import weakref
from .session_recovery import _path


class _DirectoryRelease:
    def __init__(self,descriptor,windows):
        self.descriptor,self.windows=descriptor,windows
        self.unlocked=False

    def __call__(self):
        if self.descriptor is None:return
        unlock_error=None
        if not self.unlocked:
            try:
                if self.windows:
                    import msvcrt
                    os.lseek(self.descriptor,0,os.SEEK_SET)
                    msvcrt.locking(self.descriptor,msvcrt.LK_UNLCK,1)
                else:
                    import fcntl
                    fcntl.flock(self.descriptor,fcntl.LOCK_UN)
                self.unlocked=True
            except BaseException as failure:unlock_error=failure
        try:os.close(self.descriptor)
        except BaseException as failure:
            if unlock_error is not None:failure.add_note('directory unlock also failed: '+repr(unlock_error))
            raise
        # Closing the descriptor also releases the OS advisory lock.
        self.descriptor=None


class DirectoryOwnership:
    """Hold the OS lock instead of trusting a stale PID or deleting lock files.

    The sentinel stays in place after release so other processes always lock
    the same inode. Process termination releases the OS lock automatically.
    """
    def __init__(self, directory, *, filename='.openagent-continuation.lock'):
        self.root = _path(directory).resolve(strict=True)
        if (not self.root.is_dir() or type(filename) is not str or Path(filename).name!=filename
                or filename in ('','.','..') or ':' in filename or '\0' in filename):
            raise ValueError('dedicated directory and simple sentinel filename required')
        self.path = _path(self.root/filename)
        root_info=self.root.stat();self._root_identity=(root_info.st_dev,root_info.st_ino)
        if self.path.is_symlink():
            raise ValueError('directory ownership sentinel cannot be a symlink')
        flags = os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0)
        descriptor = os.open(self.path,flags,0o600)
        acquired = False
        windows = os.name=='nt'
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size>1:
                raise ValueError('regular one-byte ownership sentinel required')
            if before.st_size==0:
                os.write(descriptor,b'\0')
                os.fsync(descriptor)
            os.lseek(descriptor,0,os.SEEK_SET)
            if windows:
                import msvcrt
                msvcrt.locking(descriptor,msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            acquired = True
            current,linked = os.fstat(descriptor),_path(self.path).stat()
            if (current.st_dev,current.st_ino)!=(linked.st_dev,linked.st_ino):
                raise ValueError('ownership sentinel replaced during acquisition')
            root_info=_path(self.root).stat()
            if (root_info.st_dev,root_info.st_ino)!=self._root_identity:
                raise ValueError('ownership directory replaced during acquisition')
        except BaseException as primary:
            cleanup=_DirectoryRelease(descriptor,windows)
            if not acquired:cleanup.unlocked=True
            try:cleanup()
            except BaseException as failure:
                primary.cleanup=cleanup
                primary.add_note('directory ownership construction cleanup retained: '+repr(failure))
            raise
        self._lock = threading.Lock()
        self._sentinel_identity=(current.st_dev,current.st_ino)
        self._release = weakref.finalize(self,_DirectoryRelease(descriptor,windows))

    def assert_held(self):
        with self._lock:
            if not self._release.alive:raise RuntimeError('directory ownership is closed')
            root_info=_path(self.root).stat();sentinel=_path(self.path).stat()
            if ((root_info.st_dev,root_info.st_ino)!=self._root_identity
                    or (sentinel.st_dev,sentinel.st_ino)!=self._sentinel_identity):
                raise ValueError('owned directory or sentinel replaced during use')

    def close(self):
        with self._lock:
            if self._release.alive:
                info=self._release.detach()
                try:info[1](*info[2],**info[3])
                except BaseException:
                    self._release=weakref.finalize(self,info[1],*info[2],**info[3])
                    raise

    @property
    def closed(self):
        return not self._release.alive


__all__ = ['DirectoryOwnership']
