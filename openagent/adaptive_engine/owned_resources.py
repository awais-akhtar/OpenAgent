"""Retryable dependency-ordered resource release for decoder ownership."""
import threading


class OwnedResources:
    """Retain failed close callbacks and all owners they may still borrow.

    Register storage first, then providers, arithmetic and decoders. A failed
    release leaves that callback at the top for an explicit subsequent retry.
    """
    def __init__(self):
        self._callbacks=[];self._lock=threading.RLock();self._releasing=False

    def callback(self,method,*args,**kwargs):
        if not callable(method):raise TypeError('callable resource release required')
        with self._lock:
            if self._releasing:raise RuntimeError('cannot add ownership during release')
            self._callbacks.append((method,args,kwargs))
        return method

    @property
    def pending(self):
        with self._lock:return len(self._callbacks)

    def close(self):
        with self._lock:
            if self._releasing:raise RuntimeError('resource release is reentrant')
            self._releasing=True
            try:
                while self._callbacks:
                    method,args,kwargs=self._callbacks[-1]
                    method(*args,**kwargs)
                    self._callbacks.pop()
            finally:self._releasing=False

    def pop_all(self):
        with self._lock:
            if self._releasing:raise RuntimeError('cannot transfer ownership during release')
            result=OwnedResources();result._callbacks,self._callbacks=self._callbacks,[]
            return result

    def __enter__(self):return self
    def __exit__(self,kind,error,traceback):
        try:self.close()
        except BaseException as cleanup_error:
            if error is None:
                cleanup_error.cleanup=self.close
                raise
            error.cleanup=self.close
            error.add_note('dependent resource release remains pending: '+str(cleanup_error))
        return False


__all__=['OwnedResources']
