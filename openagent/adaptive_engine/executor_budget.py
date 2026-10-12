"""Aggregate declared executor reservations with retained failed cleanup."""
import threading
from uuid import uuid4


class _ReservedExecutor:
    def __init__(self,owner,key,executor):
        self._budget_owner,self._budget_key,self._executor = owner,key,executor

    def __getattr__(self,name):
        if self._executor is None:
            raise RuntimeError('reserved executor is closed')
        return getattr(self._executor,name)

    def close(self):
        if self._executor is None:
            return
        self._budget_owner.release(self._budget_key)
        self._executor = None


class _ConstructionCleanup:
    """Retain the original failed constructor's cleanup and its reservation."""
    def __init__(self,cleanup): self.cleanup=cleanup
    def close(self):
        if self.cleanup is not None:
            self.cleanup();self.cleanup=None


class BudgetedExecutorFactory:
    """Reserve full per-executor budgets before construction, release on close.

    Reservations bound declared live model budgets, not allocator RSS. Borrowed
    providers included in each budget are conservatively reserved per executor.
    Construction validators run after ownership is installed so failed close
    retains the resource and reservation for cleanup retry.
    """
    def __init__(self,factory,*,executor_bytes,max_total_bytes,max_executors,validate=None):
        if not callable(factory) or (validate is not None and not callable(validate)):
            raise ValueError('concrete executor factory/validator required')
        if any(type(value) is not int or value<=0 for value in (executor_bytes,max_total_bytes,max_executors)):
            raise ValueError('explicit aggregate executor admission required')
        if executor_bytes>max_total_bytes:
            raise MemoryError('one executor exceeds aggregate model budget')
        self.factory,self.validate = factory,validate
        self.executor_bytes,self.total_limit,self.count_limit = executor_bytes,max_total_bytes,max_executors
        self._lock,self._creating = threading.RLock(),False
        self._owned,self._pending,self._closing = {},set(),set()
        self._reserved = 0

    def __call__(self):
        with self._lock:
            if self._creating or self._pending:
                raise RuntimeError('executor construction reentry or cleanup pending')
            if len(self._owned)>=self.count_limit or self.executor_bytes>self.total_limit-self._reserved:
                raise MemoryError('aggregate live executor budget exhausted')
            self._creating = True
            self._reserved += self.executor_bytes
            key = uuid4().hex
            executor = None
            try:
                executor = self.factory()
                if any(executor is value for value in self._owned.values()):
                    executor = None
                    raise ValueError('factory reused a mutable owned executor')
                if not callable(getattr(executor,'close',None)):
                    executor = None
                    raise ValueError('owned executor must expose close')
                self._owned[key] = executor
                if self.validate is not None:
                    self.validate(executor)
                return _ReservedExecutor(self,key,executor)
            except BaseException as error:
                if executor is None:
                    cleanup=getattr(error,'cleanup',None)
                    if callable(cleanup):
                        self._owned[key]=_ConstructionCleanup(cleanup);self._pending.add(key)
                        error.add_note('failed construction retains its aggregate memory reservation until cleanup succeeds')
                    else: self._reserved -= self.executor_bytes
                else:
                    try:
                        executor.close()
                    except BaseException as cleanup:
                        self._pending.add(key)
                        error.add_note('aggregate executor reservation retained for failed cleanup: '+str(cleanup))
                    else:
                        self._owned.pop(key,None)
                        self._reserved -= self.executor_bytes
                raise
            finally:
                self._creating = False

    def release(self,key):
        with self._lock:
            if key not in self._owned:
                return
            if key in self._closing:
                raise RuntimeError('executor cleanup reentry')
            self._closing.add(key)
            try:
                self._owned[key].close()
            except BaseException:
                self._pending.add(key)
                raise
            else:
                del self._owned[key]
                self._pending.discard(key)
                self._reserved -= self.executor_bytes
            finally:
                self._closing.discard(key)

    def retry_cleanup(self):
        with self._lock:
            errors = []
            for key in tuple(self._pending):
                try:
                    self.release(key)
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise RuntimeError('aggregate executor cleanup remains incomplete') from errors[0]

    def status(self):
        with self._lock:
            return dict(executors=len(self._owned),reserved_bytes=self._reserved,
                max_total_bytes=self.total_limit,pending_cleanup=len(self._pending))


__all__ = ['BudgetedExecutorFactory']
