"""Keep concrete publisher arithmetic alive across composed neural operations."""
from contextlib import nullcontext
from functools import wraps


def arithmetic_scope(provider):
    arithmetic=getattr(provider,'arithmetic',provider)
    scope=getattr(arithmetic,'execution_scope',None)
    # Lightweight equation providers may have no native/device resources.
    # Concrete publisher arithmetic always supplies an execution scope.
    return scope() if callable(scope) else nullcontext()


def owned_auxiliary(operation):
    @wraps(operation)
    def owned(self,*args,**kwargs):
        lock = getattr(self, '_lock', None)
        if lock is None:
            with arithmetic_scope(self.provider):
                return operation(self,*args,**kwargs)
        with lock:
            if getattr(self, '_busy', False):
                raise RuntimeError('auxiliary owner is reentrant')
            if getattr(self, '_closed', False):
                raise RuntimeError('auxiliary owner is closed')
            self._busy = True
            try:
                with arithmetic_scope(self.provider):
                    return operation(self,*args,**kwargs)
            finally:
                self._busy = False
    return owned
