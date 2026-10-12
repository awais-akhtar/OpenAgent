"""Ownership and atomic continuation operations for a decoder and draft stages."""
from dataclasses import dataclass
from uuid import uuid4
import threading

@dataclass(frozen=True)
class GroupedSnapshot:
    owner: str
    identity: str
    key: str
    position: int

class GroupedContinuation:
    def __init__(self,components,*,identity,max_snapshots):
        if not components or len(components)>64 or type(identity) is not str or not identity:
            raise ValueError('explicit identified continuation components required')
        if type(max_snapshots) is not int or not 1<=max_snapshots<=62:
            raise ValueError('bounded grouped snapshot count required')
        self.components=tuple(components);self.identity=identity;self.max_snapshots=max_snapshots
        self._owner=uuid4().hex;self._receipts={};self._lock=threading.RLock()

    def snapshot(self,*,_internal=False):
        with self._lock:
            if len(self._receipts)>=self.max_snapshots+(2 if _internal else 0):
                raise MemoryError('grouped snapshot count exceeds admission')
            values=[];key=uuid4().hex
            try:
                for component in self.components: values.append(component.snapshot())
            except BaseException as error:
                self._receipts[key]=values
                receipt=GroupedSnapshot(self._owner,self.identity,key,self.components[0].position)
                try: self.release_snapshot(receipt)
                except BaseException as failure:
                    error.cleanup=lambda:self.release_snapshot(receipt)
                    error.add_note('grouped partial snapshot cleanup remains pending: '+str(failure))
                raise
            self._receipts[key]=values
            return GroupedSnapshot(self._owner,self.identity,key,self.components[0].position)

    def _owned(self,receipt):
        if type(receipt) is not GroupedSnapshot or receipt.owner!=self._owner or receipt.identity!=self.identity or receipt.key not in self._receipts:
            raise ValueError('owned grouped continuation snapshot required')
        return self._receipts[receipt.key]

    def release_snapshot(self,receipt):
        with self._lock:
            values=self._owned(receipt)
            for index,(component,value) in enumerate(zip(self.components,values)):
                if value is not None: component.release_snapshot(value);values[index]=None
            del self._receipts[receipt.key]

    def restore(self,receipt):
        with self._lock:
            values=self._owned(receipt)
            if len(values)!=len(self.components) or any(value is None for value in values):
                raise ValueError('incomplete grouped snapshot cannot be restored')
            rollback=self.snapshot(_internal=True)
            try:
                for component,value in zip(self.components,values): component.restore(value)
            except BaseException as error:
                for component,value in zip(self.components,self._owned(rollback)):
                    try: component.restore(value)
                    except BaseException as failure: error.add_note('grouped rollback failed: '+str(failure))
                raise
            finally: self.release_snapshot(rollback)

    def reset(self):
        with self._lock:
            rollback=self.snapshot(_internal=True)
            try:
                for component in self.components: component.reset()
            except BaseException:
                self.restore(rollback);raise
            finally: self.release_snapshot(rollback)

    def write_cache(self,output,*,max_bytes):
        from .compound_cache import write_components
        with self._lock:
            return write_components(output,self.components,identity=self.identity,max_bytes=max_bytes)

    def restore_cache(self,source,*,max_bytes):
        from .compound_cache import restore_components
        with self._lock:
            return restore_components(source,self.components,identity=self.identity,max_bytes=max_bytes)

    def release_all(self):
        with self._lock:
            for key in tuple(self._receipts):
                self.release_snapshot(GroupedSnapshot(self._owner,self.identity,key,self.components[0].position))

__all__=['GroupedContinuation','GroupedSnapshot']
