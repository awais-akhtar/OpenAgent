"""Immutable source-name dispatch across borrowed projection owners."""
from contextlib import ExitStack, contextmanager
import hashlib
import threading
from types import MappingProxyType


class ProjectionDispatch:
    """Compose resident, packed and streamed banks without copying weights.

    Ownership stays with the caller. Closing the dispatch severs its references
    only after all calls finish; banks must be closed afterwards.
    """
    def __init__(self, assignments):
        if type(assignments) is not dict or not assignments:
            raise ValueError('explicit nonempty source-to-projection-owner mapping required')
        shapes={}; owners={}; identities={}
        for name,owner in assignments.items():
            if type(name) is not str or not name:
                raise ValueError('exact projection source name required')
            for method in ('handles','shape','linear'):
                if not callable(getattr(owner,method,None)):
                    raise TypeError('projection owner missing '+method)
            identity=getattr(owner,'identity',None)
            if type(identity) is not str or not identity or not owner.handles(name):
                raise ValueError('identified projection owner must admit its assigned source')
            shape=tuple(owner.shape(name))
            if len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape):
                raise ValueError('positive matrix projection geometry required')
            shapes[name]=shape; owners[name]=owner; identities[id(owner)]=(owner,identity)
        self._owners=MappingProxyType(owners); self._shapes=MappingProxyType(shapes)
        self._identities=tuple(identities.values())
        self.identity=hashlib.sha256(repr(tuple((name,shapes[name],owners[name].identity)
            for name in sorted(owners))).encode()).hexdigest()
        self._lock=threading.RLock();self._closed=False;self._active=0

    def _ensure_open(self):
        if self._closed:raise RuntimeError('projection dispatch closed')
        for owner,identity in self._identities:
            if owner.identity!=identity:raise RuntimeError('borrowed projection identity changed')

    def handles(self,name):
        with self._lock:
            self._ensure_open();return name in self._owners

    def shape(self,name):
        with self._lock:
            self._ensure_open();return self._shapes[name]

    def contains_projection_owner(self,owner):
        with self._lock:
            self._ensure_open()
            return any(bank is owner or (callable(getattr(bank,'contains_projection_owner',None))
                and bank.contains_projection_owner(owner)) for bank,_ in self._identities)

    @contextmanager
    def cancellation_scope(self,event):
        with self._lock:
            self._ensure_open();self._active+=1
        try:
            with ExitStack() as scopes:
                for owner,_ in self._identities:
                    scope=getattr(owner,'cancellation_scope',None)
                    if callable(scope):scopes.enter_context(scope(event))
                yield self
        finally:
            with self._lock:self._active-=1

    def _invoke(self,name,value,groups):
        with self._lock:
            self._ensure_open();owner=self._owners[name];self._active+=1
        try:
            if groups is None:return owner.linear(name,value)
            method=getattr(owner,'grouped_linear',None)
            if not callable(method):raise ValueError('assigned owner lacks grouped projection arithmetic')
            return method(name,value,groups=groups)
        finally:
            with self._lock:self._active-=1

    def linear(self,name,value):return self._invoke(name,value,None)
    def grouped_linear(self,name,value,*,groups):return self._invoke(name,value,groups)
    def close(self):
        with self._lock:
            if self._active:raise RuntimeError('cannot close an active projection dispatch')
            self._closed=True;self._owners=MappingProxyType({});self._identities=()


__all__=['ProjectionDispatch']
