"""FIFO count/byte admission with cancellation and bounded pending requests."""
from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from uuid import uuid4

@dataclass(frozen=True)
class AdmissionLease:
    scheduler: str
    key: str
    bytes: int

class AdmissionScheduler:
    def __init__(self,*,max_active,max_bytes,max_waiting):
        if any(type(v) is not int or v<=0 for v in (max_active,max_bytes,max_waiting)):
            raise ValueError('positive finite scheduler capacities required')
        self.max_active,self.max_bytes,self.max_waiting=max_active,max_bytes,max_waiting
        self.identity=uuid4().hex;self._condition=threading.Condition(threading.RLock())
        self._queue=deque();self._leases={};self._used=0;self._closed=False

    def acquire(self,amount,*,cancelled=None,timeout=None):
        if type(amount) is not int or not 0<amount<=self.max_bytes:
            raise MemoryError('request exceeds complete scheduler byte capacity')
        if cancelled is not None and not callable(cancelled):
            raise TypeError('callable admission cancellation required')
        if timeout is not None and (type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0):
            raise ValueError('positive finite admission deadline required')
        deadline=None if timeout is None else time.monotonic()+timeout
        ticket=object();queued=False
        try:
            with self._condition:
                if self._closed: raise RuntimeError('scheduler is closed')
                if len(self._queue)>=self.max_waiting: raise MemoryError('pending generation queue exhausted')
                self._queue.append(ticket);queued=True
            while True:
                # Application cancellation executes outside scheduler locks.
                if cancelled is not None and cancelled():
                    raise InterruptedError('generation cancelled during admission')
                with self._condition:
                    if self._closed: raise RuntimeError('scheduler closed during admission')
                    remaining=None if deadline is None else deadline-time.monotonic()
                    if remaining is not None and remaining<=0:
                        raise TimeoutError('generation admission deadline expired')
                    if (self._queue[0] is ticket and len(self._leases)<self.max_active and
                            amount<=self.max_bytes-self._used):
                        lease=AdmissionLease(self.identity,uuid4().hex,amount)
                        self._leases[lease.key]=lease;self._used+=amount
                        self._queue.popleft();queued=False;self._condition.notify_all()
                        return lease
                    self._condition.wait(min(0.05,remaining) if remaining is not None else 0.05)
        finally:
            if queued:
                with self._condition:
                    self._queue.remove(ticket);self._condition.notify_all()

    def release(self,lease):
        with self._condition:
            if (type(lease) is not AdmissionLease or lease.scheduler!=self.identity or
                    self._leases.get(lease.key) is not lease):
                raise ValueError('foreign, forged or released admission lease')
            del self._leases[lease.key];self._used-=lease.bytes;self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed=True;self._condition.notify_all()

    def status(self):
        with self._condition:
            return dict(active=len(self._leases),waiting=len(self._queue),reserved_bytes=self._used,
                        max_bytes=self.max_bytes,max_active=self.max_active,max_waiting=self.max_waiting,closed=self._closed)

__all__=['AdmissionScheduler','AdmissionLease']
