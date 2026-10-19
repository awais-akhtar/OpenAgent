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
    queue_wait_seconds: float = 0.0
    queue_position: int = 0


class AdmissionRejected(MemoryError):
    """The bounded queue or active reservation capacity is exhausted."""


class AdmissionTimedOut(TimeoutError):
    """A request did not acquire capacity before its wait deadline."""

class AdmissionScheduler:
    def __init__(self,*,max_active,max_bytes,max_waiting):
        if any(type(v) is not int or v<=0 for v in (max_active,max_bytes,max_waiting)):
            raise ValueError('positive finite scheduler capacities required')
        self.max_active,self.max_bytes,self.max_waiting=max_active,max_bytes,max_waiting
        self.identity=uuid4().hex;self._condition=threading.Condition(threading.RLock())
        self._queue=deque();self._leases={};self._used=0;self._closed=False
        self._admitted=self._rejected=self._timed_out=self._cancelled=0
        self._wait_seconds=0.0
        self._wait_buckets=(0.01,0.05,0.1,0.5,1.0,5.0)
        self._wait_histogram=[0]*(len(self._wait_buckets)+1)

    def try_acquire(self,amount):
        """Admit immediately or reject; never waits behind a queued request."""
        if type(amount) is not int or not 0<amount<=self.max_bytes:
            raise MemoryError('request exceeds complete scheduler byte capacity')
        with self._condition:
            if self._closed:raise RuntimeError('scheduler is closed')
            if (self._queue or len(self._leases)>=self.max_active or
                    amount>self.max_bytes-self._used):
                self._rejected+=1
                raise AdmissionRejected('request admission capacity exhausted')
            lease=AdmissionLease(self.identity,uuid4().hex,amount)
            self._leases[lease.key]=lease;self._used+=amount;self._admitted+=1
            self._wait_histogram[0]+=1
            self._condition.notify_all()
            return lease

    def acquire(self,amount,*,cancelled=None,timeout=None):
        if type(amount) is not int or not 0<amount<=self.max_bytes:
            raise MemoryError('request exceeds complete scheduler byte capacity')
        if cancelled is not None and not callable(cancelled):
            raise TypeError('callable admission cancellation required')
        if timeout is not None and (type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0):
            raise ValueError('positive finite admission deadline required')
        deadline=None if timeout is None else time.monotonic()+timeout
        ticket=object();queued=False;started=time.monotonic();position=0
        try:
            with self._condition:
                if self._closed: raise RuntimeError('scheduler is closed')
                if len(self._queue)>=self.max_waiting:
                    self._rejected+=1
                    raise AdmissionRejected('pending generation queue exhausted')
                self._queue.append(ticket);queued=True
                position=len(self._queue)
            while True:
                # Application cancellation executes outside scheduler locks.
                if cancelled is not None and cancelled():
                    with self._condition:self._cancelled+=1
                    raise InterruptedError('generation cancelled during admission')
                with self._condition:
                    if self._closed: raise RuntimeError('scheduler closed during admission')
                    remaining=None if deadline is None else deadline-time.monotonic()
                    if remaining is not None and remaining<=0:
                        self._timed_out+=1
                        raise AdmissionTimedOut('generation admission deadline expired')
                    if (self._queue[0] is ticket and len(self._leases)<self.max_active and
                            amount<=self.max_bytes-self._used):
                        waited=time.monotonic()-started
                        lease=AdmissionLease(self.identity,uuid4().hex,amount,waited,position)
                        self._leases[lease.key]=lease;self._used+=amount
                        self._admitted+=1;self._wait_seconds+=waited
                        bucket=next((i for i,limit in enumerate(self._wait_buckets)
                                     if waited<=limit),len(self._wait_buckets))
                        self._wait_histogram[bucket]+=1
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
                        max_bytes=self.max_bytes,max_active=self.max_active,max_waiting=self.max_waiting,
                        admitted=self._admitted,rejected=self._rejected,timed_out=self._timed_out,
                        cancelled=self._cancelled,wait_seconds=self._wait_seconds,
                        wait_buckets=list(self._wait_buckets),wait_histogram=list(self._wait_histogram),
                        closed=self._closed)

__all__=['AdmissionScheduler','AdmissionLease','AdmissionRejected','AdmissionTimedOut']
