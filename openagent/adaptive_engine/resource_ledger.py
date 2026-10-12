"""Declared native resource ownership across construction and cleanup.

Reservations cover each owner's complete operation budget, including staging.
They are deliberately conservative when owners borrow the same expert store.
Failed cleanup keeps both the owner and reservation reachable for retry.
"""
from dataclasses import dataclass
import threading
from uuid import uuid4


@dataclass(frozen=True)
class ResourceReceipt:
    ledger: str
    key: str
    kind: str
    bytes: int


class ResourceLedger:
    def __init__(self, *, max_bytes, max_resources=65536):
        if any(type(v) is not int or v <= 0 for v in (max_bytes, max_resources)):
            raise ValueError('positive aggregate resource limits required')
        self.max_bytes, self.max_resources = max_bytes, max_resources
        self.identity = uuid4().hex
        self._lock = threading.RLock()
        self._receipts, self._owners, self._pending = {}, {}, {}
        self._closing = set()
        self._used = 0
        self._closed = False

    def reserve(self, kind, amount):
        if type(kind) is not str or not kind or type(amount) is not int or amount <= 0:
            raise ValueError('named positive resource reservation required')
        with self._lock:
            if self._closed:
                raise RuntimeError('resource ledger no longer admits owners')
            if len(self._receipts) >= self.max_resources or amount > self.max_bytes-self._used:
                raise MemoryError('aggregate native resource admission exhausted')
            receipt = ResourceReceipt(self.identity, uuid4().hex, kind, amount)
            self._receipts[receipt.key] = receipt
            self._used += amount
            return receipt

    def _receipt(self, receipt):
        if (type(receipt) is not ResourceReceipt or receipt.ledger != self.identity or
                self._receipts.get(receipt.key) is not receipt):
            raise ValueError('foreign or released resource reservation')
        return receipt

    def adopt(self, receipt, owner):
        with self._lock:
            self._receipt(receipt)
            if receipt.key in self._owners or not callable(getattr(owner, 'close', None)):
                raise ValueError('one closeable owner required for reservation')
            if any(value is owner for value in self._owners.values()):
                raise ValueError('resource owner already belongs to this ledger')
            self._owners[receipt.key] = owner
            return LedgerResource(self, receipt, owner)

    def abandon(self, receipt):
        """Release a construction reservation which has acquired no owner."""
        with self._lock:
            self._receipt(receipt)
            if receipt.key in self._owners:
                raise RuntimeError('owned reservation must close its resource')
            del self._receipts[receipt.key]
            self._used -= receipt.bytes

    def release(self, receipt):
        with self._lock:
            if receipt.key not in self._receipts:
                return
            self._receipt(receipt)
            if receipt.key in self._closing:
                raise RuntimeError('resource cleanup reentry')
            owner = self._owners.get(receipt.key)
            if owner is None:
                self.abandon(receipt)
                return
            self._closing.add(receipt.key)
        try:
            owner.close()
        except BaseException as failure:
            with self._lock:
                self._pending[receipt.key] = failure
            raise
        else:
            with self._lock:
                del self._owners[receipt.key]
                del self._receipts[receipt.key]
                self._pending.pop(receipt.key, None)
                self._used -= receipt.bytes
        finally:
            with self._lock:
                self._closing.discard(receipt.key)

    def retry_cleanup(self):
        with self._lock:
            receipts = tuple(self._receipts[key] for key in self._pending)
        errors = []
        # Children must close before their engines, even after a failed pass.
        for receipt in sorted(receipts, key=lambda r: r.kind != 'session'):
            try:
                self.release(receipt)
            except BaseException as failure:
                errors.append(failure)
        if errors:
            raise RuntimeError('native resource cleanup remains pending') from errors[0]

    def close(self):
        with self._lock:
            self._closed = True
            receipts = tuple(self._receipts.values())
        errors = []
        for receipt in sorted(receipts, key=lambda r: r.kind != 'session'):
            try:
                self.release(receipt)
            except BaseException as failure:
                errors.append(failure)
        if errors:
            raise RuntimeError('native ledger shutdown remains pending') from errors[0]

    def status(self):
        with self._lock:
            return dict(reserved_bytes=self._used, max_bytes=self.max_bytes,
                        resources=len(self._receipts), owners=len(self._owners),
                        constructing=len(self._receipts)-len(self._owners),
                        pending_cleanup=len(self._pending), closed=self._closed)


class LedgerResource:
    def __init__(self, ledger, receipt, owner):
        self._ledger, self._receipt, self._owner = ledger, receipt, owner

    def __getattr__(self, name):
        if self._owner is None:
            raise RuntimeError('native resource is closed')
        return getattr(self._owner, name)

    def close(self):
        if self._owner is not None:
            self._ledger.release(self._receipt)
            self._owner = None


__all__ = ['ResourceLedger', 'ResourceReceipt', 'LedgerResource']
