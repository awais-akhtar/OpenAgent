import threading
import unittest
from types import SimpleNamespace

from openagent.adaptive_engine.continuation_store import DurableContinuationStore


class ContinuationStoreLockOrderTest(unittest.TestCase):
    def test_session_operations_release_lock_when_store_is_busy(self):
        store = object.__new__(DurableContinuationStore)
        store._lock = threading.RLock()
        session = SimpleNamespace(lock=threading.RLock())
        service = SimpleNamespace(_session=lambda _session_id: session)
        held = threading.Event()
        release = threading.Event()

        def hold_store_lock():
            with store._lock:
                held.set()
                release.wait(2)

        holder = threading.Thread(target=hold_store_lock)
        holder.start()
        self.assertTrue(held.wait(1))
        try:
            with self.assertRaisesRegex(RuntimeError, "store is busy"):
                store.recover_into(service, "session", "key")
            with self.assertRaisesRegex(RuntimeError, "store is busy"):
                store.save(service, "session", "key")
            self.assertTrue(session.lock.acquire(blocking=False))
            session.lock.release()
        finally:
            release.set()
            holder.join(2)
        self.assertFalse(holder.is_alive())


if __name__ == "__main__":
    unittest.main()
