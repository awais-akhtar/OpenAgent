import threading
import time
import unittest

from openagent.adaptive_engine.admission_scheduler import (
    AdmissionRejected, AdmissionScheduler, AdmissionTimedOut,
)


class AdmissionSchedulerTests(unittest.TestCase):
    def test_try_acquire_rejects_when_active_or_waiting(self):
        scheduler = AdmissionScheduler(max_active=1, max_bytes=8, max_waiting=1)
        active = scheduler.try_acquire(8)
        with self.assertRaises(AdmissionRejected):
            scheduler.try_acquire(1)
        scheduler.release(active)
        self.assertEqual(scheduler.status()['rejected'], 1)

    def test_waiting_timeout_is_counted_and_releases_queue_position(self):
        scheduler = AdmissionScheduler(max_active=1, max_bytes=8, max_waiting=2)
        active = scheduler.try_acquire(8)
        outcome = []

        def wait():
            try:
                scheduler.acquire(8, timeout=0.05)
            except BaseException as error:
                outcome.append(error)

        worker = threading.Thread(target=wait)
        worker.start()
        deadline = time.monotonic() + 1
        while scheduler.status()['waiting'] != 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(scheduler.status()['waiting'], 1)
        worker.join(1)
        scheduler.release(active)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(outcome[0], AdmissionTimedOut)
        status = scheduler.status()
        self.assertEqual((status['waiting'], status['timed_out']), (0, 1))

    def test_acquired_lease_carries_wait_and_scheduler_reports_histogram(self):
        scheduler = AdmissionScheduler(max_active=1, max_bytes=8, max_waiting=2)
        active = scheduler.try_acquire(8)
        outcome = []

        def wait():
            outcome.append(scheduler.acquire(8, timeout=1))

        worker = threading.Thread(target=wait)
        worker.start()
        deadline = time.monotonic() + 1
        while scheduler.status()['waiting'] != 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        scheduler.release(active)
        worker.join(1)
        self.assertFalse(worker.is_alive())
        lease = outcome[0]
        self.assertGreaterEqual(lease.queue_position, 1)
        self.assertGreaterEqual(lease.queue_wait_seconds, 0)
        status = scheduler.status()
        self.assertEqual(status['admitted'], 2)
        self.assertEqual(sum(status['wait_histogram']), 2)
        scheduler.release(lease)

    def test_cancelled_waiter_leaves_bounded_queue(self):
        scheduler = AdmissionScheduler(max_active=1, max_bytes=8, max_waiting=2)
        active = scheduler.try_acquire(8)
        cancelled = threading.Event()
        outcome = []

        def wait():
            try:
                scheduler.acquire(8, cancelled=cancelled.is_set)
            except BaseException as error:
                outcome.append(error)

        worker = threading.Thread(target=wait)
        worker.start()
        deadline = time.monotonic() + 1
        while scheduler.status()['waiting'] != 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        cancelled.set()
        worker.join(1)
        scheduler.release(active)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(outcome[0], InterruptedError)
        self.assertEqual(scheduler.status()['waiting'], 0)
        self.assertEqual(scheduler.status()['cancelled'], 1)

    def test_close_wakes_pending_waiters_and_rejects_new_work(self):
        scheduler = AdmissionScheduler(max_active=1, max_bytes=8, max_waiting=2)
        active = scheduler.try_acquire(8)
        outcome = []

        def wait():
            try:
                scheduler.acquire(8)
            except BaseException as error:
                outcome.append(error)

        worker = threading.Thread(target=wait)
        worker.start()
        deadline = time.monotonic() + 1
        while scheduler.status()['waiting'] != 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        scheduler.close()
        worker.join(1)
        scheduler.release(active)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(outcome[0], RuntimeError)
        self.assertEqual(scheduler.status()['waiting'], 0)
        with self.assertRaises(RuntimeError):
            scheduler.try_acquire(1)


if __name__ == '__main__':
    unittest.main()
