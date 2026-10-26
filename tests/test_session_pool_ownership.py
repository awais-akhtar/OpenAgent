"""Pool capacity, health-probe and retryable owner cleanup boundaries."""
import threading
import unittest
from unittest.mock import Mock

from openagent.adaptive_engine.session_pool import Phase2SessionPool


class _Owner:
    def __init__(self, failures=0):
        self.failures = failures
        self.closes = 0

    def close(self):
        self.closes += 1
        if self.failures:
            self.failures -= 1
            raise OSError('owner close failed')


class SessionPoolOwnershipTests(unittest.TestCase):
    def pool(self, factory, **options):
        options.setdefault('min_sessions', 0)
        options.setdefault('health_interval_s', 999)
        pool = Phase2SessionPool(factory, **options)
        self.addCleanup(pool.close)
        return pool

    def test_invalid_configuration_fails_before_starting_workers(self):
        for options in ({'min_sessions': -1}, {'min_sessions': 2, 'max_sessions': 1},
                        {'max_sessions': 0}, {'max_sessions': True},
                        {'max_idle_s': float('nan')}, {'health_interval_s': 0},
                        {'health_interval_s': float('inf')}, {'health_check_fn': 1}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                self.pool(Mock(), **options)

    def test_closed_pool_rejects_acquisition_without_opening_new_owner(self):
        factory = Mock(side_effect=_Owner)
        pool = self.pool(factory)
        pool.close()
        with self.assertRaisesRegex(RuntimeError, 'pool is closed'):
            pool.acquire(timeout=0)
        factory.assert_not_called()

    def test_concurrent_factory_openings_count_toward_capacity(self):
        entered, release = threading.Event(), threading.Event()
        owners, leases, errors = [], [], []

        def factory():
            owner = _Owner()
            owners.append(owner)
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test factory was not released')
            return owner

        pool = self.pool(factory, max_sessions=1)

        def acquire():
            try:
                leases.append(pool.acquire(timeout=1))
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=acquire)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(pool.metrics()['pool_opening'], 1)
            with self.assertRaises(TimeoutError):
                pool.acquire(timeout=0)
            self.assertEqual(len(owners), 1)
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        leases[0].close()
        self.assertEqual(pool.total_count, 1)

    def test_shutdown_during_factory_wait_retains_ownership_and_blocks_publication(self):
        entered, release = threading.Event(), threading.Event()
        owner, errors = _Owner(), []

        def factory():
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test factory was not released')
            return owner

        pool = self.pool(factory)

        def acquire():
            try:
                pool.acquire(timeout=1)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=acquire)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            with self.assertRaisesRegex(TimeoutError, 'construction or probe still active'):
                pool.close(timeout=0.1)
            self.assertEqual(owner.closes, 0)
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertEqual(owner.closes, 1)
        self.assertEqual(pool.total_count, 0)
        pool.close()

    def test_failed_idle_close_is_retained_and_retry_closes_all_remaining_owners(self):
        owners = [_Owner(failures=1), _Owner()]
        pool = self.pool(Mock(side_effect=owners), min_sessions=2)
        with self.assertRaisesRegex(OSError, 'owner close failed'):
            pool.close()
        self.assertEqual([owner.closes for owner in owners], [1, 1])
        self.assertEqual(pool.total_count, 1)
        self.assertEqual(pool.metrics()['pool_pending_cleanup'], 1)
        pool.close()
        self.assertEqual(pool.total_count, 0)
        self.assertEqual(owners[0].closes, 2)

    def test_failed_busy_lease_return_is_retryable_and_not_reusable(self):
        owner = _Owner(failures=1)
        pool = self.pool(lambda: owner)
        lease = pool.acquire(timeout=0)
        pool.close()
        self.assertEqual(owner.closes, 0)
        with self.assertRaisesRegex(OSError, 'owner close failed'):
            lease.close()
        self.assertEqual(pool.total_count, 1)
        with self.assertRaisesRegex(RuntimeError, 'returned'):
            _ = lease.session
        lease.close()
        lease.close()
        self.assertEqual(owner.closes, 2)
        self.assertEqual(pool.total_count, 0)

    def test_returned_lease_cannot_access_recycled_owner(self):
        owner = _Owner()
        pool = self.pool(lambda: owner)
        lease = pool.acquire(timeout=0)
        lease.close()
        with self.assertRaisesRegex(RuntimeError, 'returned'):
            _ = lease.session
        replacement = pool.acquire(timeout=0)
        self.assertIs(replacement.session, owner)
        replacement.close()

    def test_idle_limit_does_not_discard_session_for_time_spent_borrowed(self):
        owner = _Owner()
        pool = self.pool(lambda: owner, max_idle_s=1)
        lease = pool.acquire(timeout=0)
        entry = next(iter(pool._all.values()))
        entry.created_at -= 100
        lease.close()
        self.assertEqual(owner.closes, 0)
        self.assertEqual(pool.total_count, 1)

    def test_probe_blocks_shutdown_until_callback_releases_its_owner(self):
        entered, release = threading.Event(), threading.Event()
        owner, errors = _Owner(), []

        def probe(_session):
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test probe was not released')
            return True

        pool = self.pool(lambda: owner, min_sessions=1, health_check_fn=probe)

        def check():
            try:
                pool._run_health_check()
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=check)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            with self.assertRaisesRegex(TimeoutError, 'construction or probe still active'):
                pool.close(timeout=0.1)
            self.assertEqual(owner.closes, 0)
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(owner.closes, 1)
        self.assertEqual(pool.total_count, 0)

    def test_probe_close_failure_retains_owner_for_pool_teardown_retry(self):
        owner = _Owner(failures=1)
        pool = self.pool(lambda: owner, min_sessions=1, health_check_fn=lambda _owner: False)
        with self.assertRaisesRegex(OSError, 'owner close failed'):
            pool._run_health_check()
        self.assertEqual(pool.total_count, 1)
        self.assertEqual(pool.metrics()['pool_pending_cleanup'], 1)
        pool.close()
        self.assertEqual(owner.closes, 2)

    def test_lease_context_preserves_body_error_and_cleanup_retry(self):
        owner = _Owner(failures=1)
        pool = self.pool(lambda: owner)
        lease = pool.acquire(timeout=0)
        pool.close()
        primary = ValueError('request failed')
        with self.assertRaises(ValueError) as caught:
            with lease:
                raise primary
        self.assertIs(caught.exception, primary)
        primary.cleanup()
        self.assertEqual(owner.closes, 2)

    def test_factory_failure_refunds_capacity_reservation(self):
        owner = _Owner()
        pool = self.pool(Mock(side_effect=[OSError('factory failed'), owner]), max_sessions=1)
        with self.assertRaisesRegex(OSError, 'factory failed'):
            pool.acquire(timeout=0)
        self.assertEqual(pool.metrics()['pool_opening'], 0)
        lease = pool.acquire(timeout=0)
        self.assertIs(lease.session, owner)
        lease.close()

    def test_same_owner_cannot_be_borrowed_by_two_factory_slots(self):
        owner = _Owner()
        pool = self.pool(lambda: owner, max_sessions=2)
        first = pool.acquire(timeout=0)
        with self.assertRaisesRegex(ValueError, 'already owned session'):
            pool.acquire(timeout=0)
        self.assertEqual(owner.closes, 0)
        self.assertEqual(pool.total_count, 1)
        first.close()

    def test_owner_returned_after_shutdown_with_failed_close_remains_retryable(self):
        entered, release = threading.Event(), threading.Event()
        owner, errors = _Owner(failures=1), []

        def factory():
            entered.set()
            if not release.wait(2):
                raise TimeoutError('factory not released')
            return owner

        pool = self.pool(factory)

        def acquire():
            try:
                pool.acquire(timeout=1)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=acquire)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            with self.assertRaises(TimeoutError):
                pool.close(timeout=0.1)
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], OSError)
        self.assertEqual(pool.metrics()['pool_pending_cleanup'], 1)
        pool.close()
        self.assertEqual(owner.closes, 2)
        self.assertEqual(pool.total_count, 0)

    def test_background_health_join_timeout_does_not_close_probed_owner(self):
        entered, release = threading.Event(), threading.Event()
        owner = _Owner()

        def probe(_owner):
            entered.set()
            if not release.wait(2):
                raise TimeoutError('probe not released')
            return True

        pool = self.pool(lambda: owner, min_sessions=1,
                         health_interval_s=0.001, health_check_fn=probe)
        try:
            self.assertTrue(entered.wait(1))
            with self.assertRaisesRegex(TimeoutError, 'health worker still active'):
                pool.close(timeout=0.01)
            self.assertEqual(owner.closes, 0)
        finally:
            release.set()
            pool._health_thread.join(2)
        self.assertFalse(pool._health_thread.is_alive())
        pool.close()
        self.assertEqual(owner.closes, 1)

    def test_pool_context_preserves_body_error_and_retryable_teardown(self):
        owner = _Owner(failures=1)
        pool = self.pool(lambda: owner, min_sessions=1)
        primary = ValueError('body failed')
        with self.assertRaises(ValueError) as caught:
            with pool:
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertEqual(pool.metrics()['pool_pending_cleanup'], 1)
        primary.cleanup()
        self.assertEqual(owner.closes, 2)

    def test_invalid_acquisition_timeout_does_not_invoke_factory(self):
        factory = Mock(side_effect=_Owner)
        pool = self.pool(factory)
        for timeout in (True, -1, float('nan'), float('inf'), '1'):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                pool.acquire(timeout=timeout)
        factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
