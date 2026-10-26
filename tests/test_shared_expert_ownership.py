"""Host expert cache ownership and registry lifecycle, not accelerator placement."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.shared_expert_store import SharedExpertStoreRegistry


class SharedExpertOwnershipTests(unittest.TestCase):
    def lease(self, registry=None, **options):
        registry = registry or SharedExpertStoreRegistry()
        options.setdefault('max_ram_experts', 1)
        options.setdefault('max_vram_experts', 0)
        lease = registry.open_store('checkpoint', Path.cwd(), **options)
        self.addCleanup(lease.close)
        return lease

    def test_payload_detaches_borrowed_bytes_and_metadata_is_immutable(self):
        lease = self.lease()
        source = bytearray(b'weights')
        entry = lease.put('expert', 0, source)
        source[:] = b'changed'
        self.assertEqual(entry.weight_bytes, b'weights')
        with self.assertRaises(FrozenInstanceError):
            entry.weight_bytes = b'other'
        self.assertIs(lease.get('expert'), entry)

    def test_conflicting_expert_identity_is_rejected(self):
        lease = self.lease()
        entry = lease.put('expert', 0, b'weights')
        for layer, weights, dtype in ((1, b'weights', 'fp16'),
                                      (0, b'other', 'fp16'), (0, b'weights', 'int8')):
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                lease.put('expert', layer, weights, dtype)
        self.assertIs(lease.get('expert'), entry)

    def test_pinned_tier_exhaustion_cannot_overcommit(self):
        lease = self.lease()
        entry = lease.put('first', 0, b'first')
        lease.pin('first')
        with self.assertRaisesRegex(MemoryError, 'pinned entries'):
            lease.put('second', 0, b'second')
        self.assertEqual(lease.metrics()['ram_experts'], 1)
        self.assertIs(lease.get('first'), entry)
        lease.unpin('first')
        lease.put('second', 0, b'second')
        self.assertEqual(lease.metrics()['ram_experts'], 1)

    def test_pin_is_idempotent_and_foreign_lease_cannot_unpin(self):
        registry = SharedExpertStoreRegistry()
        first, second = self.lease(registry), self.lease(registry)
        entry = first.put('expert', 0, b'weights')
        first.pin('expert')
        first.pin('expert')
        second.unpin('expert')
        self.assertEqual(entry.pin_count, 1)
        second.pin('expert')
        first.close()
        self.assertEqual(entry.pin_count, 1)
        second.close()
        self.assertEqual(entry.pin_count, 0)
        self.assertEqual(registry.active_stores(), 0)

    def test_closed_lease_rejects_reads_writes_and_routing(self):
        lease = self.lease()
        lease.close()
        calls = (lambda: lease.get('expert'), lambda: lease.put('expert', 0, b'weights'),
                 lambda: lease.pin('expert'), lambda: lease.unpin('expert'),
                 lambda: lease.submit_prefetch(0, ['expert']),
                 lambda: lease.record_routing(['expert']), lambda: lease.hot_experts(),
                 lambda: lease.promote_expert('expert', 'vram'))
        for call in calls:
            with self.assertRaisesRegex(RuntimeError, 'lease is closed'):
                call()

    def test_registry_rejects_different_roots_and_capacities(self):
        registry = SharedExpertStoreRegistry()
        lease = self.lease(registry)
        for root, ram, vram in ((Path.cwd().parent, 1, 0), (Path.cwd(), 2, 0),
                                (Path.cwd(), 1, 1), (Path.cwd(), True, 0)):
            with self.assertRaises(ValueError):
                registry.open_store('checkpoint', root, ram, vram)
        self.assertEqual(lease.metrics()['lease_count'], 1)

    def test_disabled_tier_and_unbound_promotion_fail_closed(self):
        lease = self.lease()
        entry = lease.put('expert', 0, b'weights')
        with self.assertRaisesRegex(MemoryError, 'disabled'):
            lease.put('device-expert', 0, b'weights', tier='vram')
        with self.assertRaisesRegex(ValueError, 'physical transfer'):
            lease.promote_expert('expert', 'vram')
        self.assertEqual(entry.tier, 'ram')
        self.assertEqual(lease.metrics()['vram_experts'], 0)

    def test_last_close_clears_cache_owned_payloads_and_prefetch(self):
        lease = self.lease()
        lease.put('expert', 0, b'weights')
        lease.record_routing(['expert'])
        lease.submit_prefetch(0, ['expert'])
        shard = lease._shard
        lease.close()
        self.assertEqual(shard.metrics()['ram_experts'], 0)
        self.assertEqual(shard.metrics()['heat_tracked'], 0)
        self.assertEqual(shard.metrics()['prefetch_queue_len'], 0)

    def test_failed_lease_construction_does_not_leak_store_references(self):
        registry = SharedExpertStoreRegistry()
        first = self.lease(registry)
        with patch('openagent.adaptive_engine.shared_expert_store.SharedExpertLease',
                   side_effect=OSError('lease construction failed')):
            with self.assertRaises(OSError):
                registry.open_store('checkpoint', Path.cwd(), 1, 0)
            with self.assertRaises(OSError):
                registry.open_store('new-checkpoint', Path.cwd(), 1, 0)
        self.assertEqual(first.metrics()['lease_count'], 1)
        self.assertEqual(registry.active_stores(), 1)

    def test_release_and_reopen_are_atomic_for_expert_shard(self):
        registry = SharedExpertStoreRegistry()
        first = self.lease(registry)
        old = first._shard
        entered, release, opened = threading.Event(), threading.Event(), threading.Event()
        failures, leases = [], []
        original = old.release

        def delayed_release():
            remaining = original()
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test release not resumed')
            return remaining

        def close():
            try:
                first.close()
            except BaseException as error:
                failures.append(error)

        def reopen():
            try:
                leases.append(registry.open_store('checkpoint', Path.cwd(), 1, 0))
            except BaseException as error:
                failures.append(error)
            finally:
                opened.set()

        with patch.object(old, 'release', side_effect=delayed_release):
            closer = threading.Thread(target=close)
            closer.start()
            try:
                self.assertTrue(entered.wait(1))
                opener = threading.Thread(target=reopen)
                opener.start()
                self.assertFalse(opened.wait(0.05))
            finally:
                release.set()
                closer.join(2)
                if 'opener' in locals():
                    opener.join(2)
        self.assertEqual(failures, [])
        self.assertEqual(len(leases), 1)
        self.addCleanup(leases[0].close)
        self.assertIsNot(leases[0]._shard, old)
        self.assertEqual(registry.active_stores(), 1)
        self.assertEqual(leases[0].metrics()['lease_count'], 1)


if __name__ == '__main__':
    unittest.main()
