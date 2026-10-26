"""Shared host KV storage integrity and lease ownership, not model-route evidence."""
from dataclasses import FrozenInstanceError
import threading
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.shared_kv_cache import (
    KVBlock, SharedKVCacheRegistry, SharedKVCacheShard,
)


TOKENS = list(range(8))


def tensor(value=1, *, batch=False):
    shape = (1, 8, 1, 1) if batch else (8, 1, 1)
    return np.full(shape, value, dtype=np.float32)


class SharedKVOwnershipTests(unittest.TestCase):
    def lease(self, registry=None, **options):
        registry = registry or SharedKVCacheRegistry()
        options.setdefault('expected_layers', 1)
        lease = registry.open_cache('checkpoint', **options)
        self.addCleanup(lease.close)
        return lease

    def test_storage_is_detached_and_cannot_be_made_writeable(self):
        lease = self.lease()
        key, value = tensor(), tensor(2)
        entry = lease.store_block(TOKENS, 0, key, value)
        key.fill(9)
        value.fill(9)
        block = entry.get_block(0)
        self.assertTrue(np.all(block.key == 1))
        self.assertTrue(np.all(block.value == 2))
        with self.assertRaises(ValueError):
            block.key.flags.writeable = True
        with self.assertRaises(TypeError):
            entry.blocks[1] = block
        with self.assertRaises(FrozenInstanceError):
            block.dtype = 'fp32'
        with self.assertRaisesRegex(RuntimeError, 'through their shard'):
            entry.add_block(KVBlock(1, key, value))
        block.key.shape = (8,)
        self.assertEqual(entry.get_block(0).key.shape, (8, 1, 1))

    def test_replacing_layer_does_not_double_count_bytes(self):
        lease = self.lease(max_bytes=32)
        lease.store_block(TOKENS, 0, tensor(), tensor())
        lease.store_block(TOKENS, 0, tensor(2), tensor(2))
        self.assertEqual(lease.metrics()['kv_bytes_cached'], 32)
        self.assertEqual(lease.metrics()['kv_entries'], 1)

    def test_partial_layers_are_not_advertised_as_complete_prefix(self):
        lease = self.lease(expected_layers=2)
        entry = lease.store_block(TOKENS, 0, tensor(), tensor())
        self.assertFalse(entry.is_complete)
        self.assertIsNone(lease.find_prefix(TOKENS))
        lease.store_block(TOKENS, 1, tensor(), tensor())
        self.assertTrue(entry.is_complete)
        self.assertIs(lease.find_prefix(TOKENS + [9]), entry)

    def test_unknown_layer_inventory_does_not_claim_complete_cache(self):
        lease = self.lease(expected_layers=None)
        entry = lease.store_block(TOKENS, 0, tensor(), tensor())
        self.assertFalse(entry.is_complete)
        self.assertIsNone(lease.find_prefix(TOKENS))
        self.assertIsNone(lease.find_prefix([]))

    def test_pinned_capacity_failure_is_atomic(self):
        lease = self.lease(max_entries=1, max_bytes=32)
        entry = lease.store_block(TOKENS, 0, tensor(), tensor())
        lease.pin(entry.prefix_hash)
        before = lease.metrics()
        with self.assertRaises(MemoryError):
            lease.store_block(list(range(1, 9)), 0, tensor(), tensor())
        self.assertEqual(lease.metrics(), before)
        self.assertIs(lease.find_prefix(TOKENS), entry)
        lease.unpin(entry.prefix_hash)
        replacement = lease.store_block(list(range(1, 9)), 0, tensor(), tensor())
        self.assertNotEqual(replacement.prefix_hash, entry.prefix_hash)
        self.assertEqual(lease.metrics()['kv_bytes_cached'], 32)

    def test_growing_protected_prefix_cannot_exceed_budget(self):
        lease = self.lease(expected_layers=2, max_bytes=32)
        entry = lease.store_block(TOKENS, 0, tensor(), tensor())
        with self.assertRaises(MemoryError):
            lease.store_block(TOKENS, 1, tensor(), tensor())
        self.assertEqual(set(entry.blocks), {0})
        self.assertEqual(lease.metrics()['kv_bytes_cached'], 32)

    def test_pinned_prefix_cannot_be_mutated(self):
        lease = self.lease()
        entry = lease.store_block(TOKENS, 0, tensor(), tensor())
        lease.pin(entry.prefix_hash)
        with self.assertRaisesRegex(RuntimeError, 'pinned KV'):
            lease.store_block(TOKENS, 0, tensor(3), tensor(3))
        self.assertTrue(np.all(entry.get_block(0).key == 1))

    def test_pin_is_idempotent_and_unpin_is_lease_scoped(self):
        registry = SharedKVCacheRegistry()
        first, second = self.lease(registry), self.lease(registry)
        entry = first.store_block(TOKENS, 0, tensor(), tensor())
        first.pin(entry.prefix_hash)
        first.pin(entry.prefix_hash)
        second.unpin(entry.prefix_hash)
        self.assertEqual(entry.pin_count, 1)
        with self.assertRaisesRegex(RuntimeError, 'owned through a lease'):
            entry.unpin()
        second.pin(entry.prefix_hash)
        self.assertEqual(entry.pin_count, 2)
        first.close()
        self.assertEqual(entry.pin_count, 1)
        second.close()
        self.assertEqual(entry.pin_count, 0)
        self.assertEqual(registry.metrics()['kv_caches'], 0)

    def test_closed_lease_cannot_access_or_mutate_cache(self):
        lease = self.lease()
        lease.close()
        calls = (lambda: lease.find_prefix(TOKENS), lambda: lease.get_or_create(TOKENS),
                 lambda: lease.store_block(TOKENS, 0, tensor(), tensor()),
                 lambda: lease.pin('missing'), lambda: lease.unpin('missing'))
        for call in calls:
            with self.assertRaisesRegex(RuntimeError, 'lease is closed'):
                call()

    def test_existing_shard_requires_same_budgets_and_layer_inventory(self):
        registry = SharedKVCacheRegistry()
        self.lease(registry, max_entries=2, max_bytes=64)
        for options in ({'max_entries': 1, 'max_bytes': 64},
                        {'max_entries': 2, 'max_bytes': 32},
                        {'max_entries': 2, 'max_bytes': 64, 'expected_layers': 2}):
            options.setdefault('expected_layers', 1)
            with self.assertRaisesRegex(ValueError, 'differ'):
                registry.open_cache('checkpoint', **options)
        self.assertEqual(registry.metrics()['shards'][0]['kv_lease_count'], 1)

    def test_invalid_geometry_and_conversion_do_not_replace_good_storage(self):
        lease = self.lease()
        entry = lease.store_block(TOKENS, 0, tensor(), tensor())
        for bad in (np.ones((7, 1, 1), np.float32), np.ones((8, 1), np.float32),
                    tensor(np.nan), tensor(np.inf), tensor(1e30)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                lease.store_block(TOKENS, 0, bad, bad)
        self.assertTrue(np.all(entry.get_block(0).key == 1))
        self.assertEqual(lease.metrics()['kv_bytes_cached'], 32)

    def test_batched_storage_and_explicit_float32_budget(self):
        lease = self.lease(max_bytes=64)
        entry = lease.store_block(TOKENS, 0, tensor(batch=True), tensor(batch=True), dtype='fp32')
        self.assertEqual(entry.get_block(0).key.shape, (1, 8, 1, 1))
        self.assertEqual(lease.metrics()['kv_bytes_cached'], 64)

    def test_missing_pin_is_not_recorded_as_owned(self):
        lease = self.lease()
        with self.assertRaises(KeyError):
            lease.pin('missing')
        self.assertEqual(lease._pinned, set())

    def test_final_lease_close_releases_cache_owned_arrays(self):
        lease = self.lease()
        lease.store_block(TOKENS, 0, tensor(), tensor())
        shard = lease._shard
        lease.close()
        self.assertEqual(shard.metrics()['kv_entries'], 0)
        self.assertEqual(shard.metrics()['kv_bytes_cached'], 0)

    def test_readers_cannot_change_prefix_inventory_or_pin_accounting(self):
        lease = self.lease(expected_layers=2)
        entry = lease.store_block(TOKENS, 0, tensor(), tensor())
        for name, value in (('expected_layers', 1), ('prefix_len', 1), ('prefix_hash', 'other'),
                            ('token_ids', (9,)), ('pin_count', 0)):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(entry, name, value)
        self.assertFalse(entry.is_complete)

    def test_oversized_block_does_not_evict_existing_prefix(self):
        lease = self.lease(max_bytes=32)
        entry = lease.store_block(TOKENS, 0, tensor(), tensor())
        before = lease.metrics()
        with self.assertRaises(MemoryError):
            lease.store_block(list(range(1, 9)), 0, np.zeros((8, 2, 1), np.float32),
                              np.zeros((8, 2, 1), np.float32))
        self.assertEqual(lease.metrics(), before)
        self.assertIs(lease.find_prefix(TOKENS), entry)

    def test_failed_lease_construction_does_not_leak_reference_count(self):
        registry = SharedKVCacheRegistry()
        first = self.lease(registry)
        with patch('openagent.adaptive_engine.shared_kv_cache.SharedKVLease',
                   side_effect=OSError('lease construction failed')):
            with self.assertRaises(OSError):
                registry.open_cache('checkpoint', expected_layers=1)
            with self.assertRaises(OSError):
                registry.open_cache('new-checkpoint', expected_layers=1)
        self.assertEqual(first.metrics()['kv_lease_count'], 1)
        self.assertEqual(registry.metrics()['kv_caches'], 1)

    def test_lease_close_waits_for_inflight_store_and_clears_its_storage(self):
        lease = self.lease()
        shard = lease._shard
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        original = shard.store_block
        failures = []

        def store(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test store was not released')
            return original(*args, **kwargs)

        def write():
            try:
                lease.store_block(TOKENS, 0, tensor(), tensor())
            except BaseException as error:
                failures.append(error)

        def close():
            try:
                lease.close()
            except BaseException as error:
                failures.append(error)
            finally:
                closed.set()

        with patch.object(shard, 'store_block', side_effect=store):
            writer = threading.Thread(target=write)
            writer.start()
            try:
                self.assertTrue(entered.wait(1))
                closer = threading.Thread(target=close)
                closer.start()
                self.assertFalse(closed.wait(0.05))
            finally:
                release.set()
                writer.join(2)
                if 'closer' in locals():
                    closer.join(2)
        self.assertEqual(failures, [])
        self.assertTrue(closed.is_set())
        self.assertEqual(shard.metrics()['kv_bytes_cached'], 0)

    def test_lookup_hashes_only_existing_candidate_extent_and_returns_longest_match(self):
        lease = self.lease()
        short = lease.store_block(TOKENS, 0, tensor(), tensor())
        long_tokens = list(range(12))
        long_tensor = np.ones((12, 1, 1), np.float32)
        longest = lease.store_block(long_tokens, 0, long_tensor, long_tensor)
        import hashlib
        real = hashlib.sha256
        calls = []

        class Hasher:
            def __init__(self):
                self.inner = real()

            def update(self, value):
                calls.append(value)
                self.inner.update(value)

            def hexdigest(self):
                return self.inner.hexdigest()

        with patch('openagent.adaptive_engine.shared_kv_cache.hashlib.sha256', side_effect=Hasher):
            self.assertIs(lease.find_prefix(list(range(5000))), longest)
        self.assertEqual(len(calls), 12)
        self.assertIs(lease.find_prefix(TOKENS + [99]), short)

    def test_invalid_admission_inputs_fail_without_entries(self):
        for options in ({'max_entries': 0}, {'max_bytes': True}, {'expected_layers': 0},
                        {'expected_layers': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SharedKVCacheShard('checkpoint', **options)
        lease = self.lease()
        for tokens in ([], [True], [-1], [2**31], [1.5]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                lease.store_block(tokens, 0, tensor(), tensor())
        self.assertEqual(lease.metrics()['kv_entries'], 0)

    def test_release_and_reopen_cannot_split_one_live_checkpoint_shard(self):
        registry = SharedKVCacheRegistry()
        first = self.lease(registry)
        shard = first._shard
        entered, release, opened = threading.Event(), threading.Event(), threading.Event()
        leases, failures = [], []
        original_release = shard.release

        def delayed_release():
            remaining = original_release()
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test release not resumed')
            return remaining

        def close_first():
            try:
                first.close()
            except BaseException as error:
                failures.append(error)

        def reopen():
            try:
                leases.append(registry.open_cache('checkpoint', expected_layers=1))
            except BaseException as error:
                failures.append(error)
            finally:
                opened.set()

        with patch.object(shard, 'release', side_effect=delayed_release):
            closer = threading.Thread(target=close_first)
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
        self.assertIsNot(leases[0]._shard, shard)
        self.assertEqual(registry.metrics()['kv_caches'], 1)
        self.assertEqual(leases[0].metrics()['kv_lease_count'], 1)


if __name__ == '__main__':
    unittest.main()
