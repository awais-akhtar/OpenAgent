"""Model-free exact ownership and rollback tests for native GLM5 snapshots."""
from dataclasses import replace
import threading
import unittest

import numpy as np

from openagent.adaptive_engine.native_glm5_next import NativeGlm5NextTextDecoder


class _StableLibrary:
    def assert_unchanged(self):
        return None


def _owner(*, budget=4096):
    decoder = NativeGlm5NextTextDecoder.__new__(NativeGlm5NextTextDecoder)
    decoder._lock = threading.RLock()
    decoder._closed = False
    decoder._executing = False
    decoder._library_identity = _StableLibrary()
    decoder._snapshot_owner = object()
    decoder._snapshots = {}
    decoder._snapshot_bytes = 0
    decoder.max_snapshots = 2
    decoder._resident_bytes = 64
    decoder.max_bytes = budget
    decoder._cache = ((np.arange(6, dtype=np.float32).reshape(2, 3),
                       np.arange(4, dtype=np.float32).reshape(2, 2)),)
    decoder.position = 3
    return decoder


class NativeGlm5SnapshotOwnershipTests(unittest.TestCase):
    def test_snapshot_restore_recovers_all_physical_banks_and_position(self):
        decoder = _owner()
        receipt = decoder.snapshot()
        decoder._cache[0][0].fill(91)
        decoder._cache[0][1].fill(73)
        decoder.position = 8

        decoder.restore(receipt)

        self.assertEqual(decoder.position, 3)
        np.testing.assert_array_equal(decoder._cache[0][0], np.arange(6, dtype=np.float32).reshape(2, 3))
        np.testing.assert_array_equal(decoder._cache[0][1], np.arange(4, dtype=np.float32).reshape(2, 2))
        decoder.release_snapshot(receipt)
        self.assertEqual(decoder._snapshot_bytes, 0)

    def test_foreign_forged_and_released_receipts_reject(self):
        decoder = _owner()
        other = _owner()
        receipt = decoder.snapshot()
        with self.assertRaisesRegex(ValueError, "foreign"):
            other.restore(receipt)
        with self.assertRaisesRegex(ValueError, "forged"):
            decoder.restore(replace(receipt, position=receipt.position + 1))
        decoder.release_snapshot(receipt)
        with self.assertRaisesRegex(ValueError, "released or forged"):
            decoder.restore(receipt)

    def test_snapshot_budget_is_enforced_before_receipt_creation(self):
        decoder = _owner(budget=100)
        with self.assertRaisesRegex(MemoryError, "snapshot exceeds"):
            decoder.snapshot()
        self.assertEqual(decoder._snapshots, {})


if __name__ == "__main__":
    unittest.main()
