"""K3 FLA replacement cleanup failures quarantine the executor visibly."""
import threading
import unittest
import weakref

from openagent.adaptive_engine.k3_execution.executor import K3Execution


class _UnprintableCleanupError(RuntimeError):
    def __str__(self):
        raise AssertionError("cleanup stringification must not mask K3 failure")

    def __repr__(self):
        raise AssertionError("cleanup repr must not mask K3 failure")


class _Mixer:
    def __init__(self, failure):
        self.failure = failure
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        raise self.failure


class _PackedOwner:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.close_calls == 1:
            raise OSError("native library is still in use")


def _executor():
    executor = object.__new__(K3Execution)
    executor._fla_policy = object()
    executor._pending_kda_cleanup = []
    executor._closed = False
    return executor


class K3FLACleanupTests(unittest.TestCase):
    def test_retirement_failure_is_not_reported_as_success(self):
        executor = _executor()
        failure = _UnprintableCleanupError("buffer release failed")
        mixer = _Mixer(failure)

        with self.assertRaises(RuntimeError) as caught:
            executor._retire_fla_mixers((mixer,))

        self.assertIs(caught.exception.__cause__, failure)
        self.assertIn("K3 FLA mixer cleanup retained", str(caught.exception))
        self.assertTrue(executor._closed)
        self.assertEqual(executor._pending_kda_cleanup, [mixer])
        self.assertEqual(mixer.close_calls, 1)

    def test_cleanup_failure_preserves_primary_and_retains_retry_owner(self):
        executor = _executor()
        primary = ValueError("K3 request failed")
        mixer = _Mixer(_UnprintableCleanupError("buffer release failed"))

        executor._retire_fla_mixers((mixer,), primary=primary)

        self.assertTrue(executor._closed)
        self.assertEqual(executor._pending_kda_cleanup, [mixer])
        self.assertEqual(mixer.close_calls, 1)
        notes = getattr(primary, "__notes__", ())
        self.assertEqual(len(notes), 1)
        self.assertIn("_UnprintableCleanupError", notes[0])

    def test_executor_close_retries_packed_owner_through_its_public_close(self):
        executor = object.__new__(K3Execution)
        executor._lock = threading.RLock()
        executor._busy = False
        executor._closed = False
        executor._resources_released = False
        executor._fla_policy = None
        executor._pending_kda_cleanup = []
        executor._kda = {}
        executor._ops = None
        packed = _PackedOwner()
        executor._packed = packed
        executor._raw_libraries = []
        executor._release = weakref.finalize(executor, lambda: None)
        executor._mla = {}
        executor._weights = {}
        executor._vision = None
        executor._comparison_observations = None
        executor._expert_provider = None

        with self.assertRaisesRegex(OSError, "still in use"):
            executor.close()
        self.assertIs(executor._packed, packed)
        self.assertEqual(packed.close_calls, 1)

        executor.close()

        self.assertIsNone(executor._packed)
        self.assertEqual(packed.close_calls, 2)
        self.assertTrue(executor._resources_released)


if __name__ == "__main__":
    unittest.main()
