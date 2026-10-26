"""Brio shared-prefix cleanup must preserve failures and retire all receipts."""
import threading
import unittest

from openagent.adaptive_engine.closed_set_scoring import ClosedSetScoringMixin


class _UnprintableError(RuntimeError):
    def __str__(self):
        raise AssertionError("cleanup formatting must not replace the primary error")

    def __repr__(self):
        raise AssertionError("cleanup repr must not skip later cleanup")


class _Service:
    def __init__(self, *, prefill_error=None, restore_error=None, release_error=None):
        self.prefill_error = prefill_error
        self.restore_error = restore_error
        self.release_error = release_error
        self.calls = []
        self.receipt = object()

    def snapshot(self, _session):
        self.calls.append("snapshot")
        return self.receipt

    def prefill(self, _session, _tokens, *, cancel):
        self.calls.append("prefill")
        if self.prefill_error is not None:
            raise self.prefill_error

    def restore(self, _session, receipt):
        self.calls.append("restore")
        if receipt is not self.receipt:
            raise AssertionError("foreign rollback receipt")
        if self.restore_error is not None:
            raise self.restore_error

    def release_snapshot(self, _session, receipt):
        self.calls.append("release")
        if receipt is not self.receipt:
            raise AssertionError("foreign release receipt")
        if self.release_error is not None:
            raise self.release_error


class _Owner(ClosedSetScoringMixin):
    def __init__(self, service):
        self._lock = threading.RLock()
        self._closed = False
        self.service = service
        self.session_id = "session"

    def _artifacts_unchanged(self):
        pass


class ClosedSetScoringCleanupTests(unittest.TestCase):
    def test_unprintable_rollback_and_release_preserve_scoring_failure(self):
        primary = ValueError("scoring failed")
        service = _Service(prefill_error=primary,
            restore_error=_UnprintableError("rollback failed"),
            release_error=_UnprintableError("release failed"))
        owner = _Owner(service)

        with self.assertRaises(ValueError) as caught:
            with owner.scoring_prefix_tokens([1, 2]):
                self.fail("prefill error should abort the context")

        self.assertIs(caught.exception, primary)
        self.assertEqual(service.calls, ["snapshot", "prefill", "restore", "release"])
        notes = getattr(primary, "__notes__", ())
        self.assertEqual(len(notes), 2)
        self.assertTrue(all("_UnprintableError" in note for note in notes))

    def test_unprintable_release_failure_does_not_mask_restore_failure(self):
        restore_error = _UnprintableError("rollback failed")
        service = _Service(restore_error=restore_error,
            release_error=_UnprintableError("release failed"))
        owner = _Owner(service)

        with self.assertRaises(_UnprintableError) as caught:
            with owner.scoring_prefix_tokens([1, 2]):
                pass

        self.assertIs(caught.exception, restore_error)
        self.assertEqual(service.calls, ["snapshot", "prefill", "restore", "release"])
        self.assertEqual(len(getattr(restore_error, "__notes__", ())), 1)


if __name__ == "__main__":
    unittest.main()
