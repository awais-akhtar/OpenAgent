from __future__ import annotations

import copy
import unittest

from openagent.adaptive_engine.phase1_a05_speculative_state import SpeculativeStateManager


class Phase1A05SpeculativeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = b"session-key-for-phase1-tests"
        self.manager = SpeculativeStateManager()
        self.manager.open_session("session-1", session_key=self.key)

    def test_cancel_rolls_back_both_target_and_draft_rngs(self) -> None:
        before = self.manager.snapshot("session-1")
        self.manager.begin("session-1", "request-1")
        self.manager.capture_pre_draft("session-1", "request-1")
        self.manager.sample_draft("session-1", "request-1")
        self.manager.sample_target("session-1", "request-1")
        self.manager.rollback("session-1", "request-1")
        self.manager.release("session-1", "request-1")
        after = self.manager.snapshot("session-1")

        self.assertEqual(after["target_state"], before["target_state"])
        self.assertEqual(after["draft_state"], before["draft_state"])

    def test_rollback_is_terminal_until_owner_releases(self) -> None:
        before = self.manager.snapshot("session-1")
        self.manager.begin("session-1", "request-1")
        self.manager.capture_pre_draft("session-1", "request-1")
        self.manager.sample_draft("session-1", "request-1")
        self.manager.sample_target("session-1", "request-1")
        self.manager.rollback("session-1", "request-1")

        for operation in (
            self.manager.capture_pre_draft,
            self.manager.sample_draft,
            self.manager.sample_target,
            self.manager.accept_step,
            self.manager.commit,
        ):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(
                RuntimeError, "lease was rolled back"
            ):
                operation("session-1", "request-1")

        self.manager.release("session-1", "request-1")
        after = self.manager.snapshot("session-1")
        self.assertEqual(after["target_state"], before["target_state"])
        self.assertEqual(after["draft_state"], before["draft_state"])

    def test_successful_proposal_can_commit_then_release(self) -> None:
        self.manager.begin("session-1", "request-1")
        self.manager.capture_pre_draft("session-1", "request-1")
        self.manager.sample_draft("session-1", "request-1")
        self.manager.sample_target("session-1", "request-1")
        self.manager.accept_step("session-1", "request-1")

        self.manager.commit("session-1", "request-1")
        counts = self.manager.release("session-1", "request-1")

        self.assertEqual(counts, (1, 1))

    def test_authenticated_snapshot_restores_across_manager_instances(self) -> None:
        snapshot = self.manager.snapshot("session-1")
        restored = SpeculativeStateManager()

        restored.restore(snapshot, session_key=self.key)

        self.assertEqual(restored.snapshot("session-1"), snapshot)

    def test_foreign_or_modified_snapshot_is_rejected(self) -> None:
        snapshot = self.manager.snapshot("session-1")
        tampered = copy.deepcopy(snapshot)
        tampered["draft_state"][0] ^= 1
        restored = SpeculativeStateManager()

        with self.assertRaisesRegex(ValueError, "authentication"):
            restored.restore(tampered, session_key=self.key)
        with self.assertRaisesRegex(ValueError, "different session key"):
            restored.restore(snapshot, session_key=b"another-session-key-value")

    def test_stale_snapshot_cannot_replace_a_newer_registered_boundary(self) -> None:
        stale = self.manager.snapshot("session-1")
        self.manager.begin("session-1", "request-1")
        self.manager.capture_pre_draft("session-1", "request-1")
        self.manager.sample_target("session-1", "request-1")
        self.manager.commit("session-1", "request-1")
        self.manager.release("session-1", "request-1")
        current = self.manager.snapshot("session-1")

        with self.assertRaisesRegex(ValueError, "stale speculative snapshot sequence"):
            self.manager.restore(stale, session_key=self.key)

        self.assertEqual(self.manager.snapshot("session-1"), current)

    def test_unresolved_proposal_cannot_be_released_as_success(self) -> None:
        self.manager.begin("session-1", "request-1")
        self.manager.capture_pre_draft("session-1", "request-1")

        with self.assertRaisesRegex(RuntimeError, "call rollback"):
            self.manager.release("session-1", "request-1")

    def test_close_refuses_to_erase_an_active_rollback_owner(self) -> None:
        self.manager.begin("session-1", "request-1")
        self.manager.capture_pre_draft("session-1", "request-1")

        with self.assertRaisesRegex(RuntimeError, "active owners"):
            self.manager.close()

        self.manager.rollback("session-1", "request-1")
        self.manager.release("session-1", "request-1")
        self.manager.close()
        with self.assertRaisesRegex(RuntimeError, "manager is closed"):
            self.manager.open_session("session-2", session_key=self.key)


if __name__ == "__main__":
    unittest.main()
