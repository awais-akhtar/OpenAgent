from __future__ import annotations

import hashlib
import unittest

from openagent.adaptive_engine.phase1_a06_distributed_shard import (
    ConsensusError,
    DistributedConsensus,
    DistributedCoordinator,
    RankSummary,
    compute_shard_layout,
)


class Phase1A06DistributedShardTests(unittest.TestCase):
    def test_shard_layout_rejects_boolean_counts_and_unknown_families(self) -> None:
        with self.assertRaisesRegex(ValueError, "ranks"):
            compute_shard_layout("mixtral", True)
        with self.assertRaisesRegex(ValueError, "exact admitted family"):
            compute_shard_layout("made-up-family", 2, total_experts=4)

    def test_preflight_contract_must_agree_before_results_are_accepted(self) -> None:
        consensus = DistributedConsensus("generate", 2)
        first = hashlib.sha256(b"operation-v1").hexdigest()
        second = hashlib.sha256(b"operation-v2").hexdigest()
        consensus.submit_preflight(0, True, contract_digest=first)
        consensus.submit_preflight(1, True, contract_digest=second)

        with self.assertRaisesRegex(ConsensusError, "contract differs"):
            consensus.require_preflight_agreement()
        with self.assertRaisesRegex(ConsensusError, "mismatched operation contracts"):
            consensus.submit_result(RankSummary(
                0, "generate", (1,), 1, hashlib.sha256(b"history").hexdigest()))

    def test_result_agreement_compares_exact_history_digest(self) -> None:
        consensus = DistributedConsensus("generate", 2)
        digest = hashlib.sha256(b"operation").hexdigest()
        consensus.submit_preflight(0, True, contract_digest=digest)
        consensus.submit_preflight(1, True, contract_digest=digest)
        first = hashlib.sha256(b"history-a").hexdigest()
        second = hashlib.sha256(b"history-b").hexdigest()
        consensus.submit_result(RankSummary(0, "generate", (7,), 9, first))
        consensus.submit_result(RankSummary(1, "generate", (7,), 9, second))

        with self.assertRaisesRegex(ConsensusError, "result divergence"):
            consensus.require_result_agreement()

    def test_rank_loss_poisoning_does_not_deadlock_or_reuse_group(self) -> None:
        coordinator = DistributedCoordinator("mixtral", 2)
        coordinator.begin_operation("generate")

        coordinator.record_rank_loss(1, "peer disappeared")

        self.assertTrue(coordinator.poisoned)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            coordinator.begin_operation("snapshot")

    def test_abort_before_unanimous_preflight_is_reusable(self) -> None:
        coordinator = DistributedCoordinator("mixtral", 2)
        consensus = coordinator.begin_operation("generate")
        consensus.submit_preflight(
            0, False, "invalid shard", contract_digest=hashlib.sha256(b"op").hexdigest())

        coordinator.abort_operation("rank rejected before collective")

        self.assertFalse(coordinator.poisoned)
        self.assertIsNone(coordinator._active_consensus)
        coordinator.begin_operation("snapshot")

    def test_abort_after_unanimous_preflight_poisons_group(self) -> None:
        coordinator = DistributedCoordinator("mixtral", 2)
        consensus = coordinator.begin_operation("generate")
        digest = hashlib.sha256(b"op").hexdigest()
        consensus.submit_preflight(0, True, contract_digest=digest)
        consensus.submit_preflight(1, True, contract_digest=digest)

        with self.assertRaisesRegex(ConsensusError, "group poisoned"):
            coordinator.abort_operation("caller cancelled after agreement")

        self.assertTrue(coordinator.poisoned)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            coordinator.begin_operation("snapshot")

    def test_abort_after_any_collective_result_poisons_group(self) -> None:
        coordinator = DistributedCoordinator("mixtral", 2)
        consensus = coordinator.begin_operation("generate")
        digest = hashlib.sha256(b"op").hexdigest()
        consensus.submit_preflight(0, True, contract_digest=digest)
        consensus.submit_preflight(1, True, contract_digest=digest)
        consensus.submit_result(RankSummary(
            0, "generate", (1,), 1, hashlib.sha256(b"history").hexdigest()))

        with self.assertRaisesRegex(ConsensusError, "group poisoned"):
            coordinator.abort_operation("rank 1 result timed out")

        self.assertTrue(coordinator.poisoned)

    def test_abort_reason_must_be_bounded_and_nonempty(self) -> None:
        coordinator = DistributedCoordinator("mixtral", 2)
        coordinator.begin_operation("generate")
        for reason in ("", "x" * 513):
            with self.subTest(reason_length=len(reason)), self.assertRaisesRegex(
                ValueError, "abort reason"
            ):
                coordinator.abort_operation(reason)

    def test_recovery_uses_only_failed_rank_own_committed_cache(self) -> None:
        coordinator = DistributedCoordinator("mixtral", 2)
        digest = hashlib.sha256(b"same-token-history").hexdigest()
        coordinator.record_committed(0, 12, digest, "rank0.cache")
        coordinator.record_committed(1, 12, digest, "rank1.cache")
        coordinator.record_rank_loss(1)

        plan = coordinator.recovery_plan()

        self.assertEqual(len(plan), 1)
        self.assertTrue(plan[0]["recoverable"])
        self.assertEqual(plan[0]["cache_path"], "rank1.cache")

    def test_recovery_commit_requires_sha256_history_identity(self) -> None:
        coordinator = DistributedCoordinator("mixtral", 2)

        for invalid in ("", "not-a-digest", "A" * 64):
            with self.subTest(digest=invalid), self.assertRaisesRegex(
                ValueError, "lowercase SHA-256"
            ):
                coordinator.record_committed(0, 1, invalid, "rank0.cache")

    def test_same_position_cannot_be_rebound_to_different_history(self) -> None:
        coordinator = DistributedCoordinator("mixtral", 2)
        first = hashlib.sha256(b"first history").hexdigest()
        second = hashlib.sha256(b"different history").hexdigest()
        coordinator.record_committed(0, 12, first, "rank0.cache")

        with self.assertRaisesRegex(ValueError, "digest changed without advancing"):
            coordinator.record_committed(0, 12, second, "stale.cache")

        record = coordinator._health._records[0]
        self.assertEqual(record.last_committed_digest, first)
        self.assertEqual(record.last_committed_cache_path, "rank0.cache")

    def test_rank_cannot_claim_other_shard_checkpoint_as_its_recovery(self) -> None:
        coordinator = DistributedCoordinator("mixtral", 2)
        digest = hashlib.sha256(b"same-token-history").hexdigest()
        coordinator.record_committed(0, 12, digest, "rank0.cache")
        coordinator.record_committed(1, 11, digest, "rank1.cache")
        coordinator.record_rank_loss(1)

        entry = coordinator.recovery_plan()[0]

        self.assertFalse(entry["recoverable"])
        self.assertIsNone(entry["cache_path"])
        self.assertIn("full group restart", entry["reason"])


if __name__ == "__main__":
    unittest.main()
