from __future__ import annotations

import threading
import unittest

from openagent.adaptive_engine.expert_tiers import (
    ExpertTierManager,
    InMemoryAccelerator,
    PromotionPhase,
    ResidencyTier,
)


class ExpertTierManagerTests(unittest.TestCase):
    def test_storage_lease_releases_materialized_view(self) -> None:
        payload = bytearray(b"data")
        with ExpertTierManager(mapped_host_bytes=0, hot_host_bytes=0) as manager:
            manager.register("cold", payload)
            with manager.acquire("cold", preferred=ResidencyTier.STORAGE) as lease:
                view = lease.payload
                self.assertEqual(bytes(view), b"data")
            with self.assertRaises(ValueError):
                bytes(view)

    def test_async_promotion_records_state_and_truthful_capability(self) -> None:
        backend = InMemoryAccelerator()
        with ExpertTierManager(
            mapped_host_bytes=8,
            hot_host_bytes=8,
            accelerator_bytes=8,
            accelerator=backend,
        ) as manager:
            manager.register("layer.0/expert.3", b"abcdefgh")
            job = manager.promote("layer.0/expert.3", ResidencyTier.ACCELERATOR, asynchronous=True)
            result = job.result(timeout=5)
            self.assertTrue(result.success)
            self.assertEqual(result.expert_id, "layer.0/expert.3")
            self.assertEqual(result.resident_tier, ResidencyTier.ACCELERATOR)
            self.assertEqual(
                manager.promotion_state("layer.0/expert.3", ResidencyTier.ACCELERATOR).phase,
                PromotionPhase.RESIDENT,
            )
            capability = manager.capability()["accelerator"]
            self.assertTrue(capability["available"])
            self.assertTrue(capability["simulated"])
            self.assertFalse(capability["executes_experts"])
            status = manager.status()
            self.assertEqual(status["tiers"]["accelerator"]["used_bytes"], 8)
            self.assertEqual(status["telemetry"]["accelerator_uploads"], 1)

    def test_active_lease_prevents_budget_eviction(self) -> None:
        manager = ExpertTierManager(mapped_host_bytes=0, hot_host_bytes=4)
        self.addCleanup(manager.close)
        manager.register("first", b"1111")
        manager.register("second", b"2222")
        manager.promote("first", ResidencyTier.HOT_HOST)
        with manager.acquire("first", preferred=ResidencyTier.HOT_HOST) as lease:
            self.assertEqual(lease.expert_id, "first")
            self.assertEqual(lease.payload, b"1111")
            failed = manager.promote("second", ResidencyTier.HOT_HOST)
            self.assertFalse(failed.success)
            self.assertEqual(failed.resident_tier, ResidencyTier.STORAGE)
            self.assertIn("leased or pinned", failed.error)
            self.assertFalse(manager.evict("first", ResidencyTier.HOT_HOST))
        promoted = manager.promote("second", ResidencyTier.HOT_HOST)
        self.assertTrue(promoted.success)
        status = manager.status()
        self.assertEqual(status["tiers"]["hot_host"]["resident_experts"], ["second"])
        self.assertEqual(status["tiers"]["hot_host"]["used_bytes"], 4)
        self.assertEqual(status["telemetry"]["evictions"], 1)

    def test_accelerator_upload_failure_falls_back_with_same_identity(self) -> None:
        backend = InMemoryAccelerator(fail_uploads={"stable-id"})
        with ExpertTierManager(
            mapped_host_bytes=8,
            hot_host_bytes=8,
            accelerator_bytes=8,
            accelerator=backend,
        ) as manager:
            manager.register("stable-id", b"payload!")
            manager.promote("stable-id", ResidencyTier.HOT_HOST)
            result = manager.promote("stable-id", ResidencyTier.ACCELERATOR)
            self.assertFalse(result.success)
            self.assertTrue(result.fallback)
            self.assertEqual(result.expert_id, "stable-id")
            self.assertEqual(result.resident_tier, ResidencyTier.HOT_HOST)
            with manager.acquire("stable-id", preferred=ResidencyTier.ACCELERATOR) as lease:
                self.assertEqual(lease.expert_id, "stable-id")
                self.assertEqual(lease.tier, ResidencyTier.HOT_HOST)
                self.assertEqual(lease.payload, b"payload!")
            status = manager.status()
            self.assertEqual(status["tiers"]["accelerator"]["used_bytes"], 0)
            self.assertEqual(status["tiers"]["hot_host"]["used_bytes"], 8)
            self.assertEqual(status["telemetry"]["accelerator_upload_failures"], 2)
            self.assertEqual(status["telemetry"]["fallback_count"], 2)

    def test_each_tier_obeys_its_own_byte_budget(self) -> None:
        backend = InMemoryAccelerator()
        with ExpertTierManager(
            mapped_host_bytes=8,
            hot_host_bytes=8,
            accelerator_bytes=4,
            accelerator=backend,
        ) as manager:
            for expert_id, payload in (("a", b"aaaa"), ("b", b"bbbb"), ("c", b"cccc")):
                manager.register(expert_id, payload)
            manager.promote("a", ResidencyTier.MAPPED_HOST)
            manager.promote("b", ResidencyTier.MAPPED_HOST)
            manager.promote("c", ResidencyTier.MAPPED_HOST)
            manager.promote("a", ResidencyTier.HOT_HOST)
            manager.promote("b", ResidencyTier.HOT_HOST)
            manager.pin_hot("a", load=False)
            manager.promote("c", ResidencyTier.HOT_HOST)
            manager.promote("a", ResidencyTier.ACCELERATOR)
            manager.promote("b", ResidencyTier.ACCELERATOR)
            status = manager.status()
            self.assertLessEqual(status["tiers"]["mapped_host"]["used_bytes"], 8)
            self.assertLessEqual(status["tiers"]["hot_host"]["used_bytes"], 8)
            self.assertLessEqual(status["tiers"]["accelerator"]["used_bytes"], 4)
            self.assertIn("a", status["tiers"]["hot_host"]["resident_experts"])
            self.assertNotIn("b", status["tiers"]["hot_host"]["resident_experts"])
            self.assertEqual(status["tiers"]["accelerator"]["resident_experts"], ["b"])
            self.assertEqual(status["tiers"]["mapped_host"]["peak_bytes"], 8)
            self.assertEqual(status["tiers"]["hot_host"]["peak_bytes"], 8)
            self.assertEqual(status["tiers"]["accelerator"]["peak_bytes"], 4)
            self.assertEqual(status["telemetry"]["active_leases"], 0)

    def test_queued_promotion_exposes_async_state(self) -> None:
        gate = threading.Event()

        def source() -> bytes:
            gate.wait(timeout=5)
            return b"data"

        with ExpertTierManager(mapped_host_bytes=4, hot_host_bytes=0) as manager:
            manager.register("slow", source, size_bytes=4)
            job = manager.promote("slow", ResidencyTier.MAPPED_HOST, asynchronous=True)
            phase = manager.promotion_state("slow", ResidencyTier.MAPPED_HOST).phase
            self.assertIn(phase, {PromotionPhase.QUEUED, PromotionPhase.PROMOTING})
            gate.set()
            self.assertTrue(job.result(timeout=5).success)


if __name__ == "__main__":
    unittest.main()
