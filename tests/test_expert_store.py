from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine import DiskExpertStore, ExpertRegion, routed_swiglu
from openagent.adaptive_engine.openvino_expert import OpenVINOExpertBackend
from openagent.adaptive_engine.compiler import AdaptiveCompiler, build_memory_plan
from openagent.adaptive_engine.runtime import AdaptiveExecutionEngine
from openagent.storage import OpenAgentStore
from tests.test_adaptive_engine import hardware_profile, transformer_model


class ExpertStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "weights.bin").write_bytes(b"aaaabbbbcccc")
        self.regions = {str(i): ExpertRegion("weights.bin", i * 4, 4) for i in range(3)}

    def test_cache_hits_and_lru_eviction_preserve_exact_bytes(self):
        with DiskExpertStore(self.root, self.regions, capacity_bytes=8) as store:
            for key, expected in [("0", b"aaaa"), ("1", b"bbbb"), ("0", b"aaaa"), ("2", b"cccc")]:
                with store.acquire(key) as data:
                    self.assertEqual(data, expected)
                    self.assertTrue(data.readonly)
            self.assertEqual(store.status()["hits"], 1)
            self.assertEqual(store.status()["bytes_read"], 12)
            self.assertEqual(store.status()["resident_bytes"], 8)
            self.assertEqual(store.status()["evictions"], 1)
        self.assertEqual(store.status()["resident_bytes"], 0)

    def test_mmap_mode_releases_views_and_closes_windows_mapping(self):
        store = DiskExpertStore(self.root, self.regions, capacity_bytes=8, io_mode="mmap")
        with store.acquire("0") as data:
            self.assertEqual(data, b"aaaa")
        self.assertTrue(store.evict("0"))
        self.assertTrue(store.prefetch("1").result(timeout=5))
        with store.acquire("1") as data:
            self.assertEqual(data, b"bbbb")
        status = store.status()
        self.assertEqual(status["io_mode"], "mmap")
        self.assertEqual(status["bytes_read"], 0)
        self.assertGreaterEqual(status["mapped_bytes"], 8)
        self.assertEqual(status["mapped_files"], 1)
        store.close()
        self.assertEqual(store.status()["mapped_files"], 0)

    def test_direct_io_store_matches_declared_ranges(self):
        from openagent.adaptive_engine.expert_io import DirectIOUnsupported

        with DiskExpertStore(self.root, self.regions, capacity_bytes=8, io_mode="direct") as store:
            try:
                with store.acquire("1") as payload:
                    self.assertEqual(bytes(payload), b"bbbb")
            except DirectIOUnsupported as exc:
                self.skipTest(str(exc))
            self.assertEqual(store.status()["bytes_read"], 4)

    def test_named_storage_roots_keep_each_expert_contained(self):
        with tempfile.TemporaryDirectory() as second:
            second_root = Path(second)
            (second_root / "cold.bin").write_bytes(b"dddd")
            regions = {
                "hot": ExpertRegion("weights.bin", 0, 4),
                "cold": ExpertRegion("cold.bin", 0, 4, storage="ssd-b"),
            }
            with DiskExpertStore(
                self.root,
                regions,
                capacity_bytes=8,
                storage_roots={"ssd-b": second_root},
                prefetch_workers=2,
                prefetch_depth=4,
            ) as store:
                jobs = store.prefetch_many(["hot", "cold"])
                self.assertTrue(all(job.result(timeout=5) for job in jobs))
                with store.acquire("hot") as hot, store.acquire("cold") as cold:
                    self.assertEqual(bytes(hot), b"aaaa")
                    self.assertEqual(bytes(cold), b"dddd")
                self.assertEqual(store.status()["storage_roots"], ["", "ssd-b"])
            with self.assertRaisesRegex(ValueError, "unknown expert storage"):
                DiskExpertStore(
                    self.root,
                    {"bad": ExpertRegion("cold.bin", 0, 4, storage="missing")},
                    capacity_bytes=4,
                )

    def test_leases_cannot_be_evicted_or_closed(self):
        with DiskExpertStore(self.root, self.regions, capacity_bytes=4) as store:
            with store.acquire("0") as data:
                with self.assertRaises(MemoryError):
                    with store.acquire("1"):
                        pass
                with self.assertRaises(RuntimeError):
                    store.close()
                self.assertFalse(store.prefetch("1").result(timeout=5))
                self.assertEqual(data, b"aaaa")
            with store.acquire("1") as data:
                self.assertEqual(data, b"bbbb")
        with self.assertRaises(RuntimeError):
            with store.acquire("1"):
                pass

    def test_prefetch_is_advisory_and_checksum_failure_is_not_cached(self):
        self.regions["bad"] = ExpertRegion("weights.bin", 0, 4, "0" * 64)
        with DiskExpertStore(self.root, self.regions, capacity_bytes=8) as store:
            self.assertFalse(store.prefetch("bad").result(timeout=5))
            self.assertEqual(store.status()["resident_bytes"], 0)
            with self.assertRaisesRegex(ValueError, "checksum"):
                with store.acquire("bad"):
                    pass
            self.assertTrue(store.prefetch("0").result(timeout=5))
            with store.acquire("0") as data:
                self.assertEqual(data, b"aaaa")
            self.assertEqual(store.status()["prefetch_hits"], 1)

    def test_invalid_ranges_and_outside_paths_are_rejected(self):
        for region in [ExpertRegion("weights.bin", -1, 4), ExpertRegion("weights.bin", 0, 16),
                       ExpertRegion("weights.bin", 10, 4), ExpertRegion("weights.bin", 0, 0)]:
            with self.assertRaises(ValueError):
                DiskExpertStore(self.root, {"x": region}, capacity_bytes=8)
        with tempfile.TemporaryDirectory() as outside:
            file = Path(outside) / "other.bin"
            file.write_bytes(b"1234")
            with self.assertRaises(ValueError):
                DiskExpertStore(self.root, {"x": ExpertRegion(str(file), 0, 4)}, capacity_bytes=8)

    def test_truncated_file_fails_without_publishing_payload(self):
        with DiskExpertStore(self.root, self.regions, capacity_bytes=8) as store:
            (self.root / "weights.bin").write_bytes(b"a")
            with self.assertRaises(ValueError):
                with store.acquire("0"):
                    pass
            self.assertEqual(store.status()["resident_bytes"], 0)

    def test_pinned_entries_survive_pressure_and_prefetch_never_evicts(self):
        with DiskExpertStore(self.root, self.regions, capacity_bytes=8) as store:
            store.pin(["0"], load=True)
            with store.acquire("1"):
                pass
            self.assertFalse(store.prefetch("2").result(timeout=5))
            self.assertEqual(store.status()["resident_keys"], ["0", "1"])
            with store.acquire("2"):
                pass
            self.assertIn("0", store.status()["resident_keys"])
            self.assertNotIn("1", store.status()["resident_keys"])

    def test_streamed_cpu_swiglu_matches_resident_reference(self):
        rng = np.random.default_rng(19)
        h, intermediate = 4, 7
        x = rng.normal(size=h).astype(np.float32)
        for dtype in ("<f4", "<f2"):
            weights = [rng.normal(size=3 * h * intermediate).astype(dtype) for _ in range(3)]
            raw = b"".join(w.tobytes() for w in weights)
            (self.root / "experts.bin").write_bytes(raw)
            size = weights[0].nbytes
            regions = {str(i): ExpertRegion("experts.bin", i * size, size,
                        hashlib.sha256(w.tobytes()).hexdigest()) for i, w in enumerate(weights)}
            routes = [("2", 0.7), ("0", 0.3)]
            expected = np.zeros(h, dtype=np.float32)
            width = h * intermediate
            for key, scale in routes:
                w = weights[int(key)].astype(np.float32)
                gate = w[:width].reshape(intermediate, h) @ x
                up = w[width:2 * width].reshape(intermediate, h) @ x
                expected += scale * (w[2 * width:].reshape(h, intermediate) @ (gate / (1 + np.exp(-gate)) * up))
            with DiskExpertStore(self.root, regions, capacity_bytes=size) as store:
                actual = routed_swiglu(store, x, routes, hidden_size=h, intermediate_size=intermediate,
                                        dtype="float32" if dtype == "<f4" else "float16")
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(store.status()["bytes_read"], 2 * size)
                self.assertLessEqual(store.status()["resident_bytes"], size)
                with self.assertRaises(ValueError):
                    routed_swiglu(store, x, [("0", float("nan"))], hidden_size=h, intermediate_size=intermediate)

    def test_openvino_backend_uses_region_descriptors_and_reuses_resident_expert(self):
        class FakeClient:
            def __init__(self):
                self.loads = []
                self.runs = []

            def available(self):
                return True

            def load_expert(self, **values):
                self.loads.append(values)
                return {"status": "resident"}

            def run_expert(self, expert_id, inputs):
                self.runs.append((expert_id, inputs))
                return {"output": [value * 2 for value in inputs]}

        h, intermediate = 2, 3
        payload = np.ones(3 * h * intermediate, dtype="<f4").tobytes()
        (self.root / "expert.bin").write_bytes(payload)
        region = ExpertRegion("expert.bin", 0, len(payload), hashlib.sha256(payload).hexdigest())
        client = FakeClient()
        backend = OpenVINOExpertBackend(client, device="GPU")
        with DiskExpertStore(self.root, {"layer/0": region}, capacity_bytes=len(payload)) as store:
            first = backend.execute(
                store, [1.0, 2.0], [("layer/0", 0.5)],
                hidden_size=h, intermediate_size=intermediate, dtype="float32",
            )
            second = backend.execute(
                store, [2.0, 3.0], [("layer/0", 1.0)],
                hidden_size=h, intermediate_size=intermediate, dtype="float32",
            )
        np.testing.assert_array_equal(first, [1.0, 2.0])
        np.testing.assert_array_equal(second, [4.0, 6.0])
        self.assertEqual(len(client.loads), 1)
        self.assertEqual(len(client.runs), 2)
        self.assertEqual(client.loads[0]["path"], str((self.root / "expert.bin").resolve()))



class StreamingPlannerTests(unittest.TestCase):
    def test_existing_engine_exposes_cpu_expert_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrices = np.ones(3, dtype="<f4")
            (root / "weights.bin").write_bytes(matrices.tobytes())
            engine = AdaptiveExecutionEngine(OpenAgentStore(root / "data"))
            with engine.open_expert_store(root, {"a": ExpertRegion("weights.bin", 0, 12)}, capacity_bytes=12) as store:
                first = engine.execute_routed_experts(store, [1.0], [("a", 1.0)], hidden_size=1, intermediate_size=1)
                second = engine.execute_routed_experts(store, [1.0], [("a", 1.0)], hidden_size=1, intermediate_size=1)
                np.testing.assert_allclose(first, [1 / (1 + np.exp(-1))], rtol=1e-6)
                np.testing.assert_array_equal(first, second)
                self.assertEqual(store.status()["bytes_read"], 12)
                self.assertEqual(store.status()["hits"], 1)

    def test_gguf_memory_pressure_does_not_invent_requantized_weights(self):
        model = replace(transformer_model("gguf"), quantization_bits=4.7)
        package = AdaptiveCompiler().compile(model, hardware_profile(), memory_budget_gb=0.1)
        summary = package.constraints["precision_summary"]
        self.assertEqual(summary["planned_weight_bytes"], sum(n.parameter_bytes for n in model.graph_nodes))
        self.assertFalse(summary["conversion_required"])
        self.assertFalse(summary["memory_feasible"])

    def test_moe_storage_is_not_reported_as_no_io(self):
        model = replace(transformer_model(), expert_count=8, experts_used=2)
        result = build_memory_plan(model, hardware_profile().devices, model.weight_bytes,
                                   context_tokens=4096, budget_bytes=32_000_000, preferred_device="cpu:0")
        self.assertFalse(result["executable_without_repeated_storage_io"])
        self.assertEqual(result["storage_execution"], "backend-adapter-required")
        self.assertTrue(result["kv_budget_exceeded"])
        self.assertEqual(result["weight_tiers"]["accelerator_or_hot_ram_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
