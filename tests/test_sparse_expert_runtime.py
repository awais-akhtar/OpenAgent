from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine import (
    ExpertLayer,
    ExpertLayerBinding,
    ExpertRegion,
    ExpertTensorBinding,
    OlmoeConfig,
    OlmoeModel,
    RoutePredictor,
    SparseExpertManifest,
    SparseExpertRuntime,
    pack_safetensors_experts,
    pack_safetensors_directory_int8,
    pack_olmoe_experts,
)
from openagent.adaptive_engine.runtime import AdaptiveExecutionEngine
from openagent.storage import OpenAgentStore


class SparseExpertRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(41)
        self.hidden_size = 3
        self.intermediate_size = 5
        self.weights = [rng.normal(size=45).astype("<f4") for _ in range(4)]
        payload = b"".join(weight.tobytes() for weight in self.weights)
        (self.root / "experts.bin").write_bytes(payload)
        size = self.weights[0].nbytes
        regions = {
            str(index): ExpertRegion(
                "experts.bin",
                index * size,
                size,
                hashlib.sha256(weight.tobytes()).hexdigest(),
            )
            for index, weight in enumerate(self.weights)
        }
        self.layer = ExpertLayer(
            id="block.0",
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            experts_used=2,
            dtype="float32",
            activation="swiglu",
            regions=regions,
        )
        self.manifest = SparseExpertManifest(
            model_id="fixture-moe",
            architecture="fixture",
            fingerprint="sha256:test-fixture",
            layers=(self.layer,),
        )

    def reference(self, hidden: np.ndarray, routes: list[tuple[str, float]]) -> np.ndarray:
        output = np.zeros(self.hidden_size, dtype=np.float32)
        width = self.hidden_size * self.intermediate_size
        for expert_id, scale in routes:
            values = self.weights[int(expert_id)].astype(np.float32)
            gate = values[:width].reshape(self.intermediate_size, self.hidden_size) @ hidden
            up = values[width:2 * width].reshape(self.intermediate_size, self.hidden_size) @ hidden
            activated = gate / (1.0 + np.exp(-gate))
            output += scale * (values[2 * width:].reshape(self.hidden_size, self.intermediate_size) @ (activated * up))
        return output

    def test_manifest_round_trip_is_strict_and_atomic(self) -> None:
        path = self.manifest.write(self.root / "model.experts.json")
        parsed = SparseExpertManifest.read(path)
        self.assertEqual(parsed, self.manifest)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["layers"][0]["activation"] = "unknown"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "swiglu"):
            SparseExpertManifest.read(path)

    def test_direct_layer_dimensions_reject_non_integer_values(self) -> None:
        for name in ("hidden_size", "intermediate_size", "experts_used"):
            for value in (True, False, 3.0, "3", None):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, f"{name} must be an integer"):
                        replace(self.layer, **{name: value}).validate()

    def test_manifest_preserves_optional_storage_root(self) -> None:
        region = next(iter(self.layer.regions.values()))
        striped_layer = replace(
            self.layer,
            regions={"0": replace(region, storage="ssd-b")},
            experts_used=1,
        )
        manifest = replace(self.manifest, layers=(striped_layer,))
        parsed = SparseExpertManifest.read(manifest.write(self.root / "striped.json"))
        self.assertEqual(parsed.layers[0].regions["0"].storage, "ssd-b")

    def test_multi_layer_runtime_matches_resident_reference_and_reports_evidence(self) -> None:
        hidden = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        routes = [("2", 0.65), ("0", 0.35)]
        size = self.weights[0].nbytes
        with SparseExpertRuntime(self.root, self.manifest, capacity_bytes=2 * size) as runtime:
            actual = runtime.execute("block.0", hidden, routes)
            status = runtime.status()
        np.testing.assert_allclose(actual, self.reference(hidden, routes), rtol=1e-5, atol=1e-6)
        self.assertEqual(status["compute"]["tokens"], 1)
        self.assertEqual(status["compute"]["expert_calls"], 2)
        self.assertEqual(status["store"]["bytes_read"], 2 * size)
        self.assertEqual(status["architecture"], "fixture")

    def test_batch_union_and_transition_prediction_preserve_each_result(self) -> None:
        batch = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([-0.4, 0.2, 0.8], dtype=np.float32),
        ]
        route_batch = [[("0", 1.0)], [("1", 0.7), ("2", 0.3)]]
        with SparseExpertRuntime(self.root, self.manifest, capacity_bytes=3 * self.weights[0].nbytes) as runtime:
            first = runtime.execute_batch("block.0", batch, route_batch)
            runtime.execute("block.0", batch[0], [("0", 1.0)])
            runtime.execute("block.0", batch[1], [("1", 0.7), ("2", 0.3)])
            predicted = runtime.predictor.predict("block.0", ["0"], limit=2)
        for actual, hidden, routes in zip(first, batch, route_batch):
            np.testing.assert_allclose(actual, self.reference(hidden, routes), rtol=1e-5, atol=1e-6)
        self.assertEqual(predicted, ["1", "2"])

    def test_expert_centric_batch_loads_shared_expert_once(self) -> None:
        batch = np.asarray([[0.2, 0.4, 0.6], [0.1, -0.3, 0.7]], dtype=np.float32)
        routes = [[("2", 1.0)], [("2", 0.5)]]
        size = self.weights[0].nbytes
        with SparseExpertRuntime(self.root, self.manifest, capacity_bytes=size) as runtime:
            actual = runtime.execute_batch("block.0", batch, routes)
            status = runtime.status()
        np.testing.assert_allclose(actual[0], self.reference(batch[0], routes[0]), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual[1], self.reference(batch[1], routes[1]), rtol=1e-5, atol=1e-6)
        self.assertEqual(status["store"]["bytes_read"], size)
        self.assertEqual(status["compute"]["tokens"], 2)

    def test_hot_experts_can_be_pinned_without_silent_eviction(self) -> None:
        size = self.weights[0].nbytes
        with SparseExpertRuntime(self.root, self.manifest, capacity_bytes=2 * size) as runtime:
            hidden = np.ones(self.hidden_size, dtype=np.float32)
            runtime.execute("block.0", hidden, [("3", 1.0)])
            selected = runtime.pin_hot("block.0", limit=1, load=True)
            runtime.execute("block.0", hidden, [("0", 1.0)])
            status = runtime.store.status()
        self.assertEqual(selected, ["3"])
        self.assertIn("block.0/3", status["resident_keys"])
        self.assertEqual(status["pinned_keys"], ["block.0/3"])

    def test_route_profile_round_trip_drives_budgeted_hot_set(self) -> None:
        size = self.weights[0].nbytes
        profile = self.root / "routes.json"
        hidden = np.ones(self.hidden_size, dtype=np.float32)
        with SparseExpertRuntime(self.root, self.manifest, capacity_bytes=2 * size) as runtime:
            runtime.execute("block.0", hidden, [("2", 1.0)])
            runtime.execute("block.0", hidden, [("2", 1.0)])
            runtime.execute("block.0", hidden, [("1", 1.0)])
            runtime.save_route_profile(profile)
        with SparseExpertRuntime(self.root, self.manifest, capacity_bytes=2 * size) as restored:
            restored.load_route_profile(profile)
            selected = restored.rebalance_hot(budget_bytes=size, load=True)
            status = restored.status()
        self.assertEqual(selected, ["block.0/2"])
        self.assertEqual(status["store"]["pinned_keys"], ["block.0/2"])

    def test_route_profile_rejects_a_different_model(self) -> None:
        profile = self.root / "routes.json"
        with SparseExpertRuntime(self.root, self.manifest, capacity_bytes=self.weights[0].nbytes) as runtime:
            runtime.save_route_profile(profile)
        raw = json.loads(profile.read_text(encoding="utf-8"))
        raw["fingerprint"] = "different"
        profile.write_text(json.dumps(raw), encoding="utf-8")
        with SparseExpertRuntime(self.root, self.manifest, capacity_bytes=self.weights[0].nbytes) as runtime:
            with self.assertRaisesRegex(ValueError, "different model"):
                runtime.load_route_profile(profile)

    def test_engine_opens_runtime_from_manifest_path(self) -> None:
        path = self.manifest.write(self.root / "manifest.json")
        engine = AdaptiveExecutionEngine(OpenAgentStore(self.root / "data"))
        with engine.open_sparse_expert_runtime(
            self.root,
            path,
            capacity_bytes=2 * self.weights[0].nbytes,
        ) as runtime:
            self.assertEqual(runtime.manifest.model_id, "fixture-moe")


class RoutePredictorTests(unittest.TestCase):
    def test_transition_predictions_do_not_change_router_output(self) -> None:
        predictor = RoutePredictor()
        predictor.observe("layer", ["a"])
        predictor.observe("layer", ["b", "c"])
        predictor.observe("layer", ["a"])
        predictor.observe("layer", ["b", "c"])
        self.assertEqual(predictor.predict("layer", ["a"], limit=2), ["b", "c"])
        self.assertEqual(predictor.predict("layer", ["missing"], limit=1), ["a"])


class OlmoeContractTests(unittest.TestCase):
    def test_official_geometry_builds_exact_expert_tensor_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {
                "model_type": "olmoe",
                "architectures": ["OlmoeForCausalLM"],
                "hidden_size": 2048,
                "intermediate_size": 1024,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "vocab_size": 50304,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "norm_topk_prob": False,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            parsed = OlmoeConfig.read(root)
            bindings = parsed.expert_bindings()
        self.assertEqual(len(bindings), 16)
        self.assertEqual(len(bindings[0].experts), 64)
        self.assertEqual(bindings[15].experts[63].down, "model.layers.15.mlp.experts.63.down_proj.weight")

    def test_invalid_router_geometry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {
                "model_type": "olmoe", "architectures": ["OlmoeForCausalLM"],
                "hidden_size": 8, "intermediate_size": 4,
                "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                "num_experts": 2, "num_experts_per_tok": 3, "vocab_size": 32,
                "max_position_embeddings": 16, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exceeds"):
                OlmoeConfig.read(root)

    def test_complete_tiny_decoder_matches_prefill_and_incremental_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {
                "model_type": "olmoe", "architectures": ["OlmoeForCausalLM"],
                "hidden_size": 4, "intermediate_size": 3,
                "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                "num_experts": 2, "num_experts_per_tok": 1, "vocab_size": 8,
                "max_position_embeddings": 8, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
                "norm_topk_prob": False, "eos_token_id": 7,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            rng = np.random.default_rng(123)
            tensors = {
                "model.embed_tokens.weight": rng.normal(size=(8, 4)).astype("<f4"),
                "lm_head.weight": rng.normal(size=(8, 4)).astype("<f4"),
                "model.norm.weight": np.ones(4, dtype="<f4"),
                "model.layers.0.input_layernorm.weight": np.ones(4, dtype="<f4"),
                "model.layers.0.post_attention_layernorm.weight": np.ones(4, dtype="<f4"),
                "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(4, 4)).astype("<f4"),
                "model.layers.0.self_attn.k_proj.weight": rng.normal(size=(4, 4)).astype("<f4"),
                "model.layers.0.self_attn.v_proj.weight": rng.normal(size=(4, 4)).astype("<f4"),
                "model.layers.0.self_attn.o_proj.weight": rng.normal(size=(4, 4)).astype("<f4"),
                "model.layers.0.self_attn.q_norm.weight": np.ones(4, dtype="<f4"),
                "model.layers.0.self_attn.k_norm.weight": np.ones(4, dtype="<f4"),
                "model.layers.0.mlp.gate.weight": rng.normal(size=(2, 4)).astype("<f4"),
            }
            for expert in range(2):
                prefix = f"model.layers.0.mlp.experts.{expert}."
                tensors[prefix + "gate_proj.weight"] = rng.normal(size=(3, 4)).astype("<f4")
                tensors[prefix + "up_proj.weight"] = rng.normal(size=(3, 4)).astype("<f4")
                tensors[prefix + "down_proj.weight"] = rng.normal(size=(4, 3)).astype("<f4")
            ExpertPackingTests._write_safetensors(root / "model.safetensors", tensors)
            manifest = pack_olmoe_experts(root, root / "experts.i8", model_id="tiny-olmoe")
            capacity = max(region.size for region in manifest.layers[0].regions.values()) * 2
            with self.assertRaisesRegex(ValueError, "does not match"):
                OlmoeModel(root, replace(manifest, fingerprint="sha256:wrong"), expert_cache_bytes=capacity)
            with OlmoeModel(root, manifest, expert_cache_bytes=capacity, context_tokens=8) as model:
                prefill = model.forward([1, 2, 3])
                model.reset()
                model.forward([1])
                model.forward([2])
                incremental = model.forward([3])
                snapshot = model.snapshot_state()
                next_logits = model.forward([4])
                model.restore_state(snapshot)
                restored_logits = model.forward([4])
                committed = model.snapshot_state()
                malformed = dict(snapshot)
                malformed["values"] = [np.full_like(snapshot["values"][0], np.nan)]
                with self.assertRaisesRegex(ValueError, "value state"):
                    model.restore_state(malformed)
                self.assertEqual(model.position, committed["position"])
                for actual, expected in zip(model.snapshot_state()["keys"], committed["keys"]):
                    np.testing.assert_array_equal(actual, expected)
                for actual, expected in zip(model.snapshot_state()["values"], committed["values"]):
                    np.testing.assert_array_equal(actual, expected)
                status = model.status()
                generated = model.generate_ids([1, 2], max_tokens=2)
            self.assertEqual(prefill.shape, (8,))
            self.assertTrue(np.isfinite(prefill).all())
            np.testing.assert_allclose(prefill, incremental, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(next_logits, restored_logits, rtol=2e-5, atol=2e-5)
            self.assertLessEqual(len(generated), 2)
            self.assertEqual(status["position"], 4)


class ExpertPackingTests(unittest.TestCase):
    def test_explicit_safetensors_mapping_builds_aligned_executable_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tensors: dict[str, np.ndarray] = {}
            for expert in range(2):
                tensors[f"e{expert}.gate"] = np.full((2, 2), 1 + expert, dtype="<f4")
                tensors[f"e{expert}.up"] = np.full((2, 2), 2 + expert, dtype="<f4")
                tensors[f"e{expert}.down"] = np.full((2, 2), 3 + expert, dtype="<f4")
            source = root / "model.safetensors"
            self._write_safetensors(source, tensors)
            bindings = ExpertLayerBinding(
                id="layer0",
                hidden_size=2,
                intermediate_size=2,
                experts_used=1,
                experts=tuple(
                    ExpertTensorBinding(
                        str(index), f"e{index}.gate", f"e{index}.up", f"e{index}.down"
                    )
                    for index in range(2)
                ),
            )
            manifest = pack_safetensors_experts(
                source,
                root / "experts.bin",
                model_id="packed",
                architecture="fixture",
                fingerprint="fixture-hash",
                layers=[bindings],
                alignment=64,
            )
            regions = manifest.layers[0].regions
            self.assertEqual(regions["0"].offset, 0)
            self.assertEqual(regions["1"].offset % 64, 0)
            with SparseExpertRuntime(root, manifest, capacity_bytes=regions["0"].size) as runtime:
                actual = runtime.execute("layer0", np.ones(2, dtype=np.float32), [("1", 1.0)])
            values = np.concatenate([tensors["e1.gate"].ravel(), tensors["e1.up"].ravel(), tensors["e1.down"].ravel()])
            width = 4
            gate = values[:width].reshape(2, 2) @ np.ones(2, dtype=np.float32)
            up = values[width:2 * width].reshape(2, 2) @ np.ones(2, dtype=np.float32)
            expected = values[2 * width:].reshape(2, 2) @ (gate / (1 + np.exp(-gate)) * up)
            np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_invalid_mapping_leaves_no_partial_container(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            self._write_safetensors(source, {"wrong": np.ones((2, 2), dtype="<f4")})
            destination = root / "experts.bin"
            layer = ExpertLayerBinding(
                "layer0", 2, 2, 1,
                (ExpertTensorBinding("0", "missing", "missing", "missing"),),
            )
            with self.assertRaisesRegex(ValueError, "not found"):
                pack_safetensors_experts(
                    source, destination, model_id="m", architecture="a",
                    fingerprint="f", layers=[layer], alignment=64,
                )
            self.assertFalse(destination.exists())
            self.assertFalse((root / "experts.bin.tmp").exists())

    def test_sharded_bfloat16_experts_pack_to_executable_row_int8(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gate = np.asarray([[0.5, -1.0], [1.5, 0.25]], dtype=np.float32)
            up = np.asarray([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32)
            down = np.asarray([[0.75, -0.5], [1.25, 0.5]], dtype=np.float32)
            self._write_bfloat16_safetensors(root / "part-1.safetensors", {"e.gate": gate, "e.up": up})
            self._write_bfloat16_safetensors(root / "part-2.safetensors", {"e.down": down})
            binding = ExpertLayerBinding(
                "layer0", 2, 2, 1,
                (ExpertTensorBinding("0", "e.gate", "e.up", "e.down"),),
            )
            progress: list[tuple[int, int, str, str]] = []
            manifest = pack_safetensors_directory_int8(
                root,
                root / "experts.i8",
                model_id="olmoe-fixture",
                architecture="olmoe",
                fingerprint="fixture",
                layers=[binding],
                alignment=64,
                progress=lambda *event: progress.append(event),
            )
            self.assertEqual(manifest.layers[0].dtype, "int8-row")
            self.assertEqual(progress, [(1, 1, "layer0", "0")])
            with SparseExpertRuntime(root, manifest, capacity_bytes=manifest.layers[0].regions["0"].size) as runtime:
                hidden = np.asarray([0.4, -0.2], dtype=np.float32)
                actual = runtime.execute("layer0", hidden, [("0", 1.0)])
                batched = runtime.execute_batch("layer0", np.stack([hidden, hidden]), [[("0", 1.0)], [("0", 0.5)]])
            gate_value = gate @ hidden
            up_value = up @ hidden
            expected = down @ (gate_value / (1 + np.exp(-gate_value)) * up_value)
            np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0.01)
            np.testing.assert_allclose(batched[0], expected, rtol=0.03, atol=0.01)
            np.testing.assert_allclose(batched[1], expected * 0.5, rtol=0.03, atol=0.01)

    def test_int8_packer_stripes_experts_across_named_storage_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as second:
            root = Path(temp)
            second_root = Path(second)
            tensors: dict[str, np.ndarray] = {}
            for expert in range(2):
                for role in ("gate", "up", "down"):
                    tensors[f"e{expert}.{role}"] = np.full(
                        (2, 2), expert + 1, dtype="<f4"
                    )
            self._write_safetensors(root / "model.safetensors", tensors)
            binding = ExpertLayerBinding(
                "layer0",
                2,
                2,
                1,
                tuple(
                    ExpertTensorBinding(
                        str(expert), f"e{expert}.gate", f"e{expert}.up", f"e{expert}.down"
                    )
                    for expert in range(2)
                ),
            )
            manifest = pack_safetensors_directory_int8(
                root,
                root / "experts-a.i8",
                model_id="striped-fixture",
                architecture="fixture",
                fingerprint="fixture",
                layers=[binding],
                alignment=64,
                storage_destinations={"ssd-b": second_root / "experts-b.i8"},
            )
            regions = manifest.layers[0].regions
            self.assertEqual(regions["0"].storage, "")
            self.assertEqual(regions["1"].storage, "ssd-b")
            self.assertTrue((root / "experts-a.i8").is_file())
            self.assertTrue((second_root / "experts-b.i8").is_file())
            with SparseExpertRuntime(
                root,
                manifest,
                capacity_bytes=regions["0"].size,
                io_mode="mmap",
                storage_roots={"ssd-b": second_root},
            ) as runtime:
                output = runtime.execute(
                    "layer0", np.ones(2, dtype=np.float32), [("1", 1.0)]
                )
            self.assertEqual(output.shape, (2,))
            self.assertTrue(np.isfinite(output).all())

    @staticmethod
    def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
        header: dict[str, object] = {}
        payload = bytearray()
        for name, tensor in tensors.items():
            start = len(payload)
            payload.extend(tensor.tobytes())
            header[name] = {
                "dtype": "F32",
                "shape": list(tensor.shape),
                "data_offsets": [start, len(payload)],
            }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)

    @staticmethod
    def _write_bfloat16_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
        header: dict[str, object] = {}
        payload = bytearray()
        for name, tensor in tensors.items():
            start = len(payload)
            words = (tensor.astype("<f4", copy=False).view("<u4") >> np.uint32(16)).astype("<u2")
            payload.extend(words.tobytes())
            header[name] = {
                "dtype": "BF16",
                "shape": list(tensor.shape),
                "data_offsets": [start, len(payload)],
            }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


if __name__ == "__main__":
    unittest.main()
