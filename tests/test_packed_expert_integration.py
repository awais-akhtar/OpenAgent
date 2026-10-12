"""Packed weights through the checkpoint converter and live model runtime."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.expert_kernels import ExpertKernelRegistry, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_segments import OlmoeSegmentEngine
from tests.test_olmoe_batch import checkpoint


class PackedExpertIntegrationTests(unittest.TestCase):
    def test_cross_precision_restore_is_rejected_before_kv_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = checkpoint(root, True)
            packed = pack_olmoe_experts(root, root / "experts.i4", dtype="int4-row")
            with OlmoeModel(root, original, expert_cache_bytes=4096) as int8_model, \
                    OlmoeModel(root, packed, expert_cache_bytes=4096) as int4_model:
                int8_model.forward([1, 2])
                int4_model.forward([3])
                before = int4_model.snapshot_state()
                with self.assertRaisesRegex(ValueError, "precision/backend"):
                    int4_model.restore_state(int8_model.snapshot_state())
                self.assertEqual(int4_model.position, before["position"])
                for expected, actual in zip(before["keys"], int4_model.snapshot_state()["keys"]):
                    np.testing.assert_array_equal(actual, expected)
                legacy = dict(before, format="openagent.olmoe-state.v1")
                with self.assertRaisesRegex(ValueError, "state format"):
                    int4_model.restore_state(legacy)

    def test_default_registry_keeps_original_backend_contract(self):
        registry = ExpertKernelRegistry()
        self.assertEqual(registry.resolve(dtype="int8-row").info().name, "numpy-reference")
        self.assertEqual(registry.resolve(dtype="int4-row").info().name, "packed-int4-reference")
        self.assertNotIn("int4-row", NumPyRoutedSwiGLUBackend().info().capabilities.payload_dtypes)

    def test_checkpoint_pack_runs_prefill_and_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = checkpoint(root, True)
            packed = pack_olmoe_experts(root, root / "experts.i4", dtype="int4-row")
            packed.write(root / "experts.i4.json")
            self.assertTrue(all(layer.dtype == "int4-row" for layer in packed.layers))
            packed_bytes = sum(region.size for layer in packed.layers for region in layer.regions.values())
            original_bytes = sum(region.size for layer in original.layers for region in layer.regions.values())
            self.assertLess(packed_bytes, original_bytes)
            with OlmoeModel(root, packed, expert_cache_bytes=4096) as model:
                logits = model.forward([1, 2, 3])
                self.assertTrue(np.isfinite(logits).all())
                saved = model.snapshot_state()
                expected = model.forward([4])
                model.restore_state(saved)
                np.testing.assert_array_equal(model.forward([4]), expected)
                self.assertEqual(model.position, 4)
            # The existing segment snapshot class must not silently change precision.
            with self.assertRaisesRegex(ValueError, "manifest layer contract"):
                OlmoeSegmentEngine(root, packed, 0, 1, expert_cache_bytes=4096)

    def test_unknown_precision_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root, False)
            target = root / "bad.i3"
            with self.assertRaisesRegex(ValueError, "dtype"):
                pack_olmoe_experts(root, target, dtype="unimplemented-int3")
            self.assertFalse(target.exists())
