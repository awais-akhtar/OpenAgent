import json
from pathlib import Path
import unittest

from openagent.adaptive_engine.hf_moe import HFMoeManifest
from openagent.adaptive_engine.hf_moe_cli import expert_placement


class PlacementTests(unittest.TestCase):
    def test_mla_and_glm_placement_excludes_dense_prefix_and_shared_experts(self):
        for kind in ("deepseek_v3", "glm4_moe"):
            manifest = HFMoeManifest(Path("."), kind, json.dumps({
                "num_hidden_layers": 3, "n_routed_experts": 2,
                "first_k_dense_replace": 1,
            }))
            placement = expert_placement(manifest, "cpu", disk=True)
            self.assertEqual(set(placement), {
                "model.layers.1.mlp.experts.0", "model.layers.1.mlp.experts.1",
                "model.layers.2.mlp.experts.0", "model.layers.2.mlp.experts.1",
            })
            self.assertEqual(set(placement.values()), {"disk"})

    def test_mixtral_exact_module_paths(self):
        manifest = HFMoeManifest(Path("."), "mixtral", json.dumps({
            "num_hidden_layers": 2, "num_local_experts": 3,
        }))
        placement = expert_placement(manifest, "cpu", disk=True)
        self.assertEqual(len(placement), 6)
        self.assertEqual(placement["model.layers.1.block_sparse_moe.experts.2"], "disk")

    def test_qwen_respects_dense_layers_and_sparse_interval(self):
        manifest = HFMoeManifest(Path("."), "qwen2_moe", json.dumps({
            "num_hidden_layers": 4, "num_experts": 2,
            "decoder_sparse_step": 2, "mlp_only_layers": [1],
        }))
        self.assertEqual(expert_placement(manifest, "cuda:0", disk=False), {
            "model.layers.3.mlp.experts.0": "cuda:0",
            "model.layers.3.mlp.experts.1": "cuda:0",
        })
