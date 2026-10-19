import json
from pathlib import Path
import tempfile
import unittest

from openagent.adaptive_engine.olmoe import OlmoeConfig


class ConfigValidationTests(unittest.TestCase):
    def read(self, **overrides):
        config = dict(model_type="olmoe", architectures=["OlmoeForCausalLM"],
                      hidden_size=8, intermediate_size=12,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, num_experts=4,
                      num_experts_per_tok=2, vocab_size=16,
                      max_position_embeddings=32, rms_norm_eps=1e-5,
                      rope_theta=10000, eos_token_id=15)
        config.update(overrides)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(config))
            return OlmoeConfig.read(directory)

    def test_router_normalization_requires_boolean(self):
        for value in ("false", 0, 1, None, []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                self.read(norm_topk_prob=value)
        self.assertFalse(self.read().norm_topk_prob)
        self.assertTrue(self.read(norm_topk_prob=True).norm_topk_prob)

    def test_eos_outside_vocabulary_rejected(self):
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.read(eos_token_id=16)

    def test_odd_head_dimension_rejected(self):
        with self.assertRaisesRegex(ValueError, "even"):
            self.read(hidden_size=6)
