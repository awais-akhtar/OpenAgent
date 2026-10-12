"""Tiny independent-library oracle for the complete OLMoE decoder."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_validation import validate_olmoe


AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


@unittest.skipUnless(AVAILABLE, "optional independent reference dependencies unavailable")
class TransformersReferenceTests(unittest.TestCase):
    def test_decoder_logits_and_greedy_tokens_match_library_reference(self):
        import torch
        import transformers
        from transformers import OlmoeConfig, OlmoeForCausalLM

        torch.manual_seed(83)
        config = OlmoeConfig(vocab_size=32, hidden_size=16, intermediate_size=12,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
            norm_topk_prob=False, bos_token_id=1, eos_token_id=31, pad_token_id=0,
            attention_dropout=0.0, tie_word_embeddings=False)
        config._attn_implementation = "eager"
        reference = OlmoeForCausalLM(config).eval()
        def oracle(ids):
            with torch.inference_mode():
                return reference(input_ids=torch.tensor([ids])).logits[0].float().numpy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference.save_pretrained(root, safe_serialization=True)
            manifest = pack_olmoe_experts(root, root / "experts.i8", model_id="tiny-independent-olmoe")
            size = max(region.size for layer in manifest.layers for region in layer.regions.values())
            with OlmoeModel(root, manifest, expert_cache_bytes=size * 4, context_tokens=32) as model:
                report = validate_olmoe(model, oracle, [[1, 4, 7, 2], [1, 3, 9, 8, 6]],
                    reference_identity={"model_id": "tiny-independent-olmoe", "revision": "seed83",
                        "implementation": f"transformers {transformers.__version__} OlmoeForCausalLM",
                        "device": "cpu", "dtype": "float32"}, atol=2e-4, rtol=2e-3)
                self.assertTrue(report["passed"], report["metrics"])
                actual = model.generate_ids([1, 4, 7], max_tokens=4)
                expected = []
                ids = [1, 4, 7]
                for _ in range(4):
                    token = int(np.argmax(oracle(tuple(ids))[-1]))
                    if token == config.eos_token_id:
                        break
                    expected.append(token)
                    ids.append(token)
                self.assertEqual(actual, expected)
                self.assertFalse(report["public_checkpoint_validated"])
