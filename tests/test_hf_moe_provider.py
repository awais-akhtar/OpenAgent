import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.hf_moe_provider import HFMoeProvider
from openagent.model_gateway import ModelGateway, ProviderConfigurationError


class ProviderConfigurationTests(unittest.TestCase):
    def test_unconfigured_provider_is_unavailable(self):
        with patch.dict(os.environ, {"OPENAGENT_HF_MOE_MODEL_DIR": ""}):
            self.assertFalse(HFMoeProvider().available())

    def test_configured_provider_is_registered_and_local(self):
        with patch.dict(os.environ, {"OPENAGENT_HF_MOE_MODEL_DIR": "missing-checkpoint"}):
            gateway = ModelGateway()
            provider = gateway.providers["hf-moe-direct"]
            self.assertTrue(provider.local)
            self.assertFalse(provider.network)
            self.assertIn(provider.id, gateway.route_order(allow_external=False))
            self.assertIsNone(gateway._effective_provider_model(provider, provider_id="auto", model="unrelated"))
            with self.assertRaises(ProviderConfigurationError):
                provider.complete([], model="unrelated")

    def test_provider_hides_unadmitted_hf_execution_before_tensor_scan(self):
        import json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({
                "model_type": "mixtral", "hidden_size": 8,
                "intermediate_size": 12, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "vocab_size": 16, "num_local_experts": 2,
                "num_experts_per_tok": 1,
            }))
            with patch.dict(os.environ, {"OPENAGENT_HF_MOE_MODEL_DIR": str(root)}), \
                 patch("openagent.adaptive_engine.hf_moe_provider.importlib.util.find_spec",
                       return_value=object()), \
                 patch("openagent.adaptive_engine.hf_moe_provider._scan_directory",
                       side_effect=AssertionError("unadmitted checkpoint scan")) as scan:
                self.assertFalse(HFMoeProvider().available())
            scan.assert_not_called()


AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "accelerate", "tokenizers"))


@unittest.skipUnless(AVAILABLE, "optional HF runtime dependencies unavailable")
class LocalGenerationTests(unittest.TestCase):
    def test_gateway_provider_uses_real_local_model_and_disk_experts(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        torch.manual_seed(17)
        config = AutoConfig.for_model("qwen3_moe", vocab_size=32, hidden_size=16,
            intermediate_size=24, moe_intermediate_size=12, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            num_experts=3, num_experts_per_tok=2, max_position_embeddings=64,
            bos_token_id=None, eos_token_id=None, pad_token_id=None)
        model = AutoModelForCausalLM.from_config(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer = Tokenizer(WordLevel({f"w{i}": i for i in range(32)}, unk_token="w31"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="w31").save_pretrained(checkpoint)
            with patch.dict(os.environ, {"OPENAGENT_HF_MOE_MODEL_DIR": str(checkpoint),
                    "OPENAGENT_HF_MOE_OFFLOAD_DIR": str(root / "offload"),
                    "OPENAGENT_HF_MOE_DEVICE": "cpu", "OPENAGENT_HF_MOE_MAX_TOKENS": "3"}):
                provider = HFMoeProvider()
                self.assertTrue(provider.available())
                result = provider.complete([{"role": "user", "content": "w4 w5"}])
                self.assertTrue(result.content.strip())
                self.assertEqual(result.metadata["disk_experts"], 3)
                self.assertFalse(result.metadata["network"])
                self.assertFalse(result.metadata["streamed"])
                provider._adapter.validate_placement()
