"""Real local-checkpoint differential tests, never mock correctness evidence."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.hf_moe import (
    HFMoeAdapter, HFMoeManifest, SUPPORTED_MODEL_TYPES,
    expert_module_names,
)


class ManifestTests(unittest.TestCase):
    def test_transformers_execution_requires_an_exact_phase1_route(self):
        from openagent.adaptive_engine.hf_moe import require_hf_moe_execution_admission

        for model_type in sorted(SUPPORTED_MODEL_TYPES):
            manifest = HFMoeManifest(Path("."), model_type, json.dumps({"model_type": model_type}))
            with self.subTest(model_type=model_type), self.assertRaisesRegex(
                    ValueError, "P02 HF MoE execution is restricted"):
                require_hf_moe_execution_admission(manifest)

    def test_adapter_rejects_unadmitted_hf_route_before_library_or_checkpoint_access(self):
        from openagent.adaptive_engine.hf_moe import require_hf_moe_execution_admission

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps({
                "model_type": "mixtral", "hidden_size": 8,
                "intermediate_size": 12, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "vocab_size": 16, "num_local_experts": 2,
                "num_experts_per_tok": 1,
            }))
            with patch("openagent.adaptive_engine.hf_moe._validate_checkpoint",
                       side_effect=AssertionError("unadmitted checkpoint accessed")):
                with self.assertRaisesRegex(ValueError, "P02 HF MoE execution is restricted"):
                    HFMoeAdapter.load(directory)

    def test_exact_expert_inventory(self):
        base = dict(hidden_size=16, intermediate_size=24, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=4, vocab_size=32,
                    num_experts=3, num_local_experts=2, n_routed_experts=4,
                    num_experts_per_tok=2, moe_intermediate_size=12,
                    n_shared_experts=1, n_group=2, topk_group=1,
                    first_k_dense_replace=2, decoder_sparse_step=2, mlp_only_layers=[1],
                    kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
                    head_dim=4, linear_key_head_dim=4, linear_value_head_dim=4,
                    linear_num_key_heads=2, linear_num_value_heads=4,
                    linear_conv_kernel_dim=3, shared_expert_intermediate_size=12,
                    layer_types=["linear_attention", "full_attention"] * 2)
        cases = {"mixtral": (range(4), 2, "block_sparse_moe"),
                 "olmoe": (range(4), 3, "mlp"),
                 "qwen2_moe": ([3], 3, "mlp"), "qwen3_moe": ([3], 3, "mlp"),
                 "qwen3_next": ([3], 3, "mlp"),
                 "glm4_moe": ([2, 3], 4, "mlp"), "deepseek_v3": ([2, 3], 4, "mlp")}
        with tempfile.TemporaryDirectory() as directory:
            for model_type, (layers, count, block) in cases.items():
                with self.subTest(model_type=model_type):
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(dict(base, model_type=model_type)))
                    manifest = HFMoeManifest.read(directory)
                    path.unlink()
                    self.assertEqual(expert_module_names(manifest), tuple(
                        f"model.layers.{layer}.{block}.experts.{expert}"
                        for layer in layers for expert in range(count)))

    def test_no_newer_family_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            for model_type in ("glm5_next", "glm5_next_text", "glm_moe_dsa", "glm5_moe", "glm",
                               "inkling_mm_model", "inkling", "kimi_k3", "kimi_linear",
                               "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text",
                               "deepseek_v4", "deepseek_v41", "deepseek_v41_text"):
                with self.subTest(model_type=model_type):
                    (Path(directory) / "config.json").write_text(json.dumps({"model_type": model_type}))
                    with self.assertRaisesRegex(ValueError, "unsupported sparse"):
                        HFMoeManifest.read(directory)
                    # Complete older-family geometry must not turn a new identity
                    # into an alias, including multimodal text-config wrappers.
                    geometry = dict(hidden_size=16, intermediate_size=24,
                        moe_intermediate_size=12, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
                        num_experts=4, n_routed_experts=4, num_experts_per_tok=2,
                        n_shared_experts=1, n_group=2, topk_group=1,
                        first_k_dense_replace=1)
                    for config in (dict(geometry, model_type=model_type),
                                   dict(model_type=model_type, text_config=geometry)):
                        (Path(directory) / "config.json").write_text(json.dumps(config))
                        with self.assertRaisesRegex(ValueError, "unsupported sparse"):
                            HFMoeManifest.read(directory)

    def test_rejects_remote_and_unknown(self):
        with self.assertRaises((ValueError, OSError)):
            HFMoeManifest.read("nonexistent-hub-owner/nonexistent-model")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for raw in ({"model_type": "llama"}, [], {"model_type": "mixtral", "auto_map": {"x": "y"}}):
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaises(ValueError):
                    HFMoeManifest.read(directory)


AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "accelerate"))


@unittest.skipUnless(AVAILABLE, "optional real-model dependencies unavailable")
@unittest.skip("P02 has no admitted Transformers/Accelerate route; differential execution is restricted")
class DifferentialTests(unittest.TestCase):
    def test_dense_glm_name_collision_is_not_sparse_support(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "glm" not in CONFIG_MAPPING:
            self.skipTest("installed Transformers does not provide dense glm")
        config = AutoConfig.for_model("glm", vocab_size=16, hidden_size=8,
            intermediate_size=12, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=4, partial_rotary_factor=1.0,
            pad_token_id=0, bos_token_id=1, eos_token_id=2)
        reference = AutoModelForCausalLM.from_config(config)
        self.assertEqual(type(reference).__name__, "GlmForCausalLM")
        self.assertEqual(type(reference.model.layers[0].mlp).__name__, "GlmMLP")
        self.assertFalse(any(".experts" in name for name, _ in reference.named_modules()))
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory, safe_serialization=True)
            with self.assertRaisesRegex(ValueError, "unsupported sparse"):
                HFMoeAdapter.load(directory)

    def test_olmoe_rejects_wrong_layer_inventory(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, OlmoeForCausalLM

        config = AutoConfig.for_model("olmoe", vocab_size=16, hidden_size=8,
            intermediate_size=12, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, num_experts=2, num_experts_per_tok=1)
        reference = AutoModelForCausalLM.from_config(config)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory, safe_serialization=True)
            original = OlmoeForCausalLM.__init__

            def wrong_inventory(model, config):
                original(model, config)
                # Same expert count, wrong layer identity: must fail before dispatch.
                model.model.layers.append(model.model.layers[0])
                model.model.layers[0] = torch.nn.Identity()

            with patch.object(OlmoeForCausalLM, "__init__", wrong_inventory):
                with self.assertRaisesRegex(ValueError, "layout does not match"):
                    HFMoeAdapter.load(directory)

    def test_failed_dispatch_removes_only_new_owned_offload_directory(self):
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.for_model('mixtral', vocab_size=16, hidden_size=8,
            intermediate_size=12, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, num_local_experts=2, num_experts_per_tok=1)
        reference = AutoModelForCausalLM.from_config(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'
            offload = root / 'offload'
            offload.mkdir()
            unrelated = offload / 'keep.txt'
            unrelated.write_text('preserve')
            reference.save_pretrained(checkpoint, safe_serialization=True)
            def failed_dispatch(model, *args, **kwargs):
                folder = Path(kwargs['offload_folder'])
                (folder / 'incomplete.dat').write_bytes(b'x')
                raise RuntimeError('injected dispatch failure')
            with patch('accelerate.load_checkpoint_and_dispatch', side_effect=failed_dispatch):
                with self.assertRaisesRegex(RuntimeError, 'injected dispatch failure'):
                    HFMoeAdapter.load(checkpoint,
                        expert_placement={'model.layers.0.block_sparse_moe.experts.0': 'disk'},
                        offload_directory=offload)
            self.assertEqual(list(offload.iterdir()), [unrelated])
            self.assertEqual(unrelated.read_text(), 'preserve')

    def check_architecture(self, model_type, *, tied=False, sharded=False):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if model_type not in CONFIG_MAPPING:
            self.skipTest(f"installed Transformers does not support {model_type}")
        torch.manual_seed(17)
        config = AutoConfig.for_model(
            model_type, vocab_size=32, hidden_size=16, intermediate_size=24,
            moe_intermediate_size=12, shared_expert_intermediate_size=12,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, num_experts=3, num_local_experts=3,
            num_experts_per_tok=2, decoder_sparse_step=1,
            max_position_embeddings=64, attention_dropout=0.0,
            router_jitter_noise=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=0,
            tie_word_embeddings=tied,
        )
        config._attn_implementation = "eager"
        if model_type == "qwen3_next":
            config.layer_types = ["linear_attention", "full_attention"]
            config.linear_key_head_dim = 4
            config.linear_value_head_dim = 4
            config.linear_num_key_heads = 2
            config.linear_num_value_heads = 4
            config.linear_conv_kernel_dim = 3
            config.partial_rotary_factor = 1.0
        if model_type in ("glm4_moe", "deepseek_v3"):
            config.n_routed_experts = 4
            config.n_shared_experts = 1
            config.n_group = 2
            config.topk_group = 1
            config.first_k_dense_replace = 1
        if model_type == "deepseek_v3":
            config.num_key_value_heads = config.num_attention_heads
            config.q_lora_rank = 8
            config.kv_lora_rank = 8
            config.qk_rope_head_dim = 4
            config.qk_nope_head_dim = 4
            config.qk_head_dim = 8
            config.v_head_dim = 4
            config.head_dim = 4
        reference = AutoModelForCausalLM.from_config(config).eval()
        expert_names = tuple(name for name, _ in reference.named_modules()
                             if name.rsplit(".", 1)[-1].isdigit() and ".experts." in name)
        self.assertEqual(len(expert_names), 4 if model_type in ("glm4_moe", "deepseek_v3") else 6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            reference.save_pretrained(checkpoint, safe_serialization=True,
                                      max_shard_size="2KB" if sharded else "5GB")
            if sharded:
                self.assertTrue((checkpoint / "model.safetensors.index.json").exists())
            saved_config = (checkpoint / "config.json").read_text()
            self.assertEqual(expert_module_names(HFMoeManifest.read(checkpoint)), expert_names)
            invalid_fields = {}
            if model_type in ("glm4_moe", "deepseek_v3"):
                invalid_fields = {"first_k_dense_replace": 2, "n_group": 3,
                                  "topk_group": 3, "n_shared_experts": 0}
            if model_type == "deepseek_v3":
                invalid_fields.update(num_key_value_heads=2, qk_rope_head_dim=3, kv_lora_rank=0)
            if model_type == "qwen3_next":
                invalid_fields = {"layer_types": ["full_attention", "full_attention"],
                                  "linear_num_value_heads": 3, "linear_conv_kernel_dim": 0}
            for field, value in invalid_fields.items():
                invalid = json.loads(saved_config)
                invalid[field] = value
                (checkpoint / "config.json").write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    HFMoeManifest.read(checkpoint)
            (checkpoint / "config.json").write_text(saved_config)
            placement = {name: "disk" for name in expert_names}
            from accelerate import load_checkpoint_and_dispatch

            def checked_dispatch(model, *args, **kwargs):
                self.assertTrue(all(tensor.device.type == "meta" for tensor in model.parameters()))
                self.assertTrue(kwargs["offload_state_dict"])
                self.assertTrue(kwargs["strict"])
                self.assertEqual({name for name, device in kwargs["device_map"].items()
                                  if device == "disk"}, set(expert_names))
                return load_checkpoint_and_dispatch(model, *args, **kwargs)

            # Instrument the real call; no model loading or execution is mocked.
            with patch("accelerate.load_checkpoint_and_dispatch", side_effect=checked_dispatch):
                adapter = HFMoeAdapter.load(checkpoint, expert_placement=placement,
                                            offload_directory=root / "offload")
            self.assertIs(type(adapter.model), type(reference))
            if tied:
                self.assertIs(adapter.model.lm_head.weight, adapter.model.model.embed_tokens.weight)
            self.assertTrue(list(adapter.offload_directory.rglob("*.dat")))
            ids = torch.tensor([[1, 5, 9, 4], [1, 7, 3, 0]])
            mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
            kwargs = dict(input_ids=ids, attention_mask=mask, output_router_logits=True,
                          output_attentions=True, use_cache=True)
            routing = [[], []]
            hooks = []
            if model_type in ("glm4_moe", "deepseek_v3"):
                for index, model in enumerate((reference, adapter.model)):
                    for name, module in model.named_modules():
                        if name.endswith(".mlp.gate"):
                            hooks.append(module.register_forward_hook(
                                lambda module, args, output, index=index:
                                    routing[index].append(tuple(t.clone() for t in output))))
            with torch.inference_mode():
                expected = reference(**kwargs)
                actual = adapter.forward(**kwargs)
                for hook in hooks:
                    hook.remove()
                if model_type in ("glm4_moe", "deepseek_v3"):
                    self.assertEqual(len(routing[0]), 1)
                    for left, right in zip(routing[0], routing[1], strict=True):
                        for a, b in zip(left, right, strict=True):
                            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
                for left, right in zip(getattr(actual, "router_logits", ()) or (),
                                       getattr(expected, "router_logits", ()) or (), strict=True):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
                for left, right in zip(actual.attentions, expected.attentions, strict=True):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
                next_ids = torch.tensor([[6], [8]])
                next_mask = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], dim=1)
                cached_expected = reference(input_ids=next_ids, attention_mask=next_mask,
                                            past_key_values=expected.past_key_values)
                cached_actual = adapter.forward(input_ids=next_ids, attention_mask=next_mask,
                                                past_key_values=actual.past_key_values)
                torch.testing.assert_close(cached_actual.logits, cached_expected.logits,
                                           rtol=1e-5, atol=1e-6)
                if model_type == "qwen3_next":
                    for field in ("conv_states", "recurrent_states", "key_cache", "value_cache"):
                        for a, b in zip(getattr(cached_actual.past_key_values, field),
                                        getattr(cached_expected.past_key_values, field), strict=True):
                            if a is None or b is None:
                                self.assertIs(a, b)
                            else:
                                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                generation = dict(input_ids=ids[:1], attention_mask=mask[:1],
                                  max_new_tokens=3, do_sample=False)
                self.assertTrue(torch.equal(adapter.generate(**generation), reference.generate(**generation)))
            adapter.validate_placement()
            for name, tensor in adapter.model.named_parameters():
                self.assertEqual(tensor.device.type, "meta" if ".experts." in name else "cpu")
            # Each selected expert has genuine disk-backed weights, not a mock hook.
            offloaded = json.loads((adapter.offload_directory / "index.json").read_text())
            self.assertEqual(set(offloaded), {name for name in reference.state_dict() if ".experts." in name})
            for invalid in ({"model.layers.0": "disk"}, {expert_names[0]: "auto"},
                            {expert_names[0]: "cuda:0"}):
                with self.assertRaises(ValueError):
                    HFMoeAdapter.load(checkpoint, expert_placement=invalid)
            for device in ("auto", "disk", "cuda", "mps", "cuda:999"):
                with self.assertRaises(ValueError):
                    HFMoeAdapter.load(checkpoint, resident_device=device)
            with self.assertRaises(ValueError):
                HFMoeAdapter.load(checkpoint, expert_placement=placement)
            resident = HFMoeAdapter.load(checkpoint)
            resident.validate_placement()
            with torch.inference_mode():
                torch.testing.assert_close(resident.forward(input_ids=ids).logits,
                                           reference(input_ids=ids).logits)
            mixed = HFMoeAdapter.load(checkpoint,
                                      expert_placement={expert_names[0]: "disk"},
                                      offload_directory=root / "offload")
            self.assertNotEqual(mixed.offload_directory, adapter.offload_directory)
            with torch.inference_mode():
                torch.testing.assert_close(mixed.forward(input_ids=ids).logits,
                                           reference(input_ids=ids).logits)
            mixed.validate_placement()
            raw = json.loads((checkpoint / "config.json").read_text())
            raw["architectures"] = ["UnrecognizedForCausalLM"]
            (checkpoint / "config.json").write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                HFMoeAdapter.load(checkpoint)
            raw["architectures"] = [type(reference).__name__]
            (checkpoint / "config.json").write_text(json.dumps(raw), encoding="utf-8")
            if not sharded:
                from safetensors.torch import load_file, save_file
                state = load_file(str(checkpoint / "model.safetensors"))
                key = next(name for name in state if ".self_attn." in name and name.endswith("weight"))
                expert_key = next(name for name in state if ".experts." in name)
                for failure in ("missing", "extra", "shape", "expert_missing", "expert_shape",
                                "tied_conflict", "tied_missing"):
                    if failure.startswith("tied_") and not tied:
                        continue
                    bad = dict(state)
                    if failure == "missing":
                        del bad[key]
                    elif failure == "extra":
                        bad["unexpected.weight"] = torch.zeros(1)
                    elif failure == "shape":
                        bad[key] = bad[key][:1].clone()
                    elif failure == "expert_missing":
                        del bad[expert_key]
                    elif failure == "expert_shape":
                        bad[expert_key] = bad[expert_key][:1].clone()
                    elif failure == "tied_missing":
                        bad.pop("lm_head.weight", None)
                        bad.pop("model.embed_tokens.weight", None)
                    else:
                        alias = "lm_head.weight" if "lm_head.weight" not in bad else "model.embed_tokens.weight"
                        source = "model.embed_tokens.weight" if alias == "lm_head.weight" else "lm_head.weight"
                        bad[alias] = bad[source].clone() + 1
                    save_file(bad, str(checkpoint / "model.safetensors"), metadata={"format": "pt"})
                    with self.assertRaisesRegex(ValueError, "mismatch|inconsistent"):
                        HFMoeAdapter.load(checkpoint)
                save_file(state, str(checkpoint / "model.safetensors"), metadata={"format": "pt"})
            else:
                index_path = checkpoint / "model.safetensors.index.json"
                index = json.loads(index_path.read_text())
                index["weight_map"]["missing.weight"] = next(iter(index["weight_map"].values()))
                index_path.write_text(json.dumps(index), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "missing or extra"):
                    HFMoeAdapter.load(checkpoint)

    def test_mixtral(self):
        self.check_architecture("mixtral")

    def test_qwen2_moe(self):
        self.check_architecture("qwen2_moe")

    def test_qwen3_moe(self):
        self.check_architecture("qwen3_moe")

    def test_olmoe(self):
        self.check_architecture("olmoe")

    def test_qwen3_next(self):
        self.check_architecture("qwen3_next")

    def test_qwen3_next_tied_sharded(self):
        self.check_architecture("qwen3_next", tied=True, sharded=True)

    def test_glm4_moe(self):
        self.check_architecture("glm4_moe")

    def test_glm4_moe_tied_sharded(self):
        self.check_architecture("glm4_moe", tied=True, sharded=True)

    def test_deepseek_v3(self):
        self.check_architecture("deepseek_v3")

    def test_deepseek_v3_tied_sharded(self):
        self.check_architecture("deepseek_v3", tied=True, sharded=True)

    def test_olmoe_tied_sharded(self):
        self.check_architecture("olmoe", tied=True, sharded=True)

    def test_olmoe_tied_alias_conflict(self):
        self.check_architecture("olmoe", tied=True)

    def test_mixtral_tied_sharded(self):
        self.check_architecture("mixtral", tied=True, sharded=True)

    def test_qwen2_tied_sharded(self):
        self.check_architecture("qwen2_moe", tied=True, sharded=True)

    def test_qwen3_tied_sharded(self):
        self.check_architecture("qwen3_moe", tied=True, sharded=True)

    def test_tied_alias_conflict(self):
        self.check_architecture("mixtral", tied=True)


if __name__ == "__main__":
    unittest.main()
