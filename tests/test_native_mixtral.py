"""Independent tiny numerical oracle; no downloaded model weights."""
import importlib.util
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.native_mixtral import IndependentMixtralDecoder


AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'safetensors'))


@unittest.skipUnless(AVAILABLE, 'optional oracle dependencies unavailable')
class NativeMixtralTests(unittest.TestCase):
    def fixture(self, **kwargs):
        import torch
        import transformers
        from transformers import MixtralConfig, MixtralForCausalLM
        torch.manual_seed(731)
        config = MixtralConfig(vocab_size=37, hidden_size=24, intermediate_size=19,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=32,
            eos_token_id=36, attention_dropout=0., **kwargs)
        config._attn_implementation = 'eager'
        oracle = MixtralForCausalLM(config).eval()
        state = oracle.state_dict()
        expected_experts = {
            f'model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}.weight': shape
            for layer in range(config.num_hidden_layers)
            for expert in range(config.num_local_experts)
            for projection, shape in (('w1', (config.intermediate_size, config.hidden_size)),
                                      ('w2', (config.hidden_size, config.intermediate_size)),
                                      ('w3', (config.intermediate_size, config.hidden_size)))
        }
        actual_experts = {name: tuple(value.shape) for name, value in state.items() if '.experts.' in name}
        modulelist_experts = all(
            isinstance(getattr(getattr(layer, 'block_sparse_moe', None), 'experts', None), torch.nn.ModuleList)
            for layer in oracle.model.layers)
        if not modulelist_experts or actual_experts != expected_experts:
            self.skipTest(f'incompatible Mixtral oracle Transformers {transformers.__version__}: '
                          'requires block_sparse_moe ModuleList experts with exact w1/w2/w3 '
                          'tensor contract (validated on 4.57.6); fused/packed mlp.experts banks '
                          'are unsupported, not converted or aliased')
        tensors = {k: v.detach().numpy() for k, v in state.items()}
        return oracle, IndependentMixtralDecoder(config.to_dict(), tensors)

    def compare(self, actual, expected):
        for a, e in [(actual.logits, expected.logits),
                     *zip(actual.hidden_states, expected.hidden_states),
                     *zip(actual.router_logits, expected.router_logits),
                     *zip(actual.attentions, expected.attentions)]:
            if a.ndim == 4 and a.shape[-1] != e.shape[-1]:
                np.testing.assert_array_equal(a[..., :-e.shape[-1]], 0)
                a = a[..., -e.shape[-1]:]
            np.testing.assert_allclose(a, e.detach().numpy(), atol=3e-6, rtol=3e-5)
        import torch
        for (indices, weights), router in zip(actual.routes, expected.router_logits):
            p = torch.softmax(router.float(), dim=-1)
            ew, ei = torch.topk(p, actual.routes[0][0].shape[-1], dim=-1)
            ew /= ew.sum(dim=-1, keepdim=True)
            np.testing.assert_array_equal(indices, ei.numpy())
            np.testing.assert_allclose(weights, ew.numpy(), atol=2e-7)

    def test_full_cached_padding_positions_and_window(self):
        import torch
        for window in (None, 3):
            oracle, native = self.fixture(sliding_window=window)
            with self.subTest(window=window):
                ids = torch.tensor([[0, 1, 5, 8, 3], [1, 9, 4, 6, 2]])
                mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
                pos = (mask.cumsum(-1)-1).clamp(min=0)
                with torch.inference_mode():
                    expected = oracle(ids, attention_mask=mask, position_ids=pos,
                        output_hidden_states=True, output_router_logits=True, output_attentions=True)
                    actual = native.forward(ids.numpy(), attention_mask=mask.numpy(), position_ids=pos.numpy())
                    self.compare(actual, expected)
                    native.reset()
                    past = None
                    for start, end in ((0, 2), (2, 4), (4, 5)):
                        expected = oracle.model(ids[:, start:end], attention_mask=mask[:, :end],
                            position_ids=pos[:, start:end], past_key_values=past,
                            output_hidden_states=True, output_router_logits=True, output_attentions=True)
                        expected.logits = oracle.lm_head(expected.last_hidden_state)
                        past = expected.past_key_values
                        actual = native.forward(ids[:, start:end].numpy(), attention_mask=mask[:, :end].numpy(),
                            position_ids=pos[:, start:end].numpy())
                        self.compare(actual, expected)
                        for i, (k, v) in enumerate(native.cache):
                            retained = past.layers[i].keys.shape[2]
                            np.testing.assert_allclose(k[:, :, -retained:], past.layers[i].keys.numpy(), atol=2e-6)
                            np.testing.assert_allclose(v[:, :, -retained:], past.layers[i].values.numpy(), atol=2e-6)

    def test_generation_tied_checkpoint_and_reset(self):
        import torch
        for tied in (False, True):
            oracle, native = self.fixture(tie_word_embeddings=tied)
            with tempfile.TemporaryDirectory() as directory:
                oracle.save_pretrained(directory)
                loaded = IndependentMixtralDecoder.from_directory(directory)
                np.testing.assert_array_equal(loaded.forward([1, 3]).logits, native.forward([1, 3]).logits)
            ids, expected = [1, 3, 5], []
            with torch.inference_mode():
                for _ in range(4):
                    token = int(oracle(torch.tensor([ids])).logits[0, -1].argmax())
                    if token == oracle.config.eos_token_id:
                        break
                    expected.append(token)
                    ids.append(token)
            self.assertEqual(native.generate_ids([1, 3, 5], max_tokens=4), expected)
            native.reset()
            self.assertEqual(native.position, 0)
            self.assertEqual(native.cache, [])

    def test_reject_corrupt_contracts_and_atomic_validation(self):
        oracle, native = self.fixture()
        config = oracle.config.to_dict()
        for key, value in [('model_type', 'glm5_moe'), ('hidden_act', 'gelu'),
                           ('rope_scaling', {'type': 'linear'}), ('num_experts_per_tok', 5),
                           ('rms_norm_eps', float('nan')), ('sliding_window', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                IndependentMixtralDecoder(dict(config, **{key: value}), native.weights)
        for change in ('missing', 'extra', 'shape', 'nan'):
            tensors = dict(native.weights)
            key = 'model.norm.weight'
            if change == 'missing':
                del tensors[key]
            elif change == 'extra':
                tensors['bogus'] = np.zeros(1)
            elif change == 'shape':
                tensors[key] = np.zeros(3)
            else:
                tensors[key] = np.full_like(tensors[key], np.nan)
            with self.assertRaises(ValueError):
                IndependentMixtralDecoder(config, tensors)
        native.forward([1, 2])
        for ids in ([1.5], [-1], [37], []):
            with self.assertRaises(ValueError):
                native.forward(ids)
            self.assertEqual(native.position, 2)
        with self.assertRaises(ValueError):
            native.forward([3], attention_mask=[[0, 1, 1]])
        self.assertEqual(native.position, 2)


class MixtralPackedContractRejectionTests(unittest.TestCase):
    def test_production_decoder_rejects_fused_banks_without_aliasing(self):
        config = dict(model_type='mixtral', hidden_size=8, intermediate_size=6,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            num_local_experts=2, num_experts_per_tok=1, vocab_size=11,
            max_position_embeddings=16)
        prefix = 'model.layers.0.'
        dense = {'model.embed_tokens.weight': np.zeros((11, 8), np.float32),
                 'model.norm.weight': np.ones(8, np.float32),
                 'lm_head.weight': np.zeros((11, 8), np.float32)}
        for name in ('input_layernorm', 'post_attention_layernorm'):
            dense[prefix+name+'.weight'] = np.ones(8, np.float32)
        for projection, width in (('q', 8), ('k', 4), ('v', 4), ('o', 8)):
            dense[prefix+'self_attn.'+projection+'_proj.weight'] = np.zeros((width, 8), np.float32)
        for owner in ('mlp', 'block_sparse_moe'):
            with self.subTest(owner=owner):
                packed = dict(dense)
                packed[prefix+owner+'.gate.weight'] = np.zeros((2, 8), np.float32)
                packed[prefix+owner+'.experts.gate_up_proj'] = np.zeros((2, 12, 8), np.float32)
                packed[prefix+owner+'.experts.down_proj'] = np.zeros((2, 8, 6), np.float32)
                with self.assertRaisesRegex(ValueError, 'tensor names mismatch'):
                    IndependentMixtralDecoder(config, packed)


if __name__ == '__main__':
    unittest.main()
