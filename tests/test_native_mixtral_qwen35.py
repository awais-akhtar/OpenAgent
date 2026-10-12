"""Run with an isolated Transformers 5.16.1 oracle on PYTHONPATH."""
import importlib.util
import unittest

import numpy as np

from openagent.adaptive_engine.independent_qwen35 import IndependentQwen35TextDecoder


AVAILABLE = importlib.util.find_spec('transformers') is not None


@unittest.skipUnless(AVAILABLE, 'optional Transformers unavailable')
class Qwen35NumericalTests(unittest.TestCase):
    def fixture(self, seed=271):
        import torch
        import transformers
        if not hasattr(transformers, 'Qwen3_5MoeForCausalLM'):
            self.skipTest('requires exact Qwen3.5 MoE oracle, not Qwen3Next')
        from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
        torch.manual_seed(seed)
        c = Qwen3_5MoeTextConfig(vocab_size=37, hidden_size=24, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_key_head_dim=4, linear_value_head_dim=6, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_conv_kernel_dim=4, moe_intermediate_size=12,
            shared_expert_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
            max_position_embeddings=32, layer_types=['linear_attention', 'full_attention', 'linear_attention'],
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                             'partial_rotary_factor': .5, 'mrope_section': [1, 1, 0]}, eos_token_id=36)
        c._attn_implementation = 'eager'
        oracle = Qwen3_5MoeForCausalLM(c).eval()
        # Nonzero offsets exercise the family's (1 + weight) normalization contract.
        with torch.no_grad():
            for name, parameter in oracle.named_parameters():
                if 'layernorm.weight' in name or name == 'model.norm.weight' or '.self_attn.' in name and '_norm.weight' in name:
                    parameter.uniform_(-.15, .15)
        weights = {k: v.detach().numpy() for k, v in oracle.state_dict().items()}
        return oracle, IndependentQwen35TextDecoder(c.to_dict(), weights)

    def compare(self, actual, expected):
        for a, e in [(actual.logits, expected.logits), *zip(actual.hidden_states, expected.hidden_states)]:
            np.testing.assert_allclose(a, e.detach().numpy(), atol=4e-6, rtol=5e-5)
        if expected.attentions is not None:
            self.assertEqual(len(actual.attentions), len(expected.attentions))
            for a, e in zip(actual.attentions, expected.attentions):
                np.testing.assert_allclose(a, e.numpy(), atol=3e-6, rtol=5e-5)

    def test_mixed_decoder_full_cached_and_states(self):
        import torch
        oracle, decoder = self.fixture()
        ids = torch.tensor([[1, 3, 5, 8, 2], [1, 6, 7, 4, 9]])
        with torch.inference_mode():
            expected = oracle(ids, output_hidden_states=True, output_attentions=True)
            full = decoder.forward(ids.numpy())
            self.compare(full, expected)
            decoder.reset()
            past = None
            outputs = []
            for start, end in ((0, 2), (2, 4), (4, 5)):
                expected = oracle(ids[:, start:end], past_key_values=past, output_hidden_states=True, output_attentions=True)
                past = expected.past_key_values
                actual = decoder.forward(ids[:, start:end].numpy())
                self.compare(actual, expected)
                outputs.append(actual.logits)
                for i, kind in enumerate(decoder.config.layer_types):
                    layer = past.layers[i]
                    if kind == 'linear_attention':
                        np.testing.assert_allclose(decoder.cache[i][0], layer.conv_states[0].numpy(), atol=2e-6)
                        np.testing.assert_allclose(decoder.cache[i][1], layer.recurrent_states[0].numpy(), atol=2e-6)
                    else:
                        np.testing.assert_allclose(decoder.cache[i][0], layer.keys.numpy(), atol=2e-6)
                        np.testing.assert_allclose(decoder.cache[i][1], layer.values.numpy(), atol=2e-6)
            np.testing.assert_allclose(np.concatenate(outputs, axis=1), full.logits, atol=4e-6, rtol=5e-5)

    def test_padding_router_and_greedy(self):
        import torch
        oracle, decoder = self.fixture(277)
        routers = []
        handles = [layer.mlp.gate.register_forward_hook(lambda module, args, output: routers.append(output)) for layer in oracle.model.layers]
        ids = torch.tensor([[0, 0, 1, 3, 7], [1, 4, 8, 5, 2]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        positions = (mask.cumsum(-1)-1).clamp(min=0)
        try:
            with torch.inference_mode():
                expected = oracle(ids, attention_mask=mask, position_ids=positions, output_hidden_states=True, output_attentions=True)
                actual = decoder.forward(ids.numpy(), attention_mask=mask.numpy(), position_ids=positions.numpy())
                self.compare(actual, expected)
                for router, (indices, weights), observed in zip(actual.router_logits, actual.routes, routers):
                    # Router exposes (logits, weights, indices), independently observed.
                    logits, selected_weights, selected_indices = observed
                    np.testing.assert_allclose(router, logits.numpy(), atol=3e-6)
                    np.testing.assert_array_equal(indices, selected_indices.numpy())
                    np.testing.assert_allclose(weights, selected_weights.numpy(), atol=3e-7)
                decoder.reset()
                past = None
                for start, end in ((0, 3), (3, 4), (4, 5)):
                    expected = oracle(ids[:, start:end], attention_mask=mask[:, :end],
                        position_ids=positions[:, start:end], past_key_values=past,
                        output_hidden_states=True, output_attentions=True)
                    past = expected.past_key_values
                    self.compare(decoder.forward(ids[:, start:end].numpy(), attention_mask=mask[:, :end].numpy(),
                        position_ids=positions[:, start:end].numpy()), expected)
                decoder.reset()
                self.assertEqual(decoder.cache, [None]*3)
                prompt, expected_tokens = [1, 3, 7], []
                for _ in range(4):
                    token = int(oracle(torch.tensor([prompt])).logits[0, -1].argmax())
                    if token == 36:
                        break
                    prompt.append(token)
                    expected_tokens.append(token)
                self.assertEqual(decoder.generate_ids([1, 3, 7], max_tokens=4), expected_tokens)
        finally:
            for handle in handles:
                handle.remove()

    def test_fail_closed_contract(self):
        oracle, decoder = self.fixture()
        for key, value in [('model_type', 'qwen3_next'), ('model_type', 'qwen3_5_moe'),
                           ('layer_types', ['full_attention']), ('quantization_config', {}),
                           ('linear_num_value_heads', 3), ('rope_parameters', {'rope_type': 'yarn'})]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                IndependentQwen35TextDecoder(dict(oracle.config.to_dict(), **{key: value}), decoder.weights)
        weights = dict(decoder.weights)
        key = 'model.layers.0.mlp.experts.gate_up_proj'
        weights[key] = np.zeros((4, 24, 22), np.float32)
        with self.assertRaises(ValueError):
            IndependentQwen35TextDecoder(oracle.config.to_dict(), weights)
        self.assertFalse(decoder.status()['native_range_execution'])
        self.assertEqual(decoder.status()['implementation'], 'numpy')


if __name__ == '__main__':
    unittest.main()
