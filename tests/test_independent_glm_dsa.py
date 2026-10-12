"""Exact GLM-MoE-DSA black-box oracle; no reference code in production."""
import importlib.util
import os
import unittest

import numpy as np

from openagent.adaptive_engine.independent_glm_dsa import IndependentGlmDsaDecoder


def optional(reason):
    if os.environ.get('GLM_DSA_CERTIFY') == '1':
        raise RuntimeError('strict GLM-DSA certification: '+reason)
    raise unittest.SkipTest(reason)


def fixture(case, *, seed=619, **overrides):
    if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
        optional('optional GLM-MoE-DSA oracle unavailable')
    import torch
    import transformers
    if transformers.__version__ != '5.16.1':
        optional('requires pinned Transformers 5.16.1')
    if not hasattr(transformers, 'GlmMoeDsaForCausalLM'):
        optional('requires exact GlmMoeDsaForCausalLM oracle, not GLM4 or dense GLM')
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
    torch.manual_seed(seed)
    config = dict(vocab_size=37, hidden_size=24, intermediate_size=32, moe_intermediate_size=12,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4,
        n_shared_experts=1, num_experts_per_tok=2, q_lora_rank=12, kv_lora_rank=8,
        qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=6, index_n_heads=2,
        index_head_dim=8, index_topk=2, n_group=2, topk_group=1, first_k_dense_replace=1,
        max_position_embeddings=32, indexer_types=['full', 'shared', 'full'], eos_token_id=36,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.})
    config.update(overrides)
    c = GlmMoeDsaConfig(**config)
    c._attn_implementation = 'eager'
    model = GlmMoeDsaForCausalLM(c).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer.mlp, 'gate'):
                layer.mlp.gate.e_score_correction_bias.copy_(torch.tensor([.13, -.09, .07, -.03]))
    weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    return model, IndependentGlmDsaDecoder(c.to_dict(), weights)


def compare(actual, expected):
    for a, e in [(actual.logits, expected.logits), *zip(actual.hidden_states, expected.hidden_states),
                 *zip(actual.attentions, expected.attentions)]:
        np.testing.assert_allclose(a, e.detach().numpy(), atol=4e-6, rtol=5e-5)


class GlmDsaTests(unittest.TestCase):
    def test_yarn_baseline_full_chunk_and_rotary_contract(self):
        oracle,decoder = fixture(self,qk_rope_head_dim=16,index_head_dim=24,
            max_position_embeddings=256,rope_parameters=dict(rope_type='yarn',rope_theta=10000.,factor=4.,
                original_max_position_embeddings=64,mscale=1.2,mscale_all_dim=.7,
                beta_fast=16.,beta_slow=2.,truncate=False))
        import torch
        from openagent.adaptive_engine.independent_glm_dsa import rotary_parameters
        frequency, amplitude = rotary_parameters(16,decoder.config.rope_parameters)
        np.testing.assert_allclose(frequency,oracle.model.rotary_emb.inv_freq.numpy(),atol=1e-8,rtol=2e-7)
        self.assertAlmostEqual(amplitude,oracle.model.rotary_emb.attention_scaling,places=12)
        ids = torch.tensor([[1,3,5,7,9,2,4,6]])
        positions = torch.arange(60,68)[None]
        with torch.inference_mode():
            full = oracle(ids,position_ids=positions,output_hidden_states=True,output_attentions=True)
            compare(decoder.forward(ids.numpy(),position_ids=positions.numpy()),full)
            decoder.reset()
            past = None
            chunks = []
            for start,end in ((0,3),(3,4),(4,8)):
                expected = oracle(ids[:,start:end],position_ids=positions[:,start:end],past_key_values=past,
                    output_hidden_states=True,output_attentions=True)
                past = expected.past_key_values
                actual = decoder.forward(ids[:,start:end].numpy(),position_ids=positions[:,start:end].numpy())
                compare(actual,expected)
                chunks.append(actual.logits)
                for i,layer in enumerate(past.layers):
                    key,value = decoder.expanded_cache(i)
                    np.testing.assert_allclose(key,layer.keys.numpy(),atol=3e-6,rtol=5e-5)
                    np.testing.assert_allclose(value,layer.values.numpy(),atol=3e-6,rtol=5e-5)
                    if decoder.cache[i][2] is not None:
                        np.testing.assert_allclose(decoder.cache[i][2],layer.indexer_keys.numpy(),atol=3e-6,rtol=5e-5)
            np.testing.assert_allclose(np.concatenate(chunks,axis=1),full.logits.numpy(),atol=4e-6,rtol=5e-5)

    def test_yarn_malformed_parameters(self):
        oracle,decoder = fixture(self)
        good = dict(rope_type='yarn',rope_theta=10000.,factor=4.,original_max_position_embeddings=8)
        for key,value in (('factor',None),('factor',True),('factor',.5),('factor',float('inf')),
                ('original_max_position_embeddings',0),('original_max_position_embeddings',8.),
                ('rope_theta',1.),('beta_fast',0.),('beta_slow',1e-320),('beta_slow',float('nan')),
                ('beta_fast',.5),('attention_factor',0.),('attention_factor',True),
                ('mscale',-1.),('mscale_all_dim',float('inf')),('truncate',1),
                ('partial_rotary_factor',.5),('unknown',1),('rope_type','dynamic')):
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                IndependentGlmDsaDecoder(dict(oracle.config.to_dict(),rope_parameters=good | {key:value}),decoder.weights)
        for key in ('factor','original_max_position_embeddings'):
            bad = dict(good)
            del bad[key]
            with self.assertRaises(ValueError):
                IndependentGlmDsaDecoder(dict(oracle.config.to_dict(),rope_parameters=bad),decoder.weights)

    def test_full_cached_sparse_attention_latent_and_indexer_keys(self):
        oracle, decoder = fixture(self)
        import torch
        ids = torch.tensor([[1, 3, 5, 7, 9], [1, 4, 6, 8, 2]])
        observed = []
        handles = [layer.self_attn.register_forward_hook(lambda module, args, output: observed.append(output[2]))
                   for layer in oracle.model.layers]
        try:
            with torch.inference_mode():
                expected = oracle(ids, output_hidden_states=True, output_attentions=True)
                actual = decoder.forward(ids.numpy())
                compare(actual, expected)
                self.selections(actual, observed, np.ones(ids.shape, bool), 0)
                decoder.reset(); past = None
                for start, end in ((0, 3), (3, 4), (4, 5)):
                    observed.clear()
                    expected = oracle(ids[:, start:end], past_key_values=past, output_hidden_states=True, output_attentions=True)
                    past = expected.past_key_values
                    actual = decoder.forward(ids[:, start:end].numpy())
                    compare(actual, expected)
                    self.selections(actual, observed, np.ones((2, end), bool), start)
                    for i, layer in enumerate(past.layers):
                        key, value = decoder.expanded_cache(i)
                        np.testing.assert_allclose(key, layer.keys.numpy(), atol=3e-6)
                        np.testing.assert_allclose(value, layer.values.numpy(), atol=3e-6)
                        if decoder.config.indexer_types[i] == 'full':
                            np.testing.assert_allclose(decoder.cache[i][2], layer.indexer_keys.numpy(), atol=3e-6)
                        else:
                            self.assertIsNone(decoder.cache[i][2])
        finally:
            for handle in handles:
                handle.remove()

    def selections(self, actual, observed, mask, start):
        for a, e in zip(actual.indexer_indices, observed):
            e = e.numpy()
            for batch in range(a.shape[0]):
                for token in range(a.shape[1]):
                    valid = lambda row: {int(i) for i in row if i <= start+token and mask[batch, i]}
                    self.assertEqual(valid(a[batch, token]), valid(e[batch, token]))
        np.testing.assert_array_equal(actual.indexer_indices[0], actual.indexer_indices[1])

    def test_padding_grouped_router_and_generation(self):
        oracle, decoder = fixture(self, seed=623)
        import torch
        observed = []
        handles = [layer.mlp.gate.register_forward_hook(lambda module, args, output: observed.append(output))
                   for layer in oracle.model.layers if hasattr(layer.mlp, 'gate')]
        ids = torch.tensor([[0, 0, 1, 3, 7], [1, 4, 8, 5, 2]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        positions = (mask.cumsum(-1)-1).clamp(min=0)
        try:
            with torch.inference_mode():
                expected = oracle(ids, attention_mask=mask, position_ids=positions, output_hidden_states=True, output_attentions=True)
                actual = decoder.forward(ids.numpy(), attention_mask=mask.numpy(), position_ids=positions.numpy())
                compare(actual, expected)
                for route, logits, e in zip(actual.routes[1:], actual.router_logits[1:], observed):
                    e_logits, e_weights, e_indices = e
                    np.testing.assert_allclose(logits, e_logits.numpy(), atol=3e-6)
                    order = np.argsort(route[0], axis=-1)
                    e_order = np.argsort(e_indices.numpy(), axis=-1)
                    np.testing.assert_array_equal(np.take_along_axis(route[0], order, axis=-1), np.take_along_axis(e_indices.numpy(), e_order, axis=-1))
                    np.testing.assert_allclose(np.take_along_axis(route[1], order, axis=-1), np.take_along_axis(e_weights.numpy(), e_order, axis=-1), atol=4e-7)
                prompt, expected_ids = [1, 3, 7], []
                for _ in range(4):
                    token = int(oracle(torch.tensor([prompt])).logits[0, -1].argmax())
                    if token == 36:
                        break
                    prompt.append(token); expected_ids.append(token)
                self.assertEqual(decoder.generate_ids([1, 3, 7], max_tokens=4), expected_ids)
        finally:
            for handle in handles:
                handle.remove()

    def test_exact_identity_geometry_and_tensor_rejection(self):
        oracle, decoder = fixture(self)
        config = oracle.config.to_dict()
        for key, value in [('model_type', 'glm'), ('model_type', 'glm5_moe'),
            ('model_type', 'glm5_next_text'), ('indexer_types', ['shared']*3),
            ('num_key_value_heads', 1), ('q_lora_rank', None), ('n_group', 3),
            ('rope_parameters', {'rope_type': 'yarn'}), ('quantization_config', {})]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                IndependentGlmDsaDecoder(dict(config, **{key: value}), decoder.weights)
        key = 'model.layers.0.self_attn.indexer.k_norm.bias'
        weights = dict(decoder.weights)
        weights[key] = np.full_like(weights[key], np.nan)
        with self.assertRaises(ValueError):
            IndependentGlmDsaDecoder(config, weights)
        self.assertFalse(decoder.native_range_execution)


if __name__ == '__main__':
    unittest.main()
