"""Compiled resident Mixtral compared with a separate official model oracle."""
import ctypes
import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.native_mixtral_decoder import NativeMixtralDecoder
from tests.test_independent_qwen4 import build


class NativeMixtralDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
            raise unittest.SkipTest('optional exact Mixtral oracle unavailable')
        import torch
        import transformers
        if transformers.__version__ != '5.16.1':
            raise unittest.SkipTest('exact Mixtral oracle requires Transformers 5.16.1')
        cls.torch = torch
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.library = build(cls.directory.name, 'native_mixtral_decoder')
        cls.loaded = []
        cls.addClassCleanup(cls.unload)

    @classmethod
    def unload(cls):
        if os.name == 'nt':
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            for decoder in cls.loaded:
                ctypes.windll.kernel32.FreeLibrary(decoder.library._handle)

    def fixture(self, **options):
        from transformers import MixtralConfig, MixtralForCausalLM
        config = MixtralConfig(**(dict(vocab_size=31, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=32,
            sliding_window=4, initializer_range=.15) | options))
        config._attn_implementation = 'eager'
        with self.torch.random.fork_rng():
            self.torch.manual_seed(731)
            oracle = MixtralForCausalLM(config).float().eval()
        weights = {name: value.detach().numpy() for name, value in oracle.state_dict().items()}
        decoder = NativeMixtralDecoder(config, weights, library=self.library, max_context=16)
        self.loaded.append(decoder)
        return oracle, decoder, weights

    def compare(self, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=5e-5)

    def test_prefill_incremental_chunks_and_gqa_against_official_oracle(self):
        for options in ({}, {'sliding_window': None}, {'tie_word_embeddings': True}):
            with self.subTest(options=options):
                oracle, decoder, _ = self.fixture(**options)
                ids = [[1, 3, 5, 7, 9, 4], [2, 6, 8, 3, 1, 5]]
                with self.torch.no_grad():
                    expected = oracle(self.torch.tensor(ids), use_cache=True,
                                      output_hidden_states=True, output_router_logits=True)
                self.compare(decoder.forward(ids).logits, expected.logits.numpy())
                full_cache = decoder.cache_snapshot()
                for (keys, values), layer in zip(full_cache, expected.past_key_values.layers):
                    retained = layer.keys.shape[2]
                    self.compare(keys[:, 6-retained:6].transpose(0, 2, 1, 3), layer.keys.numpy())
                    self.compare(values[:, 6-retained:6].transpose(0, 2, 1, 3), layer.values.numpy())
                decoder.reset()
                parts = [decoder.forward(np.array(ids)[:, start:end]).logits
                         for start, end in ((0, 2), (2, 5), (5, 6))]
                self.compare(np.concatenate(parts, axis=1), expected.logits.numpy())
                for before, after in zip(full_cache, decoder.cache_snapshot()):
                    for a, b in zip(before, after):
                        self.compare(a, b)
                decoder.reset()
                parts = [decoder.forward(np.array(ids)[:, index:index+1]).logits for index in range(6)]
                self.compare(np.concatenate(parts, axis=1), expected.logits.numpy())

    def test_split_expert_layout_matches_fused_oracle(self):
        oracle, fused, weights = self.fixture()
        split = {name: value for name, value in weights.items() if '.mlp.' not in name}
        for index in range(2):
            old = f'model.layers.{index}.mlp.'
            new = f'model.layers.{index}.block_sparse_moe.'
            split[new+'gate.weight'] = weights[old+'gate.weight']
            gate_up, down = weights[old+'experts.gate_up_proj'], weights[old+'experts.down_proj']
            for expert in range(4):
                split[new+f'experts.{expert}.w1.weight'] = np.ascontiguousarray(gate_up[expert, :24])
                split[new+f'experts.{expert}.w3.weight'] = np.ascontiguousarray(gate_up[expert, 24:])
                split[new+f'experts.{expert}.w2.weight'] = np.ascontiguousarray(down[expert])
        decoder = NativeMixtralDecoder(oracle.config, split, library=self.library, max_context=16)
        self.loaded.append(decoder)
        self.compare(decoder.forward([1, 3, 5]).logits, fused.forward([1, 3, 5]).logits)

    def test_failures_preserve_committed_kv_and_position(self):
        _, decoder, _ = self.fixture()
        decoder.forward([1, 3])
        before = decoder.cache_snapshot()
        for ids in ([31], [True], [], [[1], [2]]):
            with self.assertRaises(ValueError):
                decoder.forward(ids)
        with patch.object(decoder, '_execute', return_value=4):
            with self.assertRaises(RuntimeError):
                decoder.forward([5])
        self.assertEqual(decoder.position, 2)
        for old, new in zip(before, decoder.cache_snapshot()):
            for a, b in zip(old, new):
                np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(decoder.forward([5]).logits).all())

    def test_window_one_and_fp32_normalization_overflow_reject(self):
        oracle, _, weights = self.fixture()
        with self.assertRaises(ValueError):
            NativeMixtralDecoder(oracle.config.to_dict() | {'sliding_window': 1}, weights,
                                library=self.library)
        weights = {name: value.copy() for name, value in weights.items()}
        weights['model.embed_tokens.weight'][1] = 1e20
        decoder = NativeMixtralDecoder(oracle.config, weights, library=self.library)
        self.loaded.append(decoder)
        with self.assertRaises(RuntimeError):
            decoder.forward([1])
        self.assertEqual(decoder.position, 0)
        self.assertEqual(decoder.cache_snapshot(), ())

    def test_raw_window_one_and_aggregate_normalization_overflow_reject(self):
        oracle, decoder, weights = self.fixture()
        decoder._c.window = 1
        with self.assertRaises(RuntimeError):
            decoder.forward([1])
        self.assertEqual(decoder.position, 0)
        weights = {name: value.copy() for name, value in weights.items()}
        weights['model.embed_tokens.weight'][1] = 1e19
        decoder = NativeMixtralDecoder(oracle.config, weights, library=self.library)
        self.loaded.append(decoder)
        with self.assertRaises(RuntimeError):
            decoder.forward([1])
        self.assertEqual(decoder.position, 0)
