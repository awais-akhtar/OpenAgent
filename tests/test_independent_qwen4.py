"""Exact official Qwen4 oracle, never used by the production decoder."""
import ctypes as C
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.independent_qwen4 import IndependentQwen4TextDecoder
from openagent.adaptive_engine.native_qwen4_hyper import CompiledQwen4HyperKernel
from openagent.adaptive_engine.native_qwen4 import NativeQwen4TextDecoder


def build(directory, name='native_qwen4_hyper'):
    source = Path(__file__).resolve().parents[1]/f'openagent/adaptive_engine/{name}.c'
    output = Path(directory)/(name+('.dll' if os.name == 'nt' else '.so'))
    if os.name == 'nt':
        vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
        if not vswhere.exists():
            raise unittest.SkipTest('MSVC unavailable')
        location = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
            'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
        if not location:
            raise unittest.SkipTest('MSVC unavailable')
        batch = Path(location)/'VC/Auxiliary/Build/vcvars64.bat'
        args = ['cl', '/nologo', '/W4', '/WX', '/O2', '/fp:precise', '/LD', str(source), '/link', '/OUT:'+str(output)]
        command = 'cmd.exe /d /s /c ""'+str(batch)+'" && '+subprocess.list2cmdline(args)+'"'
    else:
        compiler = shutil.which('cc')
        if compiler is None:
            raise unittest.SkipTest('C compiler unavailable')
        command = [compiler, '-std=c99', '-Wall', '-Wextra', '-Werror', '-O2', '-shared', '-fPIC', str(source), '-o', str(output), '-lm']
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return output


class Qwen4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('torch') is None or importlib.util.find_spec('transformers') is None:
            raise unittest.SkipTest('exact Qwen4 oracle unavailable')
        try:
            from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
            from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
        except ModuleNotFoundError:
            raise unittest.SkipTest('installed Transformers has no exact Qwen4 oracle')
        cls.config_class, cls.oracle_class = Qwen4ExpTextConfig, Qwen4ExpForCausalLM
        cls.directory = tempfile.TemporaryDirectory()
        try:
            cls.kernel = CompiledQwen4HyperKernel(build(cls.directory.name))
            from tests.test_native_mixtral_delta import compile_kernel
            cls.delta_kernel = compile_kernel(cls.directory.name)
            cls.native_path = build(cls.directory.name, 'native_qwen4')
            cls.native_decoders = []
        except BaseException:
            cls.directory.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        if os.name == 'nt':
            C.windll.kernel32.FreeLibrary.argtypes = [C.c_void_p]
            C.windll.kernel32.FreeLibrary(cls.kernel.library._handle)
            C.windll.kernel32.FreeLibrary(cls.delta_kernel.library._handle)
            for decoder in cls.native_decoders:
                C.windll.kernel32.FreeLibrary(decoder.library._handle)
        cls.directory.cleanup()

    def fixture(self, **overrides):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(761)
        raw = dict(vocab_size=23, hidden_size=12, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=6,
            linear_num_key_heads=1, linear_num_value_heads=2,
            linear_key_head_dim=3, linear_value_head_dim=4, linear_conv_kernel_dim=3,
            moe_intermediate_size=9, shared_expert_intermediate_size=7,
            num_experts=3, num_experts_per_tok=2, hc_count=3, hc_lowrank=5,
            layer_types=['linear_attention']*2, ple_layer_ids=[],
            max_position_embeddings=32, rms_norm_eps=1e-6, hidden_act='silu',
            initializer_range=0.15, rope_parameters={'rope_type': 'default',
                'rope_theta': 10000.0, 'partial_rotary_factor': 1.0, 'mrope_section': [1, 1, 1]},
            eos_token_id=None, pad_token_id=None, norm_topk_prob=True)
        raw.update(overrides)
        config = self.config_class(**raw)
        config._attn_implementation = 'eager'
        config._experts_implementation = 'eager'
        oracle = self.oracle_class(config).eval()
        # Nonzero norm offsets exercise Qwen4's grouped normalization contract.
        with torch.no_grad():
            for name, parameter in oracle.named_parameters():
                if 'hc_norm.weight' in name:
                    parameter.uniform_(-0.15, 0.15)
                if '.ple.conv1d.weight' in name:
                    parameter.uniform_(-0.1, 0.1)
        weights = {name: value.detach().numpy() for name, value in oracle.state_dict().items()}
        decoder = IndependentQwen4TextDecoder(config.to_dict(), weights)
        compiled = IndependentQwen4TextDecoder(config.to_dict(), weights, hyper_kernel=self.kernel,
                                               delta_kernel=self.delta_kernel)
        return oracle, decoder, compiled

    def compare(self, actual, expected):
        np.testing.assert_allclose(actual.logits, expected.logits.detach().numpy(), atol=5e-6, rtol=7e-5)
        for a, b in zip(actual.hidden_states, expected.hidden_states, strict=True):
            np.testing.assert_allclose(a, b.detach().numpy(), atol=5e-6, rtol=7e-5)

    def oracle_forward(self, oracle, ids, mask=None, cache=None, offset=0):
        import torch
        tensor = torch.tensor(ids)
        positions = torch.arange(offset, offset+tensor.shape[1])[None].expand(tensor.shape[0], -1)
        # The official all-linear cache has no get_seq_length. Supply positions
        # and mask mappings explicitly without modifying its numerical kernels.
        visible = torch.ones((tensor.shape[0], offset+tensor.shape[1]), dtype=torch.bool) if mask is None else torch.tensor(mask).bool()
        causal = torch.arange(visible.shape[1])[None, :] <= torch.arange(offset, offset+tensor.shape[1])[:, None]
        allowed = visible[:, None, None, :] & causal[None, None]
        masks = {'full_attention': torch.where(allowed, 0.0, torch.finfo(torch.float32).min),
                 'linear_attention': None if mask is None else torch.tensor(mask)[:, -tensor.shape[1]:]}
        return oracle(tensor, position_ids=positions, attention_mask=masks,
                      past_key_values=cache, output_hidden_states=True, use_cache=True)

    def test_full_chunked_hyper_states_and_compiled_oracle(self):
        import torch
        oracle, decoder, compiled = self.fixture()
        ids = [[2, 7, 4, 9], [5, 3, 8, 6]]
        observed = []
        handles = [layer.mlp.gate.register_forward_hook(
            lambda _module, _args, output: observed.append(tuple(value.detach().numpy() for value in output)))
            for layer in oracle.model.layers]
        with torch.no_grad():
            expected = self.oracle_forward(oracle, ids)
            for handle in handles:
                handle.remove()
            first = self.oracle_forward(oracle, np.array(ids)[:, :2].tolist())
            second = self.oracle_forward(oracle, np.array(ids)[:, 2:].tolist(), cache=first.past_key_values, offset=2)
        for engine in (decoder, compiled):
            actual = engine.forward(ids)
            self.compare(actual, expected)
            for logits, (indices, scores), (reference_logits, reference_scores, reference_indices) in zip(
                    actual.router_logits, actual.routes, observed, strict=True):
                np.testing.assert_allclose(logits, reference_logits, atol=4e-6)
                np.testing.assert_array_equal(indices, reference_indices)
                np.testing.assert_allclose(scores, reference_scores, atol=4e-6)
            for i, state in enumerate(engine.cache):
                layer = expected.past_key_values.layers[i]
                np.testing.assert_allclose(state[0], layer.conv_states[0].numpy(), atol=4e-6)
                np.testing.assert_allclose(state[1], layer.recurrent_states[0].numpy(), atol=4e-6)
            engine.reset()
            left = engine.forward(np.array(ids)[:, :2]).logits
            right = engine.forward(np.array(ids)[:, 2:]).logits
            np.testing.assert_allclose(np.concatenate((left, right), axis=1), actual.logits, atol=5e-6)
            np.testing.assert_allclose(right, second.logits.numpy(), atol=5e-6, rtol=7e-5)

    def test_padding_unnormalized_routes_and_greedy(self):
        import torch
        oracle, decoder, compiled = self.fixture(norm_topk_prob=False)
        ids, mask = [[2, 7, 4, 9], [5, 3, 8, 6]], [[0, 1, 1, 1], [1, 1, 1, 1]]
        with torch.no_grad():
            expected = self.oracle_forward(oracle, ids, mask)
        for engine in (decoder, compiled):
            self.compare(engine.forward(ids, attention_mask=mask), expected)
            engine.reset()
        prompt = [2, 7, 4]
        with torch.no_grad():
            tokens, expected_ids = prompt.copy(), []
            for _ in range(3):
                token = int(self.oracle_forward(oracle, [tokens]).logits[0, -1].argmax())
                expected_ids.append(token)
                tokens.append(token)
        self.assertEqual(decoder.generate_ids(prompt, max_tokens=3), expected_ids)
        self.assertEqual(compiled.generate_ids(prompt, max_tokens=3), expected_ids)

    def test_exact_contract_rejections_and_transactional_failure(self):
        _, decoder, compiled = self.fixture()
        for values in ({'model_type': 'qwen3_5_moe_text'}, {'ple_layer_ids': [1]},
                {'layer_types': ['linear_attention', 'qwen_sparse_attention']},
                {'indexer_budget': 4}, {'output_gate_type': 'relu'}, {'hc_count': 1}):
            raw = vars(decoder.config).copy()
            raw.update(values)
            with self.assertRaises(ValueError):
                IndependentQwen4TextDecoder(raw, decoder.weights)
        compiled.forward([2, 7])
        old = [(a.copy(), b.copy()) for a, b in compiled.cache]
        with patch.object(self.kernel, 'mix', side_effect=ValueError('injected native failure')):
            with self.assertRaisesRegex(ValueError, 'injected native failure'):
                compiled.forward([4])
        self.assertEqual(compiled.position, 2)
        for before, after in zip(old, compiled.cache):
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b)
        compiled.max_bytes = 1
        with self.assertRaises(MemoryError):
            compiled.forward([4])
        self.assertEqual(compiled.position, 2)

    def test_compiled_mixer_rejects_nonfinite_and_wrong_layout(self):
        _, decoder, _ = self.fixture()
        prefix = 'model.hyper_connection_mixer.'
        w = decoder.weights
        args = (np.ones((2, 36), np.float32), w[prefix+'hc_norm.weight'],
            w[prefix+'input_mix_weight_down.weight'], w[prefix+'input_mix_weight_up.weight'], None)
        result, gate = self.kernel.mix(*args, heads=3, epsilon=1e-6)
        self.assertEqual(result.shape, (2, 12))
        self.assertIsNone(gate)
        corrupt = args[0].copy()
        corrupt[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.kernel.mix(corrupt, *args[1:], heads=3, epsilon=1e-6)
        with self.assertRaises(ValueError):
            self.kernel.mix(args[0][:, ::-1], *args[1:], heads=3, epsilon=1e-6)

    def test_mixed_qsa_selection_attention_and_cached_oracle(self):
        import torch
        oracle, decoder, compiled = self.fixture(layer_types=['linear_attention', 'qwen_sparse_attention'],
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=6,
            indexer_budget=2, indexer_compress_ratio=2)
        native = NativeQwen4TextDecoder(vars(decoder.config), decoder.weights, library=self.native_path)
        self.native_decoders.append(native)
        ids = [[2, 7, 4, 9, 5, 3, 8]]
        masks = []
        handle = oracle.model.layers[1].self_attn.indexer.register_forward_hook(
            lambda _module, _args, output: masks.append(output.detach().numpy()))
        with torch.no_grad():
            expected = self.oracle_forward(oracle, ids)
        handle.remove()
        for engine in (decoder, compiled, native):
            actual = engine.forward(ids)
            self.compare(actual, expected)
            np.testing.assert_array_equal(actual.selections[0], masks[0] == 0)
            self.assertLess(actual.selections[0][0, 0, -1].sum(), len(ids[0]))
            np.testing.assert_allclose(engine.cache[1][0], expected.past_key_values.layers[1].keys.numpy(), atol=5e-6)
            np.testing.assert_allclose(engine.cache[1][2], expected.past_key_values.layers[1].indexer_keys.numpy(), atol=5e-6)
            engine.reset()
            engine.forward([2, 7, 4])
            tail = engine.forward([9, 5, 3, 8])
            np.testing.assert_allclose(tail.logits, actual.logits[:, 3:], atol=5e-6)

    def test_ple_hash_eos_dilated_convolution_and_cached_oracle(self):
        import torch
        oracle, decoder, compiled = self.fixture(ple_layer_ids=[1],
            ple_embed_dim=8, ple_conv_kernel_size=3, ngram_size=3,
            heads_per_ngram=1, ngram_vocab_size_base=17,
            make_ngram_vocab_size_divisible_by=4, seed=1729, eos_token_id=22)
        native = NativeQwen4TextDecoder(vars(decoder.config), decoder.weights, library=self.native_path)
        self.native_decoders.append(native)
        ids = [[2, 7, 22, 9, 5, 3, 8]]
        with torch.no_grad():
            expected = self.oracle_forward(oracle, ids)
            first = self.oracle_forward(oracle, [[2, 7]])
            second = self.oracle_forward(oracle, [[22, 9, 5, 3, 8]], cache=first.past_key_values, offset=2)
        for engine in (decoder, compiled, native):
            actual = engine.forward(ids)
            self.compare(actual, expected)
            layer = expected.past_key_values.layers[0]
            np.testing.assert_array_equal(engine.ple_cache[0][0], layer.conv_states[2].numpy())
            np.testing.assert_allclose(engine.ple_cache[0][1], layer.conv_states[1].numpy(), atol=5e-6)
            engine.reset()
            engine.forward([2, 7])
            result = engine.forward([22, 9, 5, 3, 8])
            self.compare(result, second)
            np.testing.assert_allclose(result.logits, actual.logits[:, 2:], atol=5e-6)

    def test_complete_compiled_linear_arithmetic_without_numpy_fallback(self):
        import torch
        oracle, independent, _ = self.fixture(norm_topk_prob=False)
        native = NativeQwen4TextDecoder(vars(independent.config), independent.weights, library=self.native_path)
        self.native_decoders.append(native)
        ids = [[2, 7, 4, 9], [3, 5, 8, 6]]
        with torch.no_grad():
            expected = self.oracle_forward(oracle, ids)
        with patch('openagent.adaptive_engine.independent_qwen4.rmsnorm', side_effect=AssertionError('NumPy norm fallback')), \
                patch('openagent.adaptive_engine.independent_qwen4.softmax', side_effect=AssertionError('NumPy router fallback')), \
                patch.object(IndependentQwen4TextDecoder, '_moe', side_effect=AssertionError('NumPy expert fallback')), \
                patch.object(IndependentQwen4TextDecoder, '_delta', side_effect=AssertionError('NumPy delta fallback')), \
                patch.object(IndependentQwen4TextDecoder, '_head', side_effect=AssertionError('NumPy head fallback')):
            actual = native.forward(ids)
            self.compare(actual, expected)
            native.reset()
            native.forward(np.array(ids)[:, :2])
            tail = native.forward(np.array(ids)[:, 2:])
            np.testing.assert_allclose(tail.logits, actual.logits[:, 2:], atol=5e-6)
        for i, state in enumerate(native.cache):
            np.testing.assert_allclose(state[0], expected.past_key_values.layers[i].conv_states[0].numpy(), atol=5e-6)
            np.testing.assert_allclose(state[1], expected.past_key_values.layers[i].recurrent_states[0].numpy(), atol=5e-6)
        self.assertTrue(native.status()['native_range_execution'])
        raw = vars(independent.config).copy()
        raw['layer_types'] = ['linear_attention', 'qwen_sparse_attention']
        with self.assertRaises(ValueError):
            NativeQwen4TextDecoder(raw, independent.weights, library=self.native_path)

    def test_compiled_mixed_qsa_ple_without_numpy_math(self):
        import torch
        oracle, independent, _ = self.fixture(layer_types=['linear_attention', 'qwen_sparse_attention'],
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=6,
            indexer_budget=2, indexer_compress_ratio=2, ple_layer_ids=[1],
            ple_embed_dim=8, ple_conv_kernel_size=3, ngram_size=3,
            heads_per_ngram=1, ngram_vocab_size_base=17,
            make_ngram_vocab_size_divisible_by=4, seed=1729, eos_token_id=22)
        native = NativeQwen4TextDecoder(vars(independent.config), independent.weights, library=self.native_path)
        self.native_decoders.append(native)
        ids = [[2, 7, 22, 9, 5, 3, 8]]
        with torch.no_grad():
            expected = self.oracle_forward(oracle, ids)
        with patch('openagent.adaptive_engine.independent_qwen4.rmsnorm', side_effect=AssertionError('NumPy norm fallback')), \
                patch('openagent.adaptive_engine.independent_qwen4.softmax', side_effect=AssertionError('NumPy softmax fallback')), \
                patch('openagent.adaptive_engine.qwen4_ple.execute', side_effect=AssertionError('NumPy PLE fallback')), \
                patch.object(IndependentQwen4TextDecoder, '_qsa', side_effect=AssertionError('NumPy QSA fallback')), \
                patch.object(IndependentQwen4TextDecoder, '_moe', side_effect=AssertionError('NumPy MoE fallback')), \
                patch.object(IndependentQwen4TextDecoder, '_delta', side_effect=AssertionError('NumPy delta fallback')):
            actual = native.forward(ids)
            self.compare(actual, expected)
            native.reset()
            native.forward([2, 7])
            tail = native.forward([22, 9, 5, 3, 8])
            np.testing.assert_allclose(tail.logits, actual.logits[:, 2:], atol=5e-6)

    def test_sigmoid_delta_gate_against_exact_oracle(self):
        import torch
        oracle, independent, compiled = self.fixture(output_gate_type='sigmoid')
        native = NativeQwen4TextDecoder(vars(independent.config), independent.weights, library=self.native_path)
        self.native_decoders.append(native)
        with torch.no_grad():
            expected = self.oracle_forward(oracle, [[2, 7, 4, 9]])
        for engine in (independent, compiled, native):
            self.compare(engine.forward([2, 7, 4, 9]), expected)

    def test_native_late_failure_preserves_all_committed_state(self):
        _, independent, _ = self.fixture(layer_types=['linear_attention', 'qwen_sparse_attention'],
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=6,
            indexer_budget=2, indexer_compress_ratio=2, ple_layer_ids=[1],
            ple_embed_dim=8, ple_conv_kernel_size=3, ngram_size=3,
            heads_per_ngram=1, ngram_vocab_size_base=17,
            make_ngram_vocab_size_divisible_by=4, seed=1729, eos_token_id=22)
        native = NativeQwen4TextDecoder(vars(independent.config), independent.weights, library=self.native_path)
        self.native_decoders.append(native)
        native.forward([2, 7])
        saved = [[tuple(a.copy() for a in state) if state else None for state in collection]
                 for collection in (native.cache, native.ple_cache)]
        positions, mask = native.positions.copy(), native.mask.copy()
        with patch.object(native, '_head', side_effect=ValueError('late head failure')):
            with self.assertRaisesRegex(ValueError, 'late head failure'):
                native.forward([4, 9])
        self.assertEqual(native.position, 2)
        np.testing.assert_array_equal(native.positions, positions)
        np.testing.assert_array_equal(native.mask, mask)
        for before, after in zip(saved, (native.cache, native.ple_cache)):
            for old, current in zip(before, after):
                if old is None:
                    self.assertIsNone(current)
                else:
                    for a, b in zip(old, current):
                        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(native.forward([4, 9]).logits).all())
