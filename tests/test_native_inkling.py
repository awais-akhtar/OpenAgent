"""Independent dense Inkling C/math/cache qualification against Transformers 5.16.1."""
import ctypes as C
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.independent_inkling import IndependentInklingDecoder, ALL_LAYER_NAMES, GLOBAL_NAMES
from openagent.adaptive_engine.native_inkling import NativeInklingDecoder, Layer, Tensor, tensor


def optional(reason):
    if os.environ.get('INKLING_CERTIFY') == '1':
        raise RuntimeError('strict Inkling certification: ' + reason)
    raise unittest.SkipTest(reason)


def build(directory):
    source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/native_inkling.c'
    output = Path(directory) / ('inkling.dll' if os.name == 'nt' else 'inkling.so')
    if os.name == 'nt':
        finder = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
        if not finder.exists():
            optional('MSVC unavailable')
        location = subprocess.check_output([str(finder), '-latest', '-products', '*', '-requires',
            'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
        if not location:
            optional('MSVC unavailable')
        setup = Path(location) / 'VC/Auxiliary/Build/vcvars64.bat'
        compile_args = ['cl', '/nologo', '/W4', '/WX', '/O2', '/fp:strict', '/LD', str(source),
                        '/link', '/OUT:' + str(output)]
        command = 'cmd.exe /d /s /c ""' + str(setup) + '" && ' + subprocess.list2cmdline(compile_args) + '"'
    else:
        cc = shutil.which('cc')
        if cc is None:
            optional('C compiler unavailable')
        command = [cc, '-std=c99', '-Wall', '-Wextra', '-Werror', '-O2', '-ffp-contract=off',
                   '-shared', '-fPIC', str(source), '-o', str(output), '-lm']
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return output


class NativeInklingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import transformers
        except ImportError:
            optional('optional oracle unavailable')
        if transformers.__version__ != '5.16.1':
            optional('qualification requires exact Transformers 5.16.1')
        cls.torch = torch
        cls.transformers = transformers
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        try:
            cls.path = build(cls.directory.name)
            cls.lib = C.CDLL(str(cls.path))
        except Exception:
            cls.directory.cleanup()
            torch.set_num_threads(cls.old_threads)
            raise

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)
        if os.name == 'nt':
            C.windll.kernel32.FreeLibrary.argtypes = [C.c_void_p]
            C.windll.kernel32.FreeLibrary(cls.lib._handle)
        cls.directory.cleanup()

    def fixture(self, **changes):
        torch = self.torch
        args = dict(vocab_size=23, hidden_size=12, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=3,
                    swa_num_attention_heads=2, swa_num_key_value_heads=1, swa_head_dim=4,
                    d_rel=3, rel_extent=3, sliding_window_size=3, conv_kernel_size=4,
                    intermediate_size=17, moe_intermediate_size=7,
                    n_routed_experts=3, num_experts_per_tok=2, n_shared_experts=1,
                    layer_types=['hybrid', 'hybrid'], mlp_layer_types=['dense', 'dense'],
                    max_position_embeddings=32, logits_mup_width_multiplier=2.5)
        args.update(changes)
        config = self.transformers.InklingTextConfig(**args)
        config._attn_implementation = 'eager'
        config._experts_implementation = 'eager'
        torch.manual_seed(461)
        model = self.transformers.InklingForCausalLM(config).eval()
        # Amplify architecture-sensitive paths beyond the tiny default initializer.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith('global_scale'):
                    parameter.fill_(1.3)
                elif 'norm.weight' in name:
                    parameter.uniform_(0.7, 1.4)
                else:
                    parameter.normal_(0, 0.22)
        weights = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
        baseline = IndependentInklingDecoder(config, weights)
        native = NativeInklingDecoder(config, weights, library=self.lib)
        return model, baseline, native

    def compare(self, actual, expected):
        np.testing.assert_allclose(actual.logits, expected.logits.numpy(), atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(actual.last_hidden_state, expected.hidden_states[-1].numpy(), atol=5e-6, rtol=2e-5)

    def compare_cache(self, native, expected):
        for i, (g, state) in enumerate(zip(native.geometry, native.cache_snapshot())):
            layer = expected.past_key_values.layers[i]
            for b in range(state.shape[0]):
                keys, values, *histories = native._views(state[b], g)
                length = layer.keys.shape[2]
                end = native.position
                for actual, reference in ((keys, layer.keys), (values, layer.values)):
                    np.testing.assert_allclose(actual[end - length:end].transpose(1, 0, 2),
                        reference[b].numpy(), atol=5e-6, rtol=2e-5)
                for j, history in enumerate(histories):
                    np.testing.assert_allclose(history.T, layer.conv_states[j][b].numpy(), atol=5e-6, rtol=2e-5)

    def exercise(self, **changes):
        model, baseline, native = self.fixture(**changes)
        torch = self.torch
        ids = torch.tensor([[1, 2, 5, 4, 7, 8, 11, 9, 6], [3, 5, 7, 9, 2, 4, 6, 8, 10]])
        observations = {}
        for i, layer in enumerate(model.model.layers):
            if model.config.mlp_layer_types[i] == 'sparse':
                layer.mlp.gate.register_forward_hook(
                    lambda module, args, output, index=i: observations.__setitem__(index, output))
        def routing(actual):
            for i, output in observations.items():
                weights, indices, shared = [v.numpy() for v in output[1:]]
                expected_weights = np.zeros_like(actual.routing_weights[:, :, i]).reshape(-1,
                    model.config.n_routed_experts + model.config.n_shared_experts)
                indices = indices.reshape(-1, model.config.num_experts_per_tok)
                weights = weights.reshape(indices.shape)
                np.put_along_axis(expected_weights[:, :model.config.n_routed_experts], indices, weights, axis=1)
                expected_weights[:, model.config.n_routed_experts:] = shared.reshape(-1, model.config.n_shared_experts)
                np.testing.assert_allclose(actual.routing_weights[:, :, i].reshape(expected_weights.shape),
                                           expected_weights, atol=3e-6, rtol=2e-5)
        with torch.inference_mode():
            expected = model(ids, use_cache=True, output_hidden_states=True)
            for result in (baseline.forward(ids.numpy()), native.forward(ids.numpy())):
                self.compare(result, expected)
                routing(result)
            self.compare_cache(native, expected)
            for chunks in ((1,) * 9, (2, 1, 4, 2), (5, 4)):
                native.reset()
                baseline.reset()
                past, offset, parts = None, 0, []
                for length in chunks:
                    piece = ids[:, offset:offset + length]
                    out = model(piece, past_key_values=past, use_cache=True, output_hidden_states=True)
                    past = out.past_key_values
                    actual = native.forward(piece.numpy())
                    self.compare(actual, out)
                    routing(actual)
                    independent = baseline.forward(piece.numpy())
                    self.compare(independent, out)
                    routing(independent)
                    self.compare_cache(native, out)
                    parts.append(actual.logits)
                    offset += length
                np.testing.assert_allclose(np.concatenate(parts, axis=1), expected.logits.numpy(), atol=3e-6, rtol=2e-5)

    def test_dense_full_and_all_four_histories(self):
        self.exercise()

    def test_cache_free_oracle_preserves_session_and_can_change_batch(self):
        for schedule in (['hybrid','hybrid'],['hybrid_sliding','hybrid']):
            model,baseline,native = self.fixture(layer_types=schedule,mlp_layer_types=['dense','sparse'],
                log_scaling_n_floor=2,log_scaling_alpha=.7,max_position_embeddings=12)
            for decoder in (baseline,native):
                for initialized in (False,True):
                    decoder.reset()
                    if initialized:
                        decoder.forward([[1,2,3,4,5,6,7,8,9,10]])
                    position,before = decoder.position,decoder.cache_snapshot()
                    ids = self.torch.tensor([[1,2,3],[3,4,5]])
                    for selection in (0,1,np.array([2,0],np.int64)):
                        keep = self.torch.from_numpy(selection) if isinstance(selection,np.ndarray) else selection
                        with self.torch.inference_mode():
                            expected = model(ids,use_cache=False,logits_to_keep=keep,output_hidden_states=True)
                        actual = decoder.forward(ids.numpy(),use_cache=False,logits_to_keep=selection)
                        self.compare(actual,expected)
                        self.unchanged(decoder,position,before)
                    decoder.forward([[11]])
                    self.assertEqual(decoder.position,position+1)

    def test_cache_free_native_failure_budget_and_boolean_are_transactional(self):
        model,_,native = self.fixture(mlp_layer_types=['sparse','dense'])
        native.forward([[1,2]])
        position,before = native.position,native.cache_snapshot()
        for bad in (None,1,'false',np.bool_(False)):
            with self.assertRaises(ValueError):
                native.forward([[3]],use_cache=bad)
            self.unchanged(native,position,before)
        execute = native._execute
        def fail(*args):
            self.assertEqual(args[-1],0)
            self.assertEqual(execute(*args),0)
            return 3
        with patch.object(native,'_execute',side_effect=fail):
            with self.assertRaises(RuntimeError):
                native.forward([[3,4]],use_cache=False)
        self.unchanged(native,position,before)
        limit = native.max_bytes
        native.max_bytes = 1
        with patch.object(native,'_execute',side_effect=AssertionError('execution before budget')):
            with self.assertRaises(MemoryError):
                native.forward([[3]],use_cache=False)
        native.max_bytes = limit
        self.unchanged(native,position,before)
        with self.torch.inference_mode():
            expected = model(self.torch.tensor([[3,4]]),use_cache=False,output_hidden_states=True)
        with patch.object(IndependentInklingDecoder,'forward',side_effect=AssertionError('fallback')), \
             patch('openagent.adaptive_engine.independent_inkling.rms',side_effect=AssertionError('numpy math')):
            self.compare(native.forward([[3,4]],use_cache=False),expected)
        self.unchanged(native,position,before)

    def test_native_normalization_overflow_rolls_back_late_token(self):
        model,baseline,_ = self.fixture()
        weights = dict(baseline.weights)
        weights[GLOBAL_NAMES[0]] = weights[GLOBAL_NAMES[0]].copy()
        weights[GLOBAL_NAMES[0]][2].fill(np.finfo(np.float32).max)
        native = NativeInklingDecoder(model.config,weights,library=self.lib)
        native.forward([[3]])
        before = native.cache_snapshot()
        for use_cache in (True,False):
            with self.assertRaisesRegex(RuntimeError,'not committed'):
                native.forward([[1,2]],use_cache=use_cache)
            self.unchanged(native,1,before)

    def test_logits_selection_oracle_cached_and_prefill(self):
        model, baseline, native = self.fixture(mlp_layer_types=['dense', 'sparse'])
        torch = self.torch
        for selection in (0, 1, 2, 99, np.array([2, 0, 2], np.int64), np.array([], np.int64)):
            with self.subTest(selection=selection):
                native.reset()
                baseline.reset()
                past = None
                for tokens in ([[1, 2, 3]], [[4, 5, 6]]):
                    oracle_selection = torch.from_numpy(selection) if isinstance(selection, np.ndarray) else selection
                    with torch.inference_mode():
                        expected = model(torch.tensor(tokens), past_key_values=past,
                                         logits_to_keep=oracle_selection, output_hidden_states=True, use_cache=True)
                    past = expected.past_key_values
                    self.compare(native.forward(tokens, logits_to_keep=selection), expected)
                    self.compare(baseline.forward(tokens, logits_to_keep=selection), expected)
                    self.compare_cache(native, expected)
                self.assertEqual(native.position, 6)
        before = native.cache_snapshot()
        for bad in (-1, True, [3], [-1], [0.5], [[0]], [0, 0, 0, 0], np.array([2**64-1], np.uint64)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                native.forward([[7, 8, 9]], logits_to_keep=bad)
            self.unchanged(native, 6, before)

    def test_dense_sliding_mixed_head_geometry_and_windows(self):
        self.exercise(layer_types=['hybrid_sliding', 'hybrid'])
        self.exercise(layer_types=['hybrid_sliding', 'hybrid_sliding'], sliding_window_size=2)

    def test_window_one_explicitly_unsupported_with_oracle_discrepancy(self):
        # Retain the qualification failure as evidence, not a relaxed tolerance.
        _, baseline, _ = self.fixture()
        raw = dict(baseline.config, layer_types=['hybrid_sliding', 'hybrid_sliding'], sliding_window_size=1)
        with self.assertRaisesRegex(ValueError, 'window=1'):
            IndependentInklingDecoder(raw, baseline.weights)
        config = self.transformers.InklingTextConfig(vocab_size=23, hidden_size=12, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
            swa_num_attention_heads=2, swa_num_key_value_heads=1, swa_head_dim=4,
            d_rel=3, rel_extent=3, sliding_window_size=1, intermediate_size=17,
            layer_types=['hybrid_sliding'], mlp_layer_types=['dense'])
        config._attn_implementation = 'eager'
        self.torch.manual_seed(461)
        model = self.transformers.InklingForCausalLM(config).eval()
        ids = self.torch.tensor([[1, 2, 3]])
        with self.torch.inference_mode():
            full = model(ids, use_cache=False).logits
            past, parts = None, []
            for i in range(3):
                result = model(ids[:, i:i + 1], past_key_values=past, use_cache=True)
                past = result.past_key_values
                parts.append(result.logits)
        self.assertGreater(float((full - self.torch.cat(parts, dim=1)).abs().max()), 1e-5)

    def test_sparse_full_and_mixed_sliding_shared_experts(self):
        self.exercise(mlp_layer_types=['sparse', 'sparse'], n_shared_experts=2)
        self.exercise(mlp_layer_types=['dense', 'sparse'], layer_types=['hybrid', 'hybrid_sliding'],
                      num_experts_per_tok=1)
        self.exercise(mlp_layer_types=['sparse', 'dense'], layer_types=['hybrid_sliding', 'hybrid'],
                      num_experts_per_tok=3, log_scaling_n_floor=3)

    def test_log_scaling_unpadded_head_and_unit_kernel(self):
        self.exercise(log_scaling_n_floor=2, log_scaling_alpha=0.7,
                      unpadded_vocab_size=19, conv_kernel_size=1)

    def test_native_does_not_call_numpy_or_oracle_math(self):
        for mlps in (['dense', 'dense'], ['sparse', 'sparse']):
            model, _, native = self.fixture(mlp_layer_types=mlps)
            with self.torch.inference_mode():
                expected = model(self.torch.tensor([[1, 2, 3]]), output_hidden_states=True)
            with patch.object(IndependentInklingDecoder, 'forward', side_effect=AssertionError('fallback')), \
                 patch.object(IndependentInklingDecoder, '_moe', side_effect=AssertionError('numpy moe')), \
                 patch('openagent.adaptive_engine.independent_inkling.rms', side_effect=AssertionError('numpy math')), \
                 patch.object(model, 'forward', side_effect=AssertionError('oracle delegation')):
                self.compare(native.forward([[1, 2, 3]]), expected)

    def unchanged(self, decoder, position, before):
        self.assertEqual(decoder.position, position)
        for old, now in zip(before, decoder.cache_snapshot()):
            np.testing.assert_array_equal(old, now)

    def test_transaction_validation_capacity_and_backend_failure(self):
        model, _, native = self.fixture(max_position_embeddings=6)
        native.forward([[1, 2]])
        before, position = native.cache_snapshot(), native.position
        for bad in ([[1, -1]], [[23]], [[1.0]], [], [[True]], [[1], [2]],
                    np.array([[2**64 - 1]], dtype=np.uint64), [[1, 2, 3, 4, 5]]):
            with self.assertRaises(ValueError):
                native.forward(bad)
            self.unchanged(native, position, before)
        for kwargs in ({'attention_mask': [[1]]}, {'position_ids': [[2]]}):
            with self.assertRaises(ValueError):
                native.forward([[3]], **kwargs)
            self.unchanged(native, position, before)
        execute = native._execute
        def fail_after_computation(*args):
            self.assertEqual(execute(*args), 0)
            return 3
        with patch.object(native, '_execute', side_effect=fail_after_computation):
            with self.assertRaisesRegex(RuntimeError, 'not committed'):
                native.forward([[3, 4]])
        self.unchanged(native, position, before)
        with patch.object(native, '_execute', side_effect=MemoryError('allocation')):
            with self.assertRaises(MemoryError):
                native.forward([[3]])
        self.unchanged(native, position, before)
        with self.torch.inference_mode():
            expected = model(self.torch.tensor([[1, 2, 3, 4]]), output_hidden_states=True)
        np.testing.assert_allclose(native.forward([[3, 4]]).logits, expected.logits.numpy()[:, 2:], atol=3e-6, rtol=2e-5)

    def test_budget_is_checked_before_state_allocation(self):
        model, _, native = self.fixture()
        native.forward([[1]])
        before = native.cache_snapshot()
        native.max_bytes = sum(a.nbytes for a in native.weights.values()) + 1
        with self.assertRaises(MemoryError):
            native.forward([[2]])
        self.unchanged(native, 1, before)

    def test_exact_identity_config_and_weight_rejection(self):
        model, baseline, native = self.fixture()
        raw = model.config.to_dict()
        overrides = [dict(model_type=t) for t in ('inkling', 'inkling_mm_model', 'inkling_audio', 'llama')]
        overrides += [dict(mlp_layer_types=['dense', 'unknown']), dict(hidden_act='gelu'),
                      dict(tie_word_embeddings=True), dict(num_mtp_layers=1),
                      dict(quantization_config={}), dict(rope_theta=10000),
                      dict(attention_dropout=0.1), dict(num_attention_heads=3),
                      dict(log_scaling_n_floor=0), dict(rms_norm_eps=float('nan')),
                      dict(unpadded_vocab_size=24), dict(layer_types=['full_attention'] * 2),
                      dict(local_layer_ids=[0]), dict(shared_expert_sink=False),
                      dict(rms_norm_eps_moe_gate=1e-5), dict(logits_scale=2),
                      dict(dtype='bfloat16'), dict(n_shared_experts=0), dict(num_experts_per_tok=4)]
        overrides += [dict(architectures=['LlamaForCausalLM']), dict(logit_scale=2),
                      dict(logits_mup_width_multiplier=0), dict(route_scale=float('inf')),
                      dict(use_cache=False), dict(return_dict=False), dict(output_attentions=True),
                      dict(output_hidden_states=True)]
        for change in overrides:
            with self.subTest(change=change), self.assertRaises(ValueError):
                NativeInklingDecoder(dict(raw, **change), baseline.weights, library=self.path)
        weights = dict(baseline.weights)
        for alteration in ('missing', 'extra', 'shape', 'dtype', 'nonfinite'):
            bad = dict(weights)
            key = GLOBAL_NAMES[0]
            if alteration == 'missing':
                del bad[key]
            elif alteration == 'extra':
                bad['alias.embed.weight'] = weights[key]
            elif alteration == 'shape':
                bad[key] = np.zeros((1, 1), np.float32)
            elif alteration == 'dtype':
                bad[key] = weights[key].astype(np.float64)
            else:
                bad[key] = weights[key].copy()
                bad[key][0, 0] = np.nan
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                IndependentInklingDecoder(raw, bad)
        with self.assertRaises(ValueError):
            native.weights[GLOBAL_NAMES[0]].setflags(write=True)

    def test_owned_inputs_reset_and_state_isolation(self):
        model, baseline, native = self.fixture()
        raw, weights = model.config.to_dict(), {k: v.copy() for k, v in baseline.weights.items()}
        other = IndependentInklingDecoder(raw, weights)
        raw['layer_types'][0] = 'bad'
        weights[GLOBAL_NAMES[0]][:] = 0
        result = native.forward([[1, 2, 3]])
        np.testing.assert_allclose(result.logits, other.forward([[1, 2, 3]]).logits, atol=3e-6)
        snapshot = native.cache_snapshot()
        snapshot[0][:] = 0
        self.assertFalse(np.array_equal(snapshot[0], native.cache_snapshot()[0]))
        native.reset()
        self.assertEqual(native.cache_snapshot(), ())
        np.testing.assert_array_equal(native.forward([[1, 2, 3]]).logits, result.logits)

    def test_c_abi_rejects_bad_count_before_mutating_staging(self):
        _, _, native = self.fixture()
        ids, pending = native._prepare([[1, 2]], None, None)
        layers = (Layer * len(native.geometry))()
        for i, g in enumerate(native.geometry):
            weights = (Tensor * 26)(*[tensor(native.weights[f'model.layers.{i}.{name}'])
                if f'model.layers.{i}.{name}' in native.weights else Tensor() for name in ALL_LAYER_NAMES])
            layers[i] = Layer(g.heads, g.kv_heads, g.dim, g.extent, g.window, 0, weights, tensor(pending[i]))
        globals_ = (Tensor * 4)(*[tensor(native.weights[n]) for n in GLOBAL_NAMES])
        hidden = np.full((1, 2, 12), -123, np.float32)
        logits = np.full((1, 2, 23), -123, np.float32)
        routing = np.full((1, 2, 2, 4), -123, np.float32)
        layers[1].weights[2].count -= 1
        code = self.lib.oa_inkling_run(C.byref(native._c), layers, globals_,
            ids.ctypes.data_as(C.POINTER(C.c_int64)), ids.size, 1, 2, 0, tensor(hidden), tensor(logits), tensor(routing))
        self.assertEqual(code, 1)
        self.assertTrue(all(np.all(a == 0) for a in pending))
        self.assertTrue(np.all(hidden == -123))
        self.assertTrue(np.all(logits == -123))

    def test_sparse_ambiguous_cutoff_rejected_transactionally(self):
        model, baseline, _ = self.fixture(mlp_layer_types=['dense', 'sparse'])
        weights = dict(baseline.weights)
        weights['model.layers.1.mlp.gate.weight'] = np.zeros_like(weights['model.layers.1.mlp.gate.weight'])
        weights['model.layers.1.mlp.gate.e_score_correction_bias'] = np.zeros(3, np.float32)
        for decoder in (IndependentInklingDecoder(model.config, weights),
                        NativeInklingDecoder(model.config, weights, library=self.lib)):
            with self.assertRaises((ValueError, RuntimeError)):
                decoder.forward([[1, 2]])
            self.assertEqual(decoder.position, 0)
            self.assertEqual(decoder.cache_snapshot(), ())

    def test_nonfinite_c_computation_does_not_commit(self):
        model, baseline, _ = self.fixture()
        weights = dict(baseline.weights)
        weights['model.layers.1.mlp.global_scale'] = np.array([3e38], np.float32)
        weights['model.layers.1.mlp.down_proj.weight'] = np.full_like(weights['model.layers.1.mlp.down_proj.weight'], 3e38)
        native = NativeInklingDecoder(model.config, weights, library=self.lib)
        with self.assertRaisesRegex(RuntimeError, 'not committed'):
            native.forward([[1, 2]])
        self.assertEqual(native.position, 0)
        self.assertEqual(native.cache_snapshot(), ())

    def test_concurrent_calls_are_serialized_without_global_c_state(self):
        _, baseline, native = self.fixture(mlp_layer_types=['sparse', 'dense'],
                                          layer_types=['hybrid_sliding', 'hybrid'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(lambda _: native.forward([[3]]).logits, range(8)))
        expected = baseline.forward([[3] * 8])
        actual = np.concatenate(outputs, axis=1)[0]
        reference = expected.logits[0]
        actual = actual[np.argsort(actual[:, 0])]
        reference = reference[np.argsort(reference[:, 0])]
        np.testing.assert_allclose(actual, reference, atol=3e-6, rtol=2e-5)
        self.assertEqual(native.position, 8)
        for a, b in zip(native.cache_snapshot(), baseline.cache_snapshot()):
            np.testing.assert_allclose(a, b, atol=5e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
