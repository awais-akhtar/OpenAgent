"""Actual compiled exact-family decoder, with independent Transformers oracle."""
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

from openagent.adaptive_engine.native_qwen35 import NativeQwen35TextDecoder
from tests import test_native_mixtral_qwen35 as qwen_oracle


def build(directory):
    source = Path(__file__).resolve().parents[1]/'openagent/adaptive_engine/native_qwen35.c'
    output = Path(directory)/('qwen35.dll' if os.name == 'nt' else 'qwen35.so')
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


class NativeQwen35Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
            raise unittest.SkipTest('optional exact Transformers Qwen3.5 oracle unavailable')
        cls.directory = tempfile.TemporaryDirectory()
        cls.decoders = []
        try:
            cls.path = build(cls.directory.name)
        except Exception:
            cls.directory.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        if os.name == 'nt':
            C.windll.kernel32.FreeLibrary.argtypes = [C.c_void_p]
            for decoder in cls.decoders:
                C.windll.kernel32.FreeLibrary(decoder.library._handle)
        cls.directory.cleanup()

    def fixture(self, seed=271):
        oracle, independent = qwen_oracle.Qwen35NumericalTests().fixture(seed)
        decoder = NativeQwen35TextDecoder(oracle.config.to_dict(), independent.weights, library=self.path)
        self.decoders.append(decoder)
        return oracle, decoder

    def compare(self, actual, expected):
        qwen_oracle.Qwen35NumericalTests().compare(actual, expected)

    def states(self, decoder, expected):
        for i, kind in enumerate(decoder.config.layer_types):
            layer = expected.past_key_values.layers[i]
            if kind == 'linear_attention':
                np.testing.assert_allclose(decoder.cache[i][0], layer.conv_states[0].numpy(), atol=3e-6)
                np.testing.assert_allclose(decoder.cache[i][1], layer.recurrent_states[0].numpy(), atol=3e-6)
            else:
                end = decoder.position
                np.testing.assert_allclose(decoder.cache[i][0][:, :, :end], layer.keys.numpy(), atol=3e-6)
                np.testing.assert_allclose(decoder.cache[i][1][:, :, :end], layer.values.numpy(), atol=3e-6)

    def test_compiled_full_prefill_chunked_padding_and_all_states(self):
        import torch
        oracle, decoder = self.fixture()
        ids = torch.tensor([[0, 0, 1, 4, 7], [1, 3, 8, 2, 6]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        positions = (mask.cumsum(-1)-1).clamp(min=0)
        with torch.inference_mode():
            expected = oracle(ids, attention_mask=mask, position_ids=positions,
                output_hidden_states=True, output_attentions=True)
            actual = decoder.forward(ids.numpy(), attention_mask=mask.numpy(), position_ids=positions.numpy())
            self.compare(actual, expected)
            self.states(decoder, expected)
            decoder.reset()
            past = None
            for start, end in ((0, 3), (3, 4), (4, 5)):
                expected = oracle(ids[:, start:end], attention_mask=mask[:, :end],
                    position_ids=positions[:, start:end], past_key_values=past,
                    output_hidden_states=True, output_attentions=True)
                past = expected.past_key_values
                self.compare(decoder.forward(ids[:, start:end].numpy(), attention_mask=mask[:, :end].numpy(),
                    position_ids=positions[:, start:end].numpy()), expected)
                self.states(decoder, expected)

    def test_router_choices_greedy_and_no_numpy_math_fallback(self):
        import torch
        oracle, decoder = self.fixture(277)
        observed = []
        handles = [layer.mlp.gate.register_forward_hook(lambda module, args, output: observed.append(output))
                   for layer in oracle.model.layers]
        ids = [1, 4, 7]
        try:
            with torch.inference_mode():
                expected = oracle(torch.tensor([ids]), output_hidden_states=True, output_attentions=True)
            with (patch('openagent.adaptive_engine.independent_qwen35.IndependentQwen35TextDecoder.forward', side_effect=AssertionError('NumPy decoder fallback')),
                 patch('numpy.exp', side_effect=AssertionError('NumPy exponent math')),
                 patch('numpy.einsum', side_effect=AssertionError('NumPy attention math'))):
                actual = decoder.forward(ids)
            self.compare(actual, expected)
            for router, (indices, weights), oracle_route in zip(actual.router_logits, actual.routes, observed):
                logits, selected_weights, selected_indices = oracle_route
                np.testing.assert_allclose(router, logits.numpy(), atol=4e-6)
                np.testing.assert_array_equal(indices, selected_indices.numpy())
                np.testing.assert_allclose(weights, selected_weights.numpy(), atol=4e-7)
            generated, prompt = [], ids.copy()
            with torch.inference_mode():
                for _ in range(4):
                    token = int(oracle(torch.tensor([prompt])).logits[0, -1].argmax())
                    if token == 36:
                        break
                    prompt.append(token)
                    generated.append(token)
            self.assertEqual(decoder.generate_ids(ids, max_tokens=4), generated)
            self.assertTrue(decoder.status()['native_range_execution'])
            self.assertEqual(decoder.status()['implementation'], 'c-qwen35-fp32-double-dot')
        finally:
            for handle in handles:
                handle.remove()

    def test_invalid_inputs_native_failure_and_memory_are_transactional(self):
        _, decoder = self.fixture()
        decoder.forward([1, 3])
        before = [(a.copy(), b.copy()) for a, b in decoder.cache]
        for ids in ([1.5], [-1], [37], []):
            with self.assertRaises(ValueError):
                decoder.forward(ids)
        with self.assertRaises(ValueError):
            decoder.forward([4], attention_mask=[[0, 1, 1]])
        original_layer = decoder._layer
        calls = []
        def fail_second(*args):
            calls.append(True)
            return 3 if len(calls) == 2 else original_layer(*args)
        with patch.object(decoder, '_layer', side_effect=fail_second), self.assertRaisesRegex(ValueError, 'execution status 3'):
            decoder.forward([4])
        self.assertEqual(len(calls), 2)
        count = decoder._bindings[0].w[3].count
        decoder._bindings[0].w[3].count -= 1
        try:
            with self.assertRaisesRegex(ValueError, 'execution status 1'):
                decoder.forward([4])
        finally:
            decoder._bindings[0].w[3].count = count
        budget = decoder.max_bytes
        decoder.max_bytes = 1
        with self.assertRaisesRegex(ValueError, 'memory budget'):
            decoder.forward([4])
        decoder.max_bytes = budget
        self.assertEqual(decoder.position, 2)
        for actual, expected in zip(decoder.cache, before):
            for a, e in zip(actual, expected):
                np.testing.assert_array_equal(a, e)
        decoder.reset()
        self.assertEqual(decoder.cache, [None]*3)

    def test_asymmetric_geometry_tied_weights_and_full_attention_schedule(self):
        import torch
        base, _ = self.fixture()
        variants = [dict(linear_num_key_heads=1, linear_num_value_heads=2,
                         linear_key_head_dim=6, linear_value_head_dim=8, head_dim=16,
                         linear_conv_kernel_dim=1, tie_word_embeddings=True),
                    dict(linear_value_head_dim=16),
                    dict(layer_types=['full_attention']*3)]
        with torch.inference_mode():
            for variant in variants:
                with self.subTest(variant=variant):
                    raw = dict(base.config.to_dict(), **variant)
                    config = type(base.config).from_dict(raw)
                    config._attn_implementation = 'eager'
                    oracle = type(base)(config).eval()
                    weights = {k: v.detach().numpy() for k, v in oracle.state_dict().items()}
                    decoder = NativeQwen35TextDecoder(config.to_dict(), weights, library=self.path)
                    self.decoders.append(decoder)
                    ids = torch.tensor([[1, 4, 7, 2]])
                    expected = oracle(ids, output_hidden_states=True, output_attentions=True)
                    full = decoder.forward(ids.numpy())
                    self.compare(full, expected)
                    self.states(decoder, expected)
                    decoder.reset()
                    past, logits = None, []
                    for row in range(ids.shape[1]):
                        expected = oracle(ids[:, row:row+1], past_key_values=past,
                            output_hidden_states=True, output_attentions=True)
                        past = expected.past_key_values
                        actual = decoder.forward(ids[:, row:row+1].numpy())
                        self.compare(actual, expected)
                        self.states(decoder, expected)
                        logits.append(actual.logits)
                    np.testing.assert_allclose(np.concatenate(logits, axis=1), full.logits, atol=4e-6, rtol=5e-5)

    def test_exact_identity_and_corrupt_weight_rejection(self):
        oracle, decoder = self.fixture()
        for model_type in ('qwen3_next', 'qwen4_exp_text', 'qwen3_5_moe', 'glm5_next_text'):
            with self.subTest(model_type=model_type), self.assertRaisesRegex(ValueError, 'expected exact'):
                NativeQwen35TextDecoder(dict(oracle.config.to_dict(), model_type=model_type),
                                        decoder.weights, library=self.path)
        for kind in ('missing', 'shape', 'nan'):
            weights = dict(decoder.weights)
            key = 'model.layers.0.mlp.experts.gate_up_proj'
            if kind == 'missing':
                del weights[key]
            elif kind == 'shape':
                weights[key] = np.zeros((4, 24, 23), np.float32)
            else:
                weights[key] = np.full_like(weights[key], np.nan)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                NativeQwen35TextDecoder(oracle.config.to_dict(), weights, library=self.path)

    def test_budget_admission_precedes_default_mask_and_position_allocations(self):
        _, decoder = self.fixture()
        decoder.forward([1, 4])
        mask = decoder.mask.copy()
        before = [[value.copy() for value in state] for state in decoder.cache]
        decoder.max_bytes = 1
        with patch('openagent.adaptive_engine.native_qwen35.np.ones',
                   side_effect=AssertionError('mask allocated before admission')), patch(
                       'openagent.adaptive_engine.native_qwen35.np.arange',
                       side_effect=AssertionError('positions allocated before admission')):
            with self.assertRaisesRegex(ValueError, 'memory budget'):
                decoder.forward(np.array([7], dtype=np.int64))
        self.assertEqual(decoder.position, 2)
        np.testing.assert_array_equal(decoder.mask, mask)
        for actual, expected in zip(decoder.cache, before):
            for a, e in zip(actual, expected):
                np.testing.assert_array_equal(a, e)

    def test_nested_mask_uses_uint8_without_full_int64_intermediate(self):
        _, decoder = self.fixture()
        original = np.asarray

        def forbid_mask_expansion(value, *args, **kwargs):
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
                raise AssertionError('full nested mask expanded before bounded conversion')
            return original(value, *args, **kwargs)

        with patch('openagent.adaptive_engine.native_qwen35.np.asarray',
                   side_effect=forbid_mask_expansion):
            decoder.forward(np.array([[1, 4]], np.int64), attention_mask=[[1, 1]])
        self.assertEqual(decoder.mask.dtype, np.dtype('uint8'))
        decoder.reset()
        for mask in ([[1, .5]], [[1, float('nan')]], [[1, 1j]], [['1', 1]]):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, 'attention_mask'):
                decoder.forward(np.array([[1, 4]], np.int64), attention_mask=mask)
        self.assertEqual(decoder.position, 0)

    def test_half_weights_admission_accounts_for_owned_fp32_conversion(self):
        oracle, decoder = self.fixture()
        half_weights = {name: value.astype(np.float16) for name, value in decoder.weights.items()}
        raw_bytes = sum(value.nbytes for value in half_weights.values())
        with patch.object(qwen_oracle.IndependentQwen35TextDecoder, '__init__',
                          side_effect=AssertionError('weight copies allocated before admission')):
            with self.assertRaisesRegex(ValueError, 'memory budget'):
                NativeQwen35TextDecoder(oracle.config.to_dict(), half_weights,
                                        library=self.path, max_bytes=raw_bytes*2)

    def test_admission_accounts_for_previous_mask_and_commit_copy(self):
        _, decoder = self.fixture()
        decoder.forward([1, 4])
        captured = []
        original = decoder._check_bytes

        def record(amount):
            captured.append(amount)
            original(amount)

        with patch.object(decoder, '_check_bytes', side_effect=record):
            decoder.forward([7])
        self.assertEqual(len(captured), 1)
        required = captured[0]
        decoder.reset()
        decoder.forward([1, 4])
        saved_mask = decoder.mask
        probe = []

        def reject_after_record(amount):
            probe.append(amount)
            raise ValueError('admission probe')

        with patch.object(decoder, '_check_bytes', side_effect=reject_after_record):
            with self.assertRaisesRegex(ValueError, 'admission probe'):
                decoder.forward([7])
            decoder.mask = None
            try:
                with self.assertRaisesRegex(ValueError, 'admission probe'):
                    decoder.forward([7])
            finally:
                decoder.mask = saved_mask
        self.assertEqual(probe[0] - probe[1], saved_mask.nbytes)
        decoder.max_bytes = required - 1
        with self.assertRaisesRegex(ValueError, 'memory budget'):
            decoder.forward([7])
        self.assertEqual(decoder.position, 2)
        decoder.max_bytes = required
        decoder.forward([7])
        self.assertEqual(decoder.position, 3)


if __name__ == '__main__':
    unittest.main()
