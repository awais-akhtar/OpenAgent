"""Compiled dense K3 fixture tests; unchanged vendor run is explicit/networked."""
import ctypes as C
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.native_kimi_k3_decoder import (
    KimiK3DenseMLAConfig, NativeKimiK3DenseMLA, tensor_shapes,
)
from openagent.adaptive_engine.independent_kimi_k3_decoder import evaluate_dense_mla


def compile_library(directory):
    root = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine'
    sources = [root/'native_kimi_k3_decoder.c', root/'native_kimi_k3_attnres.c']
    output = Path(directory)/('kimi_k3_decoder.dll' if os.name == 'nt' else 'kimi_k3_decoder.so')
    if os.name == 'nt':
        vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
        location = subprocess.check_output([str(vswhere), '-latest', '-products', '*',
            '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
            '-property', 'installationPath'], text=True).strip()
        if not location:
            raise RuntimeError('MSVC required; missing compiler is not a passing skip')
        batch = Path(location)/'VC/Auxiliary/Build/vcvars64.bat'
        args = ['cl', '/nologo', '/W4', '/WX', '/O2', '/fp:precise', '/LD',
                *map(str, sources), '/link', '/OUT:'+str(output)]
        command = 'cmd.exe /d /s /c ""'+str(batch)+'" && '+subprocess.list2cmdline(args)+'"'
    else:
        cc = shutil.which('cc')
        if not cc:
            raise RuntimeError('C compiler required')
        command = [cc, '-std=c99', '-Wall', '-Wextra', '-Werror', '-O2', '-shared', '-fPIC',
                   *map(str, sources), '-o', str(output), '-lm']
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return output


def fixture(c):
    rng = np.random.default_rng(713)
    return {name: ((1+rng.normal(0, .12, shape)) if 'norm.weight' in name else
                   rng.normal(0, .24, shape)).astype(np.float32)
            for name, shape in tensor_shapes(c).items()}


class KimiK3DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='kimi_k3_decoder_', ignore_cleanup_errors=True)
        cls.library = compile_library(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.c = KimiK3DenseMLAConfig()
        self.w = fixture(self.c)
        self.ids = [1, 4, 2, 9, 3, 8, 5]
        self.model = NativeKimiK3DenseMLA(self.library, self.c, self.w)

    def assert_state(self, actual, expected):
        self.assertEqual(actual.identity, expected.identity)
        self.assertEqual(actual.length, expected.length)
        np.testing.assert_array_equal(actual.keys, expected.keys)
        np.testing.assert_array_equal(actual.values, expected.values)

    def test_full_and_numpy_caches(self):
        expected, k, v = evaluate_dense_mla(self.c, self.w, self.ids)
        np.testing.assert_allclose(self.model.forward(self.ids), expected, atol=3e-6, rtol=3e-6)
        state = self.model.snapshot()
        np.testing.assert_allclose(state.keys[:, :len(self.ids)], k, atol=3e-6, rtol=3e-6)
        np.testing.assert_allclose(state.values[:, :len(self.ids)], v, atol=3e-6, rtol=3e-6)

    def test_cached_chunks_and_tokenwise(self):
        full = self.model.forward(self.ids)
        state = self.model.snapshot()
        for chunks in ([1]*7, [2, 1, 4], [4, 3]):
            self.model.reset()
            outputs, start = [], 0
            for count in chunks:
                outputs.append(self.model.forward(self.ids[start:start+count]))
                start += count
            np.testing.assert_array_equal(np.concatenate(outputs), full)
            self.assert_state(self.model.snapshot(), state)

    def test_rollback_restore_reset_and_detachment(self):
        self.model.forward(self.ids[:2])
        saved = self.model.snapshot()
        expected = self.model.forward(self.ids[2:])
        self.model.restore(saved)
        with self.assertRaisesRegex(RuntimeError, r'\(3\)'):
            self.model.forward(self.ids[2:], _fail_after_token=2)
        self.assert_state(self.model.snapshot(), saved)
        np.testing.assert_array_equal(self.model.forward(self.ids[2:]), expected)
        self.model.restore(saved)
        saved.keys.fill(99)
        self.assertFalse(np.any(self.model.snapshot().keys == 99))
        self.model.reset()
        self.assertEqual(self.model.snapshot().length, 0)
        self.assertFalse(self.model.snapshot().keys.any())

    def test_invalid_tokens_capacity_and_snapshot_rollback(self):
        self.model.forward(self.ids[:2])
        before = self.model.snapshot()
        for invalid in ([], [True], [-1], [self.c.vocab], [1.2], [[1]], [2**64-1]):
            with self.assertRaises(ValueError):
                self.model.forward(invalid)
            self.assert_state(self.model.snapshot(), before)
        with self.assertRaisesRegex(RuntimeError, r'\(4\)'):
            self.model.forward([1]*self.c.capacity)
        self.assert_state(self.model.snapshot(), before)
        for bad in (replace(before, length=-1), replace(before, identity='foreign'),
                    replace(before, keys=np.full_like(before.keys, np.nan)),
                    replace(before, values=np.ones_like(before.values))):
            with self.assertRaises(ValueError):
                self.model.restore(bad)
            self.assert_state(self.model.snapshot(), before)

    def test_fail_closed_geometry_and_tensor_options(self):
        for name, value in (('hidden', 65), ('layers', 9), ('rope', -1), ('qrank', 0),
                            ('block', True), ('epsilon', np.nan), ('beta', 0)):
            with self.assertRaises(ValueError):
                replace(self.c, **{name: value})
        with self.assertRaises(TypeError):
            KimiK3DenseMLAConfig(linear_attn_config={})
        with self.assertRaises(TypeError):
            self.model.forward([1], attention_mask=[1])
        for bad in ({**self.w, 'extra.weight': np.ones(1, np.float32)},
                    {k: v for k, v in self.w.items() if k != 'lm_head.weight'},
                    {**self.w, 'lm_head.weight': self.w['lm_head.weight'].astype(np.float64)}):
            with self.assertRaises(ValueError):
                NativeKimiK3DenseMLA(self.library, self.c, bad)

    def test_native_direct_rejection_preserves_output_and_state(self):
        m = self.model
        m.forward([1])
        before = m.snapshot()
        ids = np.array([2, 3], np.int32)
        output = np.full((2, self.c.vocab), 123, np.float32)
        def ptr(a):
            return a.ctypes.data_as(C.POINTER(C.c_float))
        for nw, fault in ((m._weights.size-1, -1), (m._weights.size, 1)):
            status = m._function(C.byref(m._c), ptr(m._weights), nw,
                ids.ctypes.data_as(C.POINTER(C.c_int)), 2, ptr(m._keys), ptr(m._values),
                C.byref(m._length), ptr(output), fault)
            self.assertNotEqual(status, 0)
            np.testing.assert_array_equal(output, 123)
            self.assert_state(m.snapshot(), before)

    def test_block_boundaries_and_nonrotated_coordinates(self):
        for block, layers, rope in ((1, 4, 0), (2, 5, 2), (8, 3, 1)):
            c = replace(self.c, block=block, layers=layers, rope=rope)
            w = fixture(c)
            model = NativeKimiK3DenseMLA(self.library, c, w)
            expected, _, _ = evaluate_dense_mla(c, w, self.ids)
            np.testing.assert_allclose(model.forward(self.ids), expected, atol=3e-6, rtol=3e-6)


def run_vendor_suite(vendor):
    """Called inside the hash-checked oracle probe's disposable child process."""
    import torch
    def verify():
        c = KimiK3DenseMLAConfig(layers=5, block=2)
        config = vendor.KimiLinearConfig(vocab_size=c.vocab, hidden_size=c.hidden,
            intermediate_size=c.intermediate, num_hidden_layers=c.layers,
            num_attention_heads=c.heads, num_key_value_heads=c.heads,
            q_lora_rank=c.qrank, kv_lora_rank=c.kvrank, qk_nope_head_dim=c.nope,
            qk_rope_head_dim=c.rope, v_head_dim=c.value, mla_use_nope=True,
            mla_use_output_gate=True, hidden_act='situ', activation_situ_beta=c.beta,
            activation_situ_linear_beta=c.linear_beta, attn_res_block_size=c.block,
            rms_norm_eps=c.epsilon, linear_attn_config=None, num_experts=None,
            pad_token_id=None, attn_implementation='eager')
        oracle = vendor.KimiLinearForCausalLM(config).float().eval()
        # Supported backend selection, not source rewriting or kernel substitution.
        oracle.set_attn_implementation('eager')
        w = fixture(c)
        oracle.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in w.items()}, strict=True)
        ids = [1, 4, 2, 9, 3, 8, 5]
        with torch.no_grad():
            full = oracle(torch.tensor([ids]), use_cache=True)
            expected = full.logits[0].numpy()
            numpy_logits, nk, nv = evaluate_dense_mla(c, w, ids)
            np.testing.assert_allclose(numpy_logits, expected, atol=2e-5, rtol=2e-5)
            with tempfile.TemporaryDirectory(prefix='kimi_k3_vendor_build_', ignore_cleanup_errors=True) as directory:
                model = NativeKimiK3DenseMLA(compile_library(directory), c, w)
                actual = model.forward(ids)
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                state = model.snapshot()
                for i in range(c.layers):
                    vk = full.past_key_values.key_cache[i][0].permute(1, 0, 2).numpy()
                    vv = full.past_key_values.value_cache[i][0].permute(1, 0, 2).numpy()
                    np.testing.assert_allclose(state.keys[i, :len(ids)], vk, atol=2e-5, rtol=2e-5)
                    np.testing.assert_allclose(state.values[i, :len(ids)], vv, atol=2e-5, rtol=2e-5)
                    np.testing.assert_allclose(nk[i], vk, atol=2e-5, rtol=2e-5)
                    np.testing.assert_allclose(nv[i], vv, atol=2e-5, rtol=2e-5)
                for sizes in ([1]*7, [2, 1, 4]):
                    cache, offset, results, native_results = None, 0, [], []
                    model.reset()
                    for size in sizes:
                        chunk = ids[offset:offset+size]
                        result = oracle(torch.tensor([chunk]), past_key_values=cache, use_cache=True)
                        cache = result.past_key_values
                        results.append(result.logits[0].numpy())
                        native_results.append(model.forward(chunk))
                        offset += size
                    np.testing.assert_allclose(np.concatenate(results), expected, atol=2e-5, rtol=2e-5)
                    np.testing.assert_allclose(np.concatenate(native_results), np.concatenate(results), atol=2e-5, rtol=2e-5)
                    for i in range(c.layers):
                        np.testing.assert_allclose(model.snapshot().keys[i, :len(ids)],
                            cache.key_cache[i][0].permute(1, 0, 2).numpy(), atol=2e-5, rtol=2e-5)
                        np.testing.assert_allclose(model.snapshot().values[i, :len(ids)],
                            cache.value_cache[i][0].permute(1, 0, 2).numpy(), atol=2e-5, rtol=2e-5)
                print('vendor full/cached logits + all-layer expanded K/V verified; max logit abs error=',
                      float(np.max(np.abs(actual-expected))), flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(KimiK3DecoderTests)
    suite.addTest(unittest.FunctionTestCase(verify, description='Unchanged pinned vendor dense K3 MLA oracle'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    unittest.main()
