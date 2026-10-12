"""Compiled independent GLM-DSA versus optional exact Transformers oracle."""
import ctypes as C
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from openagent.adaptive_engine.native_glm_dsa import NativeGlmDsaDecoder, tensor
from openagent.adaptive_engine.independent_glm_dsa import IndependentGlmDsaDecoder
from tests import test_independent_glm_dsa as oracle_tests


def build(directory):
    source = Path(__file__).resolve().parents[1]/'openagent/adaptive_engine/native_glm_dsa.c'
    output = Path(directory)/('glm.dll' if os.name == 'nt' else 'glm.so')
    if os.name == 'nt':
        vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
        if not vswhere.exists():
            oracle_tests.optional('MSVC unavailable')
        location = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
            'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
        if not location:
            oracle_tests.optional('MSVC unavailable')
        batch = Path(location)/'VC/Auxiliary/Build/vcvars64.bat'
        args = ['cl', '/nologo', '/W4', '/WX', '/O2', '/fp:precise', '/LD', str(source), '/link', '/OUT:'+str(output)]
        command = 'cmd.exe /d /s /c ""'+str(batch)+'" && '+subprocess.list2cmdline(args)+'"'
    else:
        compiler = shutil.which('cc')
        if compiler is None:
            oracle_tests.optional('C compiler unavailable')
        command = [compiler, '-std=c99', '-Wall', '-Wextra', '-Werror', '-O2', '-shared', '-fPIC', str(source), '-o', str(output), '-lm']
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return output


class NativeGlmDsaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
            oracle_tests.optional('optional exact Transformers/torch oracle unavailable')
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

    def fixture(self, **overrides):
        model, baseline = oracle_tests.fixture(self, **overrides)
        decoder = NativeGlmDsaDecoder(model.config.to_dict(), baseline.weights, library=self.path)
        self.decoders.append(decoder)
        return model, baseline, decoder

    def states(self, decoder, expected):
        c = decoder.config
        for i, (latent, rotary, index) in enumerate(decoder.cache):
            latent, rotary = latent[:, :decoder.position], rotary[:, :decoder.position]
            projection = decoder.weights[f'model.layers.{i}.self_attn.kv_b_proj.weight']
            kv = (latent @ projection.T).reshape(latent.shape[0], latent.shape[1], c.num_attention_heads, -1)
            key = np.concatenate((kv[..., :c.qk_nope_head_dim], np.broadcast_to(rotary[:, :, None],
                (*kv.shape[:-1], c.qk_rope_head_dim))), axis=-1).transpose(0, 2, 1, 3)
            layer = expected.past_key_values.layers[i]
            np.testing.assert_allclose(key, layer.keys.numpy(), atol=3e-6)
            np.testing.assert_allclose(kv[..., c.qk_nope_head_dim:].transpose(0, 2, 1, 3), layer.values.numpy(), atol=3e-6)
            if index is not None:
                np.testing.assert_allclose(index[:, :decoder.position], layer.indexer_keys.numpy(), atol=3e-6)

    def test_yarn_native_full_chunks_positions_and_cache_without_fallback(self):
        import torch
        variants = ({}, {'mscale':1.2,'mscale_all_dim':.7,'truncate':False,
                          'beta_fast':16.,'beta_slow':2.},
                    {'attention_factor':1.8,'mscale_all_dim':4.},
                    {'factor':1.,'original_max_position_embeddings':256,
                     'beta_fast':256/(2*np.pi),'beta_slow':256/(2*np.pi)})
        for options in variants:
            with self.subTest(options=options):
                rope = dict(rope_type='yarn',rope_theta=10000.,factor=4.,
                    original_max_position_embeddings=64) | options
                model,baseline,decoder = self.fixture(qk_rope_head_dim=16,index_head_dim=24,
                    max_position_embeddings=256,rope_parameters=rope)
                ids = torch.tensor([[1,3,5,7,9,2,4,6],[1,4,6,8,2,3,5,7]])
                positions = torch.tensor([list(range(60,68)),list(range(190,198))])
                observed = []
                handles = [layer.self_attn.register_forward_hook(
                    lambda module,args,output: observed.append(output[2])) for layer in model.model.layers]
                try:
                    with torch.inference_mode():
                        full = model(ids,position_ids=positions,output_hidden_states=True,output_attentions=True)
                        with patch.object(IndependentGlmDsaDecoder,'forward',side_effect=AssertionError('baseline fallback')), \
                             patch('openagent.adaptive_engine.independent_glm_dsa.interleaved_rotary',side_effect=AssertionError('rotary fallback')), \
                             patch('openagent.adaptive_engine.independent_glm_dsa.rotary_parameters',side_effect=AssertionError('frequency fallback')):
                            actual = decoder.forward(ids.numpy(),position_ids=positions.numpy())
                        oracle_tests.compare(actual,full)
                        oracle_tests.GlmDsaTests().selections(actual,observed,np.ones(ids.shape,bool),0)
                        self.states(decoder,full)
                        full_logits = actual.logits.copy()
                        saved = tuple(tuple(a.copy() if a is not None else None for a in s) for s in decoder.cache)
                        decoder.reset()
                        past = None
                        chunks = []
                        for start,end in ((0,3),(3,4),(4,8)):
                            observed.clear()
                            expected = model(ids[:,start:end],position_ids=positions[:,start:end],past_key_values=past,
                                output_hidden_states=True,output_attentions=True)
                            past = expected.past_key_values
                            actual = decoder.forward(ids[:,start:end].numpy(),position_ids=positions[:,start:end].numpy())
                            oracle_tests.compare(actual,expected)
                            oracle_tests.GlmDsaTests().selections(actual,observed,np.ones((2,end),bool),start)
                            self.states(decoder,expected)
                            chunks.append(actual.logits)
                        np.testing.assert_array_equal(np.concatenate(chunks,axis=1),full_logits)
                        for state,old in zip(decoder.cache,saved):
                            for a,b in zip(state,old):
                                if a is not None:
                                    np.testing.assert_array_equal(a,b)
                        if rope['factor'] > 1:
                            plain = IndependentGlmDsaDecoder(dict(model.config.to_dict(),
                                rope_parameters={'rope_type':'default','rope_theta':10000.}),baseline.weights)
                            plain.forward(ids.numpy(),position_ids=positions.numpy())
                            self.assertGreater(float(np.max(np.abs(saved[0][1][:,:8]-plain.cache[0][1]))),1e-3)
                finally:
                    for handle in handles:
                        handle.remove()

    def test_yarn_native_geometry_and_late_failure_rollback(self):
        _,_,decoder = self.fixture(rope_parameters=dict(rope_type='yarn',rope_theta=10000.,
            factor=4.,original_max_position_embeddings=8,mscale_all_dim=1.))
        decoder.forward([1,3])
        old = decoder.cache
        saved = tuple(tuple(a.copy() if a is not None else None for a in s) for s in old)
        mask,position = decoder.mask.copy(),decoder.position
        for field,value in (('factor',float('nan')),('factor',.5),('beta_slow',0.),
                            ('truncate',2),('original_context',0),('attention_factor',-1.),
                            ('beta_slow',1e-320),('attention_factor',1e-320)):
            original = getattr(decoder._c,field)
            setattr(decoder._c,field,value)
            try:
                self.assertEqual(decoder.library.oa_glm_dsa_geometry(C.byref(decoder._c)),1)
                with self.assertRaises(RuntimeError):
                    decoder.forward([4])
            finally:
                setattr(decoder._c,field,original)
        with patch.object(decoder.library,'oa_glm_dsa_head',return_value=3):
            with self.assertRaises(RuntimeError):
                decoder.forward([4,5])
        self.assertIs(decoder.cache,old)
        self.assertEqual(decoder.position,position)
        np.testing.assert_array_equal(decoder.mask,mask)
        for state,expected in zip(decoder.cache,saved):
            for a,b in zip(state,expected):
                if a is not None:
                    np.testing.assert_array_equal(a,b)
        decoder.forward([4,5])

    def test_yarn_padding_explicit_amplitude_and_stale_abi(self):
        import torch
        model,baseline,decoder = self.fixture(rope_parameters=dict(rope_type='yarn',
            rope_theta=10000.,factor=4.,original_max_position_embeddings=8,attention_factor=1.7))
        ids = torch.tensor([[0,0,1,3,7],[1,4,8,5,2]])
        mask = torch.tensor([[0,0,1,1,1],[1,1,1,1,1]])
        positions = (mask.cumsum(-1)-1).clamp(min=0)
        with torch.inference_mode():
            expected = model(ids,attention_mask=mask,position_ids=positions,
                output_hidden_states=True,output_attentions=True)
            actual = decoder.forward(ids.numpy(),attention_mask=mask.numpy(),position_ids=positions.numpy())
            oracle_tests.compare(actual,expected)
            self.states(decoder,expected)
        stale = MagicMock()
        stale.oa_glm_dsa_version.return_value = 1
        with patch('openagent.adaptive_engine.native_glm_dsa.C.CDLL',return_value=stale):
            with self.assertRaisesRegex(ValueError,'ABI'):
                NativeGlmDsaDecoder(model.config.to_dict(),baseline.weights,library=self.path)
        stale.oa_glm_dsa_layer.assert_not_called()
        for overrides in ({'factor':None},{'partial_rotary_factor':.5},{'beta_fast':True}):
            config = model.config.to_dict()
            config['rope_parameters'] = config['rope_parameters'] | overrides
            with self.assertRaises(ValueError):
                NativeGlmDsaDecoder(config,baseline.weights,library=self.path)

    def test_full_and_chunked_logits_sparse_selection_and_compressed_cache(self):
        model, _, decoder = self.fixture()
        import torch
        ids = torch.tensor([[1, 3, 5, 7, 9], [1, 4, 6, 8, 2]])
        observed = []
        handles = [layer.self_attn.register_forward_hook(lambda module, args, output: observed.append(output[2]))
            for layer in model.model.layers]
        try:
            with torch.inference_mode(), patch.object(IndependentGlmDsaDecoder, 'forward', side_effect=AssertionError('NumPy fallback')):
                expected = model(ids, output_hidden_states=True, output_attentions=True)
                actual = decoder.forward(ids.numpy())
                oracle_tests.compare(actual, expected)
                oracle_tests.GlmDsaTests().selections(actual, observed, np.ones(ids.shape, bool), 0)
                self.states(decoder, expected)
                decoder.reset()
                past = None
                for start, end in ((0, 3), (3, 4), (4, 5)):
                    observed.clear()
                    expected = model(ids[:, start:end], past_key_values=past, output_hidden_states=True, output_attentions=True)
                    past = expected.past_key_values
                    actual = decoder.forward(ids[:, start:end].numpy())
                    oracle_tests.compare(actual, expected)
                    oracle_tests.GlmDsaTests().selections(actual, observed, np.ones((2, end), bool), start)
                    self.states(decoder, expected)
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(decoder.status()['native_range_execution'])
        self.assertFalse(decoder.status()['native_streamed_range_execution'])

    def test_padding_router_multigroup_variants_and_greedy(self):
        import torch
        for normalize in (True, False):
            model, baseline, decoder = self.fixture(seed=623, topk_group=2,
                norm_topk_prob=normalize, routed_scaling_factor=1.7, n_shared_experts=2)
            ids = torch.tensor([[0, 0, 1, 3, 7], [1, 4, 8, 5, 2]])
            mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
            positions = (mask.cumsum(-1)-1).clamp(min=0)
            with torch.inference_mode():
                expected = model(ids, attention_mask=mask, position_ids=positions, output_hidden_states=True, output_attentions=True)
                actual = decoder.forward(ids.numpy(), attention_mask=mask.numpy(), position_ids=positions.numpy())
                oracle_tests.compare(actual, expected)
                independent = baseline.forward(ids.numpy(), attention_mask=mask.numpy(), position_ids=positions.numpy())
                for a, e in zip(actual.routes, independent.routes):
                    if a is not None:
                        np.testing.assert_array_equal(a[0], e[0])
                        np.testing.assert_allclose(a[1], e[1], atol=4e-7)
                self.states(decoder, expected)
                prompt, expected_ids = [1, 3, 7], []
                for _ in range(4):
                    token = int(model(torch.tensor([prompt])).logits[0, -1].argmax())
                    if token == 36:
                        break
                    prompt.append(token)
                    expected_ids.append(token)
                self.assertEqual(decoder.generate_ids([1, 3, 7], max_tokens=4), expected_ids)

    def test_budget_and_late_failure_are_transactional(self):
        _, _, decoder = self.fixture()
        decoder.forward([1, 3])
        old = decoder.cache
        mask, position = decoder.mask.copy(), decoder.position
        saved = tuple(tuple(a.copy() if a is not None else None for a in state) for state in old)
        with patch.object(decoder, '_check_bytes', side_effect=MemoryError('bounded')), \
                patch.object(decoder, '_inputs', side_effect=AssertionError('allocated before budget')):
            with self.assertRaises(MemoryError):
                decoder.forward([4])
        limit = decoder.max_bytes
        decoder.max_bytes = 1
        with self.assertRaises(MemoryError):
            decoder.forward([4])
        decoder.max_bytes = limit
        with patch.object(decoder.library, 'oa_glm_dsa_head', return_value=3):
            with self.assertRaises(RuntimeError):
                decoder.forward([4])
        self.assertIs(decoder.cache, old)
        self.assertEqual(decoder.position, position)
        np.testing.assert_array_equal(decoder.mask, mask)
        for state, expected in zip(decoder.cache, saved):
            for a, e in zip(state, expected):
                if a is not None:
                    np.testing.assert_array_equal(a, e)

    def test_private_counted_head_rejects_bad_geometry(self):
        _, _, decoder = self.fixture()
        x = np.ones((1, decoder.config.hidden_size), np.float32)
        out = np.empty_like(x)
        logits = np.empty((1, decoder.config.vocab_size), np.float32)
        bad = tensor(x)
        bad.count -= 1
        self.assertEqual(decoder.library.oa_glm_dsa_head(bad, tensor(decoder.weights['model.norm.weight']),
            tensor(decoder.weights['lm_head.weight']), decoder.config.hidden_size,
            decoder.config.vocab_size, decoder.config.rms_norm_eps, tensor(out), tensor(logits)), 1)


if __name__ == '__main__':
    unittest.main()
