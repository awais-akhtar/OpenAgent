"""Executable original KDA kernel checks; no external oracle dependency."""
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.independent_kimi import (
    IndependentKimiLinearDecoder, kda_equations, projected_kda_equations)
from openagent.adaptive_engine.native_kimi import NativeKimiKDA, NativeKimiLinearDecoder


def build(directory):
    source = Path(__file__).resolve().parents[1]/'openagent/adaptive_engine/native_kimi.c'
    output = Path(directory)/('kimi.dll' if os.name == 'nt' else 'kimi.so')
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
        cc = shutil.which('cc')
        if not cc:
            raise unittest.SkipTest('C compiler unavailable')
        command = [cc, '-std=c99', '-Wall', '-Wextra', '-Werror', '-O2', '-shared', '-fPIC', str(source), '-o', str(output), '-lm']
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return output


class NativeKimiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = build(cls.temp.name)
        cls.instances = []

    @classmethod
    def tearDownClass(cls):
        if os.name == 'nt':
            C.windll.kernel32.FreeLibrary.argtypes = [C.c_void_p]
            for model in cls.instances:
                C.windll.kernel32.FreeLibrary(model.library._handle)
        cls.temp.cleanup()

    def fixture(self, seed=914, kernel=4, n=13, **kwargs):
        rng = np.random.default_rng(seed)
        def random(shape):
            return rng.normal(0, .5, shape).astype(np.float32)
        taps = (random((2,3,kernel)), random((2,3,kernel)), random((2,5,kernel)))
        norm = random((5,))
        inputs = (random((n,2,3)), random((n,2,3)), random((n,2,5)),
                  -np.abs(random((n,2,3))), rng.random((n,2)).astype(np.float32),
                  rng.random((n,2,5)).astype(np.float32))
        model = NativeKimiKDA(taps, norm, library=self.path, capacity=64,
                             epsilon=1e-6, q_scale=3**-.5, **kwargs)
        self.instances.append(model)
        return model, inputs, taps, norm

    def assert_cache_equal(self, a, b):
        self.assertEqual(a[0], b[0])
        for x,y in zip((a[1],*a[2]), (b[1],*b[2])):
            np.testing.assert_array_equal(x,y)

    def test_equation_order_hand_computed(self):
        q = np.array([[[1., 0.]]])
        k = np.array([[[0., 1.]]])
        out, state = kda_equations(q,k,np.array([[[7.]]]),
            np.log(np.array([[[.5,.25]]])),np.array([[1.]]),np.array([[[2.],[4.]]]))
        np.testing.assert_allclose(state, [[[1.],[7.]]])
        np.testing.assert_allclose(out, [[[1.]]])

    def test_native_against_dense_equations_and_all_caches(self):
        for kernel in (1,4,8):
            model, inputs, taps, norm = self.fixture(kernel=kernel)
            before = model.snapshot()
            expected, state, history = projected_kda_equations(*inputs[:3], taps,
                *inputs[3:], norm, before[1], before[2], epsilon=model.epsilon, q_scale=model.q_scale)
            actual = model.forward(*inputs)
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
            after = model.snapshot()
            np.testing.assert_allclose(after[1], state, atol=2e-6, rtol=2e-5)
            for a,b in zip(after[2],history):
                np.testing.assert_array_equal(a,b)

    def test_chunked_and_single_token_bitwise(self):
        model, inputs, _, _ = self.fixture()
        expected = model.forward(*inputs)
        cache = model.snapshot()
        for splits in ((1,)*13,(3,2,8)):
            model.reset()
            start = 0
            rows = []
            for size in splits:
                rows.append(model.forward(*(a[start:start+size] for a in inputs)))
                start += size
            np.testing.assert_array_equal(np.concatenate(rows), expected)
            self.assert_cache_equal(cache, model.snapshot())

    def test_invalid_inputs_do_not_commit(self):
        model, inputs, _, _ = self.fixture()
        model.forward(*(a[:2] for a in inputs))
        before = model.snapshot()
        cases = []
        for index, bad in ((0,np.nan),(3,.1),(4,1.1),(5,-.1)):
            values = [a.copy() for a in inputs]
            values[index].flat[-1] = bad
            cases.append(values)
        values = list(inputs); values[0] = values[0].astype(np.float64); cases.append(values)
        values = list(inputs); values[1] = values[1][:,:,:2]; cases.append(values)
        values = [a[:0] for a in inputs]; cases.append(values)
        for values in cases:
            with self.assertRaises(ValueError):
                model.forward(*values)
            self.assert_cache_equal(before, model.snapshot())

    def test_capacity_and_memory_admission(self):
        model, inputs, _, _ = self.fixture()
        before = model.snapshot()
        model.capacity = 1
        with self.assertRaisesRegex(ValueError,'capacity'):
            model.forward(*inputs)
        self.assert_cache_equal(before, model.snapshot())
        model.capacity = 64
        model.max_bytes = model._resident*2
        with self.assertRaisesRegex(ValueError,'memory budget'):
            model.forward(*inputs)
        self.assert_cache_equal(before, model.snapshot())
        with self.assertRaisesRegex(ValueError,'memory budget'):
            self.fixture(max_bytes=1)

    def test_native_failure_after_staging_is_transactional(self):
        model, inputs, _, _ = self.fixture()
        before = model.snapshot()
        real = model._run
        def fail(*args):
            self.assertEqual(real(*args),0)
            return 5
        with patch.object(model,'_run',side_effect=fail):
            with self.assertRaisesRegex(ValueError,'cache unchanged'):
                model.forward(*inputs)
        self.assert_cache_equal(before,model.snapshot())
        model.forward(*inputs)
        self.assertEqual(model.position,13)

    def test_snapshot_and_weights_are_detached(self):
        model, inputs, taps, norm = self.fixture()
        expected = model.forward(*inputs)
        snapshot = model.snapshot()
        snapshot[1].fill(999)
        for a in snapshot[2]:
            a.fill(999)
        for a in (*taps,norm):
            a.fill(999)
        model.reset()
        np.testing.assert_array_equal(expected,model.forward(*inputs))

    def test_zero_stream_and_extreme_decay(self):
        model, inputs, _, _ = self.fixture()
        values = [a.copy() for a in inputs]
        for a in values[:3]:
            a.fill(0)
        values[3].fill(-1e30)
        out = model.forward(*values)
        np.testing.assert_array_equal(out,0)
        np.testing.assert_array_equal(model.snapshot()[1],0)

    def test_raw_abi_rejects_bad_counts_before_mutation(self):
        model, inputs, _, _ = self.fixture()
        _, state, history = model.snapshot()
        out = np.full((13,2,5),77,np.float32)
        buffers = (*inputs,model._norm,*model._taps,state,*history,out)
        counts = (C.c_uint64*15)(*(a.size for a in buffers))
        counts[0] -= 1
        status = model._run(13,2,3,5,4,1e-6,1.,1e-6,
            *(a.ctypes.data_as(C.POINTER(C.c_float)) for a in buffers),counts)
        self.assertEqual(status,2)
        np.testing.assert_array_equal(state,0)
        np.testing.assert_array_equal(out,77)

    def decoder_fixture(self, *, native=True, seed=1701, **overrides):
        config = dict(model_type='kimi_linear',architectures=['KimiLinearForCausalLM'],
            hidden_size=12,vocab_size=23,num_hidden_layers=3,first_k_dense_replace=1,
            intermediate_size=17,num_attention_heads=2,num_key_value_heads=2,
            q_lora_rank=None,kv_lora_rank=5,qk_nope_head_dim=3,qk_rope_head_dim=2,v_head_dim=4,
            num_experts=4,num_experts_per_token=2,moe_intermediate_size=7,num_shared_experts=1,
            hidden_act='silu',moe_router_activation_func='sigmoid',num_expert_group=1,
            topk_group=1,moe_renormalize=True,moe_layer_freq=1,routed_scaling_factor=1.3,
            use_grouped_topk=True,mla_use_nope=True,rms_norm_eps=1e-5,
            max_position_embeddings=32,tie_word_embeddings=False,
            linear_attn_config=dict(num_heads=2,head_dim=3,short_conv_kernel_size=4,
                                    kda_layers=[1,3],full_attn_layers=[2]))
        config.update(overrides)
        rng = np.random.default_rng(seed)
        shapes = IndependentKimiLinearDecoder.tensor_shapes(config)
        tensors = {name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()}
        for name,a in tensors.items():
            if 'norm.weight' in name:
                a[:] = rng.uniform(.6,1.4,a.shape)
        if native:
            model = NativeKimiLinearDecoder(config,tensors,library=self.path,capacity=32)
            self.instances.append(model)
        else:
            model = IndependentKimiLinearDecoder(config,tensors,capacity=32)
        return model,config,tensors

    def assert_decoder_cache(self,a,b,*,close=False):
        self.assertEqual(a[0],b[0])
        for left,right in zip(a[1],b[1]):
            for x,y in zip(left,right):
                if close:
                    np.testing.assert_allclose(x,y,atol=3e-6,rtol=3e-5)
                else:
                    np.testing.assert_array_equal(x,y)

    def test_complete_decoder_logits_and_all_states(self):
        model,config,tensors = self.decoder_fixture()
        equations = IndependentKimiLinearDecoder(config,tensors,capacity=32)
        tokens = [3,7,2,8,1,9]
        actual = model.forward(tokens)
        expected = equations.forward(tokens)
        self.assertEqual(actual.shape,(6,23))
        self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=3e-5)
        np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))
        self.assert_decoder_cache(model.snapshot(),equations.snapshot(),close=True)

    def test_decoder_prefill_incremental_and_replay(self):
        model,_,_ = self.decoder_fixture()
        tokens = [3,7,2,8,1,9]
        logits = model.forward(tokens)
        cache = model.snapshot()
        model.reset()
        rows = [model.forward(tokens[:2]),model.forward(tokens[2:5]),model.forward(tokens[5:])]
        np.testing.assert_array_equal(logits,np.concatenate(rows))
        self.assert_decoder_cache(cache,model.snapshot())
        model.reset()
        np.testing.assert_array_equal(logits,model.forward(tokens))

    def test_decoder_late_failure_rolls_back_every_layer(self):
        model,_,_ = self.decoder_fixture()
        model.forward([4,5])
        before = model.snapshot()
        real = model._mat
        def fail(w,x):
            if w is model.weights['lm_head.weight']:
                raise ValueError('forced late head failure')
            return real(w,x)
        with patch.object(model,'_mat',side_effect=fail):
            with self.assertRaisesRegex(ValueError,'late head'):
                model.forward([6,7])
        self.assert_decoder_cache(before,model.snapshot())
        model.forward([6,7])
        self.assertEqual(model.position,4)

    def test_decoder_rejects_k3_and_inexact_variants(self):
        _,config,tensors = self.decoder_fixture()
        variants = [dict(model_type='kimi_k3'),dict(hidden_act='situ'),
                    dict(attn_res_block_size=2),dict(latent_moe_use_norm=True),
                    dict(mla_use_output_gate=True),dict(q_lora_rank=5),
                    dict(num_expert_group=2),dict(mla_use_nope=False)]
        for change in variants:
            with self.assertRaises(ValueError):
                NativeKimiLinearDecoder(dict(config,**change),tensors,library=self.path,capacity=32)
        for malformed in ({**tensors,'extra':np.ones(1,np.float32)},
                          {k:v for k,v in tensors.items() if k!='model.norm.weight'}):
            with self.assertRaisesRegex(ValueError,'tensor names'):
                NativeKimiLinearDecoder(config,malformed,library=self.path,capacity=32)

    def test_decoder_invalid_ids_budget_and_capacity_rollback(self):
        model,_,_ = self.decoder_fixture()
        model.forward([2])
        before = model.snapshot()
        for ids in ([],[-1],[23],[True],[1.5],[[1]],list(range(23))*2):
            with self.assertRaises(ValueError):
                model.forward(ids)
            self.assert_decoder_cache(before,model.snapshot())
        model.max_bytes = 1
        with self.assertRaisesRegex(ValueError,'memory budget'):
            model.forward([3])
        self.assert_decoder_cache(before,model.snapshot())

    def test_mla_shared_channels_contribute_without_rope(self):
        model,_,_ = self.decoder_fixture()
        q = np.array([[0,0,0,1,2],[0,0,0,2,1]],np.float32)
        k = np.array([[[0,0,0,1,0],[0,0,0,1,0]],
                      [[0,0,0,0,1],[0,0,0,0,1]]],np.float32)
        v = np.array([[[1,0,0,0],[1,0,0,0]],[[0,1,0,0],[0,1,0,0]]],np.float32)
        out = model._attention(q,k,v)
        self.assertGreater(out[0,1],out[0,0])
        self.assertGreater(out[1,0],out[1,1])
        expected = IndependentKimiLinearDecoder._attention(model,q,k,v)
        np.testing.assert_allclose(out,expected,atol=1e-7)

    def test_router_correction_gather_and_scaling_contract(self):
        model,_,_ = self.decoder_fixture()
        scores = np.array([.2,.4,.6,.8],np.float32)
        bias = np.array([.9,0,0,0],np.float32)
        ids,weights = model._route(scores,bias)
        np.testing.assert_array_equal(ids,[0,3])
        np.testing.assert_allclose(weights,np.array([1.1,.8])/1.9*1.3,atol=1e-7)
        np.testing.assert_array_equal(scores,[np.float32(.2),np.float32(.4),np.float32(.6),np.float32(.8)])

    def test_native_decoder_has_no_numpy_arithmetic_fallback(self):
        model,_,_ = self.decoder_fixture()
        with patch('openagent.adaptive_engine.independent_kimi.projected_kda_equations',
                   side_effect=AssertionError('NumPy recurrence must not execute')):
            logits = model.forward([2,3])
        self.assertTrue(np.isfinite(logits).all())

    def test_local_checkpoint_load_f32_f16_bf16_and_shards(self):
        model,config,tensors = self.decoder_fixture()
        for dtype in ('F32','F16','BF16'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
                converted = {}
                groups = [list(tensors)[::2],list(tensors)[1::2]]
                weight_map = {}
                for shard,names in enumerate(groups):
                    filename = f'model-{shard}.safetensors'
                    header,payload = {},bytearray()
                    for name in names:
                        array = tensors[name]
                        if dtype=='BF16':
                            packed = (array.view(np.uint32)>>16).astype('<u2')
                            converted[name] = (packed.astype(np.uint32)<<16).view(np.float32)
                        else:
                            packed = array.astype('<f4' if dtype=='F32' else '<f2')
                            converted[name] = packed.astype(np.float32)
                        data = packed.tobytes()
                        header[name] = dict(dtype=dtype,shape=list(array.shape),data_offsets=[len(payload),len(payload)+len(data)])
                        payload.extend(data)
                        weight_map[name] = filename
                    raw = json.dumps(header).encode()
                    (root/filename).write_bytes(len(raw).to_bytes(8,'little')+raw+payload)
                (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=weight_map)),encoding='utf-8')
                loaded = NativeKimiLinearDecoder.from_checkpoint(root,library=self.path,capacity=32)
                self.instances.append(loaded)
                direct = NativeKimiLinearDecoder(config,converted,library=self.path,capacity=32)
                self.instances.append(direct)
                np.testing.assert_array_equal(loaded.forward([1,3,4]),direct.forward([1,3,4]))
                with self.assertRaisesRegex(ValueError,'memory budget'):
                    NativeKimiLinearDecoder.from_checkpoint(root,library=self.path,capacity=32,max_bytes=1)
                weight_map[list(tensors)[0]] = '../escape.safetensors'
                (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=weight_map)),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'local basename'):
                    NativeKimiLinearDecoder.from_checkpoint(root,library=self.path,capacity=32)

    def test_native_arithmetic_overflow_does_not_commit(self):
        _,config,tensors = self.decoder_fixture()
        for name,a in tensors.items():
            a.fill(1 if 'norm.weight' in name or name=='model.embed_tokens.weight' else 0)
        tensors['lm_head.weight'].fill(np.finfo(np.float32).max)
        model = NativeKimiLinearDecoder(config,tensors,library=self.path,capacity=32)
        self.instances.append(model)
        before = model.snapshot()
        with self.assertRaisesRegex(ValueError,'native decoder status 5'):
            model.forward([3])
        self.assert_decoder_cache(before,model.snapshot())

    def test_more_seeds_full_decoder_and_tied_embeddings(self):
        for seed in (17,43,71):
            model,config,tensors = self.decoder_fixture(seed=seed)
            equations = IndependentKimiLinearDecoder(config,tensors,capacity=32)
            ids = [1,7,8,3,4,5,6,2,9,10,11,12]
            np.testing.assert_allclose(model.forward(ids),equations.forward(ids),atol=5e-6,rtol=4e-5)
            self.assert_decoder_cache(model.snapshot(),equations.snapshot(),close=True)
        config['tie_word_embeddings'] = True
        tensors.pop('lm_head.weight')
        tied = NativeKimiLinearDecoder(config,tensors,library=self.path,capacity=32)
        self.instances.append(tied)
        self.assertTrue(np.isfinite(tied.forward([3])).all())


if __name__ == '__main__':
    unittest.main()
