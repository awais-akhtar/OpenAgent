"""Original KDA checks and an optional checksum-pinned official Linear oracle."""
import ctypes as C
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
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
        values = list(inputs)
        values[0] = values[0].astype(np.float64)
        cases.append(values)
        values = list(inputs)
        values[1] = values[1][:,:,:2]
        cases.append(values)
        values = [a[:0] for a in inputs]
        cases.append(values)
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
                    dict(num_expert_group=2),dict(mla_use_nope=False),
                    dict(use_cache=False), dict(attention_dropout=.1),
                    dict(dtype='bfloat16'), dict(torch_dtype='float16'),
                    dict(rms_norm_eps=1e-100), dict(routed_scaling_factor=1e100)]
        for change in variants:
            with self.assertRaises(ValueError):
                NativeKimiLinearDecoder(dict(config,**change),tensors,library=self.path,capacity=32)
        for malformed in ({**tensors,'extra':np.ones(1,np.float32)},
                          {k:v for k,v in tensors.items() if k!='model.norm.weight'}):
            with self.assertRaisesRegex(ValueError,'tensor names'):
                NativeKimiLinearDecoder(config,malformed,library=self.path,capacity=32)

    def test_config_bounds_and_budget_reject_before_layer_expansion(self):
        _,config,tensors = self.decoder_fixture()
        cases = (
            (dict(config,num_hidden_layers=10**12),1,'geometry'),
            (dict(config,num_hidden_layers=10**12),256*1024*1024,'geometry'),
            (config,1,'memory budget'),
            (config,0,'memory budget'),
        )
        for raw,budget,message in cases:
            with self.subTest(layers=raw['num_hidden_layers'],budget=budget):
                with patch('openagent.adaptive_engine.independent_kimi.range',
                           side_effect=AssertionError('layer expansion reached'),create=True) as expand:
                    with patch.object(IndependentKimiLinearDecoder,'tensor_shapes',
                                      side_effect=AssertionError('tensor inventory reached')) as inventory:
                        with self.assertRaisesRegex(ValueError,message):
                            IndependentKimiLinearDecoder.validate_config(raw,capacity=32,max_bytes=budget)
                        with self.assertRaisesRegex(ValueError,message):
                            NativeKimiLinearDecoder(raw,tensors,library=self.path,capacity=32,max_bytes=budget)
                        expand.assert_not_called()
                        inventory.assert_not_called()

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
        expected = model.forward([2,3,4,5])
        before = model.snapshot()
        model.reset()
        with ExitStack() as stack:
            stack.enter_context(patch('openagent.adaptive_engine.independent_kimi.projected_kda_equations',
                                      side_effect=AssertionError('NumPy recurrence must not execute')))
            for name in ('_mat', '_vec', '_attention', '_route', '_kda'):
                stack.enter_context(patch.object(IndependentKimiLinearDecoder, name,
                                                side_effect=AssertionError('NumPy arithmetic fallback')))
            logits = np.concatenate([model.forward([2,3]), model.forward([4]), model.forward([5])])
        np.testing.assert_array_equal(logits, expected)
        self.assert_decoder_cache(before, model.snapshot())
        self.assertTrue(np.isfinite(logits).all())

    def test_decoder_config_and_weights_cannot_mutate_live_contract(self):
        model, config, tensors = self.decoder_fixture()
        expected = model.forward([2,3])
        model.reset()
        config['linear_attn_config']['kda_layers'].clear()
        tensors['model.embed_tokens.weight'].fill(0)
        with self.assertRaises(AttributeError):
            model.config.hidden_size = 1
        with self.assertRaises(TypeError):
            model.config.linear_attn_config['head_dim'] = 1
        for weight in model.weights.values():
            with self.assertRaises(ValueError):
                weight.setflags(write=True)
        np.testing.assert_array_equal(model.forward([2,3]), expected)

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


ORACLE_REVISION = '6cd7ae39bdf89ece8cc11761293ed28a4b70b838'
ORACLE_HASHES = {
    'configuration_kimi_linear.py': '9ca8cab8fa7965df9a0c66c8cd6e3d1b6daa0039462268031d7d074e9ae97d33',
    'modeling_kimi_linear.py': '0fc45a9b0b286cae7be0d4047dbe473726d3c6e0e39ae7f3acaa1148de18e325',
}


class NativeKimiPinnedOracleTests(unittest.TestCase):
    """Unmodified official CPU modules, supplied locally; no network in tests."""
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('KIMI_LINEAR_ORACLE_DIR')
        if not directory:
            raise unittest.SkipTest('set KIMI_LINEAR_ORACLE_DIR to the two pinned official sources')
        import torch
        import transformers
        if transformers.__version__ != '5.16.1':
            raise unittest.SkipTest('pinned oracle dependency qualification requires Transformers 5.16.1')
        cls.torch = torch
        cls.stack = ExitStack()
        try:
            sources = {}
            for filename, digest in ORACLE_HASHES.items():
                path = Path(directory)/filename
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f'official oracle hash mismatch: {filename}')
                sources[filename] = (path, data)
            package_name = 'transformers.models.kimi_linear'
            package = types.ModuleType(package_name)
            package.__path__ = [str(Path(directory).resolve())]
            cls.stack.enter_context(patch.dict(sys.modules, {package_name: package}))
            for filename, (path, data) in sources.items():
                name = package_name+'.'+path.stem
                module = types.ModuleType(name)
                module.__file__, module.__package__ = str(path), package_name
                sys.modules[name] = module
                # Exact hashed source, with its original relative imports; no AST edits or stubs.
                exec(compile(data, str(path), 'exec'), module.__dict__)
                if filename.startswith('configuration'):
                    cls.config_class = module.KimiLinearConfig
                else:
                    cls.model_class = module.KimiLinearForCausalLM
            old_threads = torch.get_num_threads()
            cls.stack.callback(torch.set_num_threads, old_threads)
            torch.set_num_threads(1)
            cls.temp = tempfile.TemporaryDirectory()
            cls.stack.callback(cls.temp.cleanup)
            cls.path = build(cls.temp.name)
            cls.instances = []
        except BaseException:
            cls.stack.close()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            if os.name == 'nt':
                C.windll.kernel32.FreeLibrary.argtypes = [C.c_void_p]
                for model in cls.instances:
                    C.windll.kernel32.FreeLibrary(model.library._handle)
            cls.instances.clear()
        finally:
            cls.stack.close()

    def fixture(self, seed=193, **changes):
        torch = self.torch
        args = dict(vocab_size=23, hidden_size=12, intermediate_size=17, moe_intermediate_size=7,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2, kv_lora_rank=5,
            q_lora_rank=None, qk_nope_head_dim=3, qk_rope_head_dim=2, v_head_dim=4,
            num_local_experts=8, num_experts_per_tok=2, n_shared_experts=2, n_group=2, topk_group=1,
            norm_topk_prob=True, routed_scaling_factor=1.3,
            layer_types=['linear_attention', 'full_attention', 'linear_attention'],
            mlp_layer_types=['dense', 'sparse', 'sparse'],
            linear_head_dim=3, linear_num_heads=2, linear_conv_kernel_dim=4,
            max_position_embeddings=96, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        args.update(changes)
        config = self.config_class(**args)
        config._attn_implementation = 'eager'
        config._experts_implementation = 'eager'
        torch.manual_seed(seed)
        oracle = self.model_class(config).eval()
        with torch.no_grad():
            for name, parameter in oracle.named_parameters():
                if 'norm.weight' in name:
                    parameter.uniform_(.65, 1.35)
                elif name.endswith('A_log'):
                    parameter.uniform_(-2, -1)
                else:
                    parameter.normal_(0, .18)
            for name, value in oracle.named_buffers():
                if name.endswith('e_score_correction_bias'):
                    value.copy_(torch.linspace(-.23, .31, value.numel()))
        weights = {name: value.numpy().copy() for name, value in oracle.state_dict().items()}
        native = NativeKimiLinearDecoder(config.to_dict(), weights, library=self.path, capacity=96)
        self.instances.append(native)
        return oracle, native, weights

    def compare_cache(self, native, expected, oracle):
        position, state = native.snapshot()
        for i, values in enumerate(state):
            layer = expected.past_key_values.layers[i]
            if i+1 in native.config.linear_attn_config['kda_layers']:
                np.testing.assert_allclose(values[0], layer.recurrent_states[0][0].numpy(), atol=4e-6, rtol=4e-5)
                h = native.config.linear_attn_config['num_heads']
                d = native.config.linear_attn_config['head_dim']
                bank = layer.conv_states[0][0].numpy().reshape(3,h,d,-1)
                for stream, history in enumerate(values[1:]):
                    if history.shape[0]:
                        np.testing.assert_allclose(history.transpose(1,2,0), bank[stream,...,-history.shape[0]:],
                                                   atol=4e-6, rtol=4e-5)
            else:
                attn = oracle.model.layers[i].self_attn
                keys, vals = attn.expand_kv(layer.keys, layer.values)
                for actual, reference in zip(values, (keys, vals)):
                    np.testing.assert_allclose(actual[:position].transpose(1,0,2), reference[0].numpy(),
                                               atol=4e-6, rtol=4e-5)
                    np.testing.assert_array_equal(actual[position:], 0)

    def exercise(self, count=13, atol=5e-6, **changes):
        oracle, native, weights = self.fixture(**changes)
        tokens = [1+(i*7)%21 for i in range(count)]
        torch = self.torch
        baseline = IndependentKimiLinearDecoder(oracle.config.to_dict(), weights, capacity=96)
        with torch.inference_mode():
            full = oracle(torch.tensor([tokens]), use_cache=True)
            expected = full.logits[0].numpy()
            np.testing.assert_allclose(native.forward(tokens), expected, atol=atol, rtol=4e-5)
            np.testing.assert_allclose(baseline.forward(tokens), expected, atol=atol, rtol=4e-5)
            self.compare_cache(native, full, oracle)
            full_native = native.snapshot()
            for partitions in ((1,)*count, (2, count-3, 1), (count//2, count-count//2)):
                native.reset()
                past, offset, rows = None, 0, []
                for size in partitions:
                    piece = tokens[offset:offset+size]
                    out = oracle(torch.tensor([piece]), past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    actual = native.forward(piece)
                    np.testing.assert_allclose(actual, out.logits[0].numpy(), atol=atol, rtol=4e-5)
                    self.compare_cache(native, out, oracle)
                    rows.append(actual)
                    offset += size
                np.testing.assert_allclose(np.concatenate(rows), expected, atol=atol, rtol=4e-5)
                # C token arithmetic is exactly independent of caller partitions.
                NativeKimiTests.assert_decoder_cache(self, full_native, native.snapshot())

    def test_fused_hybrid_grouped_full_cached_and_every_state(self):
        self.exercise()

    def test_low_rank_q_multiple_groups_and_chunk_boundary(self):
        # The unchanged FP32 oracle's chunk and recurrent paths differ themselves
        # on this amplified fixture; retain a separate explicit long-call bound.
        self.exercise(count=67, atol=3e-5, q_lora_rank=6, n_group=4, topk_group=2, num_experts_per_tok=3)

    def test_topk_one_unrenormalized_all_experts_and_dense_schedules(self):
        self.exercise(q_lora_rank=4, num_experts_per_tok=1, linear_conv_kernel_dim=1)
        self.exercise(norm_topk_prob=False, topk_group=2, num_experts_per_tok=8,
                      mlp_layer_types=['sparse','dense','sparse'])
        self.exercise(mlp_layer_types=['dense','dense','dense'], seed=719)

    def test_grouped_router_matches_unmodified_oracle_and_not_legacy_gather(self):
        oracle, native, _ = self.fixture(q_lora_rank=6)
        gate = oracle.model.layers[1].mlp.gate
        with self.torch.inference_mode():
            x = self.torch.linspace(-.7,.9,12)[None]
            logits, weights, ids = gate(x)
            scores = logits.sigmoid()[0].numpy()
            actual_ids, actual_weights = native._route(scores, gate.e_score_correction_bias.numpy())
        order = np.argsort(ids[0].numpy())
        np.testing.assert_array_equal(actual_ids, ids[0].numpy()[order])
        np.testing.assert_allclose(actual_weights, weights[0].numpy()[order], atol=2e-7, rtol=2e-6)
        biased = scores[actual_ids]+gate.e_score_correction_bias.numpy()[actual_ids]
        biased = biased/biased.sum()*native.config.routed_scaling_factor
        self.assertGreater(float(np.max(np.abs(actual_weights-biased))), .001)

    def test_no_numpy_or_oracle_fallback_in_prefill_and_decode(self):
        oracle, native, _ = self.fixture(q_lora_rank=6)
        with self.torch.inference_mode():
            expected = oracle(self.torch.tensor([[1,2,3,4]]), use_cache=False).logits[0].numpy()
        with ExitStack() as stack:
            for name in ('_mat','_vec','_attention','_route','_kda'):
                stack.enter_context(patch.object(IndependentKimiLinearDecoder,name,side_effect=AssertionError('NumPy fallback')))
            stack.enter_context(patch('openagent.adaptive_engine.independent_kimi.projected_kda_equations',
                                      side_effect=AssertionError('NumPy recurrence')))
            stack.enter_context(patch.object(self.model_class,'forward',side_effect=AssertionError('oracle delegation')))
            actual = np.concatenate((native.forward([1,2]),native.forward([3]),native.forward([4])))
        np.testing.assert_allclose(actual,expected,atol=5e-6,rtol=4e-5)

    def test_fused_rollback_bad_inputs_ties_and_late_head_failure(self):
        oracle, native, weights = self.fixture(q_lora_rank=6)
        native.forward([1,2])
        before = native.snapshot()
        real = native._mat
        def fail(w,x):
            if w is native.weights['lm_head.weight']:
                raise ValueError('late head failure')
            return real(w,x)
        with patch.object(native,'_mat',side_effect=fail):
            with self.assertRaisesRegex(ValueError,'late head'):
                native.forward([3,4])
        NativeKimiTests.assert_decoder_cache(self,before,native.snapshot())
        for ids in ([],[True],[-1],[23],[1.5],[[1]],np.array([2**64-1],np.uint64),[1]*95):
            with self.subTest(ids=str(ids)[:30]), self.assertRaises(ValueError):
                native.forward(ids)
            NativeKimiTests.assert_decoder_cache(self,before,native.snapshot())
        scores = np.full(8,.5,np.float32)
        with self.assertRaisesRegex(ValueError,'status 6'):
            native._route(scores,np.zeros(8,np.float32))
        tied = dict(weights)
        for name in list(tied):
            if name.endswith('mlp.gate.weight') or name.endswith('e_score_correction_bias'):
                tied[name] = np.zeros_like(tied[name])
        decoder = NativeKimiLinearDecoder(oracle.config.to_dict(),tied,library=self.path,capacity=96)
        self.instances.append(decoder)
        untouched = decoder.snapshot()
        with self.assertRaisesRegex(ValueError,'status 6'):
            decoder.forward([1])
        NativeKimiTests.assert_decoder_cache(self,untouched,decoder.snapshot())

    def test_fused_config_and_tensor_boundaries_fail_closed(self):
        oracle,native,weights = self.fixture(q_lora_rank=6)
        raw = oracle.config.to_dict()
        variants = (dict(model_type='kimi_k3'),dict(q_lora_rank=True),dict(q_lora_rank=65537),
                    dict(num_expert_group=3),dict(topk_group=3),dict(num_experts_per_token=5),
                    dict(layer_types=['full_attention']),dict(mlp_layer_types=['dense']),
                    dict(attention_bias=True),dict(dtype='bfloat16'),dict(head_dim=99),
                    dict(tensor_layout='upstream-fused'),dict(mla_use_nope=False))
        for change in variants:
            with self.subTest(change=change), self.assertRaises(ValueError):
                NativeKimiLinearDecoder(dict(raw,**change),weights,library=self.path,capacity=96)
        for alteration in ('extra','missing','dtype','shape'):
            bad = dict(weights)
            name = 'model.layers.1.mlp.experts.gate_up_proj'
            if alteration == 'extra': bad['model.layers.1.block_sparse_moe.gate.weight'] = np.zeros((8,12),np.float32)
            elif alteration == 'missing': del bad[name]
            elif alteration == 'dtype': bad[name] = bad[name].astype(np.float64)
            else: bad[name] = bad[name][:,:-1].copy()
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                NativeKimiLinearDecoder(raw,bad,library=self.path,capacity=96)
        native.max_bytes = 1
        before = native.snapshot()
        with self.assertRaisesRegex(ValueError,'memory budget'):
            native.forward([1])
        NativeKimiTests.assert_decoder_cache(self,before,native.snapshot())


if __name__ == '__main__':
    unittest.main()
