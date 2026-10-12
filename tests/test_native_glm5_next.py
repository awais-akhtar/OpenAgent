"""Compiled decoder versus optional exact oracle. GLM5_NEXT_CERTIFY=1 forbids skips."""

import unittest
import importlib.util
import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np

from openagent.adaptive_engine.native_glm5_next import NativeGlm5NextTextDecoder
from openagent.adaptive_engine.independent_glm5_next import IndependentGlm5NextTextDecoder


def optional(reason):
    if os.environ.get("GLM5_NEXT_CERTIFY") == "1":
        raise RuntimeError("strict GLM5-next certification: " + reason)
    raise unittest.SkipTest(reason)


def build(directory):
    source = Path(__file__).resolve().parents[1]/"openagent/adaptive_engine/native_glm5_next.c"
    output = Path(directory)/("glm5_next.dll" if os.name == "nt" else "glm5_next.so")
    if os.name == "nt":
        vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))/"Microsoft Visual Studio/Installer/vswhere.exe"
        if not vswhere.exists():
            optional("MSVC unavailable")
        location = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
        if not location:
            optional("MSVC unavailable")
        batch = Path(location)/"VC/Auxiliary/Build/vcvars64.bat"
        args = ["cl", "/nologo", "/W4", "/WX", "/O2", "/fp:precise", "/LD", str(source), "/link", "/OUT:"+str(output)]
        command = 'cmd.exe /d /s /c ""'+str(batch)+'" && '+subprocess.list2cmdline(args)+'"'
    else:
        compiler = shutil.which("cc")
        if compiler is None:
            optional("C compiler unavailable")
        command = [compiler,"-std=c99","-Wall","-Wextra","-Werror","-O2","-shared","-fPIC",str(source),"-o",str(output),"-lm"]
    result = subprocess.run(command,cwd=directory,capture_output=True,text=True,timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return output


class Glm5NextOracleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
            optional('optional exact model oracle unavailable')
        import torch
        import transformers
        if transformers.__version__ != "5.16.1":
            optional("Contract audit requires official Transformers 5.16.1")
        from transformers import Glm5NextTextConfig, Glm5NextTextModel
        cls.torch = torch
        cls.config_class = Glm5NextTextConfig
        cls.model_class = Glm5NextTextModel
        cls.directory = tempfile.TemporaryDirectory()
        cls.decoders = []
        try:
            cls.library = build(cls.directory.name)
        except BaseException:
            cls.directory.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        if os.name == "nt":
            C.windll.kernel32.FreeLibrary.argtypes = [C.c_void_p]
            for decoder in cls.decoders:
                C.windll.kernel32.FreeLibrary(decoder.library._handle)
        cls.decoders.clear()
        cls.directory.cleanup()

    def fixture(self, **overrides):
        torch = self.torch
        torch.manual_seed(91)
        args = dict(vocab_size=32, hidden_size=16, intermediate_size=24,
            moe_intermediate_size=8, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=2,
            n_routed_experts=4, num_experts_per_tok=2,
            q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=8,
            v_head_dim=8, index_head_dim=8, index_n_heads=2,
            index_topk=4, index_kpool=2, linear_head_dim=8,
            linear_num_heads=2, hc_mult=2, pad_token_id=0,
            max_position_embeddings=128, swiglu_limit=0.3)
        args.update(overrides)
        config = self.config_class(**args)
        config._attn_implementation = "eager"
        model = self.model_class(config).eval()
        with torch.no_grad():
            for name, value in model.named_parameters():
                value.copy_(torch.randn_like(value)*0.18)
                if "norm.weight" in name or "layernorm.weight" in name:
                    value.add_(1)
                if name.endswith(".scale"):
                    value.add_(0.7)
            for name, value in model.named_buffers():
                if name.endswith("e_score_correction_bias"):
                    value.copy_(torch.linspace(-0.1,0.1,value.numel()))
        tensors = {name: value.detach().numpy().copy() for name,value in model.state_dict().items()}
        head = torch.randn(config.vocab_size,config.hidden_size)*0.2
        if config.tie_word_embeddings:
            head = model.embed_tokens.weight.detach()
        tensors["lm_head.weight"] = head.numpy().copy()
        decoder = NativeGlm5NextTextDecoder(config.to_dict(),tensors,library=self.library,max_tokens=64)
        self.decoders.append(decoder)
        return model, head, decoder, tensors

    def compare(self, model, head, decoder, ids, partitions, *, atol=1e-5, rtol=1e-5):
        torch = self.torch
        tokens = torch.tensor([ids])
        with torch.no_grad():
            full = model(tokens,use_cache=False).last_hidden_state
            expected_logits = torch.nn.functional.linear(full,head)
            cache = None
            offset = 0
            for count in partitions:
                oracle = model(tokens[:,offset:offset+count],past_key_values=cache,use_cache=True)
                cache = oracle.past_key_values
                actual = decoder.forward(tokens[:,offset:offset+count].numpy())
                np.testing.assert_allclose(actual["last_hidden_state"],full[:,offset:offset+count].numpy(),atol=atol,rtol=rtol)
                np.testing.assert_allclose(actual["last_hidden_state"],oracle.last_hidden_state.numpy(),atol=atol,rtol=rtol)
                np.testing.assert_allclose(actual["logits"],expected_logits[:,offset:offset+count].numpy(),atol=atol,rtol=rtol)
                offset += count
                self.assertEqual(decoder.position,offset)
                for i,(a,b) in enumerate(decoder.cache):
                    layer = cache.layers[i]
                    if decoder.config.layer_types[i] == "linear_attention":
                        np.testing.assert_allclose(a,layer.conv_states[0].numpy()[0],atol=atol,rtol=rtol)
                        np.testing.assert_allclose(b,layer.recurrent_states[0].numpy()[0],atol=atol,rtol=rtol)
                    else:
                        np.testing.assert_allclose(a[:offset,:,:decoder.config.qk_nope_head_dim].transpose(1,0,2),layer.keys.numpy()[0],atol=atol,rtol=rtol)
                        np.testing.assert_allclose(a[:offset,:,decoder.config.qk_nope_head_dim:].transpose(1,0,2),layer.values.numpy()[0],atol=atol,rtol=rtol)
                        if decoder.config.indexer_types[i] == "full":
                            np.testing.assert_allclose(b[:offset],layer.indexer_keys.numpy()[0,:,:-1],atol=atol,rtol=rtol)
        self.assertEqual(offset,len(ids))

    def test_native_prefill_chunks_and_single_tokens(self):
        ids = [1,7,3,9,5,2,11,4,6,13,8]
        for sizes in ((11,), (1,)*11, (2,3,1,5), (3,4,4)):
            model,head,decoder,_ = self.fixture()
            self.compare(model,head,decoder,ids,sizes)

    def test_shared_dsa_and_dense_and_sparse_schedule(self):
        model,head,decoder,_ = self.fixture(
            layer_types=["linear_attention","deepseek_sparse_attention","deepseek_sparse_attention","linear_attention"],
            indexer_types=["full","full","shared","full"],
            mlp_layer_types=["sparse","dense","sparse","dense"],index_topk=16)
        self.compare(model,head,decoder,[1,2,6,4,3,7,8,9,10],(3,1,5))

    def test_unbounded_forget_gate_and_unnormalized_routing(self):
        model,head,decoder,_ = self.fixture(linear_lower_bound=None,norm_topk_prob=False,index_topk=16)
        self.compare(model,head,decoder,[1,4,7,2,8,9,3],(1,4,2),atol=4e-5,rtol=2e-5)

    def test_softplus_accumulation_diagnostic(self):
        torch = self.torch
        model,head,decoder,_ = self.fixture(linear_lower_bound=None,norm_topk_prob=False,index_topk=16)
        tokens = torch.tensor([[1,4,7,2,8,9,3]])
        cores = []
        hook = model.layers[2].self_attn.o_norm.register_forward_pre_hook(
            lambda module,args: cores.append(args[0].detach().clone()))
        with torch.no_grad():
            full = model(tokens,use_cache=True)
            full_core = cores.pop()
            hook.remove()
            cache = None
            chunks = []
            offset = 0
            for count in (1,4,2):
                result = model(tokens[:,offset:offset+count],past_key_values=cache,use_cache=True)
                cache = result.past_key_values
                chunks.append(result.last_hidden_state)
                offset += count
            sequential = []
            single_cache = None
            for i in range(7):
                result = model(tokens[:,i:i+1],past_key_values=single_cache,use_cache=True)
                single_cache = result.past_key_values
                sequential.append(result.last_hidden_state)
        native = decoder.forward(tokens.numpy())["last_hidden_state"]
        chunk_hidden = torch.cat(chunks,dim=1)
        single_hidden = torch.cat(sequential,dim=1)
        metrics = {
            "native_vs_full_hidden": float(np.max(np.abs(native-full.last_hidden_state.numpy()))),
            "native_vs_chunk_hidden": float(np.max(np.abs(native-chunk_hidden.numpy()))),
            "native_vs_single_hidden": float(np.max(np.abs(native-single_hidden.numpy()))),
            "oracle_full_vs_chunk_hidden": (full.last_hidden_state-chunk_hidden).abs().max().item(),
            "oracle_chunk_vs_single_hidden": (chunk_hidden-single_hidden).abs().max().item(),
            "layer2_max_output_norm_gain": torch.rsqrt(full_core.square().mean(-1)+model.config.rms_norm_eps).max().item(),
            "native_vs_full_logits": float(np.max(np.abs(
                torch.nn.functional.linear(torch.from_numpy(native),head).numpy()
                -torch.nn.functional.linear(full.last_hidden_state,head).numpy()))),
        }
        for i,(_,state) in enumerate(decoder.cache):
            if i < 3:
                metrics[f"layer{i}_native_vs_chunk_memory"] = float(np.max(np.abs(state-cache.layers[i].recurrent_states[0].numpy()[0])))
                metrics[f"layer{i}_oracle_full_vs_chunk_memory"] = (full.past_key_values.layers[i].recurrent_states[0]-cache.layers[i].recurrent_states[0]).abs().max().item()
            else:
                metrics["native_vs_chunk_index"] = float(np.max(np.abs(state[:7]-cache.layers[i].indexer_keys.numpy()[0,:,:-1])))
                metrics["native_vs_full_index"] = float(np.max(np.abs(state[:7]-full.past_key_values.layers[i].indexer_keys.numpy()[0,:,:-1])))
                metrics["native_vs_single_index"] = float(np.max(np.abs(state[:7]-single_cache.layers[i].indexer_keys.numpy()[0,:,:-1])))
                metrics["oracle_full_vs_chunk_index"] = (full.past_key_values.layers[i].indexer_keys-cache.layers[i].indexer_keys).abs().max().item()
                metrics["oracle_chunk_vs_single_index"] = (single_cache.layers[i].indexer_keys-cache.layers[i].indexer_keys).abs().max().item()
        if os.environ.get("GLM5_NEXT_MEASURE") == "1":
            print("GLM5_NEXT_FP32_METRICS",metrics,flush=True)
        # Full prefill is tight. Oracle chunk/recurrent association changes
        # sub-micro-unit KDA state, amplified by near-zero gated RMS outputs.
        np.testing.assert_allclose(native,full.last_hidden_state.numpy(),atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(decoder.cache[3][1][:7],full.past_key_values.layers[3].indexer_keys.numpy()[0,:,:-1],atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(native,single_hidden.numpy(),atol=4e-5,rtol=2e-5)
        np.testing.assert_allclose(decoder.cache[3][1][:7],single_cache.layers[3].indexer_keys.numpy()[0,:,:-1],atol=4e-5,rtol=2e-5)

    def test_kpool_one_and_tied_head(self):
        model,head,decoder,_ = self.fixture(index_kpool=1,index_topk=2,tie_word_embeddings=True)
        self.compare(model,head,decoder,[1,3,2,5,4,7],(2,4))

    def test_grouped_routing_oracle_and_chunk_cache(self):
        for groups, selected, experts, top, normalize in (
                (2,1,8,2,True), (4,2,8,3,False), (2,2,8,4,True)):
            with self.subTest(groups=groups, selected=selected, normalize=normalize):
                model,head,decoder,_ = self.fixture(n_group=groups,topk_group=selected,
                    n_routed_experts=experts,num_experts_per_tok=top,
                    norm_topk_prob=normalize,index_topk=16)
                self.compare(model,head,decoder,[1,7,3,9,5,2,11],(2,1,4))

    def test_group_boundary_tie_is_transactional(self):
        model,_,_,tensors = self.fixture(n_group=2,topk_group=1,
            n_routed_experts=8,num_experts_per_tok=2)
        tensors["layers.3.mlp.gate.weight"].fill(0)
        # Unique expert scores, but equal sums of the top two in each group.
        tensors["layers.3.mlp.gate.e_score_correction_bias"][:] = [0,1,2,5,0,1,3,4]
        decoder = NativeGlm5NextTextDecoder(model.config.to_dict(),tensors,
            library=self.library,max_tokens=64)
        self.decoders.append(decoder)
        with self.assertRaisesRegex(ValueError,"boundary tie"):
            decoder.forward([[1,2]])
        self.assert_cache_equal(decoder,0,None)

    def test_group_geometry_and_tensor_names_fail_closed(self):
        model,_,_,tensors = self.fixture()
        data = model.config.to_dict()
        for overrides in ({"n_group":3}, {"n_group":4}, {"topk_group":2},
                          {"n_group":2,"topk_group":1,"num_experts_per_tok":3},
                          {"n_group":True}, {"topk_group":0}):
            with self.assertRaises(ValueError):
                IndependentGlm5NextTextDecoder(data | overrides,tensors,max_tokens=64)
        for name in (1,"language_model.model.norm.weight","model.model.norm.weight"):
            bad = dict(tensors)
            bad[name] = bad.pop("norm.weight")
            with self.assertRaises(ValueError):
                IndependentGlm5NextTextDecoder(data,bad,max_tokens=64)

    def test_normalization_overflow_cannot_commit_zero_output(self):
        _,_,decoder,_ = self.fixture()
        decoder.forward([[1,2]])
        saved = decoder.cache
        embedding = decoder.weights["embed_tokens.weight"]
        original = embedding.copy()
        embedding.flags.writeable = True
        embedding[3].fill(np.finfo(np.float32).max)
        try:
            with self.assertRaisesRegex(ValueError,"non-finite"):
                decoder.forward([[4,3]])
            self.assert_cache_equal(decoder,2,saved)
        finally:
            embedding[:] = original
            embedding.flags.writeable = False
        decoder.forward([[4,3]])

    def test_swiglu_overflow_is_not_hidden_by_clamping(self):
        model,_,_,tensors = self.fixture()
        tensors["embed_tokens.weight"].fill(1)
        tensors["layers.0.attn_hc.fn"].fill(0)
        tensors["layers.0.attn_hc.base"].fill(0)
        tensors["layers.0.ffn_hc.fn"].fill(0)
        tensors["layers.0.ffn_hc.base"].fill(0)
        tensors["layers.0.self_attn.o_proj.weight"].fill(0)
        tensors["layers.0.post_attention_layernorm.weight"].fill(1)
        tensors["layers.0.mlp.gate_proj.weight"].fill(np.finfo(np.float32).max)
        decoder = NativeGlm5NextTextDecoder(model.config.to_dict(),tensors,
            library=self.library,max_tokens=64)
        self.decoders.append(decoder)
        with self.assertRaisesRegex(ValueError,"non-finite"):
            decoder.forward([[1]])
        self.assert_cache_equal(decoder,0,None)

    def test_stale_private_abi_is_rejected(self):
        model,_,_,tensors = self.fixture()
        stale = MagicMock()
        stale.oa_glm5_next_version.return_value = 1
        with patch("openagent.adaptive_engine.native_glm5_next.C.CDLL",return_value=stale):
            with self.assertRaisesRegex(ValueError,"ABI"):
                NativeGlm5NextTextDecoder(model.config.to_dict(),tensors,
                    library=self.library,max_tokens=64)
        stale.oa_glm5_next_step.assert_not_called()

    def test_grouped_execution_has_no_python_oracle_fallback(self):
        _,_,decoder,_ = self.fixture(n_routed_experts=8,n_group=2,topk_group=1)
        with patch.object(IndependentGlm5NextTextDecoder,"forward",
                          side_effect=AssertionError("independent fallback")), \
             patch.object(self.model_class,"forward",side_effect=AssertionError("oracle fallback")), \
             patch.object(self.torch.nn.functional,"linear",side_effect=AssertionError("Torch fallback")), \
             patch.object(np,"matmul",side_effect=AssertionError("NumPy fallback")):
            full = decoder.forward([[1,2,3,4]])
            saved = decoder.cache
            decoder.reset()
            chunks = [decoder.forward([[1]]),decoder.forward([[2,3,4]])]
        for name in ("last_hidden_state","logits"):
            np.testing.assert_array_equal(full[name],np.concatenate([r[name] for r in chunks],axis=1))
        self.assert_cache_equal(decoder,4,saved)

    def assert_cache_equal(self, decoder, position, saved):
        self.assertEqual(decoder.position,position)
        if saved is None:
            self.assertIsNone(decoder.cache)
        else:
            for actual,expected in zip(decoder.cache,saved):
                for a,b in zip(actual,expected):
                    np.testing.assert_array_equal(a,b)

    def test_transaction_rolls_back_late_chunk_failure(self):
        _,_,decoder,_ = self.fixture()
        for initialized in (False,True):
            decoder.reset()
            if initialized:
                decoder.forward([[1,2]])
            position,saved = decoder.position,decoder.cache
            original = decoder._step
            calls = []
            def fail(*args):
                original(*args)
                calls.append(1)
                if len(calls) == 3:
                    raise RuntimeError("injected after C wrote candidate caches")
            with patch.object(decoder,"_step",side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,"injected"):
                    decoder.forward([[3,4,5,6]])
            self.assert_cache_equal(decoder,position,saved)
            decoder.forward([[3,4]])
            self.assertEqual(decoder.position,position+2)

    def test_native_tie_failure_rolls_back(self):
        model,_,_,tensors = self.fixture()
        for name in tensors:
            if name.endswith("mlp.gate.weight") or name.endswith("e_score_correction_bias"):
                tensors[name].fill(0)
        decoder = NativeGlm5NextTextDecoder(model.config.to_dict(),tensors,library=self.library,max_tokens=64)
        self.decoders.append(decoder)
        with self.assertRaisesRegex(ValueError,"boundary tie"):
            decoder.forward([[1,2,3]])
        self.assert_cache_equal(decoder,0,None)

    def test_pool_boundary_tie_rolls_back_late_chunk(self):
        model,_,_,tensors = self.fixture()
        tensors["layers.3.self_attn.indexer.wq_b.weight"].fill(0)
        decoder = NativeGlm5NextTextDecoder(model.config.to_dict(),tensors,library=self.library,max_tokens=64)
        self.decoders.append(decoder)
        decoder.forward([[1,2]])
        saved = decoder.cache
        # The third complete pool crosses a tied two-pool selection boundary.
        with self.assertRaisesRegex(ValueError,"boundary tie"):
            decoder.forward([[3,4,5,6]])
        self.assert_cache_equal(decoder,2,saved)

    def test_varied_head_stream_pool_and_convolution_geometry(self):
        model,head,decoder,_ = self.fixture(num_attention_heads=1,num_key_value_heads=1,
            q_lora_rank=6,kv_lora_rank=5,qk_nope_head_dim=6,v_head_dim=4,
            index_n_heads=3,index_head_dim=4,index_topk=12,index_kpool=3,
            linear_head_dim=4,linear_num_heads=3,linear_conv_kernel_dim=1,
            hc_mult=4,hc_sinkhorn_iters=1,n_shared_experts=2)
        self.compare(model,head,decoder,[1,4,7,2,8,9,3,6,5],(1,3,5))

    def test_cache_free_snapshot_and_reset(self):
        model,_,decoder,_ = self.fixture()
        decoder.forward([[1,2,3]])
        saved = decoder.cache
        actual = decoder.forward([[5,4]],use_cache=False)
        with self.torch.no_grad():
            expected = model(self.torch.tensor([[5,4]]),use_cache=False).last_hidden_state.numpy()
        np.testing.assert_allclose(actual["last_hidden_state"],expected,atol=1e-5,rtol=1e-5)
        self.assert_cache_equal(decoder,3,saved)
        snapshot = decoder.cache
        snapshot[0][0].fill(77)
        self.assert_cache_equal(decoder,3,saved)
        decoder.reset()
        self.assert_cache_equal(decoder,0,None)

    def test_no_numpy_arithmetic_fallback(self):
        _,_,decoder,_ = self.fixture()
        with patch.multiple(np,matmul=unittest.mock.DEFAULT,dot=unittest.mock.DEFAULT,
                            einsum=unittest.mock.DEFAULT,exp=unittest.mock.DEFAULT,
                            sqrt=unittest.mock.DEFAULT,sum=unittest.mock.DEFAULT,mean=unittest.mock.DEFAULT) as mocks:
            for mocked in mocks.values():
                mocked.side_effect = AssertionError("NumPy arithmetic fallback")
            decoder.forward([[1,2,3]])
            decoder.forward([[4,5]])

    def test_input_failures_leave_cache_unchanged(self):
        _,_,decoder,_ = self.fixture()
        decoder.forward([[1,2]])
        saved = decoder.cache
        for ids,kwargs in (([[32]],{}),([[-1]],{}),([[1.5]],{}),([],{}),([[1],[2]],{}),
                           ([[1]],{"attention_mask":[[1]]}),([[1]],{"position_ids":[[0]]}),
                           ([[1]],{"use_cache":1}),([[1]*63],{})):
            with self.assertRaises(ValueError):
                decoder.forward(ids,**kwargs)
            self.assert_cache_equal(decoder,2,saved)

    def test_config_and_tensor_fail_closed(self):
        model,_,_,tensors = self.fixture()
        data = model.config.to_dict()
        for name,value in (("model_type","glm_dsa"),("attention_bias",True),("hidden_act","gelu"),
                           ("qk_rope_head_dim",2),("n_group",3),("index_kpool_always_select_tail",False),
                           ("linear_head_dim",257),("layer_types",["full_attention"]*4),
                           ("indexer_types",["shared"]*4),("quantization_config",{"bits":4})):
            with self.assertRaises(ValueError,msg=name):
                NativeGlm5NextTextDecoder(data | {name:value},tensors,library=self.library)
        for bad in (dict(tensors,unexpected=np.zeros(1,np.float32)),
                    dict(tensors,**{"norm.weight":np.ones(16,np.float64)}),
                    dict(tensors,**{"norm.weight":np.full(16,np.nan,np.float32)})):
            with self.assertRaises(ValueError):
                NativeGlm5NextTextDecoder(data,bad,library=self.library)
        with self.assertRaises(ValueError):
            NativeGlm5NextTextDecoder(data,tensors,library=self.library,max_bytes=1)
        independent = IndependentGlm5NextTextDecoder(data,tensors,max_tokens=64)
        with self.assertRaises(NotImplementedError):
            independent.forward([[1]])

    def test_genuine_head_only(self):
        model,_,_,tensors = self.fixture(tie_word_embeddings=True)
        del tensors["lm_head.weight"]
        decoder = NativeGlm5NextTextDecoder(model.config.to_dict(),tensors,library=self.library,max_tokens=64)
        self.decoders.append(decoder)
        actual = decoder.forward([[1,2,3]])
        self.assertNotIn("logits",actual)
        self.assertEqual(decoder.status()["output_scope"],"text-hidden-only")
        with self.torch.no_grad():
            expected = model(self.torch.tensor([[1,2,3]]),use_cache=False).last_hidden_state.numpy()
        np.testing.assert_allclose(actual["last_hidden_state"],expected,atol=1e-5,rtol=1e-5)

    def test_native_chunk_invariance_bit_exact(self):
        _,_,decoder,_ = self.fixture()
        ids = [[1,7,3,9,5,2,11,4,6,13,8]]
        full = decoder.forward(ids)
        saved = decoder.cache
        decoder.reset()
        chunks = [decoder.forward([ids[0][:3]]),decoder.forward([ids[0][3:7]]),decoder.forward([ids[0][7:]])]
        for name in ("logits","last_hidden_state"):
            np.testing.assert_array_equal(full[name],np.concatenate([chunk[name] for chunk in chunks],axis=1))
        self.assert_cache_equal(decoder,11,saved)

    def test_native_bad_abi_and_nonfinite_rollback(self):
        _,_,decoder,_ = self.fixture()
        decoder.forward([[1,2]])
        saved = decoder.cache
        original = decoder._geometry[0]
        decoder._geometry[0] = 257
        with self.assertRaisesRegex(ValueError,"geometry"):
            decoder.forward([[3]])
        self.assert_cache_equal(decoder,2,saved)
        decoder._geometry[0] = original
        # Force finite-input arithmetic overflow after C has updated candidate state.
        norm = decoder.weights["norm.weight"]
        original_norm = norm.copy()
        norm.flags.writeable = True
        norm.fill(np.finfo(np.float32).max)
        with self.assertRaisesRegex(ValueError,"non-finite"):
            decoder.forward([[3,4]])
        self.assert_cache_equal(decoder,2,saved)
        norm[:] = original_norm
        norm.flags.writeable = False
        decoder.forward([[3]])

    def test_conditional_generation_logits_black_box(self):
        from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration, Glm5NextVisionConfig
        torch = self.torch
        model,head,decoder,tensors = self.fixture()
        vision = Glm5NextVisionConfig(depth=1,hidden_size=16,num_heads=2,
            out_hidden_size=16,intermediate_size=24,projection_intermediate_size=24,
            patch_size=2,image_size=4,spatial_merge_size=2)
        outer = Glm5NextConfig(text_config=model.config,vision_config=vision,
            image_token_id=30,video_token_id=31)
        oracle = Glm5NextForConditionalGeneration(outer).eval()
        oracle.model.language_model.load_state_dict(model.state_dict())
        with torch.no_grad():
            oracle.lm_head.weight.copy_(head)
            result = oracle(torch.tensor([[1,7,3,9,5,2]]),use_cache=False)
        actual = decoder.forward([[1,7,3,9,5,2]])
        np.testing.assert_allclose(actual["logits"],result.logits.numpy(),atol=1e-5,rtol=1e-5)
        decoder.reset()
        cache = None
        offset = 0
        ids = [1,7,3,9,5,2]
        with torch.no_grad():
            for count in (2,1,3):
                chunk = ids[offset:offset+count]
                expected = oracle(torch.tensor([chunk]),past_key_values=cache,use_cache=True)
                cache = expected.past_key_values
                actual = decoder.forward([chunk])
                np.testing.assert_allclose(actual["logits"],expected.logits.numpy(),atol=1e-5,rtol=1e-5)
                offset += count
        self.assertEqual(decoder.status()["output_scope"],"text-causal-lm-head")
        with self.assertRaises(ValueError):
            NativeGlm5NextTextDecoder(outer.to_dict(),tensors,library=self.library)

    def test_config_metadata(self):
        config = self.config_class()
        self.assertEqual(config.model_type, "glm5_next_text")
        self.assertEqual(config.layer_types[:4], ["linear_attention"] * 3 + ["deepseek_sparse_attention"])
        self.assertEqual(config.mlp_layer_types[:4], ["dense"] * 3 + ["sparse"])
        self.assertEqual(config.qk_rope_head_dim, 0)
        self.assertEqual(config.index_kpool, 16)
        self.assertEqual(config.hc_mult, 4)

    def test_black_box_prefill_and_chunks(self):
        torch = self.torch
        torch.manual_seed(7)
        config = self.config_class(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            moe_intermediate_size=8, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=2,
            n_routed_experts=4, num_experts_per_tok=2,
            q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=8,
            v_head_dim=8, index_head_dim=8, index_n_heads=2,
            index_topk=4, index_kpool=2, linear_head_dim=8,
            linear_num_heads=2, hc_mult=2, pad_token_id=0,
        )
        model = self.model_class(config).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        with torch.no_grad():
            full = model(tokens, use_cache=False).last_hidden_state
            self.assertTrue(torch.isfinite(full).all().item())
            for sizes in ((2, 3, 2), (1, 1, 1, 1, 1, 1, 1), (3, 4)):
                cache = None
                outputs = []
                offset = 0
                for size in sizes:
                    result = model(
                        tokens[:, offset:offset + size],
                        past_key_values=cache, use_cache=True,
                    )
                    cache = result.past_key_values
                    outputs.append(result.last_hidden_state)
                    offset += size
                torch.testing.assert_close(torch.cat(outputs, dim=1), full, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
