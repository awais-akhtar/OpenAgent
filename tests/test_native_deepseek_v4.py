"""Native V4 oracle certification; OA_DEEPSEEK_V4_STRICT=1 forbids skips."""

import ctypes
from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import importlib.util
from unittest.mock import patch, MagicMock

import numpy as np

from openagent.adaptive_engine.independent_deepseek_v4 import IndependentDeepseekV4Decoder, choose
from openagent.adaptive_engine.native_deepseek_v4 import NativeDeepseekV4Decoder, Tensor, tensor


def unavailable(reason):
    if os.environ.get("OA_DEEPSEEK_V4_STRICT") == "1":
        raise RuntimeError(reason)
    raise unittest.SkipTest(reason)


def require_oracle():
    if any(importlib.util.find_spec(name) is None for name in ("torch", "transformers")):
        unavailable("optional exact model oracle unavailable")
    import transformers
    if transformers.__version__ != "5.16.1":
        unavailable("oracle requires installed Transformers 5.16.1")


def build_native(directory):
    source = Path(__file__).resolve().parents[1]/"openagent/adaptive_engine/native_deepseek_v4.c"
    output = Path(directory)/("deepseek_v4.dll" if os.name == "nt" else "deepseek_v4.so")
    if os.name == "nt":
        discovery = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))/"Microsoft Visual Studio/Installer/vswhere.exe"
        if not discovery.exists():
            unavailable("native V4 certification needs MSVC")
        location = subprocess.check_output([str(discovery), "-latest", "-products", "*", "-requires",
                     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
        if not location:
            unavailable("native V4 certification needs MSVC")
        environment = Path(location)/"VC/Auxiliary/Build/vcvars64.bat"
        args = ["cl", "/nologo", "/W4", "/WX", "/O2", "/fp:precise", "/LD", str(source), "/link", "/OUT:"+str(output)]
        command = 'cmd.exe /d /s /c ""'+str(environment)+'" && '+subprocess.list2cmdline(args)+'"'
    else:
        compiler = shutil.which("cc")
        if compiler is None:
            unavailable("native V4 certification needs a C compiler")
        command = [compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2", "-ffp-contract=off",
                   "-shared", "-fPIC", str(source), "-o", str(output), "-lm"]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return output


class DeepseekV4OraclePrerequisites(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
            unavailable('optional exact model oracle unavailable')
        import torch
        import transformers

        if transformers.__version__ != "5.16.1":
            unavailable("oracle requires installed Transformers 5.16.1")
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.config = DeepseekV4Config(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            qk_rope_head_dim=4,
            partial_rotary_factor=0.5,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=1,
            index_head_dim=4,
            index_n_heads=2,
            index_topk=2,
            moe_intermediate_size=16,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            hc_mult=2,
            sliding_window=4,
            compress_rates={
                "compressed_sparse_attention": 2,
                "heavily_compressed_attention": 4,
            },
            layer_types=[
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
            mlp_layer_types=["hash_moe", "moe"],
            num_nextn_predict_layers=0,
            max_position_embeddings=128,
        )
        cls.config._attn_implementation = "eager"
        with torch.random.fork_rng():
            torch.manual_seed(7)
            cls.model = DeepseekV4ForCausalLM(cls.config).float().eval()
        cls.tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])

    def assert_close(self, actual, expected):
        self.assertTrue(self.torch.isfinite(actual).all().item())
        self.torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_config_metadata(self):
        metadata = self.config.to_dict()
        self.assertEqual(metadata["model_type"], "deepseek_v4")
        self.assertEqual(metadata["scoring_func"], "sqrtsoftplus")
        self.assertEqual(metadata["hc_sinkhorn_iters"], 20)
        self.assertEqual(metadata["compress_rope_theta"], 160000.0)

    def test_token_cache_crosses_window_and_compression_boundaries(self):
        with self.torch.no_grad():
            full = self.model(self.tokens, use_cache=False).logits
            cache = None
            parts = []
            for position in range(self.tokens.shape[1]):
                output = self.model(
                    self.tokens[:, position : position + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                parts.append(output.logits)
            self.assert_close(self.torch.cat(parts, dim=1), full)

    def test_chunked_cache(self):
        with self.torch.no_grad():
            full = self.model(self.tokens, use_cache=False).logits
            for boundaries in ((0, 3, 5, 12), (0, 4, 8, 12), (0, 7, 12)):
                cache = None
                parts = []
                for start, end in zip(boundaries, boundaries[1:]):
                    output = self.model(
                        self.tokens[:, start:end],
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = output.past_key_values
                    parts.append(output.logits)
                self.assert_close(self.torch.cat(parts, dim=1), full)

    def test_causality_and_input_sensitivity(self):
        with self.torch.no_grad():
            full = self.model(self.tokens, use_cache=False).logits
            changed = self.tokens.clone()
            changed[:, 6:] = 20
            altered = self.model(changed, use_cache=False).logits
            self.assert_close(altered[:, :6], full[:, :6])
            self.assertGreater((altered[:, 6:] - full[:, 6:]).abs().max().item(), 1e-4)


class NativeDeepseekV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_oracle()
        import torch
        cls.torch = torch
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.library = build_native(cls.directory.name)
        cls.loaded = []
        cls.addClassCleanup(cls.unload)

    @classmethod
    def unload(cls):
        if os.name == "nt":
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            for decoder in cls.loaded:
                ctypes.windll.kernel32.FreeLibrary(decoder.library._handle)

    def fixture(self, kinds=None, routing=None, **overrides):
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
        kinds = kinds or ["compressed_sparse_attention", "heavily_compressed_attention", "sliding_attention"]
        routing = routing or ["hash_moe", "moe", "moe"]
        options = dict(vocab_size=32, hidden_size=16, num_hidden_layers=len(kinds),
                       num_attention_heads=4, num_key_value_heads=1, head_dim=8,
                       qk_rope_head_dim=4, partial_rotary_factor=0.5, q_lora_rank=8,
                       o_lora_rank=6, o_groups=2, index_head_dim=6, index_n_heads=2,
                       index_topk=2, moe_intermediate_size=12, n_routed_experts=4,
                       n_shared_experts=1, num_experts_per_tok=2, hc_mult=3,
                       hc_sinkhorn_iters=5, hc_eps=2e-5, rms_norm_eps=3e-5,
                       swiglu_limit=0.35, sliding_window=4,
                       compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
                       layer_types=kinds, mlp_layer_types=routing, num_nextn_predict_layers=0,
                       max_position_embeddings=128, initializer_range=0.15)
        config = DeepseekV4Config(**(options | overrides))
        config._attn_implementation = "eager"
        torch = self.torch
        with torch.random.fork_rng():
            torch.manual_seed(31)
            model = DeepseekV4ForCausalLM(config).float().eval()
            # Nontrivial maps and biases expose transpose, overlap and routing errors.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith((".base", ".hc_base", "position_bias", ".sinks")):
                        parameter.uniform_(-0.4, 0.4)
                    if name.endswith((".scale", ".hc_scale")):
                        parameter.uniform_(0.3, 1.7)
                for name, buffer in model.named_buffers():
                    if name.endswith("tid2eid"):
                        rows = torch.arange(config.vocab_size)[:, None]
                        slots = torch.arange(config.num_experts_per_tok)[None, :]
                        buffer.copy_((rows+slots) % config.n_routed_experts)
                    if name.endswith("e_score_correction_bias"):
                        buffer.copy_(torch.linspace(-0.25, 0.3, config.n_routed_experts))
        tensors = {name: value.detach().numpy() for name, value in model.state_dict().items()}
        independent = IndependentDeepseekV4Decoder(config.to_dict(), tensors, max_context=32)
        native = NativeDeepseekV4Decoder(config.to_dict(), tensors, library=self.library, max_context=32)
        self.loaded.append(native)
        return model, independent, native

    def check(self, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=8e-6, rtol=8e-5)

    def test_all_attention_types_prefill_incremental_chunk_and_reset(self):
        model, independent, native = self.fixture()
        ids = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype=np.int64)
        with self.torch.no_grad():
            expected = model(self.torch.from_numpy(ids), use_cache=False).logits.numpy()
        self.check(independent.forward(ids).logits, expected)
        with patch.object(IndependentDeepseekV4Decoder, "_evaluate", side_effect=AssertionError("Python arithmetic fallback")):
            self.check(native.forward(ids).logits, expected)
            native.reset()
            parts = [native.forward(ids[:, i:i+1]).logits for i in range(ids.shape[1])]
            self.check(np.concatenate(parts, axis=1), expected)
            native.reset()
            parts = [native.forward(ids[:, start:end]).logits for start, end in ((0, 3), (3, 5), (5, 12))]
            self.check(np.concatenate(parts, axis=1), expected)
        self.assertEqual(native.position, 12)
        native.reset()
        self.assertEqual(native.position, 0)
        self.check(native.forward(ids).logits, expected)

    def test_hca_long_history_and_nonoverlap(self):
        model, independent, native = self.fixture(
            kinds=["heavily_compressed_attention"], routing=["moe"], hc_mult=4,
            compress_rates={"compressed_sparse_attention": 3, "heavily_compressed_attention": 3})
        ids = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=np.int64)
        with self.torch.no_grad():
            expected = model(self.torch.from_numpy(ids), use_cache=True).logits.numpy()
        self.check(independent.forward(ids).logits, expected)
        self.check(native.forward(ids).logits, expected)

    def test_relu_experts_prefill_chunks_and_component_oracle(self):
        ids = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype=np.int64)
        for modes in (["hash_moe", "moe", "moe"], ["moe", "hash_moe", "hash_moe"]):
            model, independent, native = self.fixture(hidden_act="relu", routing=modes)
            x = (4*np.random.default_rng(42).normal(size=(12, 16))).astype(np.float32)
            prefix = "model.layers.0."
            w = [independent.weights[prefix+n] if n else None for n, _ in independent.layer_specs[0]]
            table = independent.weights.get(prefix+"mlp.gate.tid2eid")
            shared_gate = x @ w[30].T
            self.assertLess(shared_gate.min(), 0)
            self.assertGreater(shared_gate.max(), independent.config.swiglu_limit)
            with self.torch.no_grad():
                expected = model(self.torch.from_numpy(ids), use_cache=False).logits.numpy()
                oracle_moe = model.model.layers[0].mlp(
                    self.torch.from_numpy(x[None]), input_ids=self.torch.from_numpy(ids)).numpy()[0]
                cache, parts = None, []
                for start, end in ((0, 3), (3, 5), (5, 12)):
                    result = model(self.torch.from_numpy(ids[:, start:end]),
                                   past_key_values=cache, use_cache=True)
                    cache = result.past_key_values
                    parts.append(result.logits.numpy())
                self.check(np.concatenate(parts, axis=1), expected)
            self.check(independent._moe(x, w, ids[0], table), oracle_moe)
            silu = IndependentDeepseekV4Decoder(model.config.to_dict() | {"hidden_act": "silu"},
                                              independent.weights, max_context=32)
            self.assertGreater(np.max(np.abs(silu.forward(ids).logits-expected)), 1e-4)
            for decoder in (independent, native):
                with patch.object(IndependentDeepseekV4Decoder, "_evaluate",
                                  side_effect=AssertionError("Python fallback")) if isinstance(
                                      decoder, NativeDeepseekV4Decoder) else nullcontext():
                    self.check(decoder.forward(ids).logits, expected)
                    decoder.reset()
                    chunks = [decoder.forward(ids[:, start:end]).logits
                              for start, end in ((0, 3), (3, 5), (5, 12))]
                    self.check(np.concatenate(chunks, axis=1), expected)
                self.assertEqual(decoder.position, 12)

    def test_activation_geometry_and_stale_abi_reject(self):
        model, _, native = self.fixture(hidden_act="relu")
        for activation in ("gelu", "unknown", None):
            with self.assertRaises(ValueError):
                IndependentDeepseekV4Decoder(model.config.to_dict() | {"hidden_act": activation}, native.weights)
        native._c.activation = 2
        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            native.forward([2])
        self.assertEqual(native.position, 0)
        native._c.activation = 1
        native.forward([2])
        stale = MagicMock()
        stale.oa_deepseek_v4_version.return_value = 1
        with patch("openagent.adaptive_engine.native_deepseek_v4.C.CDLL", return_value=stale):
            with self.assertRaisesRegex(ValueError, "ABI"):
                NativeDeepseekV4Decoder(model.config.to_dict(), native.weights,
                                       library=self.library, max_context=32)
        stale.oa_deepseek_v4_forward.assert_not_called()

    def test_tied_embeddings_full_chunk_and_official_checkpoint(self):
        from safetensors.numpy import load_file
        from transformers import DeepseekV4ForCausalLM
        model, independent, native = self.fixture(tie_word_embeddings=True)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertIn("lm_head.weight", model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            saved = load_file(str(next(Path(directory).glob("*.safetensors"))))
            self.assertNotIn("lm_head.weight", saved)
            self.assertIn("model.embed_tokens.weight", saved)
            reloaded = DeepseekV4ForCausalLM.from_pretrained(directory).float().eval()
            self.assertIs(reloaded.lm_head.weight, reloaded.model.embed_tokens.weight)
        ids = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype=np.int64)
        with self.torch.no_grad():
            expected = reloaded(self.torch.from_numpy(ids), use_cache=False).logits.numpy()
            self.check(model(self.torch.from_numpy(ids), use_cache=False).logits.numpy(), expected)
            cache, oracle_parts = None, []
            for start, end in ((0, 3), (3, 5), (5, 12)):
                result = reloaded(self.torch.from_numpy(ids[:, start:end]),
                                  past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                oracle_parts.append(result.logits.numpy())
            self.check(np.concatenate(oracle_parts, axis=1), expected)
        # Official serialization reverses layer names; normalization is loader-owned.
        normalized = {name: value.detach().numpy() for name, value in reloaded.state_dict().items()
                      if name != "lm_head.weight"}
        omitted_python = IndependentDeepseekV4Decoder(model.config.to_dict(), normalized, max_context=32)
        omitted_native = NativeDeepseekV4Decoder(model.config.to_dict(), normalized,
                                               library=self.library, max_context=32)
        self.loaded.append(omitted_native)
        for decoder in (independent, native, omitted_python, omitted_native):
            with self.subTest(implementation=decoder.implementation, omitted=decoder in (omitted_python, omitted_native)):
                self.assertIs(decoder.weights["lm_head.weight"], decoder.weights["model.embed_tokens.weight"])
                with self.assertRaises(ValueError):
                    decoder.weights["lm_head.weight"].setflags(write=True)
                with patch.object(IndependentDeepseekV4Decoder, "_evaluate",
                                  side_effect=AssertionError("Python arithmetic fallback")) if isinstance(
                                      decoder, NativeDeepseekV4Decoder) else nullcontext():
                    self.check(decoder.forward(ids).logits, expected)
                    decoder.reset()
                    parts = [decoder.forward(ids[:, start:end]).logits
                             for start, end in ((0, 3), (3, 5), (5, 12))]
                    self.check(np.concatenate(parts, axis=1), expected)
                self.assertEqual(decoder.position, 12)
        for decoder in (native, omitted_native):
            self.assertEqual(ctypes.cast(decoder._global[0].data, ctypes.c_void_p).value,
                             ctypes.cast(decoder._global[5].data, ctypes.c_void_p).value)
            self.assertEqual(decoder.implementation, "compiled-c-deepseek-v4-token-replay")

    def test_tied_manifest_equality_dtypes_and_resident_budget(self):
        model, independent, native = self.fixture(tie_word_embeddings=True)
        config, weights = model.config.to_dict(), dict(independent.weights)
        untied = IndependentDeepseekV4Decoder(config | {"tie_word_embeddings": False}, weights, max_context=32)
        self.assertEqual(untied.resident_bytes-independent.resident_bytes,
                         weights["lm_head.weight"].nbytes)
        self.assertIsNot(untied.weights["lm_head.weight"], untied.weights["model.embed_tokens.weight"])
        omitted = {n: v for n, v in weights.items() if n != "lm_head.weight"}
        mismatch = weights["lm_head.weight"].copy()
        mismatch[0, 0] = np.nextafter(mismatch[0, 0], np.float32(np.inf))
        for decoder, kwargs in ((IndependentDeepseekV4Decoder, {}),
                                (NativeDeepseekV4Decoder, {"library": self.library})):
            for head in (mismatch, weights["lm_head.weight"].astype(np.float64),
                         weights["lm_head.weight"][:-1], np.full_like(mismatch, np.nan)):
                with self.assertRaises(ValueError):
                    decoder(config, weights | {"lm_head.weight": head}, **kwargs)
            with self.assertRaises(ValueError):
                decoder(config | {"tie_word_embeddings": False}, omitted, **kwargs)
            with self.assertRaises(ValueError):
                decoder(config, {n: v for n, v in omitted.items() if n != "model.embed_tokens.weight"}, **kwargs)
            for value in (1, 0, "true", None):
                with self.assertRaises(ValueError):
                    decoder(config | {"tie_word_embeddings": value}, weights, **kwargs)
            budget = 2*independent.resident_bytes+independent.workspace_bytes(32)
            # Separate equal checkpoint arrays must still become a real shared buffer.
            supplied = weights | {"lm_head.weight": weights["lm_head.weight"].copy()}
            accepted = decoder(config, supplied, max_context=32, max_bytes=budget, **kwargs)
            self.assertIs(accepted.weights["lm_head.weight"], accepted.weights["model.embed_tokens.weight"])
            if isinstance(accepted, NativeDeepseekV4Decoder):
                self.loaded.append(accepted)
            with self.assertRaises(MemoryError):
                decoder(config, omitted, max_context=32, max_bytes=budget-1, **kwargs)

    def test_component_equations_and_indexer_visibility(self):
        model, independent, _ = self.fixture()
        rng = np.random.default_rng(918)
        streams = rng.normal(size=(12, 3, 16)).astype(np.float32)
        c = independent.config
        prefix = "model.layers.0."
        w = [independent.weights[prefix+n] if n else None for n, _ in independent.layer_specs[0]]
        collapsed, post, comb = independent._hyper(streams, w[20:23])
        with self.torch.no_grad():
            oracle_post, oracle_comb, oracle_collapsed = model.model.layers[0].attn_hc(
                self.torch.from_numpy(streams[None]))
        self.check(collapsed, oracle_collapsed.numpy()[0])
        self.check(post, oracle_post.numpy()[0])
        self.check(comb, oracle_comb.numpy()[0])
        self.assertGreater(np.max(np.abs(comb-comb.transpose(0, 2, 1))), 1e-3)
        x = rng.normal(size=(12, 16)).astype(np.float32)
        qr = rng.normal(size=(12, 8)).astype(np.float32)
        with self.torch.no_grad():
            pool, bias = model.model.layers[0].self_attn.compressor(
                self.torch.from_numpy(x[None]), self.torch.from_numpy(qr[None]),
                self.torch.arange(12)[None], None, 0)
            picks = model.model.layers[0].self_attn.compressor.indexer(
                self.torch.from_numpy(x[None]), self.torch.from_numpy(qr[None]),
                self.torch.arange(12)[None], None, 0)
        original = independent._pool(x, w[10:14], 2, 8, True, independent.thetas[1])
        self.check(original, pool.numpy()[0, 0])
        self.assertTrue(np.isneginf(bias.numpy()[0, 0, 0]).all())
        self.assertTrue((picks.numpy()[0, 0] == -1).all())
        for t in range(12):
            visible = np.flatnonzero(np.isfinite(bias.numpy()[0, 0, t]))
            self.assertTrue((visible < (t+1)//2).all())
            self.assertEqual(set(visible), set(picks.numpy()[0, t][picks.numpy()[0, t] >= 0]))

    def test_csa_indexer_and_compressor_are_not_identity_aliases(self):
        model, _, native = self.fixture(kinds=["compressed_sparse_attention"], routing=["moe"])
        ids = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype=np.int64)
        baseline = native.forward(ids).logits
        # Alter scoring head sign to make different compressed entries visible.
        with self.torch.no_grad():
            model.model.layers[0].self_attn.compressor.indexer.scorer.weights_proj.weight.mul_(-1)
            expected = model(self.torch.from_numpy(ids), use_cache=False).logits.numpy()
        weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
        changed = NativeDeepseekV4Decoder(model.config.to_dict(), weights, library=self.library, max_context=32)
        self.loaded.append(changed)
        actual = changed.forward(ids).logits
        self.check(actual, expected)
        self.assertGreater(np.max(np.abs(actual-baseline)), 1e-3)
        self.check(actual[:, :5], baseline[:, :5])
        with self.torch.no_grad():
            model.model.layers[0].self_attn.compressor.kv_norm.weight.zero_()
            expected = model(self.torch.from_numpy(ids), use_cache=False).logits.numpy()
        weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
        suppressed = NativeDeepseekV4Decoder(model.config.to_dict(), weights, library=self.library, max_context=32)
        self.loaded.append(suppressed)
        actual_suppressed = suppressed.forward(ids).logits
        self.check(actual_suppressed, expected)
        self.assertGreater(np.max(np.abs(actual_suppressed-actual)), 1e-3)

    def test_varying_geometry_seed_and_topk(self):
        for hc, group, rate, top in ((1, 1, 3, 1), (2, 4, 2, 3), (4, 2, 4, 2)):
            model, independent, native = self.fixture(hc_mult=hc, o_groups=group,
                num_experts_per_tok=top, index_topk=8,
                compress_rates={"compressed_sparse_attention": rate, "heavily_compressed_attention": rate+1})
            ids = np.array([[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]], dtype=np.int64)
            with self.torch.no_grad():
                expected = model(self.torch.from_numpy(ids), use_cache=False).logits.numpy()
            self.check(independent.forward(ids).logits, expected)
            self.check(native.forward(ids).logits, expected)

    def test_rejects_and_state_rollback(self):
        model, _, native = self.fixture()
        native.forward([2, 3, 4])
        before = native._history.copy()
        for ids, kwargs in (([32], {}), ([2.5], {}), ([[2], [3]], {}),
                            ([2], {"attention_mask": [[1]]}), ([2], {"position_ids": [[3]]}),
                            ([2]*30, {})):
            with self.assertRaises(ValueError):
                native.forward(ids, **kwargs)
            np.testing.assert_array_equal(native._history, before)
        with patch.object(native, "_evaluate", side_effect=RuntimeError("injected failure")):
            with self.assertRaises(RuntimeError):
                native.forward([5])
        np.testing.assert_array_equal(native._history, before)
        with patch.object(native, "_evaluate", return_value=np.full((4, 32), np.nan, dtype=np.float32)):
            with self.assertRaises(FloatingPointError):
                native.forward([5])
        np.testing.assert_array_equal(native._history, before)
        with self.torch.no_grad():
            expected = model(self.torch.tensor([[2, 3, 4, 5]]), use_cache=False).logits.numpy()
        self.check(native.forward([5]).logits, expected[:, -1:])

    def test_manifest_scope_budget_and_immutability(self):
        model, _, native = self.fixture()
        config = model.config.to_dict()
        weights = dict(native.weights)
        for name, value in (("attention_bias", True), ("num_nextn_predict_layers", 1),
                            ("hidden_act", "gelu"), ("n_shared_experts", 2),
                            ("num_key_value_heads", 2), ("quantization_config", {}),
                            ("norm_topk_prob", False)):
            with self.assertRaises(ValueError):
                IndependentDeepseekV4Decoder(config | {name: value}, weights)
        with self.assertRaises(MemoryError):
            IndependentDeepseekV4Decoder(config, weights, max_bytes=1024)
        first = "model.embed_tokens.weight"
        with self.assertRaises(ValueError):
            IndependentDeepseekV4Decoder(config, weights | {first: weights[first].astype(np.float64)})
        with self.assertRaises(ValueError):
            IndependentDeepseekV4Decoder(config, {n: v for n, v in weights.items() if n != first})
        with self.assertRaises(ValueError):
            weights[first].setflags(write=True)
        table_name = "model.layers.0.mlp.gate.tid2eid"
        self.assertEqual(weights[table_name].dtype, np.int64)
        duplicate = weights[table_name].copy()
        duplicate[:, 1] = duplicate[:, 0]
        with self.assertRaises(ValueError):
            IndependentDeepseekV4Decoder(config, weights | {table_name: duplicate})
        with self.assertRaises(ValueError):
            IndependentDeepseekV4Decoder(config, weights | {table_name: weights[table_name].astype(np.int32)})
        raw = {n: v.copy() for n, v in weights.items()}
        clone = IndependentDeepseekV4Decoder(config, raw, max_context=32)
        raw[first].fill(0)
        self.assertGreater(np.abs(clone.weights[first]).sum(), 0)
        self.assertEqual(native.library.oa_deepseek_v4_workspace(ctypes.byref(native._c), 32), native.workspace_bytes(32))

    def test_c_abi_bad_lengths_budget_and_kernel_tie_rollback(self):
        _, _, native = self.fixture(kinds=["sliding_attention"], routing=["moe"])
        native.forward([2])
        ids = np.array([2, 3], dtype=np.int64)
        out = np.empty((2, 32), dtype=np.float32)
        function = native.library.oa_deepseek_v4_forward
        args = (ctypes.byref(native._c), native._global, native._layers,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), 2)
        self.assertEqual(function(*args, Tensor(tensor(out).data, out.size-1), 10**8), 1)
        self.assertEqual(function(*args, tensor(out), 1), 2)
        original = Tensor(native._layers[0].w[26].data, native._layers[0].w[26].count)
        zeros = np.zeros((4, 16), dtype=np.float32)
        correction = np.zeros(4, dtype=np.float32)
        prior_correction = Tensor(native._layers[0].w[27].data, native._layers[0].w[27].count)
        native._layers[0].w[26] = tensor(zeros)
        native._layers[0].w[27] = tensor(correction)
        try:
            with self.assertRaisesRegex(ValueError, "cutoff tie"):
                native.forward([3])
            self.assertEqual(native.position, 1)
        finally:
            native._layers[0].w[26] = original
            native._layers[0].w[27] = prior_correction

    def test_fp32_metadata_and_exact_identity_rejection(self):
        model, _, native = self.fixture()
        config, weights = model.config.to_dict(), native.weights
        for decoder, kwargs in ((IndependentDeepseekV4Decoder, {}),
                                (NativeDeepseekV4Decoder, {"library": self.library})):
            for name in ("rms_norm_eps", "hc_eps", "swiglu_limit", "routed_scaling_factor"):
                for value in (1e-50, float("inf"), float("nan"), True, "0.1"):
                    with self.subTest(decoder=decoder.__name__, name=name, value=value):
                        with self.assertRaises(ValueError):
                            decoder(config | {name: value}, weights, **kwargs)
            for value in (1e-50, 1e100, True, "10000"):
                rope = {name: dict(p) for name, p in config["rope_parameters"].items()}
                rope["main"]["rope_theta"] = value
                with self.assertRaises(ValueError):
                    decoder(config | {"rope_parameters": rope}, weights, **kwargs)
            for identity in ("deepseek_v41", "deepseek_v41_text"):
                with self.assertRaisesRegex(ValueError, "model_type must be deepseek_v4"):
                    decoder(config | {"model_type": identity}, weights, **kwargs)

    def test_native_nonfinite_error_preserves_history(self):
        _, _, native = self.fixture()
        native.forward([2])
        before = native._history.copy()
        with patch.object(native.library, "oa_deepseek_v4_forward", return_value=3):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite native"):
                native.forward([3])
        np.testing.assert_array_equal(native._history, before)
        self.assertTrue(np.isfinite(native.forward([3]).logits).all())

    def test_real_router_overflow_rejects_without_commit(self):
        model, independent, native = self.fixture(kinds=["sliding_attention"], routing=["moe"])
        captured = []
        handle = model.model.layers[0].mlp.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().numpy().copy()))
        try:
            with self.torch.no_grad():
                model(self.torch.tensor([[2]]), use_cache=False)
        finally:
            handle.remove()
        weights = dict(independent.weights)
        name = "model.layers.0.mlp.gate.weight"
        gate = weights[name].copy()
        gate[-1] = np.sign(captured[0].reshape(-1, 16)[0])*np.finfo(np.float32).max
        weights[name] = gate
        python = IndependentDeepseekV4Decoder(model.config.to_dict(), weights, max_context=32)
        compiled = NativeDeepseekV4Decoder(model.config.to_dict(), weights,
                                          library=self.library, max_context=32)
        self.loaded.append(compiled)
        for decoder in (python, compiled):
            with np.errstate(over="ignore", invalid="ignore"):
                with self.assertRaises(FloatingPointError):
                    decoder.forward([2])
            self.assertEqual(decoder.position, 0)


class OptionalOraclePolicyTests(unittest.TestCase):
    def test_absent_optional_dependencies_skip_unless_strict(self):
        with patch("importlib.util.find_spec", return_value=None):
            with patch.dict(os.environ, {"OA_DEEPSEEK_V4_STRICT": "0"}):
                with self.assertRaises(unittest.SkipTest):
                    require_oracle()
            with patch.dict(os.environ, {"OA_DEEPSEEK_V4_STRICT": "1"}):
                with self.assertRaises(RuntimeError):
                    require_oracle()

    def test_indexer_cutoff_tie_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "cutoff tie"):
            choose(np.array([1, 0, 0], dtype=np.float32), 2)
        np.testing.assert_array_equal(choose(np.array([1, 0, 0], dtype=np.float32), 3), [0, 1, 2])

    def test_topk_rejects_nonfinite_unselected_candidates_and_bad_counts(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(FloatingPointError):
                choose(np.array([2, 1, value], dtype=np.float32), 1)
        for count in (0, -1, 4, True):
            with self.assertRaises(ValueError):
                choose(np.array([2, 1, 0], dtype=np.float32), count)


if __name__ == "__main__":
    unittest.main()
