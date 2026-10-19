"""Synthetic Q8_0 integration evidence, not public-checkpoint quality evidence."""

from contextlib import ExitStack, redirect_stdout, redirect_stderr
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_cli import main
from openagent.adaptive_engine.expert_runtime import SparseExpertManifest
from openagent.adaptive_engine.expert_kernels import ExpertKernelRegistry


def checkpoint(root):
    """Write real F32 SafeTensors with multiple blocks per projection row."""
    hidden, intermediate = 64, 96
    config = dict(model_type="olmoe", architectures=["OlmoeForCausalLM"], hidden_size=hidden,
                  intermediate_size=intermediate, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, num_experts=3,
                  num_experts_per_tok=2, vocab_size=23, max_position_embeddings=16,
                  rms_norm_eps=1e-5, rope_theta=10000.0, norm_topk_prob=True,
                  eos_token_id=22)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rng = np.random.default_rng(9341)
    tensors = {}

    def matrix(name, shape):
        tensors[name] = rng.normal(0, 0.08, shape).astype("<f4")

    matrix("model.embed_tokens.weight", (23, hidden))
    matrix("lm_head.weight", (23, hidden))
    tensors["model.norm.weight"] = rng.uniform(0.8, 1.2, hidden).astype("<f4")
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for name, width in (("input_layernorm", hidden), ("post_attention_layernorm", hidden),
                            ("self_attn.q_norm", hidden), ("self_attn.k_norm", hidden // 2)):
            tensors[prefix + name + ".weight"] = rng.uniform(0.8, 1.2, width).astype("<f4")
        for name, width in (("q_proj", hidden), ("k_proj", hidden // 2),
                            ("v_proj", hidden // 2), ("o_proj", hidden)):
            matrix(prefix + "self_attn." + name + ".weight", (width, hidden))
        matrix(prefix + "mlp.gate.weight", (3, hidden))
        for expert in range(3):
            for name, shape in (("gate_proj", (intermediate, hidden)),
                                ("up_proj", (intermediate, hidden)),
                                ("down_proj", (hidden, intermediate))):
                matrix(prefix + f"mlp.experts.{expert}.{name}.weight", shape)
    data, header = bytearray(), {}
    for name, tensor in tensors.items():
        start = len(data)
        data.extend(tensor.tobytes())
        header[name] = dict(dtype="F32", shape=list(tensor.shape),
                            data_offsets=[start, len(data)])
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return tensors


def scalar_decode(payload, rows, columns, offset):
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(columns // 32):
            scale = struct.unpack_from("<e", payload, offset)[0]
            codes = struct.unpack_from("32b", payload, offset + 2)
            row.extend(scale * code for code in codes)
            offset += 34
        matrix.append(row)
    return matrix, offset


def scalar_projection(matrix, vector):
    return [math.fsum(w * float(x) for w, x in zip(row, vector)) for row in matrix]


def scalar_expert(payload, vector, hidden=64, intermediate=96):
    gate, offset = scalar_decode(payload, intermediate, hidden, 0)
    up, offset = scalar_decode(payload, intermediate, hidden, offset)
    down, offset = scalar_decode(payload, hidden, intermediate, offset)
    if offset != len(payload):
        raise AssertionError("unexpected trailing Q8_0 bytes")
    g, u = scalar_projection(gate, vector), scalar_projection(up, vector)
    activation = [a / (1 + math.exp(-a)) * b for a, b in zip(g, u)]
    return np.array(scalar_projection(down, activation), dtype=np.float32)


class Q8OlmoeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.tensors = checkpoint(self.root)
        self.manifest = pack_olmoe_experts(self.root, self.root / "experts.q8", dtype="q8_0")

    def model(self, manifest=None):
        return self.stack.enter_context(OlmoeModel(
            self.root, self.manifest if manifest is None else manifest,
            expert_cache_bytes=3 * 64 * 96 // 32 * 34,
            prefetch_workers=1, prefetch_depth=1))

    def payload(self, region):
        with (self.root / region.path).open("rb") as source:
            source.seek(region.offset)
            return source.read(region.size)

    def test_real_packer_bytes_hashes_and_scalar_projection_oracle(self):
        model = self.model()
        vector = np.random.default_rng(17).normal(0, 0.4, 64).astype(np.float32)
        for layer in self.manifest.layers:
            self.assertEqual(layer.dtype, "q8_0")
            for expert, region in layer.regions.items():
                payload = self.payload(region)
                self.assertEqual(len(payload), 3 * 64 * 96 // 32 * 34)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), region.sha256)
                offset = 0
                for role, rows, columns in (("gate", 96, 64), ("up", 96, 64), ("down", 64, 96)):
                    decoded, end = scalar_decode(payload, rows, columns, offset)
                    name = f"{layer.id}.mlp.experts.{expert}.{role}_proj.weight"
                    matrix = self.tensors[name]
                    for row in range(rows):
                        for start in range(0, columns, 32):
                            block = matrix[row, start:start + 32]
                            scale = float(np.float16(np.max(np.abs(block)) / np.float32(127)))
                            codes = [max(-127, min(127, round(float(np.float32(x / scale)))))
                                     for x in block]
                            expected = struct.pack("<e32b", scale, *codes)
                            self.assertEqual(payload[offset:offset + 34], expected)
                            offset += 34
                    self.assertEqual(offset, end)
                    self.assertGreater(np.linalg.norm(np.array(decoded)), 0)
                expected = scalar_expert(payload, vector)
                actual = model.experts.execute_batch(layer.id, vector[None], [[(expert, 1.0)]])[0]
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
            routes = [("0", 0.35), ("2", 0.65)]
            expected = sum(weight * scalar_expert(self.payload(layer.regions[key]), vector)
                           for key, weight in routes)
            actual = model.experts.execute_batch(layer.id, vector[None], [routes])[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)

    def test_cached_vs_full_decode_and_snapshot_roundtrip(self):
        cached, full, resumed = self.model(), self.model(), self.model()
        ids = [1, 4, 7, 3, 9, 2]
        for index, token in enumerate(ids):
            actual = cached.forward([token])
            full.reset()
            np.testing.assert_allclose(actual, full.forward(ids[:index + 1]), rtol=3e-5, atol=3e-6)
            for field in ('keys', 'values'):
                for left, right in zip(cached.snapshot_state()[field], full.snapshot_state()[field]):
                    np.testing.assert_allclose(left, right, rtol=3e-5, atol=3e-6)
        saved = cached.snapshot_state()
        resumed.forward([8, 8])
        resumed.restore_state(saved)
        for field in ('keys', 'values'):
            for left, right in zip(saved[field], resumed.snapshot_state()[field]):
                np.testing.assert_array_equal(left, right)
        for chunk in ([5], [6, 10]):
            np.testing.assert_array_equal(cached.forward(chunk), resumed.forward(chunk))

    def test_snapshot_rejects_other_and_mixed_expert_precisions(self):
        source = self.model()
        source.forward([1, 4])
        state = source.snapshot_state()
        for dtype in ('int8-row', 'int4-row'):
            manifest = pack_olmoe_experts(self.root, self.root / f"experts.{dtype}", dtype=dtype)
            mixed = replace(self.manifest, layers=(manifest.layers[0], self.manifest.layers[1]))
            for candidate in (manifest, mixed):
                target = self.model(candidate)
                target.forward([2])
                before = target.snapshot_state()
                with self.assertRaisesRegex(ValueError, 'precision/backend'):
                    target.restore_state(state)
                after = target.snapshot_state()
                self.assertEqual(after['position'], before['position'])
                for field in ('keys', 'values'):
                    for left, right in zip(before[field], after[field]):
                        np.testing.assert_array_equal(left, right)

    @unittest.skipUnless(importlib.util.find_spec('numba'), 'optional Numba unavailable')
    def test_explicit_compiled_registry_scalar_and_model_differential(self):
        from openagent.adaptive_engine import native_q8_0_expert

        backend = native_q8_0_expert.NativeQ8_0ExpertBackend()
        if not backend.info().available:
            self.skipTest(backend.info().unavailable_reason)
        self.assertTrue(backend.info().capabilities.compiled)
        registry = ExpertKernelRegistry([backend])
        self.assertIs(registry.resolve(dtype='q8_0', preferred='q8_0-numba'), backend)
        native = self.stack.enter_context(OlmoeModel(
            self.root, self.manifest, expert_cache_bytes=3 * 64 * 96 // 32 * 34,
            kernels=registry, preferred_kernel='q8_0-numba',
            prefetch_workers=1, prefetch_depth=1))
        reference = self.stack.enter_context(OlmoeModel(
            self.root, self.manifest, expert_cache_bytes=3 * 64 * 96 // 32 * 34,
            kernels=ExpertKernelRegistry(), preferred_kernel='q8_0-numpy',
            prefetch_workers=1, prefetch_depth=1))
        vector = np.random.default_rng(48).normal(0, 0.4, 64).astype(np.float32)
        routes = [('0', 0.3), ('2', 0.7)]
        for layer in self.manifest.layers:
            expected = sum(weight * scalar_expert(self.payload(layer.regions[key]), vector)
                           for key, weight in routes)
            actual = native.experts.execute_batch(layer.id, vector[None], [routes])[0]
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-7)
        self.assertTrue(native_q8_0_expert._compiled_loop.nopython_signatures)
        ids = [1, 4, 7, 3, 9]
        for index, token in enumerate(ids):
            reference.reset()
            np.testing.assert_allclose(native.forward([token]),
                                       reference.forward(ids[:index + 1]), rtol=5e-5, atol=5e-6)
        state = native.snapshot_state()
        native.forward([6])
        native.restore_state(state)
        native_logits = native.forward([6])
        native.restore_state(state)
        np.testing.assert_array_equal(native.forward([6]), native_logits)
        with self.assertRaisesRegex(ValueError, 'precision/backend'):
            reference.restore_state(state)

    def test_cli_q8_defaults_preserve_other_precision_artifacts(self):
        artifacts = {}
        for dtype, names in (
                ('int8-row', ('openagent-experts.i8', 'openagent-experts.manifest.json')),
                ('int4-row', ('openagent-experts.i4', 'openagent-experts.i4.manifest.json')),
                ('q4_0', ('openagent-experts.q4', 'openagent-experts.q4.manifest.json'))):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(['pack', str(self.root), '--dtype', dtype]), 0)
            artifacts.update({name: (self.root / name).read_bytes() for name in names})
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(['pack', str(self.root), '--dtype', 'q8_0']), 0)
        manifest = SparseExpertManifest.read(self.root / 'openagent-experts.q8.manifest.json')
        self.assertTrue((self.root / 'openagent-experts.q8').is_file())
        self.assertTrue(all(layer.dtype == 'q8_0' for layer in manifest.layers))
        self.assertTrue(all(region.path == 'openagent-experts.q8'
                            for layer in manifest.layers for region in layer.regions.values()))
        for name, original in artifacts.items():
            self.assertEqual((self.root / name).read_bytes(), original)

    def test_nan_source_rejects_all_pack_precisions_and_cleans_outputs(self):
        source = self.root / 'model.safetensors'
        data = bytearray(source.read_bytes())
        length = struct.unpack_from('<Q', data)[0]
        header = json.loads(data[8:8 + length])
        # Fail late, after valid experts have been written to both temporary files.
        name = 'model.layers.1.mlp.experts.2.down_proj.weight'
        offset = 8 + length + header[name]['data_offsets'][0]
        struct.pack_into('<f', data, offset, float('nan'))
        source.write_bytes(data)
        for dtype in ('int8-row', 'int4-row', 'q8_0', 'q4_0', 'q4_1'):
            for existing in (False, True):
                with self.subTest(dtype=dtype, existing=existing):
                    output = self.root / f'rejected-{dtype}-{existing}.bin'
                    stripe = self.root / f'rejected-{dtype}-{existing}-stripe.bin'
                    if existing:
                        output.write_bytes(b'preserve primary')
                        stripe.write_bytes(b'preserve stripe')
                    expected_error = ('multi-container publication requires unused immutable destination names'
                                      if existing else 'finite')
                    with self.assertRaisesRegex(ValueError, expected_error):
                        pack_olmoe_experts(self.root, output, dtype=dtype,
                                          storage_destinations={'stripe': stripe})
                    for path, sentinel in ((output, b'preserve primary'),
                                           (stripe, b'preserve stripe')):
                        self.assertFalse(path.with_suffix(path.suffix + '.tmp').exists())
                        if existing:
                            self.assertEqual(path.read_bytes(), sentinel)
                        else:
                            self.assertFalse(path.exists())

    @unittest.skipUnless(importlib.util.find_spec('numba'), 'optional Numba unavailable')
    def test_gateway_loads_real_compiled_q8_runtime(self):
        from openagent.model_gateway import OlmoeDirectProvider
        from openagent.adaptive_engine import native_q8_0_expert

        backend = native_q8_0_expert.NativeQ8_0ExpertBackend()
        if not backend.info().available:
            self.skipTest(backend.info().unavailable_reason)
        manifest_path = self.root / 'gateway.q8.manifest.json'
        self.manifest.write(manifest_path)
        environment = {
            'OPENAGENT_OLMOE_MODEL_DIR': str(self.root),
            'OPENAGENT_OLMOE_MANIFEST': str(manifest_path),
            'OPENAGENT_OLMOE_CONTEXT': '16',
            'OPENAGENT_OLMOE_EXPERT_CACHE_GB': '0.25',
            'OPENAGENT_OLMOE_KERNEL': 'q8_0-numba',
            'OPENAGENT_OLMOE_IO_MODE': 'mmap',
            'OPENAGENT_OLMOE_STORAGE_ROOTS': '',
            'OPENAGENT_OLMOE_PREFETCH_WORKERS': '1',
            'OPENAGENT_OLMOE_PREFETCH_DEPTH': '1',
        }
        direct = self.stack.enter_context(OlmoeModel(
            self.root, self.manifest, expert_cache_bytes=int(0.25 * 1024**3),
            context_tokens=16, kernels=ExpertKernelRegistry([backend]),
            preferred_kernel='q8_0-numba', prefetch_workers=1, prefetch_depth=1))
        with patch.dict(os.environ, environment):
            provider = OlmoeDirectProvider()
            try:
                runtime = provider._load()
                self.assertIs(provider._runtime, runtime)
                self.assertIs(provider._load(), runtime)
                self.assertEqual(runtime.context_tokens, 16)
                self.assertEqual(runtime.experts.preferred_kernel, 'q8_0-numba')
                selected = runtime.experts.kernels.resolve(dtype='q8_0', preferred='q8_0-numba')
                self.assertIsInstance(selected, native_q8_0_expert.NativeQ8_0ExpertBackend)
                self.assertTrue(selected.info().capabilities.compiled)
                for ids in ([1, 4, 7], [3], [9, 2]):
                    np.testing.assert_array_equal(runtime.forward(ids), direct.forward(ids))
                self.assertTrue(native_q8_0_expert._compiled_loop.nopython_signatures)
            finally:
                if provider._runtime is not None:
                    provider._runtime.close()
                    provider._runtime = None


if __name__ == '__main__':
    unittest.main()
