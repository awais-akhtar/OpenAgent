"""Payload-free inspector tests and an explicit unchanged-vendor CPU probe."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.native_kimi_k3_vendor_checkpoint import (
    UnsupportedContract, VENDOR_REVISION,
    inspect_native_kimi_k3_vendor_checkpoint as inspect,
)


def document():
    return {'model_type': 'kimi_k3', 'vision_config': {}, 'text_config': {
        'model_type': 'kimi_linear', 'hidden_size': 7168, 'num_hidden_layers': 4,
        'num_experts': 896, 'first_k_dense_replace': 1,
        'quantization_config': {'format': 'mxfp4-pack-quantized'},
        'linear_attn_config': {'kda_layers': [1, 2, 3], 'full_attn_layers': [4],
                             'num_heads': 96, 'head_dim': 128}}}


class VendorInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = document()
        self.header = {'language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed':
                       {'dtype': 'U8', 'shape': [2, 16], 'data_offsets': [0, 32]}}
        self.write()

    def write(self, payload=b'\xff' * 32):
        (self.root / 'config.json').write_text(json.dumps(self.config), encoding='utf-8')
        raw = json.dumps(self.header).encode()
        (self.root / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + payload)

    def test_typed_gaps_and_one_based_schedule(self):
        result = inspect(self.root)
        self.assertEqual(result.layer_types, ('kda', 'kda', 'kda', 'mla'))
        self.assertEqual(result.payload_bytes, 32)
        self.assertFalse(result.admitted)
        self.assertFalse(result.published_checkpoint_supported)
        self.assertFalse(result.pinned_config_matches)
        self.assertEqual({g.contract for g in result.gaps}, set(UnsupportedContract))
        self.assertEqual(result.tensors[0].dtype, 'U8')

    def test_no_payload_reads(self):
        import openagent.adaptive_engine.native_kimi_k3_vendor_checkpoint as module
        original = module._open_local
        reads = []

        class Guard:
            def __init__(self, source):
                self.source = source
                self.limit = 8 + len(json.dumps(self_header).encode())

            def read(self, size=-1):
                self_position = self.source.tell()
                if size < 0 or self_position + size > self.limit:
                    raise AssertionError('tensor payload read')
                reads.append(size)
                return self.source.read(size)

            def __getattr__(self, key):
                return getattr(self.source, key)

        self_header = self.header

        def guarded(root, path, stack):
            source, stamp = original(root, path, stack)
            return (Guard(source) if path.suffix == '.safetensors' else source), stamp

        with patch.object(module, '_open_local', guarded):
            inspect(self.root)
        self.assertEqual(reads, [8, len(json.dumps(self.header).encode())])

    def test_nested_and_outer_quantization_reported(self):
        self.config['quantization_config'] = {'format': 'unknown'}
        self.write()
        self.assertEqual({g.path for g in inspect(self.root).gaps
                          if g.contract == UnsupportedContract.QUANTIZATION},
                         {'quantization_config', 'text_config.quantization_config', 'tensor_inventory'})

    def test_fixture_alias_rejected(self):
        self.config['model_type'] = 'openagent_kimi_k3_hybrid_fixture'
        self.write()
        with self.assertRaisesRegex(ValueError, 'exact publisher'):
            inspect(self.root)

    def test_bad_schedules(self):
        for kda, mla in (([0, 1, 2], [4]), ([1, 1, 2, 3], [4]),
                         ([1, 2, 3], [3, 4]), ([1, 2], [4]), ([1, 2, 3, True], [4]),
                         ([1, 2, 4], [3])):
            self.config['text_config']['linear_attn_config'].update(kda_layers=kda, full_attn_layers=mla)
            self.write()
            with self.assertRaises(ValueError):
                inspect(self.root)

    def test_invalid_ranges_shapes_and_dtypes(self):
        name = next(iter(self.header))
        original = copy.deepcopy(self.header[name])
        for delta in ({'data_offsets': [0, 31]}, {'data_offsets': [-1, 31]},
                      {'shape': [True, 32]}, {'dtype': 'F4'}, {'shape': [-1]}):
            self.header[name] = {**original, **delta}
            self.write()
            with self.assertRaises(ValueError):
                inspect(self.root)

    def test_duplicate_tensors_and_index_mismatch(self):
        (self.root / 'second.safetensors').write_bytes((self.root / 'model.safetensors').read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inspect(self.root)
        (self.root / 'second.safetensors').unlink()
        for mapping in ({}, {next(iter(self.header)): '../escape.safetensors'}):
            (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
            with self.assertRaises(ValueError):
                inspect(self.root)

    def test_duplicate_json_budget_and_empty_inventory(self):
        (self.root / 'config.json').write_text('{"model_type":"kimi_k3","model_type":"kimi_k3"}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            inspect(self.root)
        self.write()
        for budget in (True, 0, 10):
            with self.assertRaises(ValueError):
                inspect(self.root, max_metadata_bytes=budget)
        (self.root / 'model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'no SafeTensors'):
            inspect(self.root)

    def test_nonquantized_tiny_is_still_not_admitted(self):
        self.config['text_config'].update(hidden_size=32, num_experts=4, first_k_dense_replace=0)
        self.config['text_config'].pop('quantization_config')
        self.config['text_config']['linear_attn_config'].update(num_heads=2, head_dim=16)
        self.write()
        result = inspect(self.root)
        self.assertFalse(result.admitted)
        self.assertIn(UnsupportedContract.FULL_HYBRID_ORACLE, {g.contract for g in result.gaps})

    def test_header_overlap_holes_truncation_and_empty(self):
        self.header['extra'] = {'dtype': 'U8', 'shape': [1], 'data_offsets': [31, 32]}
        self.write()
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            inspect(self.root)
        self.header = {'extra': {'dtype': 'U8', 'shape': [31], 'data_offsets': [1, 32]}}
        self.write()
        with self.assertRaisesRegex(ValueError, 'noncontiguous'):
            inspect(self.root)
        self.header = {}
        self.write(payload=b'')
        with self.assertRaisesRegex(ValueError, 'no tensors'):
            inspect(self.root)
        (self.root / 'model.safetensors').write_bytes(b'\x01')
        with self.assertRaisesRegex(ValueError, 'header length'):
            inspect(self.root)

    def test_identity_change_during_inspection(self):
        import openagent.adaptive_engine.native_kimi_k3_vendor_checkpoint as module
        with patch.object(module, '_stamp', return_value=('changed',)):
            with self.assertRaisesRegex(ValueError, 'changed during inspection'):
                inspect(self.root)

    def test_pinned_digest_is_not_admission(self):
        import openagent.adaptive_engine.native_kimi_k3_vendor_checkpoint as module
        digest = hashlib.sha256((self.root / 'config.json').read_bytes()).hexdigest()
        with patch.object(module, 'VENDOR_CONFIG_SHA256', digest):
            result = inspect(self.root)
        self.assertTrue(result.pinned_config_matches)
        self.assertFalse(result.admitted)

    def test_published_schedule_and_packed_shapes(self):
        text = self.config['text_config']
        text['num_hidden_layers'] = 93
        full = list(range(4, 93, 4)) + [93]
        text['linear_attn_config'].update(full_attn_layers=full,
                                          kda_layers=[i for i in range(1, 94) if i not in full])
        self.header = {}
        position = 0
        prefix = 'language_model.model.layers.1.block_sparse_moe.experts.0.'
        for suffix, shape in (('w1.weight_packed', [3072, 1792]),
                              ('w1.weight_scale', [3072, 112]),
                              ('w2.weight_packed', [3584, 1536]),
                              ('w2.weight_scale', [3584, 96])):
            size = shape[0] * shape[1]
            self.header[prefix + suffix] = {'dtype': 'U8', 'shape': shape,
                                             'data_offsets': [position, position + size]}
            position += size
        (self.root / 'config.json').write_text(json.dumps(self.config))
        raw = json.dumps(self.header).encode()
        # Sparse extent proves inspection need not materialize packed payloads.
        with (self.root / 'model.safetensors').open('wb') as source:
            source.write(struct.pack('<Q', len(raw)) + raw)
            source.truncate(8 + len(raw) + position)
        mapping = {name: 'model.safetensors' for name in self.header}
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
        result = inspect(self.root)
        self.assertEqual(result.layer_types.count('kda'), 69)
        self.assertEqual(result.layer_types.count('mla'), 24)
        self.assertEqual(result.layer_types[-2:], ('mla', 'mla'))
        self.assertEqual(result.payload_bytes, position)
        self.assertFalse(result.admitted)


def vendor_cpu_oracle():
    """Explicit networked probe; never stub FLA or rewrite vendor source."""
    import urllib.request
    import torch
    from tests.probe_kimi_k3_oracle import HASHES
    with tempfile.TemporaryDirectory(prefix='k3_full_hybrid_', ignore_cleanup_errors=True) as directory:
        if '--provision' in sys.argv:
            import subprocess
            dependencies = Path(directory) / 'dependencies'
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                            '--target', str(dependencies), 'einops==0.8.1', 'fla-core==0.5.2',
                            'triton-windows==3.8.0.post29'], check=True, timeout=120)
            sys.path.insert(0, str(dependencies))
        package = Path(directory) / 'k3_vendor'
        package.mkdir()
        for name, digest in HASHES.items():
            url = f'https://huggingface.co/moonshotai/Kimi-K3/resolve/{VENDOR_REVISION}/{name}'
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read(2 * 1024**2 + 1)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('vendor source hash mismatch: ' + name)
            (package / name).write_bytes(raw)
        sys.path.insert(0, directory)
        try:
            vendor = importlib.import_module('k3_vendor.modeling_kimi_linear')
            config = vendor.KimiLinearConfig(
                vocab_size=64, hidden_size=32, intermediate_size=32, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=2, q_lora_rank=16, kv_lora_rank=16,
                qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                mla_use_nope=True, mla_use_output_gate=True, hidden_act='situ',
                activation_situ_beta=4., activation_situ_linear_beta=25.,
                attn_res_block_size=2, rms_norm_eps=1e-5, pad_token_id=None,
                first_k_dense_replace=1, num_experts=4, num_experts_per_token=2,
                num_shared_experts=2, routed_expert_hidden_size=16,
                latent_moe_use_norm=True, moe_intermediate_size=16,
                linear_attn_config={'kda_layers': [1, 2, 3], 'full_attn_layers': [4],
                                    'head_dim': 16, 'num_heads': 2, 'short_conv_kernel_size': 4,
                                    'gate_lower_bound': -5., 'use_full_rank_gate': True})
            model = vendor.KimiLinearForCausalLM(config).float().cpu().eval()
            model.set_attn_implementation('eager')
            print('unchanged tiny full hybrid constructed on CPU', flush=True)
            for name, tensor in model.state_dict().items():
                print(name, tuple(tensor.shape), str(tensor.dtype), flush=True)
            with torch.no_grad():
                ids = torch.tensor([[1, 2, 3]])
                full = model(ids, use_cache=True)
                cache, logits = None, []
                for token in ids[0]:
                    output = model(token.reshape(1, 1), past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    logits.append(output.logits)
                torch.testing.assert_close(torch.cat(logits, dim=1), full.logits, atol=2e-5, rtol=2e-5)
                if not torch.isfinite(full.logits).all():
                    raise ValueError('nonfinite vendor logits')
                print('unchanged vendor full/tokenwise CPU logits verified', flush=True)
        finally:
            sys.path.remove(directory)


if __name__ == '__main__':
    if '--vendor-oracle' in sys.argv:
        vendor_cpu_oracle()
    else:
        unittest.main()
