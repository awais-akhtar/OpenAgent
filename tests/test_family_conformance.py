"""Conformance of exact identities and real loader rejection, not model math."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from openagent.adaptive_engine.family_conformance import (
    REGISTRY_IDENTITIES, conformance_matrix, require_native_decoder,
)
from openagent.adaptive_engine.hf_moe import HFMoeManifest, SUPPORTED_MODEL_TYPES


def geometry():
    return dict(hidden_size=8, intermediate_size=12, vocab_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        num_experts=4, num_local_experts=4, n_routed_experts=4,
        num_experts_per_tok=1, moe_intermediate_size=6, n_shared_experts=1,
        n_group=2, topk_group=1, first_k_dense_replace=1, kv_lora_rank=4,
        qk_rope_head_dim=2, qk_nope_head_dim=2, v_head_dim=4, head_dim=4,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=1,
        linear_num_value_heads=2, linear_conv_kernel_dim=3,
        shared_expert_intermediate_size=6,
        layer_types=['linear_attention', 'full_attention'])


class FamilyConformanceTests(unittest.TestCase):
    def test_explicit_hybrid_fixture_does_not_admit_published_k3(self):
        report = conformance_matrix()
        fixture = report['native_fixture_contracts'][0]
        self.assertEqual(fixture['architecture'], 'kimi_k3')
        self.assertEqual(fixture['constructor'], 'NativeKimiK3Hybrid')
        self.assertFalse(fixture['published_checkpoint'])
        self.assertFalse(fixture['full_registry_family'])
        self.assertFalse(fixture['packed_streaming'])
        self.assertNotIn('kimi_k3', report['resident_text_loader_families'])
        with self.assertRaises(ValueError):
            require_native_decoder('kimi_k3')

    def test_registry_identity_metadata_without_executing_reference(self):
        path = Path(__file__).resolve().parents[1] / 'reference/colibri/c/family_registry.py'
        if not path.is_file():
            self.skipTest('optional reference identity metadata not present')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'FAMILIES'
                                  for t in node.targets))
        observed = []
        for descriptor in assignment.value.elts:
            fields = {keyword.arg: keyword.value for keyword in descriptor.keywords}
            observed.append(tuple(ast.literal_eval(fields[key])
                                  for key in ('id', 'model_types', 'config_section')))
        self.assertEqual(observed, [row[:3] for row in REGISTRY_IDENTITIES])

    def test_json_matrix_distinguishes_library_python_and_native(self):
        report = json.loads(json.dumps(conformance_matrix()))
        rows = report['registry_models']
        self.assertEqual(len(rows), 17)
        self.assertEqual(len({row['model_type'] for row in rows}), 17)
        self.assertEqual(report['library_allowlist'], sorted(SUPPORTED_MODEL_TYPES))
        self.assertEqual(report['native_decoder_families'], [])
        self.assertEqual(report['native_decoder_supported_scope'], 'full-registry-family')
        self.assertEqual([row['model_type'] for row in rows if row['library_allowlisted']], ['olmoe'])
        self.assertEqual([row['model_type'] for row in rows if row['independent_python_family']],
                         ['glm_moe_dsa', 'olmoe', 'qwen3_5_moe_text', 'qwen4_exp_text'])
        self.assertEqual(report['independent_python_families'],
                         ['mixtral', 'olmoe', 'qwen3_5_moe_text', 'glm_moe_dsa', 'qwen4_exp_text'])
        self.assertEqual(report['native_resident_range_families'], ['olmoe', 'qwen3_5_moe_text', 'glm_moe_dsa', 'qwen4_exp_text', 'glm5_next_text', 'inkling_text'])
        self.assertEqual([row['model_type'] for row in rows
                          if row['native_resident_range_supported']], ['glm5_next_text', 'glm_moe_dsa', 'olmoe', 'qwen3_5_moe_text', 'qwen4_exp_text'])
        self.assertEqual([row['model_type'] for row in rows if row['native_streamed_range_supported']], ['olmoe'])
        self.assertTrue(all(not row['c_owned_storage_runtime'] for row in rows))
        self.assertTrue(all(not row['native_decoder_supported'] and row['rejection_reason']
                            and not row['upstream_abi_compatible'] for row in rows))
        self.assertIsNone(report['transformers_installed'])
        self.assertFalse(report['transformers_probed'])
        self.assertEqual([row['model_type'] for row in rows if row['resident_text_loader_available']],
                         ['glm5_next_text', 'glm_moe_dsa', 'kimi_linear', 'qwen3_5_moe_text', 'qwen4_exp_text', 'deepseek_v4'])
        self.assertIn('inkling_text', report['resident_text_loader_families'])
        self.assertEqual(report['experimental_text_families'], ['kimi_linear'])

    def test_real_manifest_accepts_allowlist_and_rejects_exact_remaining(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for name in sorted(SUPPORTED_MODEL_TYPES):
                with self.subTest(accepted=name):
                    path.write_text(json.dumps(dict(geometry(), model_type=name)), encoding='utf-8')
                    self.assertEqual(HFMoeManifest.read(directory).model_type, name)
            for row in conformance_matrix()['registry_models']:
                if row['library_allowlisted']:
                    continue
                name = row['model_type']
                for raw in (dict(geometry(), model_type=name),
                            dict(model_type=name, text_config=dict(geometry(), model_type='olmoe'))):
                    with self.subTest(rejected=name, wrapper='text_config' in raw):
                        path.write_text(json.dumps(raw), encoding='utf-8')
                        with self.assertRaisesRegex(ValueError, 'unsupported sparse MoE model_type'):
                            HFMoeManifest.read(directory)

    def test_native_gate_rejects_known_unknown_and_similar_names(self):
        names = [row['model_type'] for row in conformance_matrix()['registry_models']]
        for name in names + sorted(SUPPORTED_MODEL_TYPES) + ['OLMOE', ' olmoe ', 'new_family']:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'full registry-family native execution is not certified'):
                    require_native_decoder(name)
        for value in (None, '', True, {}, 1):
            with self.assertRaises(ValueError):
                require_native_decoder(value)

    def test_resident_glm_support_does_not_enable_glm_aliases_or_streaming(self):
        rows = {row['model_type']: row for row in conformance_matrix()['registry_models']}
        exact = rows['glm_moe_dsa']
        self.assertTrue(exact['independent_python_family'])
        self.assertTrue(exact['native_resident_range_supported'])
        self.assertFalse(exact['native_streamed_range_supported'])
        self.assertFalse(exact['native_decoder_supported'])
        self.assertTrue(rows['glm5_next_text']['native_resident_range_supported'])
        self.assertFalse(rows['glm5_next_text']['independent_python_family'])
        for name in ('glm5_moe', 'glm', 'glm5_next'):
            self.assertFalse(rows[name]['native_resident_range_supported'])
            self.assertFalse(rows[name]['independent_python_family'])

    def test_registered_text_contracts_do_not_claim_full_family_support(self):
        from openagent.adaptive_engine.native_checkpoint import _DECODERS
        report = conformance_matrix()
        contracts = {row['model_type']: row for row in report['resident_text_contracts']}
        self.assertEqual(set(contracts), set(_DECODERS))
        self.assertEqual(report['native_resident_text_decoder_families'], sorted(contracts))
        for name, row in contracts.items():
            self.assertEqual((row['module'], row['constructor']), _DECODERS[name])
            self.assertTrue(row['contract'])
            self.assertFalse(row['packed_streaming'])
            self.assertFalse(row['full_registry_family'])
            self.assertFalse(row['upstream_abi_compatible'])
            self.assertEqual(row['experimental'], name == 'kimi_linear')
        for row in report['registry_models']:
            expected = contracts.get(row['model_type'])
            self.assertEqual(row['resident_text_execution_contract'],
                             expected['contract'] if expected else None)
            self.assertEqual(row['native_resident_text_decoder_supported'], expected is not None)
        self.assertNotIn('inkling_mm_model', contracts)
        self.assertNotIn('kimi_k3', contracts)
        self.assertNotIn('glm', contracts)

    def test_qwen4_partial_contract_does_not_enable_native_or_wrapper_alias(self):
        rows = {row['model_type']: row for row in conformance_matrix()['registry_models']}
        exact = rows['qwen4_exp_text']
        self.assertTrue(exact['independent_python_family'])
        self.assertEqual(exact['independent_contract'], 'linear-QSA-PLE-default-text-RoPE-FP32-resident')
        self.assertEqual(exact['native_resident_contract'], 'linear-QSA-PLE-default-text-RoPE-FP32-resident')
        self.assertTrue(exact['native_resident_range_supported'])
        self.assertFalse(exact['native_decoder_supported'])
        self.assertFalse(rows['qwen4_exp']['independent_python_family'])
        self.assertIsNone(rows['qwen4_exp']['independent_contract'])

    def test_installed_mapping_evidence(self):
        report = conformance_matrix(probe_transformers=True)
        available = importlib.util.find_spec('transformers') is not None
        self.assertEqual(report['transformers_installed'], available)
        if not available:
            self.assertIsNone(report['transformers_version'])
            return
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        for row in report['registry_models']:
            name = row['model_type']
            self.assertEqual(row['transformers_config_class'], CONFIG_MAPPING_NAMES.get(name))
            self.assertEqual(row['transformers_causal_class'], MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(name))
            self.assertFalse(row['native_decoder_supported'])
        if report['transformers_version'] == '4.57.6':
            remaining = [row for row in report['registry_models'] if row['model_type'] != 'olmoe']
            self.assertEqual(sum(row['transformers_config_class'] is None for row in remaining), 15)
            collision = next(row for row in remaining if row['model_type'] == 'glm')
            self.assertTrue(collision['identity_collision'])

    @unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('torch'),
                         'optional installed Transformers/torch required')
    def test_dense_glm_collision_is_not_moe_architecture(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        if CONFIG_MAPPING_NAMES.get('glm') != 'GlmConfig':
            self.skipTest('installed mapping no longer declares older dense GlmConfig')
        config = AutoConfig.for_model('glm', vocab_size=16, hidden_size=8,
            intermediate_size=12, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=4, max_position_embeddings=16,
            pad_token_id=0, bos_token_id=1, eos_token_id=2)
        model = AutoModelForCausalLM.from_config(config)
        self.assertEqual(type(model).__name__, 'GlmForCausalLM')
        self.assertEqual(type(model.model.layers[0].mlp).__name__, 'GlmMLP')
        self.assertFalse(any('expert' in name for name, _ in model.named_modules()))


if __name__ == '__main__':
    unittest.main()
