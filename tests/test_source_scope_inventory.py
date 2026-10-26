from types import SimpleNamespace
import ast
import hashlib
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from openagent.adaptive_engine.source_scope import (
    require_checkpoint_inventory_admission,
    require_source_family_backend_admission,
    source_matrix_rows,
    source_matrix_row,
    source_family_for_identity,
    source_scope_matrix,
    validate_phase1_source,
    validate_source_scope,
)


class SourceScopeInventoryTest(unittest.TestCase):
    def test_colibri_capability_catalog_matches_local_sse_routes(self):
        from openagent.adaptive_engine.colibri_capabilities import colibri_capabilities

        capabilities = colibri_capabilities()['capabilities']
        self.assertIn('SSE', capabilities['openai_compatible']['scope'])
        self.assertIn('SSE', capabilities['anthropic_messages']['scope'])
        self.assertIn('JSON/SSE', capabilities['colibri_queue_contract']['scope'])
        self.assertIn('protocol-specific deltas', capabilities['openai_compatible']['scope'])
        self.assertIn('named message/content-block/terminal/error events',
                      capabilities['anthropic_messages']['scope'])
        for name, field in (('openai_compatible', 'stop'),
                            ('anthropic_messages', 'stop_sequences')):
            scope = capabilities[name]['scope']
            self.assertIn(field, scope)
            self.assertIn('configured checkpoint tokenizer', scope)
            self.assertNotIn('are rejected until tokenizer-aware', scope)
        self.assertNotIn('streaming, stream_options', capabilities['anthropic_messages']['scope'])

    def test_all_production_raw_checkpoint_reader_calls_supply_admission(self):
        adaptive_root = Path(__file__).parents[1] / 'openagent' / 'adaptive_engine'
        unguarded = []
        for path in adaptive_root.rglob('*.py'):
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = (node.func.id if isinstance(node.func, ast.Name) else
                          node.func.attr if isinstance(node.func, ast.Attribute) else None)
                if target == '_read_native_checkpoint' and not any(
                        keyword.arg == 'metadata_admission' for keyword in node.keywords):
                    unguarded.append(f'{path.relative_to(adaptive_root)}:{node.lineno}')
        self.assertEqual(unguarded, [])

    def test_every_public_execution_factory_reaches_a_route_or_inventory_gate(self):
        source = (Path(__file__).parents[1] / 'openagent' / 'adaptive_engine' /
                  'execution_factories.py').read_text(encoding='utf-8')
        module = ast.parse(source)
        functions = {node.name: node for node in module.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        public_factories = {
            name for name in functions
            if name.startswith(('create_', 'open_')) and
            ('service' in name or 'execution' in name)
        }
        gates = {
            'require_checkpoint_inventory_admission',
            'require_source_inventory_admission',
            'require_source_matrix_admission',
            'require_source_family_backend_admission',
            'require_k3_payload_admission',
            '_reject_unproven_inmemory_route',
            '_reject_unproven_glm_inmemory_route',
            '_reject_unproven_qwen35_manifest_provider',
            '_reject_unproven_qwen4_manifest_provider',
        }

        def calls(node):
            result = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                if isinstance(child.func, ast.Name):
                    result.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    result.add(child.func.attr)
            return result

        reachable = {}
        def has_gate(name, seen=()):
            if name in gates:
                return True
            if name in seen or name not in functions:
                return False
            called = calls(functions[name])
            return bool(called & gates) or any(
                has_gate(target, (*seen, name))
                for target in called if target in functions
            )

        for name in sorted(public_factories):
            reachable[name] = has_gate(name)
        self.assertTrue(public_factories, 'execution factory inventory must not be empty')
        self.assertEqual([name for name, guarded in reachable.items() if not guarded], [])

    def test_canonical_identity_map_covers_every_declared_source_family(self):
        families = source_scope_matrix()['families']
        all_identities = []
        for family, row in families.items():
            for identity in row['identities']:
                all_identities.append(identity)
                self.assertEqual(source_family_for_identity(identity), family)
        self.assertEqual(len(all_identities), len(set(all_identities)))
        with self.assertRaisesRegex(ValueError, 'absent from or ambiguous'):
            source_family_for_identity('not-a-catalog-model')

    def test_source_artifact_inventory_pins_family_provenance_gate(self):
        from openagent.adaptive_engine.platform_certification.source_inventory import AREAS

        self.assertIn('family_source_binding.py', AREAS['model-families'])
        self.assertIn('family_source_binding.py', AREAS['segment-edge'])
        self.assertIn('platform_certification/source_inventory.py', AREAS['correctness-tooling'])
        self.assertIn('wholemodel_validation/adapters.py', AREAS['correctness-tooling'])
        self.assertIn('wholemodel_validation/cache_readers.py', AREAS['correctness-tooling'])
        self.assertIn('wholemodel_validation/olmoe_native.py', AREAS['correctness-tooling'])

    def test_source_artifact_inventory_pins_every_conditional_media_owner(self):
        from openagent.adaptive_engine.platform_certification.source_inventory import AREAS

        wrappers=('qwen35_multimodal.py','qwen4_multimodal.py','glm5_multimodal.py',
            'inkling_multimodal.py','multimodal_session.py')
        for source in wrappers:
            with self.subTest(source=source):
                self.assertIn(source, AREAS['model-families'])
                self.assertIn(source, AREAS['serving'])

    def test_source_inventory_stamp_normalizes_windows_birth_time(self):
        from openagent.adaptive_engine.platform_certification.source_inventory import _stamp

        path_stat=SimpleNamespace(st_dev=1,st_ino=2,st_size=3,st_mtime_ns=4,
            st_ctime_ns=5,st_birthtime_ns=6)
        descriptor_stat=SimpleNamespace(st_dev=1,st_ino=2,st_size=3,st_mtime_ns=4,
            st_ctime_ns=7,st_birthtime_ns=6)
        self.assertEqual(_stamp(path_stat),_stamp(descriptor_stat))

        posix_stat=SimpleNamespace(st_dev=1,st_ino=2,st_size=3,st_mtime_ns=4,st_ctime_ns=5)
        self.assertEqual(_stamp(posix_stat),(1,2,3,4,5))

    def test_source_inventory_distinguishes_missing_area_owners_from_source_only_receipts(self):
        from openagent.adaptive_engine.platform_certification.source_inventory import (
            AREAS, source_inventory,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'present.py'
            source.write_bytes(b'pass\n')
            report=source_inventory(root)
            self.assertTrue(report['source_only'])
            self.assertFalse(report['executed'])
            self.assertTrue(report['areas']['model-families']['missing'])
            self.assertEqual(report['inventory_sha256'],hashlib.sha256(
                json.dumps({key:value for key,value in report.items()
                    if key!='inventory_sha256'},sort_keys=True,separators=(',',':'),
                    allow_nan=False).encode()).hexdigest())

            for entries in AREAS.values():
                for relative in entries:
                    path=root/relative
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b'pass\n')
            complete=source_inventory(root)
            self.assertTrue(all(not area['missing'] for area in complete['areas'].values()))
            self.assertFalse(complete['executed'])

    def test_pinned_artifact_hashing_uses_cross_platform_descriptor_stamp(self):
        from openagent.adaptive_engine.wholemodel_validation.runner import digest_file

        with tempfile.TemporaryDirectory() as temporary:
            artifact=Path(temporary)/'dependency.bin'
            payload=b'pinned native dependency bytes'
            artifact.write_bytes(payload)
            self.assertEqual(digest_file(artifact),hashlib.sha256(payload).hexdigest())

    def test_direct_mixtral_inventory_planner_accepts_exact_local_dense_route(self):
        from openagent.adaptive_engine.resident_family_inventory import admit_resident_inventory

        inventory = {'model.embed_tokens.weight': {'shape': (1,), 'dtype': 'F32'}}
        with patch('openagent.adaptive_engine.mixtral_inventory.admit_mixtral_inventory'):
            admit_resident_inventory(
                {'model_type': 'mixtral'}, inventory, family='mixtral',
                max_bytes=1024, backend='device')

    def test_resident_text_inventory_rejects_config_only_mtp_claims(self):
        from openagent.adaptive_engine.resident_family_inventory import admit_resident_inventory

        cases = (
            ('qwen3_5_moe_text', 'qwen3_5_moe_text', 'mtp_num_hidden_layers', 1),
            ('qwen4_exp_text', 'qwen4_exp_text', 'mtp_use_dedicated_embeddings', False),
            ('glm5_next_text', 'glm5_next_text', 'num_nextn_predict_layers', 1),
        )
        for family, model_type, field, value in cases:
            with self.subTest(family=family, field=field), self.assertRaisesRegex(
                    ValueError, 'rejects MTP configuration: no admitted draft consumer'):
                admit_resident_inventory(
                    {'model_type': model_type, field: value}, {},
                    family=family, max_bytes=1024)

        with self.assertRaisesRegex(ValueError, 'rejects MTP configuration'):
            admit_resident_inventory(
                {'model_type': 'qwen4_exp_text', 'mtp_config': {'num_layers': 1}}, {},
                family='qwen4_exp_text', max_bytes=1024)

    def test_selected_text_loading_accepts_glm5_and_inkling_public_text_routes(self):
        from openagent.adaptive_engine.native_checkpoint import (
            read_native_text_checkpoint,
        )
        inventory = {'model.embed_tokens.weight': {'shape': (1,), 'dtype': 'F32'}}

        def inspect_admission(_directory, *, model_type, metadata_admission, **_kwargs):
            return metadata_admission({'model_type': model_type}, inventory, 4)

        with patch('openagent.adaptive_engine.native_checkpoint._read_native_checkpoint',
                   side_effect=inspect_admission):
            for model_type in ('glm5_next_text','inkling_text'):
                read_native_text_checkpoint(
                    'checkpoint', model_type=model_type, max_weight_bytes=1024,
                    tensor_selection=lambda _name: True, source_backend='public-range')

    def test_inkling_checkpoint_preflight_binds_wrapper_text_and_fp8_routes(self):
        cases=(
            ('inkling_mm_model','native'),
            ('inkling_mm_model','device'),
            ('inkling_mm_model','public-range'),
            ('inkling_text','native'),
            ('inkling_text','device'),
            ('inkling_text','public-range'),
        )
        for model_type,backend in cases:
            tensors={'model.embed_tokens.weight':SimpleNamespace(dtype='F16')}
            if model_type=='inkling_mm_model':
                tensors['model.layers.0.mlp.experts.0.gate_proj.weight']=SimpleNamespace(dtype='F8_E4M3')
            metadata=SimpleNamespace(tensors=tensors)
            with self.subTest(model_type=model_type,backend=backend), patch(
                    'openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                    return_value=metadata) as snapshot:
                    result=require_checkpoint_inventory_admission(
                        'fixture',family='inkling',model_type=model_type,backend=backend,
                        max_metadata_bytes=4096)
                    self.assertIs(result,metadata)
                    snapshot.assert_called_once_with('fixture',max_metadata_bytes=4096,
                        max_tensors=1_000_000,max_shards=4096)

    def test_k3_factory_requires_all_native_formats_and_exact_rows_control_execution(self):
        with self.assertRaisesRegex(ValueError, 'P02 family/backend route is not admitted'):
            require_source_family_backend_admission(
                family='kimi-k3', model_type='kimi_k3', backend='native',
                arithmetic='fp32-approximate')
        from openagent.adaptive_engine.source_scope import require_source_matrix_admission
        require_source_matrix_admission(
            family='kimi-k3',model_type='kimi_k3',backend='native',
            arithmetic='fp32-approximate',source_format='F32',
            persisted_format='float32')
        with self.assertRaisesRegex(ValueError, 'P02 source tuple is not admitted'):
            require_source_matrix_admission(
                family='kimi-k3',model_type='kimi_k3',backend='device',
                arithmetic='bf16',source_format='BF16',
                persisted_format='float32')
        require_source_matrix_admission(
            family='kimi-k3',model_type='kimi_linear',backend='native',
            arithmetic='fp32-approximate',source_format='F32',
            persisted_format='float32')

    def test_kimi_linear_checkpoint_loader_gates_before_opening_checkpoint(self):
        from openagent.adaptive_engine.native_kimi import NativeKimiLinearDecoder
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                   side_effect=ValueError('P02 source tuple is not admitted')) as admission, \
                patch('openagent.adaptive_engine.native_checkpoint._open_local') as open_local:
            with self.assertRaisesRegex(ValueError, 'P02 source tuple is not admitted'):
                NativeKimiLinearDecoder.from_checkpoint(
                    'checkpoint', library='not-opened', capacity=8, max_bytes=4096)
        admission.assert_called_once_with(
            'checkpoint',family='kimi-k3',model_type='kimi_linear',backend='native',
            arithmetic='fp32-approximate',max_metadata_bytes=4096)
        open_local.assert_not_called()

    def test_k3_factory_checks_p02_before_library_or_executor_construction(self):
        from openagent.adaptive_engine.execution_factories import (
            K3Libraries, create_k3_execution_service,
            create_public_k3_execution_service,
        )
        from openagent.adaptive_engine.k3_execution import K3Config
        from openagent.adaptive_engine.publisher_checkpoint.k3_adapter import (
            K3ExecutionPayload,
        )
        payload = K3ExecutionPayload(
            object.__new__(K3Config), {}, {}, 'fixture', (), (), ())
        object.__setattr__(payload.config, 'expert_format', 'fp32')
        with patch('openagent.adaptive_engine.publisher_checkpoint.k3_adapter.require_k3_payload_admission'):
            with self.assertRaisesRegex(ValueError, 'missing its exact source/persisted format binding'):
                create_k3_execution_service(
                    payload, K3Libraries(), runtime_max_bytes=1)
        with self.assertRaisesRegex(ValueError, 'prepared K3 execution payload required'):
            create_public_k3_execution_service(
                None, None, runtime_library=None, model_directory='.',
                checkpoint_identity='fixture', runtime_max_bytes=1,
                max_resource_bytes=1, max_work_bytes=1, max_snapshot_bytes=1)

    def test_v41_in_memory_factories_fail_closed_before_library_open(self):
        from openagent.adaptive_engine.execution_factories import (
            create_public_publisher_v41_execution_service,
            create_publisher_v41_execution_service, publisher_v41_dspark_session,
        )
        from openagent.adaptive_engine.publisher_checkpoint.v41_adapter import (
            V41ExecutionPayload,
        )
        payload = object.__new__(V41ExecutionPayload)
        def arithmetic_factory():
            return None
        with self.assertRaisesRegex(ValueError, 'must originate from admitted load_v41 checkpoint inventory'):
            create_publisher_v41_execution_service(
                payload, arithmetic_factory, library='not-opened', runtime_max_bytes=1)
        with self.assertRaisesRegex(ValueError, 'must originate from admitted load_v41 checkpoint inventory'):
            create_public_publisher_v41_execution_service(
                payload, arithmetic_factory, library='not-opened', runtime_library='not-opened',
                model_directory='.', checkpoint_identity='fixture', runtime_max_bytes=1,
                executor_bytes=1, max_resource_bytes=1, max_work_bytes=1,
                max_snapshot_bytes=1)
        with self.assertRaisesRegex(ValueError, 'must originate from admitted load_v41 checkpoint inventory'):
            publisher_v41_dspark_session(
                payload, arithmetic_factory, library='not-opened', runtime_max_bytes=1,
                draft_max_bytes=1, evidence_max_bytes=1, sampling_max_bytes=1)

    def test_candidate_family_inmemory_factories_fail_closed_before_runtime_open(self):
        from openagent.adaptive_engine.execution_factories import (
            create_device_glm5_execution_service,
            create_device_glm_dsa_execution_service,
            create_device_inkling_execution_service,
            create_device_mixtral_execution_service,
            create_device_qwen35_execution_service,
            create_device_qwen4_execution_service,
            create_glm_dsa_execution_service,
            create_mixtral_execution_service,
            create_qwen4_execution_service,
        )
        cases = (
            ('Mixtral native', create_mixtral_execution_service,
             ({}, {}), dict(library='not-opened', runtime_max_bytes=1)),
            ('Mixtral device', create_device_mixtral_execution_service,
             ({}, {}), dict(device='not-probed', runtime_max_bytes=1)),
            ('GLM-DSA native', create_glm_dsa_execution_service,
             ({}, {}), dict(library='not-opened', runtime_max_bytes=1)),
            ('GLM-DSA device', create_device_glm_dsa_execution_service,
             ({}, {}), dict(device='not-probed', runtime_max_bytes=1,
                            max_total_model_bytes=1)),
            ('Qwen4 native', create_qwen4_execution_service,
             ({}, {}), dict(library='not-opened', runtime_max_bytes=1)),
            ('Qwen4 device', create_device_qwen4_execution_service,
             ({}, {}), dict(device='not-probed', runtime_max_bytes=1)),
            ('Qwen3.5 device', create_device_qwen35_execution_service,
             ({}, {}), dict(device='not-probed', runtime_max_bytes=1)),
            ('Inkling device', create_device_inkling_execution_service,
             ({}, {}), dict(device='not-probed', runtime_max_bytes=1,
                            max_total_model_bytes=1)),
            ('GLM5-next device', create_device_glm5_execution_service,
             ({}, {}), dict(device='not-probed', runtime_max_bytes=1,
                            max_total_model_bytes=1)),
        )
        for label, factory, positional, keywords in cases:
            with self.subTest(route=label), self.assertRaisesRegex(
                    ValueError, 'in-memory factory is restricted'):
                factory(*positional, **keywords)

    def test_streamed_mixtral_factory_uses_checkpoint_admission_before_provider_work(self):
        from openagent.adaptive_engine.execution_factories import (
            create_streamed_mixtral_directory_service,
        )
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                   side_effect=ValueError('P02 source tuple rejected')) as admission:
            with self.assertRaisesRegex(ValueError, 'P02 source tuple rejected'):
                create_streamed_mixtral_directory_service(
                    'not-opened', library='not-opened', expert_provider=None,
                    max_weight_bytes=4096, runtime_max_bytes=4096)
        admission.assert_called_once_with(
            'not-opened', family='mixtral', model_type='mixtral',
            backend='native', max_metadata_bytes=4096)

    def test_streamed_mixtral_factory_rejects_unbound_provider_before_opening_checkpoint(self):
        from openagent.adaptive_engine.execution_factories import (
            create_streamed_mixtral_directory_service,
        )
        metadata=SimpleNamespace(assert_unchanged=lambda: None)
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                   return_value=metadata) as admission:
            with self.assertRaisesRegex(ValueError,'lack checkpoint source-content identity'):
                create_streamed_mixtral_directory_service(
                    'not-opened', library='not-opened', expert_provider=object(),
                    max_weight_bytes=4096, runtime_max_bytes=4096)
        admission.assert_called_once_with(
            'not-opened', family='mixtral', model_type='mixtral',
            backend='native', max_metadata_bytes=4096)

    def test_glm_dsa_directory_factories_reject_unbound_provider_before_checkpoint_work(self):
        from openagent.adaptive_engine.execution_factories import (
            open_device_glm_dsa_directory_service,
            open_native_glm_dsa_directory_service,
        )
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission') as admission:
            cases=(
                (open_device_glm_dsa_directory_service,('not-opened',),dict(
                    device='cuda:0',max_weight_bytes=4096,runtime_max_bytes=4096,
                    max_total_model_bytes=4096,expert_provider=object())),
                (open_native_glm_dsa_directory_service,('not-opened',),dict(
                    library='not-opened',max_weight_bytes=4096,runtime_max_bytes=4096,
                    max_total_model_bytes=4096,context_tokens=128,expert_provider=object())),
            )
            for factory,args,kwargs in cases:
                with self.subTest(factory=factory.__name__), self.assertRaisesRegex(
                        ValueError,'lack checkpoint source-content identity'):
                    factory(*args,**kwargs)
        admission.assert_not_called()

    def test_glm_dsa_device_loader_rejects_unbound_provider_before_header_reads(self):
        from openagent.adaptive_engine.torch_glm_dsa_decoder import TorchGlmDsaDecoder
        with patch('openagent.adaptive_engine.native_checkpoint.read_native_text_checkpoint') as read:
            with self.assertRaisesRegex(ValueError,'lack checkpoint source-content identity'):
                TorchGlmDsaDecoder.from_directory(
                    'not-opened',device='cuda:0',max_weight_bytes=4096,
                    max_bytes=4096,expert_provider=object())
        read.assert_not_called()

    def test_gateway_k3_residency_applies_cli_exact_route_admission(self):
        from openagent.adaptive_engine.publisher_provider import NativePublisherProvider
        with tempfile.TemporaryDirectory() as checkpoint:
            args = SimpleNamespace(checkpoint=checkpoint,
                                   arithmetic_device=None, fla_version=None)
            with patch('openagent.adaptive_engine.source_scope.require_source_family_backend_admission',
                       side_effect=ValueError('K3 tuple rejected')) as admission:
                with self.assertRaisesRegex(ValueError, 'K3 tuple rejected'):
                    NativePublisherProvider._residency_key(
                        {'family': 'kimi_k3'}, args)
        admission.assert_called_once_with(
            family='kimi-k3', model_type='kimi_k3', backend='native',
            arithmetic='fp32-approximate')

    def test_gateway_v41_residency_checks_each_local_bank_before_fingerprinting(self):
        from openagent.adaptive_engine.publisher_provider import NativePublisherProvider
        with tempfile.TemporaryDirectory() as checkpoint:
            args = SimpleNamespace(checkpoint=checkpoint, device=None)
            with patch('openagent.adaptive_engine.source_scope.require_source_matrix_admission',
                       side_effect=ValueError('V4.1 bank tuple rejected')) as admission:
                with self.assertRaisesRegex(ValueError, 'V4.1 bank tuple rejected'):
                    NativePublisherProvider._residency_key(
                        {'family': 'deepseek_v41'}, args)
        admission.assert_called_once_with(
            family='deepseek-v41', model_type='deepseek_v41',
            backend='native', arithmetic='publisher-bf16-fp8-fp4',
            source_format='BF16', persisted_format='bf16')

    def test_gateway_availability_is_false_for_unverified_publisher_route(self):
        from openagent.adaptive_engine.publisher_provider import NativePublisherProvider
        provider = NativePublisherProvider()
        with tempfile.TemporaryDirectory() as checkpoint:
            args = SimpleNamespace(checkpoint=checkpoint,
                                   arithmetic_device=None, fla_version=None)
            with patch.object(provider, '_configuration',
                              return_value={'family': 'kimi_k3'}), \
                    patch.object(NativePublisherProvider, '_operator',
                                 return_value=(object(), args)), \
                    patch('openagent.adaptive_engine.source_scope.require_source_family_backend_admission',
                          side_effect=ValueError('unverified K3 route')) as admission:
                self.assertFalse(provider.available())
        admission.assert_called_once()

    def test_direct_non_inkling_wrappers_reject_config_only_mtp_claims(self):
        from openagent.adaptive_engine.publisher_family_contracts import partition_weights
        weights = {
            'model.embed_tokens.weight': object(),
            'lm_head.weight': object(),
            'model.visual.patch.weight': object(),
        }
        for wrapper in ('qwen3_5_moe', 'qwen4_exp', 'glm5_next'):
            components = SimpleNamespace(
                variant=SimpleNamespace(wrapper=wrapper),
                text_config={'tie_word_embeddings': False},
                audio_config=None,
                auxiliary_config={'num_nextn_predict_layers': 1},
            )
            with self.subTest(wrapper=wrapper), self.assertRaisesRegex(
                    ValueError, 'MTP configuration requires the admitted Inkling'):
                partition_weights(weights, components)

    def test_inkling_chain_norm_bank_is_admitted_as_shared_mtp(self):
        from openagent.adaptive_engine.publisher_conditional_loader import _inkling_mtp_headers
        from openagent.adaptive_engine.publisher_family_contracts import partition_weights

        weights = {
            'model.embed_tokens.weight': object(),
            'lm_head.weight': object(),
            'model.visual.patch.weight': object(),
            'chain_norm.weight': object(),
        }
        components = SimpleNamespace(
            variant=SimpleNamespace(wrapper='inkling_mm_model'),
            text_config={'tie_word_embeddings': False}, audio_config=None,
            auxiliary_config={'num_mtp_layers': 1, 'chain_hidden_post_norm': True})
        _text, _vision, _audio, auxiliary = partition_weights(weights, components)
        self.assertIs(auxiliary['chain_norm.weight'], weights['chain_norm.weight'])

        inventory = SimpleNamespace(tensors={
            'chain_norm.weight': SimpleNamespace(shape=(4,), dtype='F32'),
            'mtp.layers.0.hnorm.weight': SimpleNamespace(shape=(4,), dtype='F32'),
            'model.layers.0.weight': SimpleNamespace(shape=(4, 4), dtype='F32'),
        })
        headers = _inkling_mtp_headers(inventory)
        self.assertEqual(headers['chain_norm.weight'], ((4,), 'F32'))
        self.assertNotIn('model.layers.0.weight', headers)
        alternate_namespace = _inkling_mtp_headers(SimpleNamespace(tensors={
            'model.layers.0.mtpBlock.attn.q_proj.weight':
                SimpleNamespace(shape=(2, 2), dtype='F32')}))
        self.assertIn('model.layers.0.mtpBlock.attn.q_proj.weight', alternate_namespace)

        from openagent.adaptive_engine import independent_inkling, inkling_mtp
        class Planner:
            def __init__(self, _config, _weights, *, _inventory_only):
                self.tensor_shapes = {
                    'model.layers.0.' + name: (2,)
                    for name in independent_inkling.LAYER_NAMES}

        complete_headers = {'chain_norm.weight': ((4,), 'F32'),
            'layers.0.enorm.weight': ((4,), 'F32'),
            'layers.0.hnorm.weight': ((4,), 'F32'),
            'layers.0.eh_proj.weight': ((4, 8), 'F32')}
        complete_headers.update({
            'layers.0.mtp_block.' + name: ((2,), 'F32')
            for name in independent_inkling.LAYER_NAMES})
        with patch.object(independent_inkling, 'IndependentInklingDecoder', Planner):
            inkling_mtp.admit_mtp_inventory(
                {'hidden_size': 4, 'vocab_size': 8},
                {'num_mtp_layers': 1, 'chain_hidden_post_norm': True},
                complete_headers, max_total_bytes=1 << 20)
            with self.assertRaisesRegex(ValueError, 'complete original Inkling MTP inventory differs'):
                inkling_mtp.admit_mtp_inventory(
                    {'hidden_size': 4, 'vocab_size': 8},
                    {'num_mtp_layers': 1, 'chain_hidden_post_norm': True},
                    alternate_namespace, max_total_bytes=1 << 20)

    def test_publisher_media_aliases_are_canonicalized_for_exact_consumers(self):
        from openagent.adaptive_engine.publisher_family_contracts import partition_weights

        vision_value = object()
        audio_value = object()
        weights = {
            'model.embed_tokens.weight': object(),
            'lm_head.weight': object(),
            'model.language_model.vision_tower.final_norm.weight': vision_value,
            'model.language_model.audio_tower.norm.weight': audio_value,
        }
        components = SimpleNamespace(
            variant=SimpleNamespace(wrapper='inkling_mm_model'),
            text_config={'tie_word_embeddings': False},
            audio_config={}, auxiliary_config={})

        _text, vision, audio, _auxiliary = partition_weights(weights, components)

        self.assertEqual(set(vision), {'model.vision_tower.final_norm.weight'})
        self.assertEqual(set(audio), {'model.audio_tower.norm.weight'})
        self.assertIs(vision['model.vision_tower.final_norm.weight'], vision_value)
        self.assertIs(audio['model.audio_tower.norm.weight'], audio_value)

    def test_publisher_media_alias_collision_is_rejected_during_partition(self):
        from openagent.adaptive_engine.publisher_family_contracts import partition_weights

        weights = {
            'model.embed_tokens.weight': object(),
            'lm_head.weight': object(),
            'model.vision_tower.final_norm.weight': object(),
            'model.language_model.vision_tower.final_norm.weight': object(),
        }
        components = SimpleNamespace(
            variant=SimpleNamespace(wrapper='inkling_mm_model'),
            text_config={'tie_word_embeddings': False},
            audio_config=None, auxiliary_config={})

        with self.assertRaisesRegex(ValueError, 'duplicate mapped publisher checkpoint tensor'):
            partition_weights(weights, components)

    def test_publisher_component_contract_rejects_unknown_mtp_config_claims(self):
        from openagent.adaptive_engine.publisher_family_contracts import component_contract
        configs = (
            {'model_type': 'qwen3_5_moe', 'vision_config': {},
             'text_config': {'model_type': 'qwen3_5_moe_text',
                             'mtp_prediction_depth': 2}},
            {'model_type': 'qwen4_exp', 'vision_config': {},
             'text_config': {'model_type': 'qwen4_exp_text', 'hidden_size': 8,
                             'layer_types': ['full_attention'],
                             'mtp_prediction_depth': 2}},
            {'model_type': 'inkling_mm_model', 'vision_config': {},
             'text_config': {'model_type': 'inkling_text', 'num_hidden_layers': 1,
                             'local_layer_ids': [0], 'dense_mlp_idx': 0,
                             'unowned_nextn_prediction_depth': 2}},
        )
        for config in configs:
            with self.subTest(family=config['model_type']), self.assertRaisesRegex(
                    ValueError, 'unadmitted publisher MTP/next-token configuration'):
                component_contract(config, arithmetic='fp32')

    def test_direct_non_inkling_wrappers_reject_mtp_tensor_banks(self):
        from openagent.adaptive_engine.publisher_family_contracts import partition_weights
        from openagent.adaptive_engine.publisher_family_contracts import is_mtp_tensor_name
        mtp_names = ('mtp.layers.0.projection.weight',
                     'model.layers.0.mtp_block.attn.q_proj.weight',
                     'model.layers.0.mtpBlock.attn.q_proj.weight')
        self.assertTrue(is_mtp_tensor_name(mtp_names[1]))
        self.assertFalse(is_mtp_tensor_name('model.layers.0.self_attn.q_proj.weight'))
        for mtp_name in mtp_names:
            weights = {
                'model.embed_tokens.weight': object(),
                'lm_head.weight': object(),
                'model.visual.patch.weight': object(),
                mtp_name: object(),
            }
            for wrapper in ('qwen3_5_moe', 'qwen4_exp', 'glm5_next'):
                components = SimpleNamespace(
                    variant=SimpleNamespace(wrapper=wrapper),
                    text_config={'tie_word_embeddings': False},
                    audio_config=None,
                    auxiliary_config={},
                )
                with self.subTest(wrapper=wrapper, tensor=mtp_name), self.assertRaisesRegex(
                        ValueError, 'MTP tensors require the admitted Inkling'):
                    partition_weights(weights, components)

    def test_qwen_conditional_constructors_reject_unconsumed_mtp_banks(self):
        from openagent.adaptive_engine import qwen35_multimodal, qwen4_multimodal
        for module, decoder in (
                (qwen35_multimodal, qwen35_multimodal.Qwen35MultimodalDecoder),
                (qwen4_multimodal, qwen4_multimodal.Qwen4MultimodalDecoder)):
            wrapper = ('qwen3_5_moe' if decoder is qwen35_multimodal.Qwen35MultimodalDecoder
                       else 'qwen4_exp')
            components = SimpleNamespace(
                variant=SimpleNamespace(wrapper=wrapper),
                original_config={}, text_config={}, vision_config={},
                auxiliary_config={}, audio_config=None)
            with self.subTest(decoder=decoder.__name__), \
                    patch.object(module, 'component_contract', return_value=components), \
                    patch.object(module, 'partition_weights', return_value=(
                        {}, {}, {}, {'mtp.layers.0.projection.weight': object()})):
                with self.assertRaisesRegex(ValueError, 'no admitted draft consumer'):
                    decoder({}, {}, library=None, text_library=None, max_bytes=1024,
                            **({'context_tokens': 8} if decoder is qwen35_multimodal.Qwen35MultimodalDecoder else {}))

    def test_qwen4_public_wrapper_requires_inventory_receipt_before_native_registration(self):
        from openagent.adaptive_engine import qwen4_multimodal
        components = SimpleNamespace(
            variant=SimpleNamespace(wrapper='qwen4_exp'), original_config={},
            text_config={}, vision_config={}, auxiliary_config={}, audio_config=None)
        with patch.object(qwen4_multimodal, 'component_contract', return_value=components), \
                patch.object(qwen4_multimodal, 'partition_weights', return_value=({}, {}, {}, {})):
            with self.assertRaisesRegex(ValueError, 'publisher loader checkpoint inventory receipt'):
                qwen4_multimodal.Qwen4MultimodalDecoder(
                    {}, {}, library=None, text_library=None, max_bytes=1024,
                    native_composition=True, segment_runtime_library='runtime',
                    segment_model_directory='.', segment_work_bytes=128)

    def test_glm5_inventory_rejects_config_only_mtp_before_tensor_planning(self):
        from openagent.adaptive_engine.publisher_glm5_loader import (
            admit_publisher_glm5_inventory,
        )
        components = SimpleNamespace(
            variant=SimpleNamespace(wrapper='glm5_next'),
            original_config={},
            auxiliary_config={'num_nextn_predict_layers': 1},
            audio_config=None,
        )
        inventory = SimpleNamespace(config={}, tensors={})
        with self.assertRaisesRegex(ValueError, 'GLM5 publisher rejects auxiliary/MTP configuration'):
            admit_publisher_glm5_inventory(inventory, components)

    def test_glm5_inventory_rejects_per_layer_mtp_namespace_before_mapping(self):
        from openagent.adaptive_engine.publisher_glm5_loader import (
            admit_publisher_glm5_inventory,
        )
        from openagent.adaptive_engine.independent_glm5_next import Glm5NextTextContract
        components = SimpleNamespace(
            variant=SimpleNamespace(wrapper='glm5_next'), original_config={},
            text_config={}, vision_config={}, audio_config=None, auxiliary_config={})
        contract = SimpleNamespace(tie_word_embeddings=False, vocab_size=1,
                                   hidden_size=1, mlp_layer_types=('dense',),
                                   n_routed_experts=0, moe_intermediate_size=1)
        inventory = SimpleNamespace(
            config={}, tensors={'model.layers.0.mtp_block.attn.weight':
                                SimpleNamespace(dtype='F32', shape=(1,))})
        with patch.object(Glm5NextTextContract, 'parse', return_value=contract), \
                patch('openagent.adaptive_engine.publisher_glm5_loader.tensor_layout',
                      return_value=({}, {})), \
                patch('openagent.adaptive_engine.glm5_vision.vision_shapes',
                      return_value=({}, {})), \
                patch('openagent.adaptive_engine.source_scope.require_source_inventory_admission'):
            with self.assertRaisesRegex(ValueError, 'GLM5 publisher loader rejects MTP tensors'):
                admit_publisher_glm5_inventory(inventory, components)

    def test_conditional_service_rejects_unverified_inkling_mtp_before_checkpoint_open(self):
        from openagent.adaptive_engine.execution_factories import (
            open_native_conditional_directory_service,
        )
        with self.assertRaisesRegex(ValueError, 'generated drafts are target-verified'):
            open_native_conditional_directory_service(
                'not-opened', family='inkling_mm_model', library=None,
                text_library=None, context_tokens=8, max_weight_bytes=4096,
                runtime_max_bytes=8192, max_total_model_bytes=8192,
                max_media_bytes=1024, mtp_max_bytes=512, mtp_work_bytes=128)

    def test_segment_edge_factory_requires_admitted_explicit_route(self):
        from openagent.adaptive_engine.execution_factories import (
            create_segment_edge_execution_service,
        )
        # The public conditional wrapper route is now source-bound for the
        # ordinary Qwen4 publisher identity, so use the text identity's
        # deliberately unsupported F8 decoder tuple to prove admission runs
        # before range validation.
        inventory = SimpleNamespace(tensors={
            'model.layers.0.weight': SimpleNamespace(dtype='F8_E4M3'),
        })
        with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                   return_value=inventory):
            with self.assertRaisesRegex(ValueError, 'P02 source tuple is not admitted'):
                create_segment_edge_execution_service(
                    None, edge_adapter='fixture', model_directory='.',
                    checkpoint_identity='fixture', ranges=(), vocabulary=1,
                    context_tokens=1, native_memory_bytes=1,
                    source_family='qwen4', model_type='qwen4_exp_text',
                    arithmetic='fp32')

    def test_segment_edge_factory_binds_the_checkpoint_inventory(self):
        from openagent.adaptive_engine.execution_factories import (
            create_segment_edge_execution_service,
        )
        with patch('openagent.adaptive_engine.source_scope.require_source_family_backend_admission') as family_admission, \
                patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission') as inventory:
            with self.assertRaisesRegex(ValueError, 'bounded native layer range inventory'):
                create_segment_edge_execution_service(
                    None, edge_adapter='fixture', model_directory='checkpoint',
                    checkpoint_identity='fixture', ranges=(), vocabulary=1,
                    context_tokens=1, native_memory_bytes=1,
                    source_family='qwen4', model_type='qwen4_exp_text',
                    arithmetic='fp32')
        inventory.assert_called_once_with('checkpoint',family='qwen4',
            model_type='qwen4_exp_text',backend='public-range',arithmetic='fp32',
            max_metadata_bytes=256*1024**2)
        family_admission.assert_not_called()

    def test_public_family_directory_factory_uses_and_owns_exact_text_route(self):
        from openagent.adaptive_engine.execution_factories import (
            open_native_family_public_directory_service,
        )
        for family, source_family, context in (
                ('glm5_next_text','glm5-next',128),
                ('inkling_text','inkling',None)):
            decoder=SimpleNamespace(close=Mock(),config={'max_position_embeddings':128})
            service=object()
            with self.subTest(family=family), \
                    patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission') as admission, \
                    patch('openagent.adaptive_engine.native_checkpoint.load_native_text_decoder',
                          return_value=decoder) as loader, \
                    patch('openagent.adaptive_engine.execution_factories.create_native_family_public_execution_service',
                          return_value=service) as create:
                result=open_native_family_public_directory_service(
                    'checkpoint',runtime_library='public-runtime',kernel_library='family-kernel',
                    family=family,max_weight_bytes=4096,runtime_max_bytes=8192,
                    max_total_model_bytes=16384,context_tokens=128,max_work_bytes=2048)
                self.assertIs(result,service)
                admission.assert_called_once_with(
                    'checkpoint',family=source_family,model_type=family,
                    backend='public-range',arithmetic='fp32',max_metadata_bytes=4096)
                loader.assert_called_once_with(
                    'checkpoint',model_type=family,library='family-kernel',
                    max_weight_bytes=4096,max_bytes=8192,context_tokens=context,
                    source_backend='public-range')
                self.assertIs(create.call_args.args[1],decoder)
                self.assertTrue(create.call_args.kwargs['own_decoder'])
                decoder.close.assert_not_called()

    def test_public_family_directory_factory_releases_decoder_if_composition_fails(self):
        from openagent.adaptive_engine.execution_factories import (
            open_native_family_public_directory_service,
        )
        decoder=SimpleNamespace(close=Mock(),config={'max_position_embeddings':128})
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission'), \
                patch('openagent.adaptive_engine.native_checkpoint.load_native_text_decoder',
                      return_value=decoder), \
                patch('openagent.adaptive_engine.execution_factories.create_native_family_public_execution_service',
                      side_effect=RuntimeError('composition failed')):
            with self.assertRaisesRegex(RuntimeError,'composition failed'):
                open_native_family_public_directory_service(
                    'checkpoint',runtime_library='public-runtime',kernel_library='family-kernel',
                    family='glm5_next_text',max_weight_bytes=4096,runtime_max_bytes=8192,
                    max_total_model_bytes=16384,context_tokens=128,max_work_bytes=2048)
        decoder.close.assert_called_once_with()

    def test_native_family_operator_routes_public_runtime_for_admitted_text_identities(self):
        from openagent.adaptive_engine.native_family_cli import _FlatOwner, _source_backend, parser
        args=parser().parse_args([
            '--family','glm5_next_text','--checkpoint','checkpoint',
            '--public-runtime-library','public-runtime','--kernel-library','family-kernel',
            '--context','128','--runtime-mib','64','--staging-mib','128',
            '--snapshot-mib','8','--transfer-mib','8','--layers-per-range','2',
            '--token-ids','[1,2]'])
        self.assertEqual(args.public_runtime_library,Path('public-runtime'))
        self.assertEqual(_source_backend(args),'public-range')
        owner=_FlatOwner(Path('checkpoint'),args)
        service=object()
        with patch('openagent.adaptive_engine.execution_factories.open_native_family_public_directory_service',
                   return_value=service) as factory:
            result=owner.create_execution_service({'model_type':'glm5_next_text'},
                context_tokens=128,max_new_tokens=32,max_sessions=2,
                prefill_chunk_tokens=16,max_snapshots_per_session=4,
                max_transfer_bytes=8*1024**2)
        self.assertIs(result,service)
        self.assertEqual(factory.call_args.args,(Path('checkpoint'),))
        self.assertEqual(factory.call_args.kwargs['runtime_library'],Path('public-runtime'))
        self.assertEqual(factory.call_args.kwargs['kernel_library'],Path('family-kernel'))
        self.assertEqual(factory.call_args.kwargs['max_work_bytes'],64*1024**2)

    def test_flat_native_cli_rejects_unconsumed_runtime_library(self):
        from openagent.adaptive_engine.native_family_cli import PublisherSession, parser

        args=parser().parse_args([
            '--family','glm5_next_text','--checkpoint','checkpoint',
            '--runtime-library','unconsumed-runtime','--kernel-library','family-kernel',
            '--context','128','--runtime-mib','64','--staging-mib','128',
            '--snapshot-mib','8','--transfer-mib','8','--layers-per-range','2',
            '--token-ids','[1,2]'])
        with self.assertRaisesRegex(ValueError,'does not consume --runtime-library'):
            PublisherSession(args)

    def test_public_family_composition_preflights_the_public_range_tuple(self):
        from openagent.adaptive_engine.execution_factories import (
            create_native_family_public_execution_service,
        )
        from openagent.adaptive_engine.native_glm5_next import NativeGlm5NextTextDecoder
        decoder=object.__new__(NativeGlm5NextTextDecoder)
        owner=SimpleNamespace(edge_id='public-edge',segment_id='public-segment',
            root=Path('checkpoint'),identity='checkpoint-pin',layers=3,
            vocabulary=17,context=32,close=Mock())
        service=SimpleNamespace(_factory_cleanup=Mock())
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission') as admission, \
                patch('openagent.adaptive_engine.native_family_public_adapters.NativeFamilyPublicAdapters',
                      return_value=owner), \
                patch('openagent.adaptive_engine.execution_factories.create_segment_edge_execution_service',
                      return_value=service):
            result=create_native_family_public_execution_service(
                'public-runtime',decoder,model_directory='checkpoint',
                max_resource_bytes=8192,max_work_bytes=1024)
        self.assertIs(result,service)
        admission.assert_called_once_with('checkpoint',family='glm5-next',
            model_type='glm5_next_text',backend='public-range',
            max_metadata_bytes=8192)
        self.assertIs(service.native_model_owner,owner)

    def test_public_family_text_matrix_rows_use_identity_specific_media_restrictions(self):
        for family,model_type in (
                ('glm5-next','glm5_next_text'),('inkling','inkling_text')):
            row=source_matrix_row(family=family,model_type=model_type,
                backend='public-range',arithmetic='fp32',source_format='F16',
                persisted_format='float32')
            self.assertEqual(row['admission'],'verified-admitted')
            self.assertEqual(row['cells']['media']['classification'],
                'contract-restriction')
            self.assertEqual(row['cells']['media']['references'],
                ('native_family_public_adapters.NativeFamilyPublicAdapters',))
            self.assertIn('native_family_public_adapters.NativeFamilyPublicAdapters',
                row['callable_chain']['native_abi'])

    def test_mixed_checkpoint_cannot_hide_unsupported_float_encoding(self):
        inventory = SimpleNamespace(tensors={
            "resident.weight": SimpleNamespace(dtype="BF16"),
            "optional.weight": SimpleNamespace(dtype="F8_E5M2"),
        })
        with patch("openagent.adaptive_engine.publisher_checkpoint.storage.snapshot",
                   return_value=inventory):
            with self.assertRaisesRegex(ValueError, "unsupported floating source encoding"):
                require_checkpoint_inventory_admission(
                    "unused", family="qwen4", model_type="qwen4_exp",
                    backend="native", max_metadata_bytes=4096)

    def test_exact_local_publisher_and_text_routes_pass_checkpoint_gate(self):
        routes = (
            ('qwen35','qwen3_5_moe','native',{'F32','F8_E4M3'}),
            ('qwen4','qwen4_exp','device',{'BF16','F8_E4M3'}),
            ('glm5-next','glm5_next','native',{'F16','F8_E4M3'}),
            ('qwen35','qwen3_5_moe_text','device',{'BF16'}),
            ('qwen4','qwen4_exp_text','native',{'F16'}),
            ('glm5-next','glm5_next_text','device',{'F32'}),
            ('glm5-next','glm5_next_text','public-range',{'F32','F16','BF16'}),
        )
        for family,model_type,backend,dtypes in routes:
            with self.subTest(route=(family,model_type,backend)):
                inventory=SimpleNamespace(tensors={
                    f'tensor-{index}':SimpleNamespace(dtype=dtype)
                    for index,dtype in enumerate(sorted(dtypes))
                })
                with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                           return_value=inventory):
                    admitted=require_checkpoint_inventory_admission(
                        'unused',family=family,model_type=model_type,backend=backend,
                        max_metadata_bytes=4096)
                self.assertIs(admitted,inventory)

    def test_text_only_local_route_rejects_unimplemented_fp8_conversion(self):
        inventory=SimpleNamespace(tensors={
            'resident.weight':SimpleNamespace(dtype='F8_E4M3'),
        })
        with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                   return_value=inventory):
            with self.assertRaisesRegex(ValueError,'P02 source tuple is not admitted'):
                require_checkpoint_inventory_admission(
                    'unused',family='qwen4',model_type='qwen4_exp_text',
                    backend='device',max_metadata_bytes=4096)

    def test_checkpoint_preflight_allows_only_named_e8m0_fp8_scale_companions(self):
        inventory=SimpleNamespace(tensors={
            'model.layers.0.proj.weight':SimpleNamespace(dtype='F8_E4M3'),
            'model.layers.0.proj.weight_scale_inv':SimpleNamespace(dtype='F8_E8M0'),
        })
        with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                   return_value=inventory):
            admitted=require_checkpoint_inventory_admission(
                'unused',family='qwen4',model_type='qwen4_exp',backend='native',
                max_metadata_bytes=4096)
        self.assertIs(admitted,inventory)

        for scale_name in ('orphan.weight_scale_inv','model.layers.0.other_scale_inv'):
            with self.subTest(scale_name=scale_name):
                invalid=SimpleNamespace(tensors={
                    'model.layers.0.proj.weight':SimpleNamespace(dtype='F8_E4M3'),
                    scale_name:SimpleNamespace(dtype='F8_E8M0'),
                })
                with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                           return_value=invalid):
                    with self.assertRaisesRegex(ValueError,
                            'unsupported non-floating source encoding'):
                        require_checkpoint_inventory_admission(
                            'unused',family='qwen4',model_type='qwen4_exp',
                            backend='native',max_metadata_bytes=4096)

    def test_qwen35_native_dense_row_names_direct_multimodal_loader_chain(self):
        row=source_matrix_row(family='qwen35',model_type='qwen3_5_moe',
            backend='native',arithmetic='fp32',source_format='F32',
            persisted_format='float32')
        self.assertEqual(row['admission'],'verified-admitted')
        self.assertIn('native_checkpoint.load_native_multimodal_decoder',
                      row['callable_chain']['loading'])
        self.assertIn('native_checkpoint._admit_qwen35_multimodal_source',
                      row['callable_chain']['loading'])
        self.assertIn('native_qwen35_vision.NativeQwen35Vision',
                      row['callable_chain']['execution'])
        self.assertIn('native_qwen35_vision.c',row['callable_chain']['native_abi'])
        fp8_row=source_matrix_row(family='qwen35',model_type='qwen3_5_moe',
            backend='native',arithmetic='fp32',source_format='F8_E4M3',
            persisted_format='fp8-e4m3-b128-f32')
        self.assertNotIn('native_checkpoint.load_native_multimodal_decoder',
                         fp8_row['callable_chain']['loading'])

    def test_olmoe_manifest_rows_name_the_selected_leased_consumers(self):
        from openagent.adaptive_engine.source_scope import source_matrix_row

        native=source_matrix_row(family='olmoe',model_type='olmoe',backend='native',
            arithmetic='fp32',source_format='BF16',persisted_format='q4_0')
        self.assertEqual(native['admission'],'verified-admitted')
        self.assertIn('native_leased_olmoe_segments.NativeLeasedOlmoeSegmentEngine',
                      native['callable_chain']['execution'])
        self.assertIn('native_segment_executor.NativeSegmentExecutor',
                      native['callable_chain']['execution'])
        self.assertIn('native/segment_executor.c',native['callable_chain']['native_abi'])

        public=source_matrix_row(family='olmoe',model_type='olmoe',backend='public-range',
            arithmetic='fp32',source_format='BF16',persisted_format='int8-row')
        self.assertEqual(public['admission'],'verified-admitted')
        self.assertIn('segment_edge_adapters.NativeOlmoePublicAdapters',
                      public['callable_chain']['native_abi'])

    def test_checkpoint_preflight_rejects_unmapped_integer_and_packed_encodings(self):
        for dtype in ('I32','U8','BOOL',None):
            with self.subTest(dtype=dtype):
                inventory=SimpleNamespace(tensors={
                    'resident.weight':SimpleNamespace(dtype='BF16'),
                    'unmapped.tensor':SimpleNamespace(dtype=dtype),
                })
                with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                           return_value=inventory):
                    with self.assertRaisesRegex(ValueError,'unsupported non-floating source encoding'):
                        require_checkpoint_inventory_admission(
                            'unused',family='qwen4',model_type='qwen4_exp',
                            backend='native',max_metadata_bytes=4096)

    def test_olmoe_native_edge_safetensors_dtype_resolves_to_exact_admitted_row(self):
        row=source_matrix_row(
            family='olmoe',model_type='olmoe',backend='native',arithmetic='fp32',
            source_format='F32',persisted_format='float32')
        self.assertEqual(row['admission'],'verified-admitted')
        self.assertEqual(row['source_review'],
            'p02-olmoe-native-edge-dense-fp32-chain-2026-10-18')
        self.assertEqual(row['source_dtype'],'F32')
        inventory=SimpleNamespace(tensors={
            'model.embed_tokens.weight':SimpleNamespace(dtype='F32'),
        })
        with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                   return_value=inventory):
            admitted=require_checkpoint_inventory_admission(
                'unused',family='olmoe',model_type='olmoe',backend='native',
                max_metadata_bytes=4096)
        self.assertIs(admitted,inventory)

    def test_olmoe_checkpoint_preflight_binds_manifest_persisted_format(self):
        inventory=SimpleNamespace(tensors={
            'model.embed_tokens.weight':SimpleNamespace(dtype='BF16'),
            'model.layers.0.mlp.experts.0.gate_proj.weight':SimpleNamespace(dtype='BF16'),
        })
        with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                   return_value=inventory):
            admitted=require_checkpoint_inventory_admission(
                'unused',family='olmoe',model_type='olmoe',backend='native',
                persisted_format='q4_0',max_metadata_bytes=4096)
        self.assertIs(admitted,inventory)

    def test_wholemodel_runner_requires_dependency_artifact_pin(self):
        from openagent.adaptive_engine.wholemodel_validation.runner import pinned_artifacts
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[]
            for kind in ('source','config','weights','tokenizer'):
                path=root/(kind+'.bin')
                path.write_bytes(kind.encode())
                rows.append(dict(kind=kind,path=path.name,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            with self.assertRaisesRegex(ValueError,'dependency pins required'):
                pinned_artifacts(dict(artifacts=rows),root)

    def test_wholemodel_runner_rejects_unvalidated_artifact_claim_fields(self):
        from openagent.adaptive_engine.wholemodel_validation.runner import pinned_artifacts
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[]
            for kind in ('source','config','weights','tokenizer','dependency'):
                path=root/(kind+'.bin')
                path.write_bytes(kind.encode())
                rows.append(dict(kind=kind,path=path.name,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            rows[0]['runtime_evidence_verified']=True
            with self.assertRaisesRegex(ValueError,'source/config/weights/tokenizer/dependency pins required'):
                pinned_artifacts(dict(artifacts=rows),root)

    def test_complete_source_matrix_catalog_materializes_and_validates(self):
        rows = source_matrix_rows()
        self.assertGreater(len(rows), 0)
        self.assertTrue(all(row["cells"].get("execution") for row in rows))
        scope = source_scope_matrix()
        self.assertTrue(validate_source_scope(scope))
        self.assertTrue(validate_phase1_source(scope=scope))

    def test_phase1_source_validator_accepts_repository_root(self):
        repository = Path(__file__).parents[1]
        self.assertTrue(validate_phase1_source(root=repository))

    def test_every_unverified_generated_execution_tuple_fails_closed(self):
        from openagent.adaptive_engine.source_scope import (
            require_source_family_backend_admission,
            require_source_matrix_admission,
        )

        rows=source_matrix_rows()
        self.assertTrue(rows)
        admitted=source_matrix_rows(admitted_only=True)
        self.assertEqual(len(admitted),171)
        mixtral = [row for row in admitted if row['family']=='mixtral']
        glm_dsa = [row for row in admitted if row['family']=='glm-dsa']
        inkling = [row for row in admitted if row['family']=='inkling']
        v41 = [row for row in admitted if row['family']=='deepseek-v41']
        local_publishers = [row for row in admitted if row['family'] in (
            'qwen35','qwen4','glm5-next') and row['backend'] in ('native','device')]
        olmoe = [row for row in admitted if row['family']=='olmoe']
        self.assertEqual(len(mixtral),18)
        self.assertTrue(all(row['model_type']=='mixtral' and
            row['backend'] in ('native','device','public-range') and
            row['arithmetic']=='fp32' and row['persisted_format']=='float32' and
            row['source_review']=='p02-mixtral-local-dense-fp32-chain-2026-10-18'
            for row in mixtral))
        self.assertEqual(len(glm_dsa),9)
        self.assertTrue(all(row['model_type']=='glm_moe_dsa' and
            row['backend'] in ('native','device','public-range') and
            row['arithmetic']=='fp32' and row['persisted_format']=='float32' and
            row['source_review']=='p02-glm-dsa-local-dense-fp32-chain-2026-10-18'
            for row in glm_dsa))
        self.assertEqual(len(inkling),21)
        self.assertTrue(all(row['backend'] in ('native','device','public-range') and
            row['arithmetic']=='fp32' and row['source_review'].startswith(
                'p02-inkling-local-') for row in inkling))
        self.assertEqual(len(v41),9)
        self.assertTrue(all(row['model_type']=='deepseek_v41' and
            row['backend'] in ('native','device','public-range') and
            row['source_review']=='p02-v41-local-bank-chain-2026-10-18'
            for row in v41))
        self.assertEqual(len(local_publishers),42)
        self.assertTrue(all(row['backend'] in ('native','device') and
            row['arithmetic']=='fp32' and row['source_review'].startswith(
                'p02-'+row['family']+'-local-')
            for row in local_publishers))
        self.assertEqual(len(olmoe),45)
        self.assertTrue(all(row['backend'] in ('native','public-range') and
            row['persisted_format'] in ({'float32','int8-row','int2-row',
                'int3-grouped-g64','int4-row','int4-grouped-g64',
                'mxfp4-e2m1-g32-ue8m0','q4_0','q4_1','q5_0','q5_1',
                'q8_0','q8_1','fp8-e4m3-b128-f32'} if row['backend']=='native'
                else {'int8-row'}) and
            row['source_review']==(
                'p02-olmoe-native-edge-dense-fp32-chain-2026-10-18'
                if row['persisted_format']=='float32' else
                'p02-olmoe-direct-gateway-manifest-consumer-chain-2026-10-23'
                if row['backend']=='native' else
                'p02-olmoe-leased-'+row['backend']+'-chain-2026-10-18')
            for row in olmoe))
        segment_text = [row for row in admitted if row['model_type'] in (
            'qwen3_5_moe_text','qwen4_exp_text') and row['backend']=='public-range']
        self.assertEqual(len(segment_text),6)
        self.assertTrue(all(
            row['persisted_format']=='float32' and
            row['source_review']=='p02-'+row['family']+'-native-segment-text-chain-2026-10-20'
            for row in segment_text))
        public_text = [row for row in admitted if row['model_type'] in (
            'glm5_next_text','inkling_text') and row['backend']=='public-range']
        self.assertEqual(len(public_text),6)
        self.assertTrue(all(
            row['persisted_format']=='float32' and
            row['source_review'].endswith('local-public-text-chain-2026-10-18') and
            'execution_factories.open_native_family_public_directory_service'
                in row['callable_chain']['factory'] and
            'native_family_public_adapters.NativeFamilyPublicAdapters'
                in row['callable_chain']['native_abi'] and
            row['cells']['media']['classification']=='contract-restriction'
            for row in public_text))
        conditional_public = [row for row in admitted
            if row['family'] in ('qwen4','glm5-next') and
            row['model_type'] in ('qwen4_exp','glm5_next') and
            row['backend']=='public-range']
        self.assertEqual(len(conditional_public),8)
        self.assertTrue(all(
            row['source_format'] in ('F32','F16','BF16','F8_E4M3') and
            row['persisted_format']==(
                'fp8-e4m3-b128-f32' if row['source_format']=='F8_E4M3'
                else 'float32') and
            row['source_review']=='p02-'+row['family']+
                '-local-public-conditional-chain-2026-10-23'
            for row in conditional_public))
        kimi_linear = [row for row in admitted
            if row['family']=='kimi-k3' and row['model_type']=='kimi_linear']
        self.assertEqual(len(kimi_linear),2)
        self.assertTrue(all(
            row['backend']=='native' and row['arithmetic']=='fp32-approximate' and
            row['source_format'] in ('F32','BF16') and
            row['persisted_format']=='float32' and
            row['source_review']=='p02-kimi-linear-native-cpu-service-chain-2026-10-23'
            for row in kimi_linear))
        kimi_k3 = [row for row in admitted
            if row['family']=='kimi-k3' and row['model_type']=='kimi_k3']
        self.assertEqual(len(kimi_k3),8)
        self.assertTrue(all(
            row['backend'] in ('native','public-range') and
            row['arithmetic']=='fp32-approximate' and
            (row['source_format'],row['persisted_format']) in {
                ('F32','float32'),('BF16','float32'),
                ('MXFP4','float32'),
                ('MXFP4','mxfp4-e2m1-g32-ue8m0')} and
            row['source_review']==
                'p02-k3-publisher-adapter-exact-format-factory-chain-2026-10-23'
            for row in kimi_k3))
        from openagent.adaptive_engine.source_scope import source_matrix_row
        first=next(row for row in rows if row['admission']!='verified-admitted')
        key={name:first[name] for name in (
            'family','model_type','backend','arithmetic','source_format','persisted_format')}
        detached=source_matrix_row(**key)
        detached['admission']='verified-admitted'
        self.assertEqual(source_matrix_row(**key)['admission'],first['admission'])
        tuples=set()
        groups=set()
        for row in rows:
            tuples.add(tuple(row[name] for name in (
                'family','model_type','backend','arithmetic','source_format','persisted_format')))
            groups.add(tuple(row[name] for name in (
                'family','model_type','backend','arithmetic')))
        for family,model_type,backend,arithmetic,source_format,persisted_format in sorted(tuples):
            with self.subTest(route=(family,model_type,backend,source_format,persisted_format)):
                if (family=='deepseek-v41' and model_type=='deepseek_v41' and
                        backend in ('native','device','public-range') and
                        (source_format,persisted_format) in {
                            ('BF16','bf16'),('FP8','fp8'),('FP4','fp4')}):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (family=='mixtral' and model_type=='mixtral' and
                        backend in ('native','device','public-range') and
                        source_format in ('F32','F16','BF16') and
                        persisted_format=='float32'):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (family=='glm-dsa' and model_type=='glm_moe_dsa' and
                        backend in ('native','device','public-range') and
                        source_format in ('F32','F16','BF16') and
                        persisted_format=='float32'):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (model_type in ('glm5_next_text','qwen3_5_moe_text','qwen4_exp_text') and
                        backend=='public-range' and source_format in ('F32','F16','BF16') and
                        persisted_format=='float32'):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (family in ('qwen4','glm5-next') and
                        model_type in ('qwen4_exp','glm5_next') and
                        backend=='public-range' and
                        (source_format,persisted_format) in {
                            ('F32','float32'),('F16','float32'),
                            ('BF16','float32'),
                            ('F8_E4M3','fp8-e4m3-b128-f32')}):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (family=='kimi-k3' and model_type=='kimi_linear' and
                        backend=='native' and arithmetic=='fp32-approximate' and
                        source_format in ('F32','BF16') and persisted_format=='float32'):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (family=='kimi-k3' and model_type=='kimi_k3' and
                        backend in ('native','public-range') and
                        arithmetic=='fp32-approximate' and
                        (source_format,persisted_format) in {
                            ('F32','float32'),('BF16','float32'),
                            ('MXFP4','float32'),
                            ('MXFP4','mxfp4-e2m1-g32-ue8m0')}):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (family=='inkling' and
                        ((model_type=='inkling_mm_model' and backend in (
                            'native','device','public-range') and
                          (source_format,persisted_format) in {
                              ('F32','float32'),('F16','float32'),('BF16','float32'),
                              ('F8_E4M3','fp8-e4m3-b128-f32')}) or
                         (model_type=='inkling_text' and backend in (
                            'native','device','public-range') and
                          (source_format,persisted_format) in {
                              ('F32','float32'),('F16','float32'),('BF16','float32')}))):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (family in ('qwen35','qwen4','glm5-next') and
                        backend in ('native','device') and
                        ((model_type in ('qwen3_5_moe','qwen4_exp','glm5_next') and
                          (source_format,persisted_format) in {
                              ('F32','float32'),('F16','float32'),('BF16','float32'),
                              ('F8_E4M3','fp8-e4m3-b128-f32')}) or
                         (model_type in ('qwen3_5_moe_text','qwen4_exp_text','glm5_next_text') and
                          (source_format,persisted_format) in {
                              ('F32','float32'),('F16','float32'),('BF16','float32')}))):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (family=='olmoe' and model_type=='olmoe' and
                        backend=='native' and source_format in ('F32','F16','BF16') and
                        persisted_format=='float32'):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                elif (family=='olmoe' and model_type=='olmoe' and
                        backend in ('native','public-range') and source_format in (
                            'F32','F16','BF16') and
                        (persisted_format in {
                            'int8-row','int2-row','int3-grouped-g64','int4-row',
                            'int4-grouped-g64','mxfp4-e2m1-g32-ue8m0',
                            'q4_0','q4_1','q5_0','q5_1','q8_0','q8_1',
                            'fp8-e4m3-b128-f32'} if backend=='native'
                         else persisted_format=='int8-row')):
                    result=require_source_matrix_admission(family=family,model_type=model_type,
                        backend=backend,arithmetic=arithmetic,source_format=source_format,
                        persisted_format=persisted_format)
                    self.assertEqual(result['admission'],'verified-admitted')
                else:
                    with self.assertRaisesRegex(ValueError,'P02 source tuple is not admitted'):
                        require_source_matrix_admission(family=family,model_type=model_type,
                            backend=backend,arithmetic=arithmetic,source_format=source_format,
                            persisted_format=persisted_format)
        for family,model_type,backend,arithmetic in sorted(groups):
            with self.subTest(route=(family,model_type,backend)):
                if family=='deepseek-v41' and model_type=='deepseek_v41' and backend in (
                        'native','device','public-range'):
                    self.assertEqual(len(require_source_family_backend_admission(family=family,
                        model_type=model_type,backend=backend,arithmetic=arithmetic)),3)
                elif family=='deepseek-v41' and model_type=='deepseek_v41' and backend=='distributed':
                    with self.assertRaisesRegex(ValueError,'restricted pending peer-loss consensus'):
                        require_source_family_backend_admission(family=family,
                            model_type=model_type,backend=backend,arithmetic=arithmetic)
                else:
                    with self.assertRaisesRegex(ValueError,'P02 family/backend route is not admitted'):
                        require_source_family_backend_admission(family=family,
                            model_type=model_type,backend=backend,arithmetic=arithmetic)

    def test_v41_distributed_routes_fail_with_process_loss_reason(self):
        from openagent.adaptive_engine.source_scope import require_source_matrix_admission

        with self.assertRaisesRegex(ValueError,'process-loss cancellation and recovery consensus'):
            require_source_matrix_admission(family='deepseek-v41',model_type='deepseek_v41',
                backend='distributed',arithmetic='publisher-bf16-fp8-fp4',
                source_format='BF16',persisted_format='bf16')

    def test_v41_gateway_selects_only_source_admitted_local_bank_routes(self):
        from argparse import Namespace
        from openagent.adaptive_engine.publisher_provider import NativePublisherProvider

        for device in (None,'cuda:0'):
            with self.subTest(device=device):
                NativePublisherProvider._require_source_route(
                    {'family':'deepseek_v41'},
                    Namespace(family='deepseek_v41',device=device),
                    Path('unused-checkpoint'))

    def test_v41_checkpoint_preflight_uses_full_publisher_contract_inventory(self):
        from openagent.adaptive_engine.source_scope import require_checkpoint_inventory_admission

        metadata=SimpleNamespace(assert_unchanged=lambda: None)
        checkpoint=SimpleNamespace(_metadata=metadata,report=SimpleNamespace(unsupported=(
            SimpleNamespace(contract='executor',path='deepseek_v41',detail='external runtime'),
            SimpleNamespace(contract='mapping_coverage',path='deepseek_v41',detail='P13 only'))))
        with patch('openagent.adaptive_engine.publisher_checkpoint.service.PublisherCheckpoint.open',
                   return_value=checkpoint) as open_checkpoint:
            result=require_checkpoint_inventory_admission('fixture',family='deepseek-v41',
                model_type='deepseek_v41',backend='public-range',
                arithmetic='publisher-bf16-fp8-fp4',max_metadata_bytes=1024)
        self.assertIs(result,metadata)
        open_checkpoint.assert_called_once_with('fixture',family='deepseek_v41',
            max_metadata_bytes=1024,max_tensors=1_000_000,max_shards=4096)

    def test_media_rows_without_comparison_adapters_name_enforced_restriction(self):
        rows = [row for row in source_matrix_rows()
                if row['family'] in ('glm5-next', 'inkling')
                and row['evidence_binding']['row_binding']['media']]
        self.assertTrue(rows)
        for row in rows:
            evidence = row['evidence_binding']
            self.assertIsNone(evidence['media_adapter'])
            self.assertTrue(evidence['media_adapter_missing'])
            self.assertIn('runner.validate_execution_binding', evidence['media_restriction'])
            self.assertFalse(evidence['completion_claim'])

    def test_selected_device_e8_route_names_decode_and_activation_owners(self):
        rows = [row for row in source_matrix_rows()
                if row['backend'] == 'device'
                and row['persisted_format'] == 'e8-lattice-caller-v1'
                and row['family'] in ('mixtral', 'qwen35', 'qwen4')]
        self.assertTrue(rows)
        for row in rows:
            route = row['cells']['format_route']
            self.assertEqual(route['device_consumer'],
                             'device_packed_decode.decode_device_expert')
            self.assertEqual(route['auxiliary'],
                             'device_e8_rotation.rotate_device_e8')
            self.assertIn(route['auxiliary'], row['callable_chain']['execution'])
            # The decode owners stay traceable, but publisher packed manifests
            # are rejected by native_family_cli and the factory guards.
            self.assertEqual(row['admission'], 'rejected-unadmitted')
            self.assertEqual(row['combination_status'], 'restricted')
            self.assertTrue(row['cells']['execution']['contract'])
        self.assertEqual(len({row['family'] for row in rows}), 3)

    def test_media_evidence_restriction_reason_cannot_be_weakened(self):
        scope = source_scope_matrix()
        row = next(row for row in scope['execution_cases']
                   if row['family'] in ('glm5-next', 'inkling')
                   and row['evidence_binding']['row_binding']['media'])
        row['evidence_binding']['media_restriction'] = 'not implemented yet'
        with self.assertRaisesRegex(ValueError, 'source execution case contract differs'):
            validate_source_scope(scope)

    def test_media_runner_rejection_reports_row_bound_restriction_reason(self):
        from openagent.adaptive_engine.wholemodel_validation.runner import (
            validate_execution_binding,
        )
        case = next(row for row in source_matrix_rows()
                    if row['family'] == 'glm5-next'
                    and row['evidence_binding']['row_binding']['media'])
        contract = dict(case['evidence_binding']['row_binding'])
        binding = dict(contract=contract, oracle_factory='oracle',
            candidate_factory='candidate', artifact_pins={},
            binding_sha256='0' * 64, receipt_required=True)
        manifest = dict(execution_binding=binding,
            oracle=dict(factory='oracle', artifacts=[]),
            candidate=dict(factory='candidate', artifacts=[]))
        with self.assertRaisesRegex(ValueError, 'media execution contract lacks a row-bound media adapter'):
            validate_execution_binding(manifest)


if __name__ == "__main__":
    unittest.main()
