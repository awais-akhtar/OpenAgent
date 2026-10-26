"""P02 candidate triage: every restriction is enforced by the owner it names.

``source_scope._tuple_restriction`` moves a generated candidate row to
``restricted`` only when a concrete consumer rejects that exact tuple before
payload use.  These tests drive each named consumer directly (with the P02
admission gate patched out where it would otherwise fire first) so a
restriction cannot survive the consumer gaining support, and a remaining
candidate cannot silently hide a restriction.
"""
from argparse import Namespace
import ast
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openagent.adaptive_engine import source_scope
from openagent.adaptive_engine.source_scope import (
    require_checkpoint_inventory_admission,
    require_source_matrix_admission,
    source_matrix_rows,
)

KEY = ('family', 'model_type', 'backend', 'arithmetic', 'source_format', 'persisted_format')

# Keep this empty unless a genuinely unreviewed generated route is added.
EXPECTED_CANDIDATES = {}


def _safetensors(path, tensors):
    header, offset, payload = {}, 0, b''
    for name, (dtype, shape, data) in tensors.items():
        header[name] = dict(dtype=dtype, shape=list(shape), data_offsets=[offset, offset+len(data)])
        offset += len(data)
        payload += data
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


class MatrixShapeTests(unittest.TestCase):
    def test_counts_after_triage(self):
        rows = source_matrix_rows()
        counts = {}
        for row in rows:
            counts[row['combination_status']] = counts.get(row['combination_status'], 0) + 1
        self.assertEqual(len(rows), 1334)
        self.assertEqual(counts, {'verified-admitted': 171, 'restricted': 877,
                                  'rejected-unadmitted': 286})
        self.assertTrue(source_scope.validate_phase1_source())

    def test_k3_native_public_rows_bind_and_incomplete_device_hybrids_are_restricted(self):
        rows = [row for row in source_matrix_rows()
                if row['family'] == 'kimi-k3' and row['model_type'] == 'kimi_k3']
        admitted = [row for row in rows if row['combination_status'] == 'verified-admitted']
        self.assertEqual(len(admitted), 8)
        self.assertEqual({(row['backend'], row['source_format'], row['persisted_format'])
                          for row in admitted}, {
            (backend, source, persisted)
            for backend in ('native', 'public-range')
            for source, persisted in (
                ('F32', 'float32'), ('BF16', 'float32'),
                ('MXFP4', 'float32'), ('MXFP4', 'mxfp4-e2m1-g32-ue8m0'))})
        restricted = {(row['backend'], row['source_format'], row['persisted_format'])
                      for row in rows if row['combination_status'] == 'restricted'
                      and row['backend'] == 'device'
                      and 'mixed BF16/FLA state owner' in row['cells']['execution']['contract']}
        self.assertEqual(restricted, {
            ('device', 'BF16', 'float32'),
            ('device', 'MXFP4', 'float32'),
            ('device', 'MXFP4', 'mxfp4-e2m1-g32-ue8m0')})
        for row in rows:
            if (row['backend'], row['source_format'], row['persisted_format']) in restricted:
                self.assertIn('mixed BF16/FLA state owner',
                              row['cells']['execution']['contract'])
        dense_packed = [row for row in rows
                        if row['persisted_format'] == 'mxfp4-e2m1-g32-ue8m0'
                        and row['source_format'] in ('F32', 'BF16')]
        self.assertEqual(len(dense_packed), 7)
        self.assertTrue(all(row['combination_status'] == 'restricted'
                            for row in dense_packed))

    def test_k3_loader_rejects_dense_to_packed_before_config_or_tensor_load(self):
        from openagent.adaptive_engine.publisher_checkpoint.k3_adapter import load_k3
        checkpoint = SimpleNamespace(report=SimpleNamespace(family='kimi_k3'))
        with patch('openagent.adaptive_engine.publisher_checkpoint.k3_adapter.k3_source_format',
                   return_value='F32'), \
             patch('openagent.adaptive_engine.publisher_checkpoint.k3_adapter.k3_config') as config:
            with self.assertRaisesRegex(ValueError, 'P02 source tuple is not admitted'):
                load_k3(checkpoint, capacity=8, max_weight_bytes=1,
                    expert_format='e2m1-low-first-e8m0-g32', source_backend='native',
                    source_arithmetic='fp32-approximate')
        config.assert_not_called()

    def test_k3_device_hybrid_restrictions_fail_before_config_or_tensor_load(self):
        from openagent.adaptive_engine.publisher_checkpoint.k3_adapter import load_k3
        checkpoint = SimpleNamespace(report=SimpleNamespace(family='kimi_k3'))
        cases = (('BF16', 'fp32'), ('MXFP4', 'fp32'),
                 ('MXFP4', 'e2m1-low-first-e8m0-g32'))
        with patch('openagent.adaptive_engine.publisher_checkpoint.k3_adapter.k3_source_format',
                   side_effect=lambda _checkpoint: active_source[0]), \
             patch('openagent.adaptive_engine.publisher_checkpoint.k3_adapter.k3_config') as config:
            for source_format, expert_format in cases:
                active_source = [source_format]
                with self.subTest(source_format=source_format, expert_format=expert_format):
                    with self.assertRaisesRegex(ValueError, 'mixed BF16/FLA state owner'):
                        load_k3(checkpoint, capacity=8, max_weight_bytes=1,
                            expert_format=expert_format, source_backend='device',
                            source_arithmetic='bf16')
        config.assert_not_called()

    def test_k3_source_classifier_distinguishes_quantized_and_homogeneous_dense(self):
        from openagent.adaptive_engine.publisher_checkpoint.k3_adapter import k3_source_format
        for dtype in ('F32', 'BF16'):
            checkpoint = SimpleNamespace(report=SimpleNamespace(
                family='kimi_k3', contracts=(SimpleNamespace(encoding='float32',sources=('w',)),)),
                tensors={'w': SimpleNamespace(dtype=dtype)})
            self.assertEqual(k3_source_format(checkpoint), dtype)
        quantized = SimpleNamespace(report=SimpleNamespace(
            family='kimi_k3', contracts=(SimpleNamespace(encoding='mxfp4-g32',sources=('w',)),)),
            tensors={'w': SimpleNamespace(dtype='U8')})
        self.assertEqual(k3_source_format(quantized), 'MXFP4')
        mixed = SimpleNamespace(report=SimpleNamespace(
            family='kimi_k3', contracts=(SimpleNamespace(encoding='float32',sources=('a','b')) ,)),
            tensors={'a': SimpleNamespace(dtype='F32'), 'b': SimpleNamespace(dtype='BF16')})
        with self.assertRaisesRegex(ValueError, 'homogeneous'):
            k3_source_format(mixed)

    def test_olmoe_direct_gateway_formats_are_bound_and_float16_is_restricted(self):
        direct = {
            'int8-row', 'int2-row', 'int3-grouped-g64', 'int4-row',
            'int4-grouped-g64', 'mxfp4-e2m1-g32-ue8m0', 'q4_0', 'q4_1',
            'q5_0', 'q5_1', 'q8_0', 'q8_1', 'fp8-e4m3-b128-f32'}
        rows = [row for row in source_matrix_rows()
                if row['family']=='olmoe' and row['model_type']=='olmoe'
                and row['backend']=='native']
        for row in rows:
            if row['persisted_format'] in direct:
                self.assertEqual(row['combination_status'], 'verified-admitted')
                self.assertIn('model_gateway.OlmoeDirectProvider', row['callable_chain']['gateway'])
                self.assertIn('olmoe.OlmoeModel', row['callable_chain']['execution'])
            elif row['persisted_format'] == 'float16':
                self.assertEqual(row['combination_status'], 'restricted')
                self.assertIn('no OLMoE routed-expert execution owner',
                              row['cells']['format_route']['contract'])

    def test_olmoe_direct_model_runs_inventory_gate_before_dense_weight_reads(self):
        from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
        from tests.test_q8_0_olmoe import checkpoint
        class AdmissionStop(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root/'experts.q4', dtype='q4_0')
            with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                       side_effect=AdmissionStop) as admission, \
                 patch('openagent.adaptive_engine.olmoe._read_float_matrix') as read_tensor:
                with self.assertRaises(AdmissionStop):
                    OlmoeModel(root, manifest, expert_cache_bytes=1024*1024)
            admission.assert_called_once_with(
                root, family='olmoe', model_type='olmoe', backend='native',
                arithmetic='fp32', persisted_format='q4_0')
            read_tensor.assert_not_called()

    def test_remaining_candidates_are_exactly_the_reviewed_set(self):
        found = {}
        for row in source_matrix_rows():
            if row['combination_status'] == 'candidate-unverified':
                group = tuple(row[name] for name in KEY[:4])
                found.setdefault(group, set()).add(
                    row['source_format'] + ':' + row['persisted_format'])
        self.assertEqual(found, EXPECTED_CANDIDATES)

    def test_every_restriction_is_on_its_row_and_fails_closed_with_reason(self):
        checked = 0
        for row in source_matrix_rows():
            identity = {name: row[name] for name in KEY}
            reason = source_scope._case_format_reason(
                row['family'], row['model_type'], row['backend'], row['source_format'],
                row['persisted_format'],
                route_admitted=row['cells']['format_route']['classification'] != 'rejected-unadmitted',
                source_verified=row['combination_status'] == 'verified-admitted')
            if row['combination_status'] == 'candidate-unverified':
                self.assertIsNone(source_scope._tuple_restriction(
                    row['family'], row['model_type'], row['backend'],
                    row['source_format'], row['persisted_format']))
                continue
            if reason is None:
                continue
            checked += 1
            self.assertEqual(row['combination_status'], 'restricted')
            self.assertEqual(row['admission'], 'rejected-unadmitted')
            self.assertIn(reason, row['cells']['execution']['contract'])
            self.assertEqual(row['cells']['format_route']['contract'], reason)
            with self.assertRaisesRegex(ValueError, 'P02 source tuple is not admitted') as raised:
                require_source_matrix_admission(**identity)
            self.assertIn(reason[:40], str(raised.exception))
        backend_only = sum(
            1 for row in source_matrix_rows()
            if row['combination_status'] == 'restricted' and
            row['cells']['format_route']['contract'] is None and
            (row['family'], row['backend']) in source_scope._BACKEND_RESTRICTIONS)
        self.assertEqual(checked + backend_only, 877)

    def test_validator_rejects_a_candidate_that_hides_a_restriction(self):
        scope = source_scope.source_scope_matrix()
        row = next(row for row in scope['execution_cases']
                   if row['family'] == 'olmoe' and row['backend'] == 'public-range'
                   and row['persisted_format'] == 'q4_0')
        self.assertEqual(row['combination_status'], 'restricted')
        row['combination_status'] = row['admission'] = 'candidate-unverified'
        with self.assertRaises(ValueError):
            source_scope.validate_source_scope(scope)


class OlmoeConsumerTests(unittest.TestCase):
    def setUp(self):
        from tests.test_q8_0_olmoe import checkpoint
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'olmoe'
        self.root.mkdir()
        checkpoint(self.root)
        self.library = Path(temporary.name) / 'unused-library'
        self.library.write_bytes(b'')

    def _manifest(self, dtype):
        from openagent.adaptive_engine.olmoe import pack_olmoe_experts
        packed = pack_olmoe_experts(self.root, self.root / ('experts.' + dtype), dtype=dtype)
        path = self.root / (dtype + '.manifest.json')
        packed.write(path)
        return path

    def test_float16_is_not_even_an_olmoe_packing_format(self):
        from openagent.adaptive_engine.olmoe import pack_olmoe_experts
        with self.assertRaisesRegex(ValueError, 'unsupported expert packing dtype'):
            pack_olmoe_experts(self.root, self.root / 'experts.f16', dtype='float16')

    def test_direct_gateway_consumers_bind_formats_outside_the_leased_abi(self):
        # The direct gateway executes these formats through OlmoeModel and
        # SparseExpertRuntime. The universal leased C adapter has its own
        # explicit preflight and must not silently claim those formats.
        from openagent.adaptive_engine.native_leased_olmoe_segments import _APPLY_ATTRIBUTES
        self.assertEqual(set(_APPLY_ATTRIBUTES), set(source_scope._OLMOE_NATIVE_FORMATS))
        unsupported = ('int4-row', 'int2-row', 'int3-grouped-g64',
                       'int4-grouped-g64', 'mxfp4-e2m1-g32-ue8m0',
                       'fp8-e4m3-b128-f32')
        for source_format in ('F32', 'F16', 'BF16'):
            for dtype in unsupported:
                with self.subTest(source_format=source_format, dtype=dtype):
                    self.assertIsNone(source_scope._tuple_restriction(
                        'olmoe', 'olmoe', 'native', source_format, dtype))
            with self.subTest(source_format=source_format, dtype='float16'):
                self.assertIn('no float16 expert-manifest consumer', source_scope._tuple_restriction(
                    'olmoe', 'olmoe', 'native', source_format, 'float16'))
        for dtype in _APPLY_ATTRIBUTES:
            self.assertIsNone(source_scope._tuple_restriction(
                'olmoe', 'olmoe', 'native', 'BF16', dtype))

    def test_public_adapter_rejects_every_non_int8_manifest_itself(self):
        from openagent.adaptive_engine.segment_edge_adapters import NativeOlmoePublicAdapters
        inventory = SimpleNamespace(tensors={}, assert_unchanged=lambda: None)
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                   return_value=inventory), \
                patch('openagent.adaptive_engine.olmoe.validate_olmoe_tensors'):
            for dtype in ('q4_0', 'q8_0', 'int4-row', 'int2-row', 'mxfp4-e2m1-g32-ue8m0',
                          'fp8-e4m3-b128-f32', 'int4-grouped-g64', 'int3-grouped-g64'):
                with self.subTest(dtype=dtype):
                    with self.assertRaisesRegex(ValueError, 'public OLMoE INT8 expert manifest differs'):
                        NativeOlmoePublicAdapters(
                            self.library, kernel_library=self.library,
                            model_directory=self.root, manifest=self._manifest(dtype),
                            expert_cache_bytes=1 << 20, max_staging_bytes=1 << 20,
                            max_snapshot_bytes=1 << 20)


class PublisherConsumerTests(unittest.TestCase):
    def test_v41_checkpoint_service_payload_loader_delegates_to_gated_instance(self):
        import ast

        source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/publisher_checkpoint/service.py'
        module = ast.parse(source.read_text(encoding='utf-8'))
        service = next(node for node in module.body
                       if isinstance(node, ast.ClassDef) and
                       node.name == 'PublisherCheckpointService')
        loader = next(node for node in service.body
                      if isinstance(node, ast.FunctionDef) and
                      node.name == 'load_v41_execution_payload')
        calls = [node for node in ast.walk(loader) if isinstance(node, ast.Call)]
        checkpoint_open = [node for node in calls if isinstance(node.func, ast.Attribute) and
                           node.func.attr == 'open' and
                           isinstance(node.func.value, ast.Name) and
                           node.func.value.id == 'PublisherCheckpoint']
        instance_load = [node for node in calls if isinstance(node.func, ast.Attribute) and
                         node.func.attr == 'load_v41_execution_payload']
        self.assertTrue(checkpoint_open and instance_load)
        self.assertLess(min(node.lineno for node in checkpoint_open),
                        min(node.lineno for node in instance_load))

    def test_v41_checkpoint_instance_gates_inventory_before_mapped_load(self):
        import ast

        source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/publisher_checkpoint/service.py'
        module = ast.parse(source.read_text(encoding='utf-8'))
        checkpoint = next(node for node in module.body
                          if isinstance(node, ast.ClassDef) and node.name == 'PublisherCheckpoint')
        loader = next(node for node in checkpoint.body
                      if isinstance(node, ast.FunctionDef) and
                      node.name == 'load_v41_execution_payload')
        calls = [node for node in ast.walk(loader) if isinstance(node, ast.Call)]
        gate = next(node for node in calls if isinstance(node.func, ast.Name) and
                    node.func.id == 'require_checkpoint_inventory_admission')
        keywords = {item.arg: ast.unparse(item.value) for item in gate.keywords}
        self.assertEqual(keywords['family'], "'deepseek-v41'")
        self.assertEqual(keywords['model_type'], "'deepseek_v41'")
        self.assertEqual(keywords['arithmetic'], "'publisher-bf16-fp8-fp4'")
        load = next(node for node in calls if isinstance(node.func, ast.Name) and
                    node.func.id == 'load_v41')
        self.assertLess(gate.lineno, load.lineno)

    def test_v41_service_factory_gates_full_inventory_before_checkpoint_open(self):
        import ast

        source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/execution_factories.py'
        module = ast.parse(source.read_text(encoding='utf-8'))
        factory = next(node for node in module.body
                       if isinstance(node, ast.FunctionDef) and
                       node.name == 'open_publisher_v41_service')
        calls = [node for node in ast.walk(factory) if isinstance(node, ast.Call)]
        gate = next(node for node in calls
                    if isinstance(node.func, ast.Name) and
                    node.func.id == 'require_checkpoint_inventory_admission')
        keywords = {item.arg: ast.unparse(item.value) for item in gate.keywords}
        self.assertEqual(keywords['family'], "'deepseek-v41'")
        self.assertEqual(keywords['model_type'], "'deepseek_v41'")
        self.assertEqual(keywords['arithmetic'], "'publisher-bf16-fp8-fp4'")
        self.assertEqual(keywords['backend'],
                         "'distributed' if distributed is not None else 'device' if device is not None else 'native'")
        gate_line = gate.lineno
        checkpoint_open = [node.lineno for node in calls
                           if isinstance(node.func, ast.Attribute) and
                           node.func.attr == 'open' and
                           isinstance(node.func.value, ast.Name) and
                           node.func.value.id == 'PublisherCheckpoint']
        payload_load = [node.lineno for node in calls
                        if isinstance(node.func, ast.Name) and node.func.id == 'load_v41']
        self.assertTrue(checkpoint_open and payload_load)
        self.assertLess(gate_line, min(checkpoint_open + payload_load))

    def test_v41_cli_gates_full_source_inventory_before_runtime_loading(self):
        import ast

        source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/publisher_v41_cli.py'
        module = ast.parse(source.read_text(encoding='utf-8'))
        session = next(node for node in module.body
                       if isinstance(node, ast.ClassDef) and node.name == 'PublisherSession')
        constructor = next(node for node in session.body
                           if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        calls = [node for node in ast.walk(constructor) if isinstance(node, ast.Call)]
        gate = next(node for node in calls
                    if isinstance(node.func, ast.Name) and
                    node.func.id == 'require_checkpoint_inventory_admission')
        keywords = {item.arg: ast.unparse(item.value) for item in gate.keywords}
        self.assertEqual(keywords['family'], "'deepseek-v41'")
        self.assertEqual(keywords['model_type'], "'deepseek_v41'")
        self.assertEqual(keywords['arithmetic'], "'publisher-bf16-fp8-fp4'")
        self.assertEqual(keywords['backend'],
                         "'distributed' if distributed is not None else 'device' if device is not None else 'native'")
        gate_line = gate.lineno
        loader_lines = [node.lineno for node in calls
                        if isinstance(node.func, ast.Name) and node.func.id == 'load_v41']
        checkpoint_lines = [node.lineno for node in calls
                            if isinstance(node.func, ast.Attribute) and node.func.attr == 'open' and
                            isinstance(node.func.value, ast.Name) and
                            node.func.value.id == 'PublisherCheckpoint']
        self.assertTrue(loader_lines and checkpoint_lines)
        self.assertLess(gate_line, min(loader_lines + checkpoint_lines))

    def test_native_family_cli_rejects_every_packed_manifest(self):
        import sys
        from openagent.adaptive_engine import native_family_cli
        # The rejection precedes any tokenizer use beyond parsing; a stub
        # keeps this independent of the optional tokenizers wheel.
        stub = SimpleNamespace(Tokenizer=SimpleNamespace(from_str=lambda text: object()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'model_type': 'mixtral', 'vocab_size': 4}))
            (root / 'tokenizer.json').write_text('{}')
            for name in ('runtime', 'kernel', 'expert-kernel'):
                (root / name).write_bytes(b'')
            manifest = root / 'experts.manifest.json'
            manifest.write_text('{}')
            args = native_family_cli.parser().parse_args([
                '--family', 'mixtral', '--checkpoint', str(root),
                '--runtime-library', str(root / 'runtime'), '--kernel-library', str(root / 'kernel'),
                '--context', '8', '--runtime-mib', '1', '--staging-mib', '1',
                '--snapshot-mib', '1', '--transfer-mib', '1', '--layers-per-range', '1',
                '--max-tokens', '1', '--token-ids', '[]',
                '--expert-manifest', str(manifest), '--expert-kernel', 'native',
                '--expert-cache-mib', '1', '--expert-work-mib', '1'])
            parsed = SimpleNamespace(architecture='mixtral', layers=())
            with patch.dict(sys.modules, {'tokenizers': stub}), \
                    patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission'), \
                    patch('openagent.adaptive_engine.expert_runtime.SparseExpertManifest.read',
                          return_value=parsed):
                with self.assertRaisesRegex(ValueError, 'packed expert manifest lacks source-dtype'):
                    native_family_cli.PublisherSession(args)

    def test_direct_factories_reject_packed_providers(self):
        from openagent.adaptive_engine import execution_factories as factories
        from openagent.adaptive_engine.torch_qwen35_experts import TorchQwen35Experts
        from openagent.adaptive_engine.torch_qwen4_experts import TorchQwen4Experts
        with self.assertRaisesRegex(ValueError, 'Mixtral packed experts lack'):
            factories._reject_unproven_mixtral_manifest_provider(object())
        with self.assertRaisesRegex(ValueError, 'Qwen3.5 packed manifest provider'):
            factories._reject_unproven_qwen35_manifest_provider(object.__new__(TorchQwen35Experts))
        with self.assertRaisesRegex(ValueError, 'Qwen4 packed manifest provider'):
            factories._reject_unproven_qwen4_manifest_provider(object.__new__(TorchQwen4Experts))
        with self.assertRaisesRegex(ValueError, 'GLM-DSA packed experts lack'):
            factories.open_device_glm_dsa_directory_service(
                'unused', device='cpu', max_weight_bytes=1, runtime_max_bytes=1,
                max_total_model_bytes=1, expert_provider=object())

    def test_conditional_cli_has_no_packed_manifest_input(self):
        from openagent.adaptive_engine import native_conditional_cli
        options = {action.dest for action in native_conditional_cli.parser()._actions}
        self.assertFalse({name for name in options if 'manifest' in name and 'expert' in name})

    def test_text_reader_rejects_fp8_and_packed_headers(self):
        from openagent.adaptive_engine.native_checkpoint import _header
        with tempfile.TemporaryDirectory() as directory:
            for dtype in ('F8_E4M3', 'U8'):
                path = Path(directory) / (dtype + '.safetensors')
                _safetensors(path, {'w': (dtype, (2,), b'\0\0')})
                with self.subTest(dtype=dtype), path.open('rb') as source:
                    with self.assertRaisesRegex(ValueError, 'unsupported weight metadata'):
                        _header(source, (0, 0, path.stat().st_size))

    def test_fp8_sources_bind_only_to_the_block_route(self):
        inventory = SimpleNamespace(tensors={
            'a.weight': SimpleNamespace(dtype='BF16'),
            'b.weight': SimpleNamespace(dtype='F8_E4M3'),
            'b.weight_scale_inv': SimpleNamespace(dtype='F8_E8M0'),
        })
        calls = []
        with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                   return_value=inventory), \
                patch('openagent.adaptive_engine.source_scope.require_source_inventory_admission',
                      side_effect=lambda **kw: calls.append((kw['persisted_format'],
                                                             sorted(kw['inventory'])))):
            require_checkpoint_inventory_admission(
                'unused', family='qwen4', model_type='qwen4_exp', backend='native',
                max_metadata_bytes=4096)
        self.assertEqual(calls, [('float32', ['a.weight']),
                                 ('fp8-e4m3-b128-f32', ['b.weight'])])

    def test_qwen35_wrapper_has_no_public_segment_loader(self):
        from openagent.adaptive_engine import publisher_conditional_loader as loader
        config = {'model_type': 'qwen3_5_moe'}
        raw = json.dumps(config).encode()
        inventory = SimpleNamespace(tensors={}, config_sha256=hashlib.sha256(raw).hexdigest())
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(loader, 'snapshot', return_value=inventory), \
                patch.object(loader, 'read_regular', return_value=raw), \
                patch.object(loader, 'component_contract',
                             return_value=SimpleNamespace(auxiliary_config=None)), \
                patch('openagent.adaptive_engine.source_scope.require_source_inventory_admission'):
            with self.assertRaisesRegex(ValueError, 'requires an implemented publisher wrapper'):
                loader.load_publisher_conditional_decoder(
                    directory, family='qwen3_5_moe', library=None, text_library=None,
                    max_weight_bytes=1, max_bytes=1, context_tokens=1, max_conversion_bytes=1,
                    segment_runtime_library='segment', segment_work_bytes=1)

    def test_nested_v41_text_identity_cannot_be_opened(self):
        from openagent.adaptive_engine.publisher_checkpoint.contracts import build_contracts
        with self.assertRaisesRegex(ValueError, 'requires its exact model identity'):
            require_checkpoint_inventory_admission(
                'unused', family='deepseek-v41', model_type='deepseek_v41_text',
                backend='native', arithmetic='publisher-bf16-fp8-fp4', max_metadata_bytes=4096)
        with self.assertRaisesRegex(ValueError, 'exact publisher wrapper'):
            build_contracts({'model_type': 'deepseek_v41_text', 'text_config': {}}, max_contracts=1)

    def test_kimi_linear_is_wired_to_native_cli_factory_and_recovery_owner(self):
        package = Path(source_scope.__file__).parent
        owners = set()
        for path in package.rglob('*.py'):
            relative = path.relative_to(package).as_posix()
            if relative.startswith(('phase_1_audit/', 'phase1_')) or relative == 'source_scope.py':
                continue
            if 'kimi_linear' in path.read_text(encoding='utf-8', errors='replace'):
                owners.add(relative)
        self.assertEqual(owners, {'independent_kimi.py', 'native_kimi.py',
                                  'text_decoder_registry.py', 'family_conformance.py',
                                  'native_libraries.py', 'native_checkpoint.py',
                                  'execution_factories.py', 'native_family_cli.py',
                                  '__init__.py',
                                  'native_kimi_k3_vendor_checkpoint.py',
                                  'publisher_checkpoint/contracts.py'})
        text = (package / 'native_kimi.py').read_text(encoding='utf-8')
        self.assertIn("model_type='kimi_linear',backend='native'", text)
        cli = (package / 'native_family_cli.py').read_text(encoding='utf-8')
        factory = (package / 'execution_factories.py').read_text(encoding='utf-8')
        decoder = (package / 'independent_kimi.py').read_text(encoding='utf-8')
        self.assertIn("'kimi_linear'))", cli)
        self.assertIn('open_native_kimi_linear_directory_service', cli)
        self.assertIn('def open_native_kimi_linear_directory_service', factory)
        self.assertIn('def restore(self, snapshot)', decoder)
        self.assertIn("'p02-kimi-linear-native-cpu-service-chain-2026-10-23'", 
                      (package / 'source_scope.py').read_text(encoding='utf-8'))
        from openagent.adaptive_engine.native_checkpoint import load_native_text_decoder
        with self.assertRaisesRegex(ValueError, 'only its exact native CPU route'):
            load_native_text_decoder('unused', model_type='kimi_linear', library='unused',
                                     max_weight_bytes=1, source_backend='public-range')
        with patch('openagent.adaptive_engine.native_kimi.NativeKimiLinearDecoder.from_checkpoint',
                   return_value='kimi-decoder') as open_kimi:
            result = load_native_text_decoder(
                'checkpoint', model_type='kimi_linear', library='kernel',
                max_weight_bytes=4096, max_bytes=8192, context_tokens=128)
        self.assertEqual(result, 'kimi-decoder')
        open_kimi.assert_called_once_with(
            'checkpoint', library='kernel', capacity=128, max_bytes=4096)

    def test_kimi_linear_native_formats_bind_to_the_checkpoint_service_gate(self):
        from openagent.adaptive_engine.execution_factories import (
            open_native_kimi_linear_directory_service,
        )
        from openagent.adaptive_engine.source_scope import require_source_matrix_admission
        for source_format in ('F32', 'BF16'):
            with self.subTest(source_format=source_format):
                require_source_matrix_admission(
                    family='kimi-k3', model_type='kimi_linear', backend='native',
                    arithmetic='fp32-approximate', source_format=source_format,
                    persisted_format='float32')
                row = next(row for row in source_matrix_rows()
                    if row['family']=='kimi-k3' and row['model_type']=='kimi_linear'
                    and row['backend']=='native' and row['source_format']==source_format
                    and row['persisted_format']=='float32')
                self.assertEqual(row['combination_status'], 'verified-admitted')
        with patch('openagent.adaptive_engine.source_scope.require_checkpoint_inventory_admission',
                   side_effect=ValueError('P02 source tuple is not admitted')) as admission:
            with self.assertRaisesRegex(ValueError, 'P02 source tuple is not admitted'):
                open_native_kimi_linear_directory_service(
                    'checkpoint', library='kernel', max_weight_bytes=4096,
                    runtime_max_bytes=8192, context_tokens=8)
        admission.assert_called_once_with(
            'checkpoint', family='kimi-k3', model_type='kimi_linear',
            backend='native', arithmetic='fp32-approximate',
            max_metadata_bytes=4096)

    def test_kimi_linear_physical_restore_validates_before_commit_without_runtime_build(self):
        import threading
        import numpy as np
        from openagent.adaptive_engine.independent_kimi import IndependentKimiLinearDecoder
        decoder = object.__new__(IndependentKimiLinearDecoder)
        decoder.capacity = 4
        decoder.config = SimpleNamespace(
            num_hidden_layers=2, num_attention_heads=1,
            qk_nope_head_dim=1, qk_rope_head_dim=1, v_head_dim=2,
            linear_attn_config={'kda_layers': (1,), 'num_heads': 1,
                                'head_dim': 2, 'short_conv_kernel_size': 2})
        decoder._lock = threading.RLock()
        decoder._closed = False
        decoder._position = 0
        decoder._cache = []
        snapshot = (2, (
            (np.ones((1,2,2),np.float32), np.ones((1,1,2),np.float32),
             np.ones((1,1,2),np.float32), np.ones((1,1,2),np.float32)),
            (np.ones((4,1,2),np.float32), np.ones((4,1,2),np.float32))))
        decoder.restore(snapshot)
        self.assertEqual(decoder.position, 2)
        self.assertTrue(all(array.flags.c_contiguous for layer in decoder.snapshot()[1]
                            for array in layer))
        before = decoder.snapshot()
        malformed = list(snapshot[1])
        malformed[0] = (np.full((1,2,2), np.nan, np.float32), *malformed[0][1:])
        with self.assertRaisesRegex(ValueError, 'nonfinite state'):
            decoder.restore((snapshot[0], tuple(malformed)))
        self.assertEqual(decoder.position, before[0])
        for left_layer, right_layer in zip(before[1], decoder.snapshot()[1]):
            for left, right in zip(left_layer, right_layer):
                np.testing.assert_array_equal(left, right)


class QwenTextSegmentRouteTests(unittest.TestCase):
    """The admitted native Qwen text route loads through public-range rows."""

    def test_native_text_route_loads_and_serves_through_public_range(self):
        from openagent.adaptive_engine import native_checkpoint
        from openagent.adaptive_engine.native_qwen35_ranges import NativeQwen35Range
        from openagent.adaptive_engine.native_qwen4_ranges import NativeQwen4Range
        seen = []

        class Stop(Exception):
            pass

        def capture(directory, **options):
            seen.append((options['model_type'], options['source_backend']))
            raise Stop

        with patch.object(native_checkpoint, 'read_native_text_checkpoint', side_effect=capture):
            for decoder in (NativeQwen35Range, NativeQwen4Range):
                with self.assertRaises(Stop):
                    decoder.from_directory('unused', begin=0, end=1, library=None,
                                           max_weight_bytes=1, max_bytes=1)
        self.assertEqual(seen, [('qwen3_5_moe_text', 'public-range'),
                                ('qwen4_exp_text', 'public-range')])
        for family, model_type in (('qwen35', 'qwen3_5_moe_text'), ('qwen4', 'qwen4_exp_text')):
            inventory = SimpleNamespace(tensors={'w': SimpleNamespace(dtype='BF16')})
            with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',
                       return_value=inventory):
                self.assertIs(require_checkpoint_inventory_admission(
                    'unused', family=family, model_type=model_type, backend='public-range',
                    max_metadata_bytes=4096), inventory)

    def test_qwen4_ple_substituted_shards_match_provider_inventory_geometry(self):
        from types import SimpleNamespace
        from openagent.adaptive_engine import native_checkpoint
        from openagent.adaptive_engine import native_qwen4_ranges

        config = {'num_hidden_layers': 1, 'tie_word_embeddings': False}
        table = 'model.layers.0.ple.ple_embedding.ngram_embedding.weight'
        shard = 'model.layers.0.ple.shard0'
        layer_weight = 'model.layers.0.self_attn.weight'
        shapes = {table: (4, 2), layer_weight: (2, 2)}
        provider_record = SimpleNamespace(shape=(4, 2), dtype='F32')

        class Snapshot:
            tensors = {shard: provider_record}

            def assert_unchanged(self):
                return None

        provider = SimpleNamespace(
            bindings={0: ((shard,), (0, 4))},
            config=config,
            snapshot=Snapshot(),
            max_bytes=64,
        )
        inventory = {
            layer_weight: {'shape': (2, 2), 'dtype': 'F32'},
            shard: {'shape': (3, 2), 'dtype': 'F32'},
        }

        class Stop(Exception):
            pass

        def capture(directory, **options):
            options['tensor_selection'](config, inventory)
            raise Stop

        with patch.object(native_qwen4_ranges, 'qwen4_tensor_inventory',
                          return_value=(shapes, {})), \
             patch.object(native_checkpoint, 'read_native_text_checkpoint', side_effect=capture):
            with self.assertRaisesRegex(ValueError, 'PLE source shard inventory differs'):
                native_qwen4_ranges.NativeQwen4Range.from_directory(
                    'unused', begin=0, end=1, library=None, max_weight_bytes=1,
                    max_bytes=1024, ple_provider=provider)

    def test_explicit_public_runtime_stays_rejected_for_qwen_text(self):
        from openagent.adaptive_engine.execution_factories import (
            open_native_family_public_directory_service)
        with self.assertRaisesRegex(ValueError, 'text-only for GLM5 or Inkling'):
            open_native_family_public_directory_service(
                'unused', runtime_library=None, kernel_library=None, family='qwen3_5_moe_text',
                max_weight_bytes=1, runtime_max_bytes=1, max_total_model_bytes=1,
                context_tokens=1, max_work_bytes=1)


class ConditionalPublicRouteTests(unittest.TestCase):
    def test_glm5_and_qwen4_wrapper_public_routes_bind_all_admitted_source_encodings(self):
        expected = {('glm5-next', 'glm5_next'), ('qwen4', 'qwen4_exp')}
        source_pairs = {
            ('F32', 'float32'), ('F16', 'float32'), ('BF16', 'float32'),
            ('F8_E4M3', 'fp8-e4m3-b128-f32'),
        }
        bindings = source_scope._SOURCE_VERIFIED_ROUTE_BINDINGS
        for family, identity in expected:
            for source_format, persisted_format in source_pairs:
                route = (family, identity, 'public-range', 'fp32',
                         source_format, persisted_format)
                self.assertIn(route, bindings)
                self.assertEqual(
                    bindings[route],
                    'p02-'+family+'-local-public-conditional-chain-2026-10-23')
        rows = {tuple(row[name] for name in KEY): row for row in source_matrix_rows()}
        for family, identity in expected:
            for source_format, persisted_format in source_pairs:
                row = rows[(family, identity, 'public-range', 'fp32',
                            source_format, persisted_format)]
                self.assertEqual(row['combination_status'], 'verified-admitted')
                self.assertTrue(row['callable_chain']['loading'])
                self.assertTrue(row['callable_chain']['execution'])
                self.assertIn('native_conditional_cli.PublisherSession',
                              row['callable_chain']['cli'])
        self.assertTrue(source_scope.validate_phase1_source())


class K3ToolingAdmissionTests(unittest.TestCase):
    def test_k3_weight_loading_apis_require_explicit_source_route(self):
        from openagent.adaptive_engine.publisher_checkpoint.k3_adapter import (
            load_k3, load_k3_projection_shard)
        from openagent.adaptive_engine.publisher_checkpoint.service import PublisherCheckpoint

        with self.assertRaisesRegex(TypeError, 'source_backend.*source_arithmetic'):
            load_k3(object(), capacity=8, max_weight_bytes=1024)
        with self.assertRaisesRegex(TypeError, 'source_backend.*source_arithmetic'):
            load_k3_projection_shard(object(), 'layer.q_proj.weight', rank=0,
                world_size=1, max_weight_bytes=1024)
        with self.assertRaisesRegex(TypeError, 'source_backend.*source_arithmetic'):
            PublisherCheckpoint.load_k3_execution_payload(
                object(), capacity=8, max_weight_bytes=1024)
        with self.assertRaisesRegex(TypeError, 'source_backend.*source_arithmetic'):
            PublisherCheckpoint.load_k3_execution_payload(
                'unused', capacity=8, max_weight_bytes=1024)

    def test_candidate_factory_admits_exact_route_before_payload_loading(self):
        source = (Path(__file__).parents[1] / 'openagent' / 'adaptive_engine' /
                 'wholemodel_validation' / 'k3.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        factory = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and
                       node.name == 'K3CandidateFactory')
        call = next(node for node in factory.body
                    if isinstance(node, ast.FunctionDef) and node.name == '__call__')
        route_calls = [node for node in ast.walk(call)
                       if isinstance(node, ast.Call) and
                       isinstance(node.func, ast.Name) and
                       node.func.id == 'k3_route_identity']
        loader_calls = [node for node in ast.walk(call)
                        if isinstance(node, ast.Call) and
                        isinstance(node.func, ast.Name) and node.func.id == 'load_k3']
        self.assertEqual(len(route_calls), 1)
        self.assertEqual(len(loader_calls), 1)
        self.assertLess(route_calls[0].lineno, loader_calls[0].lineno)
        self.assertEqual({keyword.arg for keyword in loader_calls[0].keywords},
                         {'capacity', 'max_weight_bytes', 'expert_format',
                          'projection_epsilon', 'l2_epsilon', 'rotary_mode',
                          'rotary_theta', 'include_vision',
                          'vision_tower_epsilon', 'source_backend',
                          'source_arithmetic'})


if __name__ == '__main__':
    unittest.main()
