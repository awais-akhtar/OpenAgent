"""Fail-closed regression checks for the P15 closure verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.phase1_closure_integrated import (
    P02NativeCPUConverter,
    Phase1ValidatorComplete,
    SourceScopeConverter,
)
from openagent.adaptive_engine.phase1_p13_p14_p15_complete import Phase1ClosureValidator as LegacyClosureValidator


def _scope(rows):
    return SimpleNamespace(
        validate_phase1_source=lambda: True,
        source_scope_matrix=lambda: {'execution_cases': rows},
    )


def _pin(path: Path, role: str | None = None, kind: str | None = None):
    result = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if role is not None:
        result['role'] = role
    if kind is not None:
        result['kind'] = kind
    return result


class Phase1ClosureGateTests(unittest.TestCase):
    def test_legacy_validator_delegates_to_canonical_receipt_gate(self):
        assertions = {
            'A10_evidence_requirements': {'passed': False, 'description': 'missing route receipts', 'error': 'missing'},
        }
        for family in LegacyClosureValidator.PHASE1_FAMILIES:
            assertions[f'A0X_{family}_route_complete'] = {
                'passed': True, 'description': 'source route bound', 'error': None,
            }
        canonical_result = {
            'phase1_closed': False, 'checks_passed': 10, 'total_checks': 17,
            'assertions': assertions, 'blocked_on_a10': 'missing route receipts',
        }
        with patch('openagent.adaptive_engine.phase1_closure_integrated.Phase1ValidatorComplete') as canonical:
            canonical.return_value.validate_all.return_value = canonical_result
            result = LegacyClosureValidator().validate_all()
        self.assertFalse(result['phase1_closed'])
        self.assertEqual(result['checks_passed'], 10)
        self.assertEqual(result['total_checks'], 17)
        self.assertFalse(any(result['families_ready'].values()))

    def test_legacy_presence_only_stage_check_refuses_certification(self):
        with self.assertRaisesRegex(RuntimeError, 'not authoritative'):
            LegacyClosureValidator().validate_p04_media_lifecycle()

    def test_candidate_cannot_be_promoted_by_legacy_converters(self):
        candidate = {'admission': 'candidate-unverified', 'family': 'olmoe',
                     'backend': 'native', 'persisted_format': 'float32'}
        with self.assertRaisesRegex(RuntimeError, 'disabled'):
            SourceScopeConverter.convert_candidate_to_verified(candidate)
        with self.assertRaisesRegex(RuntimeError, 'disabled'):
            P02NativeCPUConverter.convert_candidate_to_verified(candidate)
        self.assertEqual(candidate['admission'], 'candidate-unverified')

    def test_candidate_and_missing_receipts_prevent_closure(self):
        row = {'case_id': 'candidate', 'family': 'olmoe',
               'combination_status': 'candidate-unverified'}
        with TemporaryDirectory() as directory:
            validator = Phase1ValidatorComplete(
                _scope([row]), Path(directory) / 'missing.json')
            result = validator.validate_all()
        self.assertFalse(result['assertions']['A01_priority_tuples']['passed'])
        self.assertFalse(result['assertions']['A0X_olmoe_route_complete']['passed'])
        self.assertFalse(result['assertions']['A10_evidence_requirements']['passed'])
        self.assertFalse(result['phase1_closed'])

    def test_placeholder_manifest_is_not_runtime_evidence(self):
        row = {'case_id': 'olmoe/example/native/fp32/F32/float32',
               'family': 'olmoe', 'combination_status': 'verified-admitted',
               'source_review': 'reviewed', 'chain_order': ('loading', 'execution'),
               'callable_chain': {'loading': ('loader',), 'execution': ('consumer',)}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'phase1_runtime_receipts.json'
            manifest.write_text(json.dumps({
                'schema': 'openagent.phase1-runtime-receipts.v1',
                'routes': [{'case_id': row['case_id'], 'artifacts': [
                    {'role': 'config', 'path': 'test_config/config.json', 'sha256': 'a' * 64},
                    {'role': 'weights', 'path': 'test_config/model.bin', 'sha256': 'b' * 64}],
                    'cache_schema': {'version': 1},
                    'placement_constraints': {'device': 'cpu'},
                    'comparison_adapter': 'reference'}],
            }), encoding='utf-8')
            validator = Phase1ValidatorComplete(_scope([row]), manifest)
            self.assertFalse(validator._check_a09_observation())
            self.assertFalse(validator._check_a10_evidence())

    def test_missing_route_receipt_is_reported_on_binding_and_evidence_checks(self):
        row = {'case_id': 'olmoe/example/native/fp32/BF16/int8-row',
               'family': 'olmoe', 'combination_status': 'verified-admitted',
               'source_review': 'reviewed', 'chain_order': ('loading', 'execution'),
               'callable_chain': {'loading': ('loader',), 'execution': ('consumer',)}}
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / 'phase1_runtime_receipts.json'
            manifest.write_text(json.dumps({
                'schema': 'openagent.phase1-runtime-receipts.v1', 'routes': [],
            }), encoding='utf-8')
            result = Phase1ValidatorComplete(_scope([row]), manifest).validate_all()
        binding = result['assertions']['A09_observation_binding']
        evidence = result['assertions']['A10_evidence_requirements']
        self.assertFalse(binding['passed'])
        self.assertFalse(evidence['passed'])
        self.assertIn('1 of 1 admitted routes lack runtime receipts', binding['error'])
        self.assertEqual(binding['error'], evidence['error'])

    def test_generic_file_reused_for_every_kind_is_rejected(self):
        row = {'case_id': 'olmoe/example/native/fp32/F32/float32',
               'family': 'olmoe', 'combination_status': 'verified-admitted',
               'source_review': 'reviewed', 'chain_order': ('loading', 'execution'),
               'callable_chain': {'loading': ('loader',), 'execution': ('consumer',)}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            weights = root / 'model.bin'
            observation = root / 'observation.json'
            config.write_text('{}', encoding='utf-8')
            weights.write_bytes(b'weights')
            observation.write_text('{"observed":true}', encoding='utf-8')
            manifest = root / 'phase1_runtime_receipts.json'
            record = {
                'case_id': row['case_id'],
                'route': [row.get(name) for name in ('family', 'model_type', 'backend',
                                                   'arithmetic', 'source_format', 'persisted_format')],
                'artifacts': [_pin(config, role='config'), _pin(weights, role='weights')],
                'cache_schema': {'version': 1},
                'placement_constraints': {'device': 'cpu'},
                'comparison_adapter': 'independent-oracle',
                'observations': [_pin(observation, kind=kind) for kind in
                                 ('tokens', 'logits', 'cache', 'routing', 'recovery',
                                  'comparison', 'placement')],
                'full_generation_success': True,
                'numerical_parity_verified': True,
                'platform_certified': True,
            }
            manifest.write_text(json.dumps({
                'schema': 'openagent.phase1-runtime-receipts.v1', 'routes': [record],
            }), encoding='utf-8')
            validator = Phase1ValidatorComplete(_scope([row]), manifest)
            # Real files with matching pins prove only that bytes exist: one
            # generic file reused for every kind is not an observation.
            self.assertTrue(validator._check_a09_observation())
            self.assertFalse(validator._check_a10_evidence())
            self.assertIn('one file cannot serve as several observation kinds',
                          validator._verified_receipts()[1])
            result = validator.validate_all()
            self.assertFalse(result['phase1_closed'])
            self.assertIn('one file cannot serve',
                          result['assertions']['A10_evidence_requirements']['error'])


if __name__ == '__main__':
    unittest.main()
