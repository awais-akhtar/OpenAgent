"""Phase 1 A10 — CPU Runtime Evidence Capture Tests.

Verifies that A10 evidence schemas and capture functions work on CPU.
Real checkpoints required for full end-to-end; these tests verify the machinery.
"""

import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile


class Phase1A10CPUEvidenceTest(unittest.TestCase):
    """Verify A10 evidence capture on CPU without GPU."""

    def test_execution_receipt_schema_valid(self):
        """A10: ExecutionReceipt schema is complete and JSON-serializable."""
        from openagent.adaptive_engine.phase1_a10_cpu_evidence import ExecutionReceipt

        receipt = ExecutionReceipt(
            route_key='qwen3.5/qwen3_5_moe/native/fp32/float32/float32',
            checkpoint_path='/models/qwen3.5-test',
            config_hash='abc123def456',
            input_tokens=[1, 2, 3],
            output_shape=(128000,),
            output_dtype='float32',
            output_min=-5.0,
            output_max=10.0,
            output_mean=0.5,
            output_std=1.2,
            forward_ms=45.3,
            memory_peak_mb=512.5,
            cache_shape={'k': (1, 8, 64), 'v': (1, 8, 64)},
            cache_hash='cache_hash_123',
            timestamp='2026-10-19T22:22:00Z',
        )

        # Verify it's serializable
        receipt_dict = receipt.to_dict()
        self.assertIsInstance(receipt_dict, dict)
        json_str = json.dumps(receipt_dict)
        self.assertIsNotNone(json_str)

        # Verify key fields are present
        self.assertEqual(receipt_dict['route_key'], 'qwen3.5/qwen3_5_moe/native/fp32/float32/float32')
        self.assertEqual(receipt_dict['output_shape'], [128000])
        self.assertEqual(receipt_dict['output_dtype'], 'float32')
        self.assertIn('output_stats', receipt_dict)
        self.assertIn('performance_ms', receipt_dict)
        self.assertIn('cache_shape', receipt_dict)

    def test_recovery_receipt_schema_valid(self):
        """A10: RecoveryReceipt schema captures snapshot/restore evidence."""
        from openagent.adaptive_engine.phase1_a10_cpu_evidence import RecoveryReceipt

        recovery = RecoveryReceipt(
            route_key='qwen3.5/qwen3_5_moe/native/fp32/float32/float32',
            gen1_tokens=[1, 2, 3],
            gen1_logits_hash='logits_hash_1',
            gen1_cache_hash='cache_hash_1',
            snapshot_bytes=2048,
            snapshot_hash='snapshot_hash_123',
            gen2_tokens=[4, 5],
            gen2_logits_hash='logits_hash_1',  # Same = verified match
            gen2_cache_hash='cache_hash_1',
            logits_match=True,
            cache_match=True,
            timestamp='2026-10-19T22:22:00Z',
        )

        recovery_dict = recovery.to_dict()
        json_str = json.dumps(recovery_dict)
        self.assertIsNotNone(json_str)

        # Verify verification fields
        self.assertTrue(recovery_dict['verification']['logits_match'])
        self.assertTrue(recovery_dict['verification']['cache_match'])

    def test_p13_evidence_binding_schema_valid(self):
        """A10: P13EvidenceBinding aggregates multiple receipts into complete evidence."""
        from openagent.adaptive_engine.phase1_a10_cpu_evidence import (
            P13EvidenceBinding, ExecutionReceipt
        )

        receipt = ExecutionReceipt(
            route_key='qwen3.5/qwen3_5_moe/native/fp32/float32/float32',
            checkpoint_path='/models/qwen3.5-test',
            config_hash='config_123',
            input_tokens=[1, 2, 3],
            output_shape=(128000,),
            output_dtype='float32',
            output_min=-5.0,
            output_max=10.0,
            output_mean=0.5,
            output_std=1.2,
            forward_ms=45.3,
            memory_peak_mb=512.5,
            cache_shape=None,
            cache_hash=None,
            timestamp='2026-10-19T22:22:00Z',
        )

        binding = P13EvidenceBinding(
            route_key='qwen3.5/qwen3_5_moe/native/fp32/float32/float32',
            family='qwen3.5',
            model_type='qwen3_5_moe',
            backend='native',
            checkpoint_hash='checkpoint_123',
            config_hash='config_123',
            execution_receipts=[receipt],
            recovery_receipts=[],
            parity_verified=True,
            tolerance_threshold=1e-5,
            full_generation_success=False,
            generation_tokens=[],
            timestamp='2026-10-19T22:22:00Z',
        )

        binding_dict = binding.to_dict()
        json_str = json.dumps(binding_dict)
        self.assertIsNotNone(json_str)

        # Verify aggregation
        self.assertEqual(len(binding_dict['execution_receipts']), 1)
        self.assertTrue(binding_dict['parity_verified'])
        self.assertEqual(binding_dict['family'], 'qwen3.5')

    def test_save_evidence_manifest_to_file(self):
        """A10: Evidence manifests can be saved and loaded from JSON."""
        from openagent.adaptive_engine.phase1_a10_cpu_evidence import (
            P13EvidenceBinding, ExecutionReceipt, save_evidence_manifest
        )

        receipt = ExecutionReceipt(
            route_key='qwen3.5/qwen3_5_moe/native/fp32/float32/float32',
            checkpoint_path='/models/qwen3.5-test',
            config_hash='config_123',
            input_tokens=[1, 2, 3],
            output_shape=(128000,),
            output_dtype='float32',
            output_min=-5.0,
            output_max=10.0,
            output_mean=0.5,
            output_std=1.2,
            forward_ms=45.3,
            memory_peak_mb=512.5,
            cache_shape=None,
            cache_hash=None,
            timestamp='2026-10-19T22:22:00Z',
        )

        binding = P13EvidenceBinding(
            route_key='qwen3.5/qwen3_5_moe/native/fp32/float32/float32',
            family='qwen3.5',
            model_type='qwen3_5_moe',
            backend='native',
            checkpoint_hash='checkpoint_123',
            config_hash='config_123',
            execution_receipts=[receipt],
            recovery_receipts=[],
            parity_verified=True,
            tolerance_threshold=1e-5,
            full_generation_success=False,
            generation_tokens=[],
            timestamp='2026-10-19T22:22:00Z',
        )

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'a10_evidence_test.json'
            save_evidence_manifest(binding, output_path)

            # Verify file exists and is valid JSON
            self.assertTrue(output_path.exists())
            manifest_text = output_path.read_text()
            manifest = json.loads(manifest_text)

            # Verify structure
            self.assertIn('phase1_a10_evidence', manifest)
            self.assertIn('schema_version', manifest)
            self.assertEqual(manifest['schema_version'], '1.0')

    def test_capture_functions_importable(self):
        """A10: Evidence capture functions exist and are callable."""
        from openagent.adaptive_engine.phase1_a10_cpu_evidence import (
            capture_execution_evidence,
            capture_recovery_evidence,
            save_evidence_manifest,
        )

        self.assertTrue(callable(capture_execution_evidence))
        self.assertTrue(callable(capture_recovery_evidence))
        self.assertTrue(callable(save_evidence_manifest))

        # Verify docstrings mention A10
        self.assertIn('A10', capture_execution_evidence.__doc__)
        self.assertIn('A10', capture_recovery_evidence.__doc__)


if __name__ == '__main__':
    unittest.main()
