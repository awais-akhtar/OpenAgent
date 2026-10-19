"""Phase 1 A10 — Intel Arc GPU Evidence Capture Tests.

Tests Arc GPU detection and evidence schema without requiring actual GPU execution.
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


class Phase1A10ArcGPUTest(unittest.TestCase):
    """Verify A10 Arc GPU detection and evidence schemas."""

    def test_arc_gpu_info_schema_valid(self):
        """A10: ArcGPUInfo schema captures GPU hardware info."""
        from openagent.adaptive_engine.phase1_a10_arc_gpu_evidence import ArcGPUInfo

        gpu_info = ArcGPUInfo(
            device_name='Intel Arc A770',
            device_id='xpu:0',
            compute_capability='Arc (XPU)',
            total_memory_mb=16384,
            driver_version='2024.1',
            oneapi_available=True,
            npu_available=True,
        )

        info_dict = gpu_info.to_dict()
        self.assertEqual(info_dict['device_name'], 'Intel Arc A770')
        self.assertEqual(info_dict['total_memory_mb'], 16384)
        self.assertTrue(info_dict['oneapi_available'])
        self.assertTrue(info_dict['npu_available'])

    def test_arc_gpu_execution_receipt_schema_valid(self):
        """A10: ArcGPUExecutionReceipt schema with GPU metrics and parity."""
        from openagent.adaptive_engine.phase1_a10_arc_gpu_evidence import (
            ArcGPUExecutionReceipt, ArcGPUInfo
        )

        gpu_info = ArcGPUInfo(
            device_name='Intel Arc A770',
            device_id='xpu:0',
            compute_capability='Arc (XPU)',
            total_memory_mb=16384,
            driver_version='2024.1',
            oneapi_available=True,
            npu_available=True,
        )

        receipt = ArcGPUExecutionReceipt(
            route_key='qwen3.5/qwen3_5_moe/arc_gpu/bf16/float32/float32',
            checkpoint_path='/models/qwen3.5-0.8b',
            config_hash='config_123',
            device_info=gpu_info,
            input_tokens=[1, 2, 3],
            output_shape=(128000,),
            output_dtype='float32',
            output_min=-5.0,
            output_max=10.0,
            output_mean=0.5,
            output_std=1.2,
            forward_ms=8.5,  # GPU much faster than CPU (45ms)
            gpu_memory_used_mb=2048.5,
            gpu_memory_peak_mb=3500.0,
            cache_on_gpu=True,
            cache_shape={'k': (1, 8, 64), 'v': (1, 8, 64)},
            cache_hash='cache_hash_123',
            gpu_cpu_logits_match=True,  # Parity verified
            gpu_cpu_tolerance_used=1e-4,
            timestamp='2026-10-19T22:27:00Z',
        )

        receipt_dict = receipt.to_dict()
        json_str = json.dumps(receipt_dict)
        self.assertIsNotNone(json_str)

        # Verify GPU-specific fields
        self.assertEqual(receipt_dict['device_info']['device_name'], 'Intel Arc A770')
        self.assertEqual(receipt_dict['performance_ms'], 8.5)  # Faster than CPU
        self.assertEqual(receipt_dict['gpu_memory_used_mb'], 2048.5)
        self.assertTrue(receipt_dict['gpu_cpu_parity']['logits_match'])
        self.assertEqual(receipt_dict['gpu_cpu_parity']['tolerance'], 1e-4)
        self.assertTrue(receipt_dict['cache_on_gpu'])

    def test_arc_gpu_p13_evidence_binding_valid(self):
        """A10: ArcGPUP13EvidenceBinding aggregates GPU execution evidence."""
        from openagent.adaptive_engine.phase1_a10_arc_gpu_evidence import (
            ArcGPUP13EvidenceBinding, ArcGPUExecutionReceipt, ArcGPUInfo
        )

        gpu_info = ArcGPUInfo(
            device_name='Intel Arc A770',
            device_id='xpu:0',
            compute_capability='Arc (XPU)',
            total_memory_mb=16384,
            driver_version='2024.1',
            oneapi_available=True,
            npu_available=True,
        )

        receipt = ArcGPUExecutionReceipt(
            route_key='qwen3.5/qwen3_5_moe/arc_gpu/bf16/float32/float32',
            checkpoint_path='/models/qwen3.5-0.8b',
            config_hash='config_123',
            device_info=gpu_info,
            input_tokens=[1, 2, 3],
            output_shape=(128000,),
            output_dtype='float32',
            output_min=-5.0,
            output_max=10.0,
            output_mean=0.5,
            output_std=1.2,
            forward_ms=8.5,
            gpu_memory_used_mb=2048.5,
            gpu_memory_peak_mb=3500.0,
            cache_on_gpu=True,
            cache_shape={'k': (1, 8, 64), 'v': (1, 8, 64)},
            cache_hash='cache_hash_123',
            gpu_cpu_logits_match=True,
            gpu_cpu_tolerance_used=1e-4,
            timestamp='2026-10-19T22:27:00Z',
        )

        binding = ArcGPUP13EvidenceBinding(
            route_key='qwen3.5/qwen3_5_moe/arc_gpu/bf16/float32/float32',
            family='qwen3.5',
            model_type='qwen3_5_moe',
            backend='arc_gpu',
            checkpoint_hash='checkpoint_123',
            config_hash='config_123',
            device_info=gpu_info,
            gpu_execution_receipts=[receipt],
            gpu_parity_verified=True,
            gpu_cpu_parity_verified=True,
            tolerance_threshold=1e-4,
            full_generation_success=True,
            generation_tokens=[15, 23, 8, 42],
            generation_device='xpu:0',
            timestamp='2026-10-19T22:27:00Z',
        )

        binding_dict = binding.to_dict()
        json_str = json.dumps(binding_dict)
        self.assertIsNotNone(json_str)

        # Verify GPU aggregation
        self.assertEqual(binding_dict['backend'], 'arc_gpu')
        self.assertEqual(len(binding_dict['gpu_execution_receipts']), 1)
        self.assertTrue(binding_dict['parity_verification']['gpu_parity_verified'])
        self.assertTrue(binding_dict['parity_verification']['gpu_cpu_parity_verified'])
        self.assertEqual(binding_dict['generation_device'], 'xpu:0')

    def test_save_arc_gpu_evidence_manifest(self):
        """A10: Arc GPU evidence manifests can be saved and loaded."""
        from openagent.adaptive_engine.phase1_a10_arc_gpu_evidence import (
            ArcGPUP13EvidenceBinding, ArcGPUExecutionReceipt, ArcGPUInfo,
            save_arc_gpu_evidence_manifest
        )

        gpu_info = ArcGPUInfo(
            device_name='Intel Arc A770',
            device_id='xpu:0',
            compute_capability='Arc (XPU)',
            total_memory_mb=16384,
            driver_version='2024.1',
            oneapi_available=True,
            npu_available=True,
        )

        receipt = ArcGPUExecutionReceipt(
            route_key='qwen3.5/qwen3_5_moe/arc_gpu/bf16/float32/float32',
            checkpoint_path='/models/qwen3.5-0.8b',
            config_hash='config_123',
            device_info=gpu_info,
            input_tokens=[1, 2, 3],
            output_shape=(128000,),
            output_dtype='float32',
            output_min=-5.0,
            output_max=10.0,
            output_mean=0.5,
            output_std=1.2,
            forward_ms=8.5,
            gpu_memory_used_mb=2048.5,
            gpu_memory_peak_mb=3500.0,
            cache_on_gpu=True,
            cache_shape=None,
            cache_hash=None,
            gpu_cpu_logits_match=True,
            gpu_cpu_tolerance_used=1e-4,
            timestamp='2026-10-19T22:27:00Z',
        )

        binding = ArcGPUP13EvidenceBinding(
            route_key='qwen3.5/qwen3_5_moe/arc_gpu/bf16/float32/float32',
            family='qwen3.5',
            model_type='qwen3_5_moe',
            backend='arc_gpu',
            checkpoint_hash='checkpoint_123',
            config_hash='config_123',
            device_info=gpu_info,
            gpu_execution_receipts=[receipt],
            gpu_parity_verified=True,
            gpu_cpu_parity_verified=True,
            tolerance_threshold=1e-4,
            full_generation_success=True,
            generation_tokens=[15, 23, 8],
            generation_device='xpu:0',
            timestamp='2026-10-19T22:27:00Z',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'a10_arc_gpu_evidence.json'
            save_arc_gpu_evidence_manifest(binding, output_path)

            self.assertTrue(output_path.exists())
            manifest_text = output_path.read_text()
            manifest = json.loads(manifest_text)

            self.assertIn('phase1_a10_arc_gpu_evidence', manifest)
            self.assertEqual(manifest['schema_version'], '1.0')
            self.assertEqual(
                manifest['phase1_a10_arc_gpu_evidence']['device_info']['device_name'],
                'Intel Arc A770'
            )

    def test_arc_gpu_helper_functions_importable(self):
        """A10: Arc GPU utility functions are callable."""
        from openagent.adaptive_engine.phase1_a10_arc_gpu_evidence import (
            detect_arc_gpu,
            move_to_arc_gpu,
            move_to_cpu,
            capture_arc_gpu_execution_evidence,
            save_arc_gpu_evidence_manifest,
        )

        self.assertTrue(callable(detect_arc_gpu))
        self.assertTrue(callable(move_to_arc_gpu))
        self.assertTrue(callable(move_to_cpu))
        self.assertTrue(callable(capture_arc_gpu_execution_evidence))
        self.assertTrue(callable(save_arc_gpu_evidence_manifest))

        # Verify docstrings
        self.assertIn('Arc GPU', detect_arc_gpu.__doc__)
        self.assertIn('A10', capture_arc_gpu_execution_evidence.__doc__)


if __name__ == '__main__':
    unittest.main()
