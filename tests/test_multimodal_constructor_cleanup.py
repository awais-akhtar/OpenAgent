"""Owned provider cleanup when publisher media inventory admission fails."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openagent.adaptive_engine import qwen4_multimodal, qwen35_multimodal


class _Provider:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class ConditionalConstructorCleanupTests(unittest.TestCase):
    @staticmethod
    def _components(wrapper):
        return SimpleNamespace(
            variant=SimpleNamespace(wrapper=wrapper),
            original_config={},
            auxiliary_config={},
            text_config={},
        )

    def test_qwen4_closes_owned_provider_when_tensor_partition_fails(self):
        provider = _Provider()
        with patch.object(qwen4_multimodal, 'component_contract',
                          return_value=self._components('qwen4_exp')), \
             patch.object(qwen4_multimodal, 'partition_weights',
                          side_effect=ValueError('injected inventory failure')):
            with self.assertRaisesRegex(ValueError, 'injected inventory failure'):
                qwen4_multimodal.Qwen4MultimodalDecoder(
                    {}, {}, library=None, text_library=None, max_bytes=1024,
                    expert_provider=provider, own_expert_provider=True)
        self.assertEqual(provider.close_calls, 1)

    def test_qwen35_closes_owned_provider_when_audio_bank_is_rejected(self):
        provider = _Provider()
        with patch.object(qwen35_multimodal, 'component_contract',
                          return_value=self._components('qwen3_5_moe')), \
             patch.object(qwen35_multimodal, 'partition_weights',
                          return_value=({}, {}, {'audio.weight': object()}, {})):
            with self.assertRaisesRegex(ValueError, 'no admitted audio tower'):
                qwen35_multimodal.Qwen35MultimodalDecoder(
                    {}, {}, library=None, text_library=None, max_bytes=1024,
                    expert_provider=provider, own_expert_provider=True)
        self.assertEqual(provider.close_calls, 1)


if __name__ == '__main__':
    unittest.main()
