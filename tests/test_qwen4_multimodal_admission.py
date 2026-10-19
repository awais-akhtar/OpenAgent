"""Public Qwen4 preflight must validate checkpoint paths before composition."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openagent.adaptive_engine.qwen4_multimodal import Qwen4MultimodalDecoder


class Qwen4MultimodalAdmissionTests(unittest.TestCase):
    def test_public_runtime_preflight_checks_inventory_root_before_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = SimpleNamespace(root=root, assert_unchanged=Mock())
            components = SimpleNamespace(
                variant=SimpleNamespace(wrapper='qwen4_exp'),
                original_config={}, vision_config={}, text_config={})
            with patch('openagent.adaptive_engine.qwen4_multimodal.component_contract',
                       return_value=components), \
                 patch('openagent.adaptive_engine.qwen4_multimodal.partition_weights',
                       side_effect=RuntimeError('reached partition')):
                with self.assertRaisesRegex(RuntimeError, 'reached partition'):
                    Qwen4MultimodalDecoder({}, {}, library=None, text_library=None,
                        max_bytes=1024, native_composition=True,
                        segment_runtime_library='segment-runtime',
                        segment_model_directory=root, segment_work_bytes=1,
                        _checkpoint_inventory=inventory)


if __name__ == '__main__':
    unittest.main()
