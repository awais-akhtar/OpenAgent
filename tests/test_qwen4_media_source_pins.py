from pathlib import Path
import tempfile
import unittest

from openagent.adaptive_engine.wholemodel_validation.qwen4_media import _require_source_pin


class Qwen4MediaSourcePinTests(unittest.TestCase):
    def test_qwen4_wrapper_implementation_must_be_explicitly_source_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = Path(temporary) / 'qwen4_multimodal.py'
            wrapper.write_text('class Qwen4MultimodalDecoder: pass\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'actual Qwen4 wrapper'):
                _require_source_pin([], wrapper)
            with self.assertRaisesRegex(ValueError, 'actual Qwen4 wrapper'):
                _require_source_pin([
                    {'kind': 'dependency', 'resolved_path': str(wrapper)},
                ], wrapper)
            _require_source_pin([
                {'kind': 'source', 'resolved_path': str(wrapper)},
            ], wrapper)


if __name__ == '__main__':
    unittest.main()
