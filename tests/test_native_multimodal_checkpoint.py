"""Exact wrapper loading, explicit libraries and compiled image/text execution."""
import gc
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.native_checkpoint import (
    load_native_multimodal_decoder, read_native_multimodal_checkpoint,
    read_native_text_checkpoint,
)
from tests import test_native_checkpoint as checkpoint_tests


class NativeMultimodalCheckpointTests(unittest.TestCase):
    def test_wrapper_identity_is_not_a_text_alias(self):
        for name in ('qwen3_5_moe_text', 'qwen4_exp', 'glm5_next', 'kimi_k3', None, True):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'exact native multimodal'):
                read_native_multimodal_checkpoint('missing', model_type=name, max_weight_bytes=1024)
        with self.assertRaisesRegex(ValueError, 'exact native text'):
            read_native_text_checkpoint('missing', model_type='qwen3_5_moe', max_weight_bytes=1024)

    def test_explicit_library_and_budget_validation_precede_payload_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / 'vision.dll'
            library.write_bytes(b'not loaded during rejection tests')
            with patch('openagent.adaptive_engine.native_checkpoint.read_native_multimodal_checkpoint',
                       side_effect=AssertionError('payload loading')):
                with self.assertRaises(FileNotFoundError):
                    load_native_multimodal_decoder(root, model_type='qwen3_5_moe', library=library,
                        text_library=root/'missing.dll', max_weight_bytes=1024)
                for budget in (True, 0, -1):
                    with self.assertRaisesRegex(ValueError, 'positive integer'):
                        load_native_multimodal_decoder(root, model_type='qwen3_5_moe',
                            library=library, text_library=library, max_weight_bytes=1024, max_bytes=budget)

    def test_direct_wrapper_loader_checks_exact_p02_source_format_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / 'vision.dll'
            library.write_bytes(b'not loaded during source rejection')
            inventory = {'weight': {'dtype': 'F32'}}
            decoder = object()
            with patch('openagent.adaptive_engine.native_checkpoint._admit_qwen35_multimodal_inventory'), \
                 patch('openagent.adaptive_engine.native_checkpoint._read_native_checkpoint',
                       side_effect=lambda _directory, **options: (
                           options['metadata_admission']({}, inventory, 0), {})) as reader, \
                 patch('openagent.adaptive_engine.native_qwen35_vision.NativeQwen35Vision',
                       return_value=decoder):
                result=load_native_multimodal_decoder(
                    root, model_type='qwen3_5_moe', library=library, text_library=library,
                    max_weight_bytes=1024, max_bytes=4096)

            reader.assert_called_once()
            self.assertIs(result,decoder)

    def test_local_wrapper_loader_admits_exact_p02_tuple_before_payload(self):
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
            self.skipTest('optional exact multimodal oracle unavailable')
        import transformers
        if transformers.__version__ != '5.16.1':
            self.skipTest('exact multimodal oracle requires Transformers 5.16.1')
        from tests.test_native_qwen35_vision import Qwen35VisionTests
        case_type = Qwen35VisionTests
        case_type.setUpClass()
        _native = oracle = None
        try:
            case = case_type()
            oracle, _native, weights = case.fixture()
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                checkpoint_tests.NativeCheckpointExecutionTests.write_checkpoint(root, oracle.config.to_dict(), weights)
                with patch('openagent.adaptive_engine.native_checkpoint._read_tensor',
                           side_effect=AssertionError('rejected payload was read')):
                    with self.assertRaisesRegex(ValueError, 'memory budget'):
                        load_native_multimodal_decoder(root, model_type='qwen3_5_moe',
                            library=case.vision_path, text_library=case.text_path,
                            max_weight_bytes=64*1024**2, max_bytes=1024)
                    with patch('openagent.adaptive_engine.native_qwen35_vision.NativeQwen35Vision._geometry',
                               {'hidden_size': 99}):
                        with self.assertRaisesRegex(ValueError, 'unsupported vision geometry'):
                            load_native_multimodal_decoder(root, model_type='qwen3_5_moe',
                                library=case.vision_path, text_library=case.text_path,
                                max_weight_bytes=64*1024**2)
                # The exact dense local route is now source-admitted. Load the
                # fixture through its own header planner and native vision
                # consumer; source admission still does not claim parity.
                decoder=load_native_multimodal_decoder(root, model_type='qwen3_5_moe',
                    library=case.vision_path, text_library=case.text_path,
                    max_weight_bytes=64*1024**2, max_bytes=256*1024**2)
                self.assertTrue(callable(getattr(decoder,'prefill',None)))
                decoder.close()
        finally:
            _native = oracle = None
            gc.collect()
            case_type.tearDownClass()
