"""Exact wrapper loading, explicit libraries and compiled image/text execution."""
import gc
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

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

    def test_local_shards_preserve_wrapper_and_run_native_image_text(self):
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
            self.skipTest('optional exact multimodal oracle unavailable')
        import torch
        import transformers
        if transformers.__version__ != '5.16.1':
            self.skipTest('exact multimodal oracle requires Transformers 5.16.1')
        from tests.test_native_qwen35_vision import Qwen35VisionTests
        case_type = Qwen35VisionTests
        case_type.setUpClass()
        native = oracle = loaded = None
        try:
            case = case_type()
            oracle, native, weights = case.fixture()
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
                loaded = load_native_multimodal_decoder(root, model_type='qwen3_5_moe',
                    library=case.vision_path, text_library=case.text_path,
                    max_weight_bytes=64*1024**2, max_bytes=256*1024**2)
                ids = torch.tensor([[1, 32, 35, 33, 7]])
                grid = torch.tensor([[1, 2, 2]])
                pixels = torch.tensor([[.2, -.4, .7], [.9, .1, -.8],
                                       [-.6, .3, .5], [.4, -.2, -.1]])
                with torch.no_grad():
                    expected = oracle(ids, pixel_values=pixels, image_grid_thw=grid,
                                      mm_token_type_ids=(ids == 35).long(), use_cache=True)
                actual = loaded.forward(ids.numpy(), pixel_values=pixels.numpy(), image_grid_thw=grid.numpy())
                np.testing.assert_allclose(actual.logits, expected.logits.numpy(), atol=4e-6, rtol=5e-5)
                with torch.no_grad():
                    tail = oracle(torch.tensor([[8]]), past_key_values=expected.past_key_values)
                np.testing.assert_allclose(loaded.forward([8]).logits, tail.logits.numpy(), atol=4e-6, rtol=5e-5)
        finally:
            loaded = native = oracle = None
            gc.collect()
            case_type.tearDownClass()
