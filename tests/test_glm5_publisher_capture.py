import numpy as np
import unittest
from unittest.mock import patch, Mock
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile

from openagent.adaptive_engine.glm5_publisher_capture import (
    _routes_from_result,
    glm5_publisher_case_id,
)
from openagent.adaptive_engine.independent_glm5_vision import IndependentGlm5VisionEncoder
from openagent.adaptive_engine.glm5_vision import Glm5VisionEncoder, vision_shapes


class Glm5PublisherCaptureTests(unittest.TestCase):
  def test_device_capture_reaches_real_loader_signature_with_explicit_null_text_library(self):
    from openagent.adaptive_engine.glm5_publisher_capture import capture_glm5_publisher_route
    class ReachedLoader(Exception):pass
    with tempfile.TemporaryDirectory() as folder:
      root=Path(folder);checkpoint=root/'checkpoint';checkpoint.mkdir()
      (checkpoint/'config.json').write_text(json.dumps({'model_type':'glm5_next',
          'text_config':{'num_hidden_layers':1}}))
      media=root/'media.dll';media.write_bytes(b'boundary-only-test-library')
      image=root/'input.png';image.write_bytes(b'boundary-only-test-input')
      inventory=SimpleNamespace(tensors={'model.weight':SimpleNamespace(dtype='F32')})
      processor=Mock()
      processor.prepare.return_value=((1,),{'pixel_values':np.ones((1,1),np.float32)})
      with patch('openagent.adaptive_engine.publisher_checkpoint.storage.snapshot',return_value=inventory), \
           patch('openagent.adaptive_engine.source_scope.require_source_inventory_admission',return_value=('F32',)), \
           patch('openagent.adaptive_engine.publisher_media_processor.PublisherMediaProcessor',return_value=processor), \
           patch('openagent.adaptive_engine.publisher_conditional_loader.load_publisher_conditional_decoder',
                 autospec=True,side_effect=ReachedLoader('signature accepted')) as loader:
        with self.assertRaisesRegex(ReachedLoader,'signature accepted'):
          capture_glm5_publisher_route(checkpoint,source_format='F32',backend='device',
              media_library=media,reference_library=media,device='cuda:0',
              prompt='test',image_paths=[image],max_tokens=1,recovery_tokens=1,context_tokens=8)
        self.assertIn('text_library',loader.call_args.kwargs)
        self.assertIsNone(loader.call_args.kwargs['text_library'])
      processor.close.assert_called_once()

  def test_glm5_publisher_capture_resolves_all_and_only_twelve_admitted_routes(self):
    formats = ("F32", "F16", "BF16", "F8_E4M3")
    backends = ("native", "device", "public-range")
    cases = {glm5_publisher_case_id(backend, source_format)
             for backend in backends for source_format in formats}
    self.assertEqual(len(cases), 12)
    self.assertTrue(all("zai-org/GLM-5.3-Flash/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a" in case
                        for case in cases))

  def test_glm5_publisher_capture_rejects_unadmitted_tuple(self):
    from openagent.adaptive_engine.device_capture import CaptureError
    for backend, source_format in (("bogus", "F32"), ("native", "I8")):
      with self.subTest(backend=backend, source_format=source_format):
        with self.assertRaises(CaptureError):
          glm5_publisher_case_id(backend, source_format)

  def test_independent_vision_uses_numpy_equations_not_production_forward(self):
    config = {
        "depth": 1, "hidden_size": 4, "num_heads": 1, "in_channels": 1,
        "patch_size": 1, "temporal_patch_size": 1, "spatial_merge_size": 1,
        "out_hidden_size": 2, "intermediate_size": 2,
        "projection_intermediate_size": 2, "rms_norm_eps": 1e-5,
        "swiglu_limit": 4.0, "attention_bias": False,
        "attention_dropout": 0.0, "hidden_act": "silu",
    }
    _, shapes = vision_shapes(config)
    tensors = {"model.visual." + name: np.zeros(shape, dtype=np.float32)
               for name, shape in shapes.items()}
    with patch.object(Glm5VisionEncoder, "forward",
                      side_effect=AssertionError("production vision forward was used")):
      self._run_independent_vision(config, tensors)

  def _run_independent_vision(self, config, tensors):
    reference = IndependentGlm5VisionEncoder(config, tensors, max_bytes=1024 * 1024)
    try:
        result = reference.forward(np.ones((1, 1), dtype=np.float32),
                                   np.ones((1, 3), dtype=np.int64))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (1, 2))
        self.assertTrue(np.isfinite(result[0]).all())
    finally:
        reference.close()

  def test_glm_route_observation_is_executed_topk_with_exact_geometry(self):
    class Contract:
        n_routed_experts = 4
        num_experts_per_tok = 2
        mlp_layer_types = ("sparse",)

    class Decoder:
        contract = Contract()

    weights = np.array([[[[0.1, 0.8, 0.0, 0.1]],
                         [[0.0, 0.2, 0.7, 0.1]]]], dtype=np.float32)
    result = _routes_from_result({"routing_weights": weights}, Decoder())
    self.assertEqual(result, [[[1, 0], [2, 1]]])

  def test_dense_glm_mlp_layers_record_empty_route_rows(self):
    class Contract:
        n_routed_experts = 4
        num_experts_per_tok = 2
        mlp_layer_types = ("dense", "sparse")

    class Decoder:
        contract = Contract()

    weights = np.array([[[[0.0, 0.0, 0.0, 0.0], [0.1, 0.8, 0.0, 0.1]],
                         [[0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.7, 0.1]]]], dtype=np.float32)
    self.assertEqual(_routes_from_result({"routing_weights": weights}, Decoder()),
                     [[[], []], [[1, 0], [2, 1]]])
