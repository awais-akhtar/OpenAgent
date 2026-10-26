"""Exercise capture output and continuation wiring without writing model evidence."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from openagent.adaptive_engine.qwen4_media_capture import measure_qwen4_media_route


class _Owner:
    def __init__(self):self.position=0
    def snapshot(self):return self.position
    def restore(self,value):self.position=value
    def release_snapshot(self,value):pass
    def forward(self,ids,**kwargs):
        callback=kwargs.get('cancelled')
        if callback is not None and (callback() or callback()):raise InterruptedError('cancelled')
        count=np.asarray(ids).size;self.position+=count
        return SimpleNamespace(logits=np.tile(np.array([0,2,0,0],np.float32),(1,count,1)),
            routing_weights=np.tile(np.array([.8,.2],np.float32),(1,count,1,1)))


class QwenMediaMeasurementTests(unittest.TestCase):
    def test_prefill_emits_one_logit_row_per_generated_token_and_retains_rope_delta(self):
        candidate=_Owner();reference=_Owner()
        config={'num_hidden_layers':1,'num_experts':2,'num_experts_per_tok':1,
                'vocab_size':4,'max_position_embeddings':32}
        text_type=type('NativeQwen4TextDecoder',(),{'__module__':'openagent.adaptive_engine.native_qwen4'})
        candidate.text=text_type();candidate.text._cache=(np.ones((1,2),np.float32),)
        candidate.implementation='test-measurement-owner'
        candidate.config={'image_token_id':2,'video_token_id':3}
        candidate.components=SimpleNamespace(vision_config={})
        deltas=[]
        def independent(text,vision,ids,**kwargs):
            deltas.append(kwargs.get('rope_delta',0))
            return text.forward(ids),None,7
        route=dict(family='qwen4',model_type='qwen4_exp',backend='native',
                   arithmetic='fp32',source_format='F32',persisted_format='float32')
        with patch('openagent.adaptive_engine.qwen4_media_capture.independent_qwen4_media_forward',
                   side_effect=independent):
            result=measure_qwen4_media_route(candidate=candidate,reference_text=reference,
                reference_vision=object(),token_ids=[0,2,0],media={
                    'pixel_values':np.ones((1,1),np.float32),'image_grid_thw':np.ones((1,3),np.int64)},
                route=route,config=config,artifacts=[],max_tokens=2,recovery_tokens=1,tolerance=1e-3)
        self.assertIsInstance(result['candidate_logits'],np.ndarray)
        self.assertEqual(result['candidate_logits'].shape,(2,4))
        self.assertEqual(result['reference_logits'].shape,(2,4))
        self.assertEqual(result['routing_layers']['model.layers.0'],[[0],[0],[0],[0]])
        self.assertEqual(deltas,[0,7,7])
        self.assertEqual(candidate.position,4)


if __name__=='__main__':unittest.main()
