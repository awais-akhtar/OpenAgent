"""Compiled direct Q8 blocks, native leased ranges and transaction evidence."""
import ctypes as C
from contextlib import ExitStack
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.native_segment_executor import NativeSegmentExecutor, _tensor
from openagent.adaptive_engine.native_leased_olmoe_segments import NativeLeasedOlmoeSegmentEngine, Q8_NUMERIC_CLASS
from openagent.adaptive_engine.native_olmoe_segments import NativeOlmoeEdge
from openagent.adaptive_engine.native_segment_snapshot import write_native_snapshot, restore_native_snapshot
from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.q8_0_expert import pack_expert
from tests.test_q8_0_olmoe import checkpoint, scalar_expert
from tests.test_native_segment_executor import build_executor


class NativeQ8Segments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = build_executor(tempfile.mkdtemp(prefix='oa-native-q8-'))

    def test_direct_blocks_against_scalar_and_raw_rejections(self):
        executor = NativeSegmentExecutor(self.library)
        rng = np.random.default_rng(901)
        for h,i in ((32,32),(64,96)):
            payload = pack_expert(*(rng.normal(0,.03,shape).astype(np.float32)
                for shape in ((i,h),(i,h),(h,i))))
            x = rng.normal(0,.4,(3,h)).astype(np.float32)
            expected = np.stack([scalar_expert(payload,row,h,i) for row in x])
            np.testing.assert_allclose(executor.q8_swiglu(x,payload,intermediate=i),expected,atol=2e-6,rtol=4e-5)
            # Signed/zero/subnormal scales and -128 are valid, not row-INT8 rules.
            modified = bytearray(payload)
            for offset,value in ((0,-.01),(34,0.),(68,2**-24)):
                struct.pack_into('<e',modified,offset,value)
            modified[2] = 128
            expected = np.stack([scalar_expert(modified,row,h,i) for row in x])
            np.testing.assert_allclose(executor.q8_swiglu(x,modified,intermediate=i),expected,atol=2e-6,rtol=4e-5)
            invalid = bytearray(payload)
            struct.pack_into('<H',invalid,0,0x7c00)
            for bad in (payload[:-1],payload+b'x',bytes(invalid)):
                output = np.full_like(x,51.)
                self.assertNotEqual(executor._packed_q8(_tensor(x),len(x),h,i,C.c_char_p(bad),len(bad),
                    output.ctypes.data,output.size,executor.max_bytes),0)
                np.testing.assert_array_equal(output,51.)
            ids = np.array([[0,0]]*len(x),np.uint64)
            weights = np.full(ids.shape,.5,np.float32)
            accumulator = np.zeros_like(x)
            output = np.full_like(x,51.)
            self.assertNotEqual(executor._apply_q8(_tensor(x),len(x),h,i,ids.ctypes.data,
                weights.ctypes.data,ids.size,2,3,0,C.c_char_p(payload),len(payload),
                _tensor(accumulator),output.ctypes.data,output.size,executor.max_bytes),0)
            np.testing.assert_array_equal(output,51.)
        executor._packed_q8 = None
        with self.assertRaisesRegex(RuntimeError,'Q8_0 exports'):
            executor.q8_swiglu(x,payload,intermediate=i)

    def test_full_composed_q8_decoder_resume_without_python_math(self):
        with ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            checkpoint(root)
            manifest = pack_olmoe_experts(root,root/'experts.q8',dtype='q8_0')
            def engine(begin,end):
                return stack.enter_context(NativeLeasedOlmoeSegmentEngine(root,manifest,begin,end,
                    self.library,1024**2,manifest.layers[0].regions['0'].size))
            full = engine(0,2)
            with patch.object(NativeSegmentExecutor,'require_q8',side_effect=RuntimeError('Q8_0 exports missing')):
                with patch('openagent.adaptive_engine.native_leased_olmoe_segments._Q8DenseLoader',
                           side_effect=AssertionError('loaded before export check')):
                    with self.assertRaisesRegex(RuntimeError,'exports missing'):
                        engine(0,2)
            composed = stack.enter_context(NativeLeasedOlmoeSegmentEngine.compose([engine(0,1),engine(1,2)]))
            edge = stack.enter_context(NativeOlmoeEdge(root,self.library,1024**2))
            a = stack.enter_context(full.create_decoder_session(edge))
            b = stack.enter_context(composed.create_decoder_session(edge))
            reference = stack.enter_context(OlmoeModel(root,manifest,expert_cache_bytes=65536))
            for tokens in ([1,2,3],[7],[4,8]):
                expected = np.stack([reference.forward([token]) for token in tokens])
                with patch.object(OlmoeModel,'_attention',side_effect=AssertionError('Python math')):
                    actual = a.forward(tokens)
                    split = b.forward(tokens)
                np.testing.assert_allclose(actual,expected,atol=8e-6,rtol=1e-4)
                np.testing.assert_array_equal(actual,split)
            self.assertEqual(b.snapshot_state()['range_state']['numeric_class'],Q8_NUMERIC_CLASS)
            wire = io.BytesIO()
            write_native_snapshot(b,wire)
            other = stack.enter_context(composed.create_decoder_session(edge))
            restore_native_snapshot(other,io.BytesIO(wire.getvalue()))
            np.testing.assert_array_equal(b.forward([5]),other.forward([5]))
            before = b.snapshot_state()
            incompatible = dict(before,range_state=dict(before['range_state'],numeric_class='int8-row'))
            with self.assertRaisesRegex(ValueError,'numeric_class'):
                b.restore_state(incompatible)
            with patch.object(edge,'final_head',side_effect=ValueError('head failure')):
                with self.assertRaises(ValueError):
                    b.forward([3])
            self.assertEqual(b.position,before['range_state']['position'])
            for old,new in zip(before['range_state']['keys'],b.snapshot_state()['range_state']['keys']):
                np.testing.assert_array_equal(old,new)
            self.assertLessEqual(full.status()['expert_cache_used_bytes'],full.expert_cache_bytes)


if __name__=='__main__':
    unittest.main()
