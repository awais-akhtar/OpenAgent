"""Real MSVC C INT8-row math, bounded storage leases and range transactions."""
import ctypes as C
from contextlib import ExitStack, contextmanager
import copy
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.native_segment_executor import NativeSegmentExecutor, NativeExecutionError, _tensor
from openagent.adaptive_engine.native_leased_olmoe_segments import NativeLeasedOlmoeSegmentEngine
from openagent.adaptive_engine.native_olmoe_segments import NativeOlmoeEdge, NativeOlmoeSegmentEngine
from openagent.adaptive_engine.olmoe import OlmoeModel, _read_float_matrix
from openagent.adaptive_engine.olmoe_segments import OlmoeSegmentEngine
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_runtime import SparseExpertRuntime
from openagent.adaptive_engine.expert_store import DiskExpertStore
from tests.test_native_segment_executor import build_executor
from tests.test_native_expert import encode
from tests.test_olmoe_batch import checkpoint
from openagent.adaptive_engine.native_segment_snapshot import write_native_snapshot, restore_native_snapshot


class NativeLeasedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix="oa-native-leased-")
        cls.library = build_executor(cls.directory)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = checkpoint(self.root,True)
        self.payload_bytes = self.manifest.layers[0].regions["0"].size
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

    def engine(self,begin=0,end=2,**kwargs):
        return self.stack.enter_context(NativeLeasedOlmoeSegmentEngine(self.root,self.manifest,
            begin,end,self.library,kwargs.pop("max_resident_bytes",100000),
            kwargs.pop("expert_cache_bytes",self.payload_bytes),**kwargs))

    def edge(self):
        return self.stack.enter_context(NativeOlmoeEdge(self.root,self.library,100000))

    def session(self,engine):
        return self.stack.enter_context(engine.create_session())

    def assert_state(self,actual,expected):
        self.assertEqual(set(actual),set(expected))
        for field in actual:
            if field in ("keys","values"):
                for a,b in zip(actual[field],expected[field]):
                    np.testing.assert_array_equal(a,b)
            else:
                self.assertEqual(actual[field],expected[field])

    def test_native_packed_primitive_odd_shapes_against_numpy_and_raw_errors(self):
        executor = NativeSegmentExecutor(self.library)
        rng = np.random.default_rng(150)
        for h,i in ((1,1),(3,7),(17,11)):
            matrices = [rng.normal(0,.2,shape).astype(np.float32) for shape in ((i,h),(i,h),(h,i))]
            packed = encode(matrices,"int8-row")
            x = rng.normal(size=(4,h)).astype(np.float32)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource({"e":packed}),x,
                [[("e",1.)]]*len(x),hidden_size=h,intermediate_size=i,dtype="int8-row")
            np.testing.assert_allclose(executor.packed_swiglu(x,packed,intermediate=i),expected,
                                       atol=3e-6,rtol=4e-5)
            for bad in (packed[:-1],packed+b"\0"):
                out = np.full_like(x,74.)
                self.assertEqual(executor._packed(_tensor(x),len(x),h,i,C.c_char_p(bad),len(bad),
                    out.ctypes.data,out.size,executor.max_bytes),1)
                np.testing.assert_array_equal(out,74.)
            bad = bytearray(packed)
            struct.pack_into("<f",bad,3*h*i,float("nan"))
            out = np.full_like(x,74.)
            self.assertEqual(executor._packed(_tensor(x),len(x),h,i,C.c_char_p(bytes(bad)),len(bad),
                out.ctypes.data,out.size,executor.max_bytes),1)
            np.testing.assert_array_equal(out,74.)

    def test_real_full_and_split_ranges_match_olmoe_both_routing_modes(self):
        base = self.root
        for normalized in (False,True):
            self.root = base/str(normalized)
            self.root.mkdir()
            self.manifest = checkpoint(self.root,normalized)
            engine,edge = self.engine(),self.edge()
            full = self.session(engine)
            split = [self.session(self.engine(index,index+1)) for index in range(2)]
            reference = self.stack.enter_context(OlmoeModel(self.root,self.manifest,expert_cache_bytes=4096))
            for tokens in ([1,2,3],[7],[4,8]):
                expected = reference.forward(tokens)
                hidden = edge.embeddings(tokens)
                with patch.object(OlmoeSegmentEngine,"forward",side_effect=AssertionError("Python range")), \
                     patch.object(OlmoeModel,"_attention",side_effect=AssertionError("Python attention")), \
                     patch.object(SparseExpertRuntime,"execute_batch",side_effect=AssertionError("Python experts")), \
                     patch("openagent.adaptive_engine.expert_kernels._decode_payload",side_effect=AssertionError("Python decode")):
                    result = full.forward(hidden)
                    for session in split:
                        hidden = session.forward(hidden)
                np.testing.assert_allclose(edge.final_head(result)[-1],expected,atol=4e-6,rtol=5e-5)
                np.testing.assert_allclose(edge.final_head(hidden)[-1],expected,atol=4e-6,rtol=5e-5)
                state = full.snapshot_state()
                for index in range(2):
                    np.testing.assert_allclose(state["keys"][index],reference._k[index][:,:reference.position],
                                               atol=3e-6,rtol=4e-5)
                self.assertLessEqual(engine.status()["expert_cache_used_bytes"],self.payload_bytes)

    def test_loads_no_expert_payloads_then_only_selected_leases(self):
        acquire = DiskExpertStore.acquire
        calls = []
        @contextmanager
        def record(store,key):
            calls.append(key)
            self.assertFalse(any(entry.leases for entry in store._entries.values()))
            with acquire(store,key) as payload:
                self.assertEqual(sum(entry.leases for entry in store._entries.values()),1)
                yield payload
        with patch.object(DiskExpertStore,"acquire",record), \
             patch("openagent.adaptive_engine.olmoe._read_float_matrix",wraps=_read_float_matrix) as read, \
             patch("openagent.adaptive_engine.native_olmoe_segments._decode_expert",side_effect=AssertionError("resident decode")):
            engine = self.engine(0,1)
            self.assertEqual(calls,[])
            self.assertEqual(len(read.call_args_list),9)
            self.assertTrue(all(call.args[1].startswith("model.layers.0.") for call in read.call_args_list))
            self.assertFalse(engine._loader.experts.store._entries)
            s = self.session(engine)
            s.forward(np.ones((3,8),np.float32))
        self.assertEqual(calls,["model.layers.0/0","model.layers.0/1"])
        self.assertTrue(engine.status()["sparse_streamed_experts"])
        self.assertEqual(engine.status()["resident_expert_weight_bytes"],0)
        self.assertEqual(sum(a.nbytes for a in engine._range._weights),engine.resident_weight_bytes)
        for layer in engine._range._layers:
            for name in ("gate","up","down"):
                self.assertIsNone(getattr(layer,name).data)
                self.assertEqual(getattr(layer,name).count,0)
        self.assertFalse(any(entry.leases for entry in engine._loader.experts.store._entries.values()))

    def test_exact_dense_admission_cache_and_transaction_limits(self):
        engine = self.engine()
        required = engine.resident_weight_bytes
        with patch("openagent.adaptive_engine.native_leased_olmoe_segments.OlmoeSegmentEngine",
                   side_effect=AssertionError("loaded before admission")):
            with self.assertRaises(MemoryError):
                self.engine(max_resident_bytes=required-1)
            with self.assertRaises(MemoryError):
                self.engine(expert_cache_bytes=self.payload_bytes-1)
        exact = self.engine(max_resident_bytes=required)
        amount = exact.staging_required(2,exact.context_tokens)
        limited = self.engine(max_staging_bytes=amount-1)
        s = self.session(limited)
        with patch.object(limited._loader.experts.store,"acquire",side_effect=AssertionError("lease before admission")):
            with self.assertRaisesRegex(MemoryError,"max_staging_bytes"):
                s.forward(np.ones((2,8),np.float32))
        self.assertEqual(s.position,0)
        admitted = self.session(self.engine(max_staging_bytes=amount))
        admitted.forward(np.ones((2,8),np.float32))
        self.assertEqual(admitted.position,2)

    def test_late_layer_lease_failure_rolls_back_entire_range(self):
        engine = self.engine()
        s = self.session(engine)
        x = np.ones((2,8),np.float32)
        s.forward(x)
        before = s.snapshot_state()
        full_keys,full_values = s._native._keys.copy(),s._native._values.copy()
        acquire = engine._loader.experts.store.acquire
        @contextmanager
        def fail(key):
            if key.startswith("model.layers.1/"):
                raise OSError("late layer lease failed")
            with acquire(key) as payload:
                yield payload
        with patch.object(engine._loader.experts.store,"acquire",fail):
            with self.assertRaisesRegex(OSError,"late layer"):
                s.forward(x)
        self.assert_state(s.snapshot_state(),before)
        np.testing.assert_array_equal(s._native._keys,full_keys)
        np.testing.assert_array_equal(s._native._values,full_values)
        self.assertFalse(any(entry.leases for entry in engine._loader.experts.store._entries.values()))
        s.forward(x)
        self.assertEqual(s.position,4)

    def test_cancellation_between_expert_leases_rolls_back_and_releases(self):
        engine = self.engine()
        session = self.session(engine)
        x = np.ones((2,8),np.float32)
        session.forward(x)
        before = session.snapshot_state()
        calls = []
        acquire = engine._loader.experts.store.acquire

        @contextmanager
        def record(key):
            calls.append(key)
            with acquire(key) as payload:
                yield payload

        with patch.object(engine._loader.experts.store,"acquire",record):
            with self.assertRaises(InterruptedError):
                session.forward(x,cancelled=lambda: bool(calls))
        self.assertEqual(len(calls),1)
        self.assert_state(session.snapshot_state(),before)
        self.assertFalse(any(entry.leases for entry in engine._loader.experts.store._entries.values()))
        session.forward(x)
        self.assertEqual(session.position,4)

    def test_checksum_rechecked_on_every_lease_and_failure_rolls_back(self):
        engine = self.engine(0,1)
        s = self.session(engine)
        s.forward(np.ones((1,8),np.float32))
        before = s.snapshot_state()
        full_keys,full_values = s._native._keys.copy(),s._native._values.copy()
        store = engine._loader.experts.store
        # Cache permits one payload only. Last selected expert is 1, already
        # verified by the store; mutate cache to test adapter's EVERY-lease hash.
        entry = store._entries["model.layers.0/1"]
        entry.payload[0] ^= 1
        acquire = store.acquire
        @contextmanager
        def serve(key):
            if key=="model.layers.0/1":
                yield memoryview(entry.payload)
            else:
                with acquire(key) as payload:
                    yield payload
        with patch.object(store,"acquire",serve):
            with self.assertRaisesRegex(ValueError,"checksum"):
                s.forward(np.ones((1,8),np.float32))
        self.assert_state(s.snapshot_state(),before)
        np.testing.assert_array_equal(s._native._keys,full_keys)
        np.testing.assert_array_equal(s._native._values,full_values)

    def test_bad_scale_with_valid_hash_fails_native_and_releases_lease(self):
        path = self.root/"experts.i8"
        region = self.manifest.layers[0].regions["0"]
        with path.open("r+b") as stream:
            stream.seek(region.offset)
            packed = bytearray(stream.read(region.size))
            struct.pack_into("<f",packed,3*8*6,0.)
            stream.seek(region.offset)
            stream.write(packed)
        regions = dict(self.manifest.layers[0].regions)
        regions["0"] = replace(region,sha256=hashlib.sha256(packed).hexdigest())
        self.manifest = replace(self.manifest,layers=[replace(self.manifest.layers[0],regions=regions),
                                                    self.manifest.layers[1]])
        engine = self.engine()
        s = self.session(engine)
        before = s.snapshot_state()
        with self.assertRaises(NativeExecutionError):
            s.forward(np.ones((2,8),np.float32))
        self.assert_state(s.snapshot_state(),before)
        self.assertFalse(any(entry.leases for entry in engine._loader.experts.store._entries.values()))
        s.close()
        engine.close()
        path.unlink()

    def test_snapshots_numeric_class_content_binding_and_cancellation(self):
        engine = self.engine()
        a,b = self.session(engine),self.session(engine)
        x = np.ones((2,8),np.float32)
        a.forward(x)
        state = a.snapshot_state()
        b.restore_state(state)
        np.testing.assert_array_equal(a.forward(x),b.forward(x))
        before = a.snapshot_state()
        pending = iter([False,True])
        with self.assertRaises(InterruptedError):
            a.forward(x,cancelled=lambda: next(pending))
        self.assert_state(a.snapshot_state(),before)
        bad = copy.deepcopy(before)
        bad["numeric_class"] = "c-fp32-weights-double-dot-resident-experts-v1"
        with self.assertRaisesRegex(ValueError,"numeric_class"):
            a.restore_state(bad)
        with self.assertRaises(ValueError):
            NativeOlmoeSegmentEngine.compose([engine])
        composed = self.stack.enter_context(NativeLeasedOlmoeSegmentEngine.compose([engine]))
        self.assertTrue(composed.status()["sparse_streamed_experts"])

    def test_composed_decoder_differential_resume_and_collective_rollback(self):
        left,right = self.engine(0,1),self.engine(1,2)
        merged = self.stack.enter_context(NativeLeasedOlmoeSegmentEngine.compose([left,right]))
        edge = self.edge()
        decoder = self.stack.enter_context(merged.create_decoder_session(edge))
        reference = self.stack.enter_context(OlmoeModel(self.root,self.manifest,expert_cache_bytes=4096))
        for tokens in ([1,2,3],[7],[4,8]):
            expected = np.stack([reference.forward([token]) for token in tokens])
            np.testing.assert_allclose(decoder.forward(tokens),expected,atol=4e-6,rtol=5e-5)
        before = decoder.snapshot_state()
        with patch.object(right._loader.experts.store,"acquire",side_effect=OSError("second range failed")):
            with self.assertRaisesRegex(OSError,"second range"):
                decoder.forward([3])
        self.assert_state(decoder.snapshot_state()["range_state"],before["range_state"])
        with patch.object(edge,"final_head",side_effect=ValueError("head failed")):
            with self.assertRaisesRegex(ValueError,"head failed"):
                decoder.forward([3])
        self.assert_state(decoder.snapshot_state()["range_state"],before["range_state"])
        calls = []
        acquire = right._loader.experts.store.acquire
        @contextmanager
        def record(key):
            calls.append(key)
            with acquire(key) as payload:
                yield payload
        with patch.object(right._loader.experts.store,"acquire",record):
            with self.assertRaises(InterruptedError):
                decoder.forward([3],cancelled=lambda: bool(calls))
        self.assertTrue(calls)
        self.assert_state(decoder.snapshot_state()["range_state"],before["range_state"])
        for e in (left,right):
            self.assertFalse(any(entry.leases for entry in e._loader.experts.store._entries.values()))
            with self.assertRaisesRegex(RuntimeError,"compositions"):
                e.close()
        resumed = self.stack.enter_context(merged.create_decoder_session(edge))
        resumed.restore_state(before)
        np.testing.assert_array_equal(decoder.forward([3]),resumed.forward([3]))

    def test_composition_admission_identity_and_nested_lifecycle(self):
        left,right = self.engine(0,1),self.engine(1,2)
        for sources in ([],[left,left],[right,left],[left,self.engine(0,1)]):
            with self.assertRaises((TypeError,ValueError)):
                NativeLeasedOlmoeSegmentEngine.compose(sources)
        budget = left.resident_weight_bytes+right.resident_weight_bytes
        with self.assertRaises(MemoryError):
            NativeLeasedOlmoeSegmentEngine.compose([left,right],max_resident_bytes=budget-1)
        with self.assertRaises(ValueError):
            NativeLeasedOlmoeSegmentEngine.compose([left,right],context_tokens=left.context_tokens+1)
        self.assertFalse(left._compositions)
        original = right.manifest
        right.manifest = replace(original,fingerprint="different-checkpoint")
        try:
            with self.assertRaisesRegex(ValueError,"manifests"):
                NativeLeasedOlmoeSegmentEngine.compose([left,right])
        finally:
            right.manifest = original
        self.assertFalse(left._compositions)
        merged = self.stack.enter_context(NativeLeasedOlmoeSegmentEngine.compose([left,right],max_resident_bytes=budget))
        nested = self.stack.enter_context(NativeLeasedOlmoeSegmentEngine.compose([merged]))
        with self.assertRaisesRegex(RuntimeError,"compositions"):
            merged.close()
        nested.close()
        session = self.session(merged)
        needed = merged.staging_required(2,session.context_tokens)
        merged.max_staging_bytes = needed-1
        with self.assertRaises(MemoryError):
            session.forward(np.ones((2,8),np.float32))
        self.assertEqual(session.position,0)
        merged.max_staging_bytes = needed
        session.forward(np.ones((2,8),np.float32))
        state = session.snapshot_state()
        other = self.stack.enter_context(NativeLeasedOlmoeSegmentEngine.compose([left,right]))
        self.assertEqual(merged.execution_identity,other.execution_identity)
        target = self.session(other)
        target.restore_state(state)
        altered = copy.deepcopy(state)
        altered["execution_identity"] = "0"*64
        with self.assertRaisesRegex(ValueError,"execution_identity"):
            target.restore_state(altered)

    def test_native_binary_composed_decoder_resume_and_corruption_rollback(self):
        sources = [self.engine(0,1),self.engine(1,2)]
        merged = self.stack.enter_context(NativeLeasedOlmoeSegmentEngine.compose(sources))
        edge = self.edge()
        a = self.stack.enter_context(merged.create_decoder_session(edge))
        b = self.stack.enter_context(merged.create_decoder_session(edge))
        a.forward([1,2,3])
        wire = io.BytesIO()
        with patch.object(a,"snapshot_state",side_effect=AssertionError("full snapshot copy")):
            write_native_snapshot(a,wire)
        blob = wire.getvalue()
        restore_native_snapshot(b,io.BytesIO(blob))
        np.testing.assert_array_equal(a.forward([7]),b.forward([7]))
        before = b.snapshot_state()
        corrupt = bytearray(blob)
        corrupt[-1] ^= 1
        nonfinite = bytearray(blob)
        payload_start = 12+struct.unpack_from('<I',nonfinite,8)[0]
        struct.pack_into('<f',nonfinite,payload_start,float('nan'))
        nonfinite[-32:] = hashlib.sha256(nonfinite[:-32]).digest()
        for data in (bytes(corrupt),bytes(nonfinite),blob[:-1],blob+b'x',b'badmagic'+blob[8:]):
            with self.assertRaises(ValueError):
                restore_native_snapshot(b,io.BytesIO(data))
            self.assert_state(b.snapshot_state()["range_state"],before["range_state"])
        with self.assertRaises(MemoryError):
            restore_native_snapshot(b,io.BytesIO(blob),max_bytes=len(blob)-1)
        self.assert_state(b.snapshot_state()["range_state"],before["range_state"])
        original = edge.execution_identity
        edge.execution_identity = '0'*64
        try:
            with self.assertRaisesRegex(ValueError,"identity"):
                restore_native_snapshot(b,io.BytesIO(blob))
        finally:
            edge.execution_identity = original

    def test_native_binary_empty_range_short_writes_and_budget(self):
        engine = self.engine()
        a,b = self.session(engine),self.session(engine)
        class ShortWriter(io.BytesIO):
            def write(self,data):
                return super().write(data[:7])
        output = ShortWriter()
        write_native_snapshot(a,output)
        restore_native_snapshot(b,io.BytesIO(output.getvalue()))
        self.assertEqual(b.position,0)
        blocked = io.BytesIO()
        with self.assertRaises(MemoryError):
            write_native_snapshot(a,blocked,max_bytes=44)
        self.assertEqual(blocked.getvalue(),b'')
        a.forward(np.ones((2,8),np.float32))
        output = ShortWriter()
        write_native_snapshot(a,output)
        restore_native_snapshot(b,io.BytesIO(output.getvalue()))
        np.testing.assert_array_equal(a.forward(np.ones((1,8),np.float32)),b.forward(np.ones((1,8),np.float32)))

    def test_leased_token_to_logit_head_failure_atomicity_and_resume(self):
        engine,edge = self.engine(),self.edge()
        with self.assertRaisesRegex(RuntimeError,"leased ranges"):
            edge.create_session(engine)
        decoder = self.stack.enter_context(engine.create_decoder_session(edge))
        reference = self.stack.enter_context(OlmoeModel(self.root,self.manifest,expert_cache_bytes=4096))
        for tokens in ([1,2,3],[7],[4,8]):
            expected = np.stack([reference.forward([token]) for token in tokens])
            np.testing.assert_allclose(decoder.forward(tokens),expected,atol=4e-6,rtol=5e-5)
        before = decoder.snapshot_state()
        with patch.object(edge,"final_head",side_effect=ValueError("late head failure")):
            with self.assertRaisesRegex(ValueError,"late head"):
                decoder.forward([3])
        self.assert_state(decoder.snapshot_state()["range_state"],before["range_state"])
        norm,head = edge.final_norm,edge.lm_head
        edge.final_norm = np.full(8,np.finfo(np.float32).max,np.float32)
        edge.lm_head = np.full((13,8),np.finfo(np.float32).max,np.float32)
        with self.assertRaises(NativeExecutionError):
            decoder.forward([3])
        self.assert_state(decoder.snapshot_state()["range_state"],before["range_state"])
        edge.final_norm,edge.lm_head = norm,head
        other = self.stack.enter_context(engine.create_decoder_session(edge))
        other.restore_state(before)
        np.testing.assert_array_equal(other.forward([3]),decoder.forward([3]))
        with self.assertRaisesRegex(RuntimeError,"live"):
            edge.close()

    def test_routed_c_contract_rejects_duplicate_ids_without_output_mutation(self):
        executor = NativeSegmentExecutor(self.library)
        packed = encode([np.ones(shape,np.float32) for shape in ((3,2),(3,2),(2,3))],"int8-row")
        x = np.ones((2,2),np.float32)
        accumulator = np.zeros_like(x)
        out = np.full_like(x,41.)
        ids = np.array([[0,0],[0,1]],np.uint64)
        weights = np.full((2,2),.5,np.float32)
        self.assertEqual(executor._apply_packed(_tensor(x),2,2,3,ids.ctypes.data,weights.ctypes.data,
            ids.size,2,3,0,C.c_char_p(packed),len(packed),_tensor(accumulator),
            out.ctypes.data,out.size,executor.max_bytes),1)
        np.testing.assert_array_equal(out,41.)

    def test_prepare_raw_envelopes_preserve_all_phase_outputs_and_kv(self):
        engine = self.engine(0,1)
        session = self.session(engine)
        cfg = type(engine._range.config).from_buffer_copy(engine._range.config)
        x = np.ones((2,8),np.float32)
        residual,normalized = np.full_like(x,52.),np.full_like(x,53.)
        ids,weights = np.full((2,2),99,np.uint64),np.full((2,2),54.,np.float32)
        keys,values = session._native._keys[0],session._native._values[0]
        args = [C.byref(cfg),C.byref(engine._range._layers[0]),_tensor(x),2,
            residual.ctypes.data,normalized.ctypes.data,x.size,ids.ctypes.data,
            weights.ctypes.data,ids.size,keys.ctypes.data,values.ctypes.data,keys.size,0,engine.max_bytes]
        for index,value,status in ((0,None,1),(1,None,1),(6,15,1),(9,3,1),(12,keys.size-1,1),
                                   (13,11,5),(14,4,5)):
            bad = list(args)
            bad[index] = value
            self.assertEqual(engine.executor._prepare(*bad),status)
            np.testing.assert_array_equal(residual,52.)
            np.testing.assert_array_equal(normalized,53.)
            np.testing.assert_array_equal(ids,99)
            np.testing.assert_array_equal(weights,54.)
            np.testing.assert_array_equal(keys,0.)
            np.testing.assert_array_equal(values,0.)
        cfg.layers = 2
        self.assertEqual(engine.executor._prepare(*args),1)
        np.testing.assert_array_equal(residual,52.)

    def test_packed_late_row_overflow_is_transactional(self):
        executor = NativeSegmentExecutor(self.library)
        packed = encode([np.ones(shape,np.float32) for shape in ((3,2),(3,2),(2,3))],"int8-row")
        x = np.array([[1.,1.],[np.finfo(np.float32).max,np.finfo(np.float32).max]],np.float32)
        out = np.full_like(x,65.)
        self.assertEqual(executor._packed(_tensor(x),2,2,3,C.c_char_p(packed),len(packed),
            out.ctypes.data,out.size,executor.max_bytes),1)
        np.testing.assert_array_equal(out,65.)

    def test_leased_snapshot_binds_manifest_expert_hash_before_any_leases(self):
        engine = self.engine()
        source = self.session(engine)
        source.forward(np.ones((1,8),np.float32))
        regions = dict(self.manifest.layers[0].regions)
        regions["0"] = replace(regions["0"],sha256="0"*64)
        self.manifest = replace(self.manifest,layers=[replace(self.manifest.layers[0],regions=regions),
                                                    self.manifest.layers[1]])
        target_engine = self.engine()
        target = self.session(target_engine)
        before = target.snapshot_state()
        with patch.object(target_engine._loader.experts.store,"acquire",side_effect=AssertionError("lease during restore")):
            with self.assertRaisesRegex(ValueError,"execution_identity"):
                target.restore_state(source.snapshot_state())
        self.assert_state(target.snapshot_state(),before)


if __name__ == "__main__":
    unittest.main()
