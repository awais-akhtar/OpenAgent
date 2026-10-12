"""Real checkpoint loading, bounded resident conversion and native-only calls."""
import copy
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import native_olmoe_segments as native
from openagent.adaptive_engine.native_segment_executor import NativeExecutionError
from openagent.adaptive_engine.olmoe import OlmoeModel, _read_float_matrix
from openagent.adaptive_engine.olmoe_segments import OlmoeSegmentEngine
from openagent.adaptive_engine.expert_runtime import SparseExpertRuntime
from openagent.adaptive_engine.expert_store import DiskExpertStore
from tests.test_native_segment_executor import build_executor
from tests.test_olmoe_batch import checkpoint


class NativeOlmoeSegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_directory = tempfile.mkdtemp(prefix="oa-native-checkpoint-")
        cls.library = build_executor(cls.build_directory)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = checkpoint(self.root,True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

    def engine(self,begin=0,end=2,**kwargs):
        return self.stack.enter_context(native.NativeOlmoeSegmentEngine(
            self.root,self.manifest,begin,end,self.library,kwargs.pop("max_resident_bytes",100000),**kwargs))

    def edge(self,**kwargs):
        return self.stack.enter_context(native.NativeOlmoeEdge(self.root,self.library,
            kwargs.pop("max_resident_bytes",100000),**kwargs))

    def session(self,engine,**kwargs):
        return self.stack.enter_context(engine.create_session(**kwargs))

    def assert_state(self,actual,expected):
        self.assertEqual(set(actual),set(expected))
        for field in actual:
            if field in ("keys","values"):
                for a,b in zip(actual[field],expected[field]):
                    np.testing.assert_array_equal(a,b)
            else:
                self.assertEqual(actual[field],expected[field])

    def test_loaded_native_chain_matches_model_and_composed_ranges(self):
        base = self.root
        for normalize in (False,True):
            # Fresh manifest fingerprints follow each fixture's config.
            self.root = base / str(normalize)
            self.root.mkdir()
            self.manifest = checkpoint(self.root,normalize)
            edge = self.edge()
            engine = self.engine()
            parts = [self.engine(layer_index,layer_index+1) for layer_index in range(2)]
            full = self.session(engine)
            split = [self.session(e) for e in parts]
            reference = self.stack.enter_context(OlmoeModel(self.root,self.manifest,expert_cache_bytes=4096))
            for ids in ([1,2,3],[7],[4,8],[0]):
                expected = reference.forward(ids)
                with patch.object(OlmoeSegmentEngine,"forward",side_effect=AssertionError("Python range")), \
                     patch.object(OlmoeModel,"_attention",side_effect=AssertionError("Python attention")), \
                     patch.object(SparseExpertRuntime,"execute_batch",side_effect=AssertionError("Python experts")):
                    x = edge.embeddings(ids)
                    direct = full.forward(x)
                    for s in split:
                        x = s.forward(x)
                    np.testing.assert_allclose(edge.final_head(direct)[-1],expected,atol=3e-6,rtol=4e-5)
                    np.testing.assert_allclose(edge.final_head(x)[-1],expected,atol=3e-6,rtol=4e-5)
                snapshot = full.snapshot_state()
                for layer_index in range(2):
                    np.testing.assert_allclose(snapshot["keys"][layer_index],reference._k[layer_index][:,:reference.position],
                                               atol=2e-6,rtol=3e-5)
                    np.testing.assert_allclose(snapshot["values"][layer_index],reference._v[layer_index][:,:reference.position],
                                               atol=2e-6,rtol=3e-5)

    def test_only_requested_payloads_and_disk_closed_once_weights_owned(self):
        leased,closed = [],[]
        acquire,close = DiskExpertStore.acquire,SparseExpertRuntime.close

        def record_acquire(store,key):
            leased.append(key)
            return acquire(store,key)

        def record_close(runtime):
            closed.append(runtime)
            return close(runtime)

        with patch("openagent.adaptive_engine.olmoe._read_float_matrix",wraps=_read_float_matrix) as read, \
             patch.object(DiskExpertStore,"acquire",record_acquire), \
             patch.object(SparseExpertRuntime,"close",record_close):
            engine = self.engine(1,2)
        self.assertEqual(len(read.call_args_list),9)
        self.assertTrue(all(call.args[1].startswith("model.layers.1.") for call in read.call_args_list))
        self.assertEqual(leased,[f"model.layers.1/{e}" for e in range(3)])
        self.assertEqual(len(closed),1)
        self.assertTrue(closed[0].store._closed)
        self.assertEqual(closed[0].store._entries,{})
        self.assertFalse(engine.status()["disk_runtime_open"])
        # Weight execution remains usable after its expert file disappears.
        (self.root / "experts.i8").unlink()
        s = self.session(engine)
        self.assertTrue(np.isfinite(s.forward(np.ones((1,8),np.float32))).all())

    def test_exact_aggregate_admission_before_any_weight_allocation(self):
        c = native.OlmoeConfig.read(self.root)
        per_layer = sum(native._bytes(s) for s in native._shapes(c).values())
        with patch.object(native,"OlmoeSegmentEngine",side_effect=AssertionError("loaded too early")), \
             patch.object(np,"empty",side_effect=AssertionError("allocated too early")), \
             patch("openagent.adaptive_engine.olmoe._read_float_matrix",side_effect=AssertionError("read too early")):
            with self.assertRaisesRegex(MemoryError,f"require {2*per_layer} bytes"):
                self.engine(max_resident_bytes=2*per_layer-1)
            with self.assertRaisesRegex(MemoryError,"max_conversion_bytes"):
                self.engine(max_conversion_bytes=1)
            with self.assertRaisesRegex(MemoryError,"max_bytes"):
                self.engine(max_bytes=16)
        engine = self.engine(max_resident_bytes=2*per_layer)
        self.assertEqual(engine.resident_weight_bytes,2*per_layer)
        self.assertEqual(sum(a.nbytes for a in engine._range._weights),2*per_layer)
        # Dense arrays are retained directly, not duplicated by the array facade.
        self.assertTrue(all(not a.flags.writeable for a in engine._range._weights))
        self.assertFalse(engine.status()["sparse_streamed_experts"])

    def test_edge_selected_loading_admission_and_close(self):
        with patch("openagent.adaptive_engine.olmoe._read_float_matrix",side_effect=AssertionError("read early")):
            with self.assertRaisesRegex(MemoryError,"max_conversion_bytes"):
                self.edge(max_conversion_bytes=1)
        for mode,names,required in (
            ("embedding",{"model.embed_tokens.weight"},13*8*4),
            ("head",{"lm_head.weight","model.norm.weight"},(13*8+8)*4),
            ("both",{"model.embed_tokens.weight","lm_head.weight","model.norm.weight"},(2*13*8+8)*4)):
            with patch("openagent.adaptive_engine.olmoe._read_float_matrix",side_effect=AssertionError("read early")):
                with self.assertRaises(MemoryError):
                    self.edge(mode=mode,max_resident_bytes=required-1)
            with patch("openagent.adaptive_engine.olmoe._read_float_matrix",wraps=_read_float_matrix) as read:
                edge = self.edge(mode=mode,max_resident_bytes=required)
            self.assertEqual({call.args[1] for call in read.call_args_list},names)
            self.assertEqual(edge.status()["resident_weight_bytes"],required)
            if mode == "embedding":
                with self.assertRaises(RuntimeError):
                    edge.final_head(np.ones((1,8),np.float32))
            if mode == "head":
                with self.assertRaises(RuntimeError):
                    edge.embeddings([1])
            edge.close()
            edge.close()
            with self.assertRaises(RuntimeError):
                edge.embeddings([1])

    def test_snapshots_continuation_isolation_reset_and_capacities(self):
        engine,edge = self.engine(),self.edge()
        a,b = self.session(engine,context_tokens=7),self.session(engine,context_tokens=7)
        a.forward(edge.embeddings([1,2,3]))
        state = a.snapshot_state()
        b.restore_state(state)
        self.assert_state(b.snapshot_state(),state)
        state["keys"][0].fill(123)
        self.assertFalse(np.any(b.snapshot_state()["keys"][0] == 123))
        x = edge.embeddings([4])
        np.testing.assert_array_equal(a.forward(x),b.forward(x))
        saved = a.snapshot_state()
        b.reset()
        self.assertEqual(b.position,0)
        b.restore_state(saved)
        np.testing.assert_array_equal(a.forward(x),b.forward(x))
        self.assertEqual(a.position,5)
        small = self.session(engine,context_tokens=2)
        before = small.snapshot_state()
        with self.assertRaises(NativeExecutionError):
            small.forward(edge.embeddings([1,2,3]))
        self.assert_state(small.snapshot_state(),before)
        for capacity in (0,True,11,1.5):
            with self.assertRaises(ValueError):
                engine.create_session(context_tokens=capacity)

    def test_cancellation_and_c_failures_do_not_commit(self):
        engine,edge = self.engine(),self.edge()
        s = self.session(engine)
        s.forward(edge.embeddings([1,2]))
        before = s.snapshot_state()
        x = edge.embeddings([3])
        for values in ((True,),(False,True)):
            check = iter(values)
            with self.assertRaises(InterruptedError):
                s.forward(x,cancelled=lambda: next(check))
            self.assert_state(s.snapshot_state(),before)
        def raises():
            raise RuntimeError("cancel callback failed")
        checks = iter([False,raises])
        def callback():
            value = next(checks)
            return value() if callable(value) else value
        with self.assertRaisesRegex(RuntimeError,"callback failed"):
            s.forward(x,cancelled=callback)
        self.assert_state(s.snapshot_state(),before)
        s.forward(x,cancelled=lambda: False)
        self.assertEqual(s.position,3)
        before = s.snapshot_state()
        for cancel in (None,lambda: False):
            with self.assertRaises(NativeExecutionError):
                s.forward(np.full((1,8),np.nan,np.float32),cancelled=cancel)
            self.assert_state(s.snapshot_state(),before)

    def test_malformed_snapshot_and_numeric_class_rejected_transactionally(self):
        engine,edge = self.engine(),self.edge()
        s = self.session(engine)
        s.forward(edge.embeddings([1]))
        before = s.snapshot_state()
        for field,value in (("numeric_class","numpy-fp32-dense-int8-row-experts"),
                            ("execution_identity","wrong"),("format","openagent.olmoe-segment.v2"),
                            ("fingerprint","wrong"),("range",(False,2)),("position",True),
                            ("position",11),("context_tokens",True),("keys",[])):
            bad = copy.deepcopy(before)
            bad[field] = value
            with self.assertRaises(ValueError):
                s.restore_state(bad)
            self.assert_state(s.snapshot_state(),before)

    def test_cancellation_callback_cannot_reenter_or_mutate_active_session(self):
        engine = self.engine()
        s = self.session(engine)
        x = np.ones((1,8),np.float32)
        before = s.snapshot_state()
        for action in (s.reset,s.close,s.snapshot_state,lambda: s.forward(x),
                       lambda: s.restore_state(before)):
            for calls in ((action,),(None,action)):
                pending = iter(calls)
                def callback():
                    operation = next(pending)
                    if operation is not None:
                        operation()
                    return False
                with self.assertRaisesRegex(RuntimeError,"executing"):
                    s.forward(x,cancelled=callback)
                self.assert_state(s.snapshot_state(),before)
        s.forward(x,cancelled=lambda: False)
        self.assertEqual(s.position,1)
        before = s.snapshot_state()
        for value in (np.zeros((1,1,1),np.float32),before["values"][0].astype(np.float64),
                      np.full_like(before["values"][0],np.nan)):
            bad = copy.deepcopy(before)
            bad["values"][-1] = value
            with self.assertRaises(ValueError):
                s.restore_state(bad)
            self.assert_state(s.snapshot_state(),before)

    def test_snapshots_bound_to_dense_contents_and_requested_range(self):
        engine = self.engine(0,1)
        source = self.session(engine)
        source.forward(np.ones((1,8),np.float32))
        changed = self.root / "changed"
        changed.mkdir()
        for file in self.root.iterdir():
            if file.is_file():
                shutil.copy2(file,changed / file.name)
        shard = changed / "model.safetensors"
        payload = bytearray(shard.read_bytes())
        length = int.from_bytes(payload[:8],"little")
        header = json.loads(payload[8:8+length])
        offset = header["model.layers.0.input_layernorm.weight"]["data_offsets"][0]
        struct.pack_into("<f",payload,8+length+offset,2.)
        shard.write_bytes(payload)
        other = self.stack.enter_context(native.NativeOlmoeSegmentEngine(changed,self.manifest,
            0,1,self.library,100000))
        self.assertEqual(engine.manifest.fingerprint,other.manifest.fingerprint)
        target = self.session(other)
        before = target.snapshot_state()
        with self.assertRaisesRegex(ValueError,"execution_identity"):
            target.restore_state(source.snapshot_state())
        self.assert_state(target.snapshot_state(),before)
        # Changing a NONselected dense layer cannot affect range weight identity.
        second = self.engine(1,2)
        second_changed = self.stack.enter_context(native.NativeOlmoeSegmentEngine(changed,self.manifest,
            1,2,self.library,100000))
        self.assertEqual(second.execution_identity,second_changed.execution_identity)

    def test_snapshots_bound_to_decoded_expert_weights(self):
        engine = self.engine(0,1)
        source = self.session(engine)
        path = self.root / "experts.i8"
        region = self.manifest.layers[0].regions["0"]
        with path.open("r+b") as stream:
            stream.seek(region.offset)
            payload = bytearray(stream.read(region.size))
            payload[0] = (payload[0] + 1) % 256
            stream.seek(region.offset)
            stream.write(payload)
        regions = dict(self.manifest.layers[0].regions)
        regions["0"] = replace(region,sha256=hashlib.sha256(payload).hexdigest())
        changed = replace(self.manifest,layers=[replace(self.manifest.layers[0],regions=regions),
                                               self.manifest.layers[1]])
        other = self.stack.enter_context(native.NativeOlmoeSegmentEngine(self.root,changed,0,1,
                                                                        self.library,100000))
        self.assertNotEqual(engine.execution_identity,other.execution_identity)
        with self.assertRaisesRegex(ValueError,"execution_identity"):
            self.session(other).restore_state(source.snapshot_state())

    def test_snapshot_memory_bounds_and_engine_lifetimes(self):
        engine = self.engine(max_snapshot_bytes=1)
        s = self.session(engine)
        empty = s.snapshot_state()
        with self.assertRaises(MemoryError):
            s.restore_state(empty)
        s.forward(np.ones((1,8),np.float32))
        with self.assertRaises(MemoryError):
            s.snapshot_state()
        with self.assertRaises(RuntimeError):
            engine.close()
        s.close()
        s.close()
        engine.close()
        engine.close()
        self.assertTrue(engine.status()["closed"])
        for operation in (lambda: engine.create_session(),lambda: s.forward(np.ones((1,8),np.float32)),
                          s.snapshot_state,s.reset):
            with self.assertRaises(RuntimeError):
                operation()

    def test_decode_failure_closes_runtime_and_releases_leases(self):
        closed = []
        close = SparseExpertRuntime.close
        def record(runtime):
            close(runtime)
            closed.append(runtime)
        with patch.object(SparseExpertRuntime,"close",record), \
             patch.object(native,"_decode_expert",side_effect=ValueError("decode failed")):
            with self.assertRaisesRegex(ValueError,"decode failed"):
                self.engine()
        self.assertEqual(len(closed),1)
        self.assertTrue(closed[0].store._closed)
        self.assertFalse(closed[0].store._entries)

    def test_invalid_manifest_checksum_and_int8_decode_cleanup(self):
        regions = dict(self.manifest.layers[0].regions)
        regions["0"] = replace(regions["0"],sha256="0"*64)
        wrong = replace(self.manifest,layers=[replace(self.manifest.layers[0],regions=regions),
                                             self.manifest.layers[1]])
        with self.assertRaisesRegex(ValueError,"checksum"):
            native.NativeOlmoeSegmentEngine(self.root,wrong,0,1,self.library,100000)
        # Corrupt a scale, then update the checksum to reach decode validation.
        region = self.manifest.layers[0].regions["0"]
        path = self.root / "experts.i8"
        with path.open("r+b") as stream:
            stream.seek(region.offset)
            payload = bytearray(stream.read(region.size))
            struct.pack_into("<f",payload,3*8*6,0.)
            stream.seek(region.offset)
            stream.write(payload)
        regions["0"] = replace(region,sha256=hashlib.sha256(payload).hexdigest())
        invalid = replace(self.manifest,layers=[replace(self.manifest.layers[0],regions=regions),
                                               self.manifest.layers[1]])
        with self.assertRaisesRegex(ValueError,"scales"):
            native.NativeOlmoeSegmentEngine(self.root,invalid,0,1,self.library,100000)
        # Closed runtime must not keep the expert file locked after either error.
        path.unlink()

    def test_atomic_decoder_full_checkpoint_prefill_decode_and_resume(self):
        engine,edge = self.engine(),self.edge()
        decoder = self.stack.enter_context(edge.create_session(engine))
        reference = self.stack.enter_context(OlmoeModel(self.root,self.manifest,expert_cache_bytes=4096))
        for ids in ([1,2,3],[7],[4,8]):
            expected = np.stack([reference.forward([token]) for token in ids])
            with patch.object(OlmoeSegmentEngine,"forward",side_effect=AssertionError("Python range")), \
                 patch.object(OlmoeModel,"_attention",side_effect=AssertionError("Python attention")), \
                 patch.object(SparseExpertRuntime,"execute_batch",side_effect=AssertionError("Python expert")), \
                 patch.object(edge,"embeddings",side_effect=AssertionError("separate embed call")), \
                 patch.object(edge,"final_head",side_effect=AssertionError("separate head call")):
                out = decoder.forward(ids)
            np.testing.assert_allclose(out,expected,atol=4e-6,rtol=4e-5)
            self.assertEqual(decoder.position,reference.position)
        saved = decoder.snapshot_state()
        resumed = self.stack.enter_context(edge.create_session(engine))
        resumed.restore_state(saved)
        np.testing.assert_array_equal(resumed.forward([9]),decoder.forward([9]))
        decoder.reset()
        decoder.restore_state(saved)
        saved["range_state"]["keys"][0].fill(999)
        self.assertFalse(np.any(decoder.snapshot_state()["range_state"]["keys"][0] == 999))

    def test_atomic_decoder_head_failure_cancellation_reentry_and_lifetime(self):
        engine,edge = self.engine(),self.edge()
        decoder = self.stack.enter_context(edge.create_session(engine))
        decoder.forward([1])
        before = decoder.snapshot_state()
        old_norm,old_head = edge.final_norm,edge.lm_head
        edge.final_norm = np.full(8,np.finfo(np.float32).max,np.float32)
        edge.lm_head = np.full((13,8),np.finfo(np.float32).max,np.float32)
        with self.assertRaises(NativeExecutionError):
            decoder.forward([2])
        self.assert_state(decoder.snapshot_state()["range_state"],before["range_state"])
        edge.final_norm,edge.lm_head = old_norm,old_head
        for values in ((True,),(False,True)):
            pending = iter(values)
            with self.assertRaises(InterruptedError):
                decoder.forward([2],cancelled=lambda: next(pending))
            self.assert_state(decoder.snapshot_state()["range_state"],before["range_state"])
        for action in (decoder.reset,decoder.close,decoder.snapshot_state,lambda: decoder.forward([2])):
            pending = iter([False,action])
            def callback():
                value = next(pending)
                return value() if callable(value) else value
            with self.assertRaisesRegex(RuntimeError,"executing"):
                decoder.forward([2],cancelled=callback)
            self.assert_state(decoder.snapshot_state()["range_state"],before["range_state"])
        for close in (edge.close,engine.close):
            with self.assertRaisesRegex(RuntimeError,"live"):
                close()
        decoder.forward([2],cancelled=lambda: False)
        decoder.close()
        decoder.close()
        self.assertEqual(edge.status()["live_decoder_sessions"],0)
        with self.assertRaisesRegex(RuntimeError,"closed"):
            decoder.forward([3])

    def test_decoder_requires_full_range_and_snapshots_bind_edge_contents(self):
        engine,edge = self.engine(),self.edge()
        with self.assertRaisesRegex(ValueError,"full layer range"):
            edge.create_session(self.engine(0,1))
        with self.assertRaises(RuntimeError):
            self.edge(mode="head").create_session(engine)
        with patch.object(engine.executor,"_decode",None):
            with self.assertRaisesRegex(RuntimeError,"lacks atomic decode"):
                edge.create_session(engine)
        self.assertFalse(engine._sessions)
        source = self.stack.enter_context(edge.create_session(engine))
        source.forward([1,2])
        shard = self.root / "model.safetensors"
        payload = bytearray(shard.read_bytes())
        length = int.from_bytes(payload[:8],"little")
        header = json.loads(payload[8:8+length])
        offset = header["lm_head.weight"]["data_offsets"][0]
        struct.pack_into("<f",payload,8+length+offset,2.)
        shard.write_bytes(payload)
        other_edge = self.edge()
        target = self.stack.enter_context(other_edge.create_session(engine))
        before = target.snapshot_state()
        with self.assertRaisesRegex(ValueError,"edge_execution_identity"):
            target.restore_state(source.snapshot_state())
        self.assert_state(target.snapshot_state()["range_state"],before["range_state"])
        for ids in ([],[True],[-1],[13],[[1]]):
            with self.assertRaises((ValueError,NativeExecutionError)):
                target.forward(ids)
            self.assertEqual(target.position,0)

    def test_composed_ranges_atomic_head_rollback_and_shared_weight_lifetime(self):
        sources = [self.engine(0,1),self.engine(1,2)]
        merged = self.stack.enter_context(native.NativeOlmoeSegmentEngine.compose(sources))
        self.assertTrue(merged.status()["shares_resident_weights"])
        for original,shared in zip([a for e in sources for a in e._range._weights],merged._range._weights):
            self.assertIs(original,shared)
        for engine in sources:
            engine.close()
        edge = self.edge()
        decoder = self.stack.enter_context(edge.create_session(merged))
        reference = self.stack.enter_context(OlmoeModel(self.root,self.manifest,expert_cache_bytes=4096))
        np.testing.assert_allclose(decoder.forward([1,2,3])[-1],reference.forward([1,2,3]),atol=4e-6,rtol=4e-5)
        before = decoder.snapshot_state()
        original_norm,original_head = edge.final_norm,edge.lm_head
        edge.final_norm = np.full(8,np.finfo(np.float32).max,np.float32)
        edge.lm_head = np.full((13,8),np.finfo(np.float32).max,np.float32)
        with self.assertRaises(NativeExecutionError):
            decoder.forward([4])
        self.assert_state(decoder.snapshot_state()["range_state"],before["range_state"])
        edge.final_norm,edge.lm_head = original_norm,original_head
        np.testing.assert_allclose(decoder.forward([4])[-1],reference.forward([4]),atol=4e-6,rtol=4e-5)
        resumed = self.stack.enter_context(edge.create_session(merged))
        resumed.restore_state(decoder.snapshot_state())
        np.testing.assert_array_equal(resumed.forward([7]),decoder.forward([7]))

    def test_composition_rejects_gaps_duplicates_order_and_budgets(self):
        a,b = self.engine(0,1),self.engine(1,2)
        for engines in ([],[a,a],[b,a],[a,self.engine()]):
            with self.assertRaises((ValueError,TypeError)):
                native.NativeOlmoeSegmentEngine.compose(engines)
        required = a.resident_weight_bytes+b.resident_weight_bytes
        with self.assertRaises(MemoryError):
            native.NativeOlmoeSegmentEngine.compose([a,b],max_resident_bytes=required-1)
        with self.assertRaises(MemoryError):
            native.NativeOlmoeSegmentEngine.compose([a,b],max_bytes=1)
        merged = self.stack.enter_context(native.NativeOlmoeSegmentEngine.compose([a,b],
            max_resident_bytes=required,max_bytes=4096,context_tokens=3))
        self.assertEqual(merged.executor.max_bytes,4096)
        decoder = self.stack.enter_context(self.edge().create_session(merged))
        with self.assertRaises(NativeExecutionError):
            decoder.forward([1,2,3,4])
        self.assertEqual(decoder.position,0)

    def test_atomic_decoder_shared_session_concurrent_calls_are_serialized(self):
        engine,edge = self.engine(),self.edge()
        shared = self.stack.enter_context(edge.create_session(engine))
        serial = self.stack.enter_context(edge.create_session(engine))
        expected = np.concatenate([serial.forward([2]) for _ in range(8)])
        barrier = threading.Barrier(8)
        def forward(index):
            barrier.wait(timeout=30)
            return shared.forward([2])
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = np.concatenate(list(pool.map(forward,range(8))))
        np.testing.assert_array_equal(actual[np.lexsort(actual.T[::-1])],
                                      expected[np.lexsort(expected.T[::-1])])
        self.assertEqual(shared.position,8)
        self.assert_state(shared.snapshot_state()["range_state"],serial.snapshot_state()["range_state"])


if __name__ == "__main__":
    unittest.main()
