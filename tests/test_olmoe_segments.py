"""Differential checks using genuine OLMoE SafeTensors and packed experts."""

import copy
from dataclasses import replace
import io
import json
import shutil
import struct
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.olmoe import OlmoeModel
from openagent.adaptive_engine.olmoe_segments import OlmoeEdge, OlmoeSegmentEngine
from tests.test_olmoe_batch import checkpoint


class OlmoeSegmentTests(unittest.TestCase):
    def test_snapshot_rejects_dense_changes_with_identical_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory) / 'original'
            root.mkdir()
            manifest = checkpoint(root, True)
            changed = Path(directory) / 'changed'
            shutil.copytree(root, changed)
            shard = next(changed.glob('*.safetensors'))
            payload = bytearray(shard.read_bytes())
            length = int.from_bytes(payload[:8], 'little')
            header = json.loads(payload[8:8 + length])
            offset = header['model.layers.0.input_layernorm.weight']['data_offsets'][0]
            struct.pack_into('<f', payload, 8 + length + offset, 2.0)
            shard.write_bytes(payload)
            full = stack.enter_context(OlmoeModel(root, manifest, expert_cache_bytes=4096))
            other_full = stack.enter_context(OlmoeModel(changed, manifest, expert_cache_bytes=4096))
            self.assertEqual(full.manifest.fingerprint, other_full.manifest.fingerprint)
            full.forward([1])
            before = other_full.snapshot_state()
            with self.assertRaisesRegex(ValueError, 'different model weights'):
                other_full.restore_state(full.snapshot_state())
            self.assert_state(other_full.snapshot_state(), before)
            engine = stack.enter_context(OlmoeSegmentEngine(
                root, manifest, 0, 1, expert_cache_bytes=4096))
            other = stack.enter_context(OlmoeSegmentEngine(
                changed, manifest, 0, 1, expert_cache_bytes=4096))
            source = stack.enter_context(engine.create_session())
            target = stack.enter_context(other.create_session())
            with self.assertRaisesRegex(ValueError, 'execution_identity'):
                target.restore_state(source.snapshot_state())

    def test_snapshot_rejects_different_expert_content_before_mutating_kv(self):
        from openagent.adaptive_engine.segment_snapshot import (
            write_segment_snapshot, restore_segment_snapshot)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            manifest = checkpoint(root, True)
            for capacity in (0, True, 1.5, '10'):
                with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                    OlmoeModel(root, manifest, expert_cache_bytes=4096,
                               context_tokens=capacity)
            regions = dict(manifest.layers[0].regions)
            missing_hashes = dict(regions)
            missing_hashes['0'] = replace(regions['0'], sha256='')
            unhashed = replace(manifest, layers=[
                replace(manifest.layers[0], regions=missing_hashes), *manifest.layers[1:]])
            for constructor in (
                    lambda: OlmoeModel(root, unhashed, expert_cache_bytes=4096),
                    lambda: OlmoeSegmentEngine(root, unhashed, 0, 1, expert_cache_bytes=4096)):
                with self.assertRaisesRegex(ValueError, 'content hashes'):
                    constructor()
            renamed = dict(regions)
            renamed['invalid'] = renamed.pop('0')
            invalid_ids = replace(manifest, layers=[
                replace(manifest.layers[0], regions=renamed), *manifest.layers[1:]])
            with self.assertRaisesRegex(ValueError, 'does not match'):
                OlmoeModel(root, invalid_ids, expert_cache_bytes=4096)
            regions['0'] = replace(regions['0'], sha256='0' * 64)
            changed = replace(manifest, layers=[
                replace(manifest.layers[0], regions=regions), *manifest.layers[1:]])
            original = stack.enter_context(OlmoeSegmentEngine(
                root, manifest, 0, 1, expert_cache_bytes=4096))
            other = stack.enter_context(OlmoeSegmentEngine(
                root, changed, 0, 1, expert_cache_bytes=4096))
            source = stack.enter_context(original.create_session())
            target = stack.enter_context(other.create_session())
            edge = stack.enter_context(OlmoeEdge(root))
            source.forward(edge.embeddings([1]))
            before = target.snapshot_state()
            with self.assertRaisesRegex(ValueError, 'execution_identity'):
                target.restore_state(source.snapshot_state())
            self.assert_state(target.snapshot_state(), before)
            encoded = io.BytesIO()
            write_segment_snapshot(source, encoded)
            with self.assertRaisesRegex(ValueError, 'execution_identity'):
                restore_segment_snapshot(target, io.BytesIO(encoded.getvalue()))
            self.assert_state(target.snapshot_state(), before)
            legacy = source.snapshot_state()
            legacy['format'] = 'openagent.olmoe-segment.v1'
            legacy.pop('execution_identity')
            with self.assertRaises(ValueError):
                source.restore_state(legacy)

    def test_final_head_rejects_finite_weights_that_overflow_logits(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            checkpoint(root, True)
            edge = stack.enter_context(OlmoeEdge(root))
            edge.final_norm.fill(1)
            edge.lm_head.fill(np.float32(3e38))
            with self.assertRaisesRegex(ValueError, 'nonfinite final logits'):
                edge.final_head(np.ones((1, edge.config.hidden_size), np.float32))

    def assert_state(self, actual, expected):
        for field in actual:
            if field in ('keys', 'values'):
                for a, b in zip(actual[field], expected[field]):
                    np.testing.assert_array_equal(a, b)
            else:
                self.assertEqual(actual[field], expected[field])

    def test_composed_ranges_match_full_model_and_isolate_sessions(self):
        for normalized in (False, True):
            with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                manifest = checkpoint(root, normalized)
                budget = max(region.size for layer in manifest.layers for region in layer.regions.values())
                edge = stack.enter_context(OlmoeEdge(root))
                engines = [stack.enter_context(OlmoeSegmentEngine(
                    root, manifest, i, i + 1, 10, expert_cache_bytes=budget)) for i in range(2)]
                sessions = [[stack.enter_context(e.create_session()) for e in engines] for _ in range(2)]
                full = [stack.enter_context(OlmoeModel(root, manifest, expert_cache_bytes=budget)) for _ in range(2)]
                for chunks in (([1, 2, 3], [4]), ([5], [6, 7]), ([8, 9], [2])):
                    for index, ids in enumerate(chunks):
                        untouched = [s.snapshot_state() for s in sessions[1 - index]]
                        hidden = edge.embeddings(ids)
                        for session in sessions[index]:
                            hidden = session.forward(hidden)
                        np.testing.assert_allclose(edge.final_head(hidden)[-1], full[index].forward(ids),
                                                   atol=2e-6, rtol=2e-5)
                        for s, before in zip(sessions[1 - index], untouched):
                            self.assert_state(s.snapshot_state(), before)
                        for layer, s in enumerate(sessions[index]):
                            np.testing.assert_array_equal(s.snapshot_state()['keys'][0],
                                                          full[index].snapshot_state()['keys'][layer])
                            np.testing.assert_array_equal(s.snapshot_state()['values'][0],
                                                          full[index].snapshot_state()['values'][layer])
                for group in sessions:
                    for layer, s in enumerate(group):
                        saved = s.snapshot_state()
                        resumed = stack.enter_context(engines[layer].create_session())
                        resumed.restore_state(saved)
                        self.assert_state(resumed.snapshot_state(), saved)
                        s.reset()
                        s.restore_state(saved)
                        self.assert_state(s.snapshot_state(), saved)
                        saved['keys'][0].fill(999)
                        self.assertFalse(np.any(s._k[0] == 999))
                for index, group in enumerate(sessions):
                    hidden = edge.embeddings([3])
                    for s in group:
                        hidden = s.forward(hidden)
                    np.testing.assert_allclose(edge.final_head(hidden)[-1], full[index].forward([3]),
                                               atol=2e-6, rtol=2e-5)

    def test_range_only_reads_and_capabilities(self):
        from openagent.adaptive_engine.olmoe import _read_float_matrix
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            manifest = checkpoint(root, True)
            for options in ({'kernels': object()}, {'preferred_kernel': 'vulkan-wgpu'}):
                with self.assertRaisesRegex(ValueError, 'numeric class'):
                    OlmoeSegmentEngine(root, manifest, 0, 1, expert_cache_bytes=4096, **options)
            with patch('openagent.adaptive_engine.olmoe._read_float_matrix', wraps=_read_float_matrix) as read:
                engine = stack.enter_context(OlmoeSegmentEngine(root, manifest, 1, 2, expert_cache_bytes=4096))
                self.assertEqual(len(read.call_args_list), 9)
                self.assertTrue(all(c.args[1].startswith('model.layers.1.') for c in read.call_args_list))
            self.assertEqual(set(engine.experts.layers), {'model.layers.1'})
            self.assertTrue(all(k.startswith('model.layers.1/') for k in engine.experts.store.regions))
            self.assertFalse(engine.status()['native_range_execution'])
            with patch('openagent.adaptive_engine.olmoe._read_float_matrix', wraps=_read_float_matrix) as read:
                stack.enter_context(OlmoeEdge(root))
                self.assertEqual({c.args[1] for c in read.call_args_list},
                                 {'model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight'})
            for begin, end in ((-1, 1), (0, 0), (1, 0), (0, 3), (True, 2), (0, 1.0)):
                with self.assertRaises(ValueError):
                    OlmoeSegmentEngine(root, manifest, begin, end, expert_cache_bytes=4096)

    def test_transactional_errors_cancellation_and_lifetime(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            manifest = checkpoint(root, True)
            edge = stack.enter_context(OlmoeEdge(root))
            engine = stack.enter_context(OlmoeSegmentEngine(root, manifest, 0, 2, 5, expert_cache_bytes=4096))
            s = stack.enter_context(engine.create_session())
            s.forward(edge.embeddings([1, 2]))
            before = s.snapshot_state()
            original_k = [a.copy() for a in s._k]
            original_v = [a.copy() for a in s._v]
            def unchanged():
                self.assert_state(s.snapshot_state(), before)
                for a, b in zip(s._k + s._v, original_k + original_v):
                    np.testing.assert_array_equal(a, b)
            for field, value in (('fingerprint', 'wrong'), ('range', (1, 2)),
                                 ('range', (False, 2)), ('numeric_class', 'float16'),
                                 ('context_tokens', 4), ('context_tokens', True),
                                 ('position', True), ('position', 6), ('format', 'wrong'),
                                 ('keys', [])):
                bad = copy.deepcopy(before)
                bad[field] = value
                with self.assertRaises(ValueError):
                    s.restore_state(bad)
                unchanged()
            for array in (np.zeros((1, 2, 3), np.float32), before['values'][0].astype(np.float64),
                          np.full_like(before['values'][0], np.nan)):
                bad = copy.deepcopy(before)
                bad['values'][-1] = array
                with self.assertRaises(ValueError):
                    s.restore_state(bad)
                unchanged()
            execute = engine.experts.execute_batch
            def fail(layer, hidden, routes):
                if layer.endswith('.1'):
                    raise RuntimeError('injected failure')
                return execute(layer, hidden, routes)
            with patch.object(engine.experts, 'execute_batch', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    s.forward(edge.embeddings([3]))
            unchanged()
            for answers in ((True,), (False, True)):
                check = iter(answers)
                with patch.object(engine.experts, 'execute_batch', wraps=execute) as observed:
                    with self.assertRaises(InterruptedError):
                        s.forward(edge.embeddings([3]), cancelled=lambda: next(check))
                    self.assertEqual(observed.call_count, 0 if answers[0] else 2)
                unchanged()
            foreign_engine = stack.enter_context(OlmoeSegmentEngine(
                root, manifest, 1, 2, 5, expert_cache_bytes=4096))
            foreign = stack.enter_context(foreign_engine.create_session())
            with self.assertRaises(ValueError):
                engine.forward(foreign, edge.embeddings([3]))
            with self.assertRaises(ValueError):
                foreign.restore_state(before)
            smaller = stack.enter_context(engine.create_session(context_tokens=4))
            with self.assertRaises(ValueError):
                smaller.restore_state(before)
            smaller.close()
            for hidden in (np.zeros((0, 8), np.float32), np.zeros((1, 7), np.float32),
                           np.zeros((1, 8), np.float64), np.full((1, 8), np.inf, np.float32),
                           np.zeros((4, 8), np.float32)):
                with self.assertRaises(ValueError):
                    s.forward(hidden)
                unchanged()
            with self.assertRaises(RuntimeError):
                engine.close()
            s.forward(edge.embeddings([3]))
            s.close()
            with self.assertRaises(RuntimeError):
                s.snapshot_state()
            engine.close()
            with self.assertRaises(RuntimeError):
                engine.create_session()
            edge.close()
            with self.assertRaises(RuntimeError):
                edge.embeddings([1])


if __name__ == '__main__':
    unittest.main()
