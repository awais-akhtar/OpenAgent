"""Portable snapshots exercised against tiny real OLMoE ranges."""

from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.olmoe_segments import OlmoeEdge, OlmoeSegmentEngine
from openagent.adaptive_engine.segment_snapshot import (
    restore_segment_snapshot, write_segment_snapshot)
from tests.test_olmoe_batch import checkpoint


class ShortStream(io.BytesIO):
    def read(self, size=-1):
        if not 0 <= size <= 1024 * 1024:
            raise AssertionError('unbounded read')
        return super().read(min(size, 7))

    def write(self, data):
        if len(data) > 1024 * 1024:
            raise AssertionError('unbounded write')
        return super().write(data[:11])


class SegmentSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        manifest = checkpoint(root, True)
        self.edge = self.stack.enter_context(OlmoeEdge(root))
        self.engines = [self.stack.enter_context(OlmoeSegmentEngine(
            root, manifest, i, i + 1, 10, expert_cache_bytes=4096)) for i in range(2)]
        self.sessions = [self.stack.enter_context(e.create_session()) for e in self.engines]

    def encode(self, session):
        output = ShortStream()
        write_segment_snapshot(session, output)
        return output.getvalue()

    def assert_state(self, session, state):
        actual = session.snapshot_state()
        for name in state:
            if name in ('keys', 'values'):
                for a, b in zip(actual[name], state[name]):
                    np.testing.assert_array_equal(a, b)
            else:
                self.assertEqual(actual[name], state[name])

    def test_composed_range_continuation_and_empty(self):
        resumed = [self.stack.enter_context(e.create_session()) for e in self.engines]
        for s, r in zip(self.sessions, resumed):
            restore_segment_snapshot(r, ShortStream(self.encode(s)))
            self.assert_state(r, s.snapshot_state())
        hidden = self.edge.embeddings([1, 2, 3])
        for s in self.sessions:
            hidden = s.forward(hidden)
        for s, r in zip(self.sessions, resumed):
            data = self.encode(s)
            restore_segment_snapshot(r, ShortStream(data), max_bytes=4096)
            self.assert_state(r, s.snapshot_state())
        for ids in ([4], [5, 6]):
            a = self.edge.embeddings(ids)
            b = a.copy()
            for s, r in zip(self.sessions, resumed):
                a = s.forward(a)
                b = r.forward(b)
            np.testing.assert_array_equal(self.edge.final_head(a), self.edge.final_head(b))

    def rewrite(self, data, change, payload_change=None):
        length = struct.unpack('<I', data[8:12])[0]
        header = json.loads(data[12:12 + length])
        change(header)
        encoded = json.dumps(header, separators=(',', ':')).encode()
        payload = data[12 + length:-32]
        if payload_change:
            payload = payload_change(payload)
        body = data[:8] + struct.pack('<I', len(encoded)) + encoded + payload
        return body + hashlib.sha256(body).digest()

    def test_active_destination_restore_respects_kv_staging_budget(self):
        engine = self.stack.enter_context(OlmoeSegmentEngine(
            self.engines[0].root, self.engines[0].manifest, 0, 2, 10,
            expert_cache_bytes=4096))
        source = self.stack.enter_context(engine.create_session())
        target = self.stack.enter_context(engine.create_session())
        source.forward(self.edge.embeddings([1] * 10))
        target.forward(self.edge.embeddings([2] * 10))
        expected = source.snapshot_state()
        before = target.snapshot_state()
        data = self.encode(source)
        cache_bytes = sum(a.nbytes for a in target._k + target._v)
        payload_bytes = sum(a.nbytes for a in expected['keys'] + expected['values'])
        budget = cache_bytes + payload_bytes
        self.assertLess(len(data), budget)
        allocated = []
        empty, zeros_like = np.empty, np.zeros_like

        def track(function):
            def allocate(*args, **kwargs):
                array = function(*args, **kwargs)
                allocated.append(array.nbytes)
                return array
            return allocate

        with patch.object(target, 'snapshot_state', side_effect=AssertionError('KV copy')), \
                patch('openagent.adaptive_engine.segment_snapshot.np.empty', side_effect=track(empty)), \
                patch('openagent.adaptive_engine.olmoe_segments.np.zeros_like', side_effect=track(zeros_like)):
            metadata = target.snapshot_metadata()
            self.assertNotIn('keys', metadata)
            self.assertNotIn('values', metadata)
            self.assertEqual(allocated, [])
            with self.assertRaisesRegex(ValueError, 'exceeds max_bytes'):
                restore_segment_snapshot(target, ShortStream(data), max_bytes=budget - 1)
            self.assertEqual(allocated, [])
            restore_segment_snapshot(target, ShortStream(data), max_bytes=budget)
        self.assertEqual(sum(allocated), budget)
        self.assert_state(target, expected)
        self.assertFalse(np.array_equal(before['keys'][0], expected['keys'][0]))

    def test_malformed_and_fuzz_are_transactional(self):
        session = self.sessions[0]
        session.forward(self.edge.embeddings([1, 2]))
        data = self.encode(session)
        before = session.snapshot_state()
        bad = [data[:i] for i in range(len(data))]
        bad += [data + b'x', b'wrong!!!' + data[8:],
                data[:8] + struct.pack('<I', 65537) + data[12:]]
        for field, value in [('fingerprint', 'wrong'), ('range', [1, 2]),
                             ('range', [False, 1]), ('numeric_class', 'fp16'),
                             ('context_tokens', True), ('position', True),
                             ('position', -1), ('position', 11), ('version', 1),
                             ('execution_identity', 'wrong'),
                             ('shapes', [[1, 2, 1000000000]]), ('lengths', [4]),
                             ('shapes', None), ('format', 'wrong')]:
            bad.append(self.rewrite(data, lambda h, f=field, v=value: h.update({f: v})))
        for value in (np.nan, np.inf, -np.inf):
            bad.append(self.rewrite(data, lambda h: None,
                                   lambda p, v=value: struct.pack('<f', v) + p[4:]))
        rng = random.Random(17)
        for _ in range(100):
            corrupted = bytearray(data)
            index = rng.randrange(len(data))
            corrupted[index] ^= 1 << rng.randrange(8)
            bad.append(bytes(corrupted))
        with patch.object(session, 'restore_state', wraps=session.restore_state) as commit:
            for malformed in bad:
                with self.assertRaises(ValueError):
                    restore_segment_snapshot(session, ShortStream(malformed))
                self.assert_state(session, before)
            commit.assert_not_called()
        with patch('openagent.adaptive_engine.segment_snapshot.np.empty') as allocate:
            with self.assertRaises(ValueError):
                restore_segment_snapshot(session, io.BytesIO(data), max_bytes=100)
            with self.assertRaises(ValueError):
                restore_segment_snapshot(self.sessions[1], io.BytesIO(data))
            allocate.assert_not_called()

    def test_header_schema_and_writer_finiteness(self):
        session = self.sessions[0]
        data = self.encode(session)
        for encoded in (b'{"version":1,"version":1}', b'[]', b'{',
                        b'\xff', b'[' * 2000, b'{"position":NaN}'):
            body = data[:8] + struct.pack('<I', len(encoded)) + encoded
            with self.assertRaises(ValueError):
                restore_segment_snapshot(session, io.BytesIO(body))
        session.forward(self.edge.embeddings([1]))
        session._k[0][0, 0, 0] = np.nan
        output = io.BytesIO()
        with self.assertRaises(ValueError):
            write_segment_snapshot(session, output)
        self.assertEqual(output.getvalue(), b'')


if __name__ == '__main__':
    unittest.main()
