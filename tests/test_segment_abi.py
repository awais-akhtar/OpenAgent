"""Compile the actual C bridge, then test callbacks against real tiny weights."""
import ctypes as C
from contextlib import ExitStack
import gc
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.olmoe import OlmoeModel
from openagent.adaptive_engine.olmoe_segments import OlmoeEdge, OlmoeSegmentEngine
from openagent.adaptive_engine.segment_snapshot import write_segment_snapshot
from openagent.adaptive_engine.segment_abi import (
    SegmentBridge, NativeSegmentMath, SegmentABIError, SegmentABICloseError, CALLBACK, U64, VERSION, SESSION,
    EDGE, OK, INVALID, BAD_VERSION, BAD_HANDLE, BUSY, LIMIT, CALLBACK_ERROR,
    CANCELLED, CLOSE, FORWARD, EMBED, RESET, SNAPSHOT)
from tests.test_olmoe_batch import checkpoint


def build_bridge(directory):
    """No compiler download; unavailable skips, compiler failure fails tests."""
    source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/native/segment_bridge.c'
    math_source = source.with_name('segment_math.c')
    output = Path(directory) / ('segment_bridge.dll' if os.name == 'nt' else 'segment_bridge.so')
    compiler = next((shutil.which(name) for name in ('clang', 'gcc', 'cc') if shutil.which(name)), None)
    if compiler:
        command = [compiler, '-std=c99', '-Wall', '-Wextra', '-Werror', '-shared', str(source), str(math_source), '-o', str(output)]
        if os.name != 'nt':
            command.insert(1, '-fPIC')
            command.append('-lm')
    elif os.name == 'nt':
        cl = shutil.which('cl')
        if cl:
            command = [cl, '/nologo', '/W4', '/WX', '/LD', str(source), str(math_source), '/link', '/OUT:' + str(output)]
        else:
            vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
            if not vswhere.is_file():
                raise unittest.SkipTest('Native Segment ABI unavailable: no Windows C compiler found')
            location = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
                'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
            if not location:
                raise unittest.SkipTest('Native Segment ABI unavailable: MSVC C tools are not installed')
            bat = Path(location) / 'VC/Auxiliary/Build/vcvars64.bat'
            command = (f'cmd.exe /d /s /c ""{bat}" && '
                f'cl /nologo /W4 /WX /LD "{source}" "{math_source}" /link /OUT:"{output}""')
    else:
        raise unittest.SkipTest('Native Segment ABI unavailable: no C compiler found')
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode or not output.is_file():
        raise RuntimeError('C bridge build failed:\n' + result.stdout + result.stderr)
    return output


class SegmentABITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Windows cannot unlink a loaded DLL; retain the temporary directory
        # for the process lifetime and let the OS temporary-file policy clean it.
        cls.build_directory = tempfile.mkdtemp(prefix='oa-segment-abi-')
        cls.library = build_bridge(cls.build_directory)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = checkpoint(self.root, True)
        self.bridge = SegmentBridge(self.library, max_bytes=8192)
        self.handles = []
        self.addCleanup(self.cleanup_handles)

    def cleanup_handles(self):
        self.bridge.close()

    def register(self, backend, **kwargs):
        handle = self.bridge.register(backend, **kwargs)
        self.handles.append(handle)
        return handle

    def engine(self, begin=0, end=2):
        return self.register(OlmoeSegmentEngine(self.root, self.manifest, begin, end,
            10, expert_cache_bytes=4096))

    def session(self, engine):
        handle = self.bridge.create_session(engine)
        self.handles.append(handle)
        return handle

    def status(self, expected, function, *args, **kwargs):
        with self.assertRaises(SegmentABIError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.status, expected)
        return caught.exception

    def test_composed_ranges_match_model_and_snapshot_continuation(self):
        edge = self.register(OlmoeEdge(self.root))
        engines = [self.engine(i, i+1) for i in range(2)]
        sessions = [self.session(e) for e in engines]
        restored = [self.session(e) for e in engines]
        with OlmoeModel(self.root, self.manifest, expert_cache_bytes=4096) as full:
            for ids in ([1, 2], [3], [4, 5]):
                hidden = self.bridge.embeddings(edge, ids)
                for session in sessions:
                    hidden = self.bridge.hidden(session, hidden)
                logits = self.bridge.hidden(edge, hidden, final_head=True)
                np.testing.assert_allclose(logits[-1], full.forward(ids), atol=2e-6, rtol=2e-5)
            for session, target in zip(sessions, restored):
                wire = self.bridge.snapshot(session)
                reference = io.BytesIO()
                write_segment_snapshot(self.bridge._objects[session][0], reference)
                self.assertEqual(wire, reference.getvalue())
                self.bridge.restore(target, wire)
                self.assertEqual(self.bridge.snapshot(target), wire)
            a = b = self.bridge.embeddings(edge, [6])
            for left, right in zip(sessions, restored):
                a = self.bridge.hidden(left, a)
                b = self.bridge.hidden(right, b)
            np.testing.assert_array_equal(a, b)

    def test_lifetimes_stale_handles_and_isolation(self):
        engine = self.engine()
        session = self.session(engine)
        other = self.session(engine)
        before = self.bridge.snapshot(other)
        self.status(BUSY, self.bridge.release, engine)
        self.bridge.hidden(session, np.ones((2, 8), np.float32))
        self.assertEqual(before, self.bridge.snapshot(other))
        self.bridge.reset(session)
        self.assertEqual(self.bridge._objects[session][0].position, 0)
        self.bridge.release(session)
        new = self.session(engine)
        self.assertNotEqual(session, new)
        self.status(BAD_HANDLE, self.bridge.reset, session)
        self.assertEqual(self.bridge.lib.oa_segment_release(session), BAD_HANDLE)
        gc.collect()
        self.bridge.reset(new)

    def test_cancellation_failures_and_corruption_are_transactional(self):
        engine = self.engine()
        session = self.session(engine)
        hidden = np.ones((1, 8), np.float32)
        self.bridge.hidden(session, hidden)
        before = self.bridge.snapshot(session)
        for answers in ((True,), (False, True)):
            answers = iter(answers)
            self.status(CANCELLED, self.bridge.hidden, session, hidden, cancelled=lambda: next(answers))
            self.assertEqual(before, self.bridge.snapshot(session))
        backend = self.bridge._objects[engine][0]
        with patch.object(backend.experts, 'execute_batch', side_effect=RuntimeError('injected')):
            error = self.status(CALLBACK_ERROR, self.bridge.hidden, session, hidden)
            self.assertIsInstance(error.__cause__, RuntimeError)
        self.assertEqual(before, self.bridge.snapshot(session))
        bad = before[:-1] + bytes([before[-1] ^ 1])
        self.status(INVALID, self.bridge.restore, session, bad)
        self.assertEqual(before, self.bridge.snapshot(session))
        self.status(INVALID, self.bridge.hidden, session, np.ones((10, 8), np.float32))
        self.assertEqual(before, self.bridge.snapshot(session))

    def test_bounds_preflight_and_kind_validation(self):
        engine = self.engine()
        session = self.session(engine)
        before = self.bridge.snapshot(session)
        self.status(LIMIT, self.bridge.call, session, FORWARD,
                    np.ones((1, 8), '<f4').tobytes(), rows=1, capacity=31)
        self.status(LIMIT, self.bridge.call, session, SNAPSHOT, capacity=8193)
        self.status(LIMIT, self.bridge.call, session, SNAPSHOT, capacity=10)
        self.status(INVALID, self.bridge.call, engine, RESET)
        self.status(INVALID, self.bridge.call, session, EMBED)
        self.status(INVALID, self.bridge.call, session, FORWARD, b'x', rows=1, capacity=32)
        self.assertEqual(before, self.bridge.snapshot(session))
        for value in (-1, True, 1 << 64):
            with self.assertRaises(ValueError):
                self.bridge.call(session, RESET, rows=value)

    def test_raw_c_contract_and_callback_reentrancy(self):
        lib = self.bridge.lib
        calls = []
        live = U64()
        def callback(ctx, op, rows, src, size, dst, cap, written):
            calls.append(op)
            written[0] = 0
            self.assertEqual(lib.oa_segment_release(live.value), BUSY)
            return OK
        cb = CALLBACK(callback)
        with self.bridge.lock:
            self.assertEqual(lib.oa_segment_register(99, EDGE, 0, 128, cb, None, C.byref(live)), BAD_VERSION)
            self.assertEqual(lib.oa_segment_register(VERSION, SESSION, 999999, 128, cb, None, C.byref(live)), BAD_HANDLE)
            self.assertEqual(lib.oa_segment_register(VERSION, EDGE, 0, 128, cb, None, C.byref(live)), OK)
            try:
                written = U64(999)
                self.assertEqual(lib.oa_segment_call(live, EMBED, 1, None, 8, None, 0, C.byref(written)), INVALID)
                self.assertEqual(written.value, 0)
                self.assertEqual(lib.oa_segment_call(live, EMBED, 1, None, 0, None, 129, C.byref(written)), INVALID)
                buffer = C.create_string_buffer(8)
                self.assertEqual(lib.oa_segment_call(live, EMBED, 1, buffer, 8, buffer, 8, C.byref(written)), OK)
                self.assertEqual(calls, [EMBED])
            finally:
                self.assertEqual(lib.oa_segment_release(live), OK)

    def test_callback_invalid_status_and_length(self):
        lib = self.bridge.lib
        for response, length in ((-1, 0), (99, 0), (OK, 9)):
            def callback(ctx, op, rows, src, size, dst, cap, written):
                written[0] = length if op != CLOSE else 0
                return response if op != CLOSE else OK
            cb = CALLBACK(callback)
            handle = U64()
            with self.bridge.lock:
                self.assertEqual(lib.oa_segment_register(VERSION, EDGE, 0, 128, cb, None, C.byref(handle)), OK)
                try:
                    written = U64(999)
                    buffer = C.create_string_buffer(8)
                    self.assertEqual(lib.oa_segment_call(handle, EMBED, 1, buffer, 8, buffer, 8, C.byref(written)), CALLBACK_ERROR)
                    self.assertEqual(written.value, 0)
                finally:
                    self.assertEqual(lib.oa_segment_release(handle), OK)

    def test_parent_identity_and_close_failure_retry(self):
        first = self.engine()
        second = self.engine()
        backend = self.bridge._objects[first][0]
        session = backend.create_session()
        try:
            with self.assertRaises(ValueError):
                self.bridge.register(session, parent=second)
        finally:
            session.close()
        with patch.object(backend, 'close', side_effect=RuntimeError('close failure')):
            self.status(CALLBACK_ERROR, self.bridge.release, first)
        self.bridge.release(first)

    def test_python_callback_reentry_is_busy(self):
        session = self.session(self.engine())
        def cancelled():
            self.status(BUSY, self.bridge.reset, session)
            self.status(BUSY, self.bridge.release, session)
            return True
        self.status(CANCELLED, self.bridge.hidden, session, np.ones((1, 8), np.float32), cancelled=cancelled)

    def test_adapter_is_pinned_until_native_release(self):
        bridge = SegmentBridge(self.library, max_bytes=8192)
        handle = bridge.register(OlmoeEdge(self.root))
        reference = weakref.ref(bridge)
        del bridge
        gc.collect()
        pinned = reference()
        self.assertIsNotNone(pinned)
        try:
            self.assertEqual(pinned.embeddings(handle, [1]).shape, (1, 8))
        finally:
            pinned.release(handle)
        del pinned
        gc.collect()
        self.assertIsNone(reference())

    def test_raw_registry_capacity_and_transfer_limit(self):
        lib = self.bridge.lib
        calls = []
        def callback(ctx, op, rows, src, size, dst, cap, written):
            calls.append(op)
            written[0] = 0
            return OK
        cb = CALLBACK(callback)
        handles = []
        with self.bridge.lock:
            try:
                # Registry storage is a documented 65,536 live-handle bound;
                # page size is an implementation detail, not a capacity
                # contract. Exercise the maximum through the public constant.
                for _ in range(65536):
                    handle = U64()
                    self.assertEqual(lib.oa_segment_register(VERSION, EDGE, 0, 16, cb, None, C.byref(handle)), OK)
                    handles.append(handle.value)
                extra = U64(999)
                self.assertEqual(lib.oa_segment_register(VERSION, EDGE, 0, 16, cb, None, C.byref(extra)), LIMIT)
                self.assertEqual(extra.value, 0)
                output = C.create_string_buffer(17)
                written = U64(999)
                self.assertEqual(lib.oa_segment_call(handles[0], EMBED, 1, None, 0,
                    output, 17, C.byref(written)), LIMIT)
                self.assertEqual(written.value, 0)
                self.assertEqual(calls, [])
            finally:
                for handle in handles:
                    self.assertEqual(lib.oa_segment_release(handle), OK)

    def test_close_releases_children_first_and_is_terminal_idempotent(self):
        edge = self.register(OlmoeEdge(self.root))
        engine = self.engine()
        sessions = [self.session(engine), self.session(engine)]
        order = []
        backends = [record[0] for record in self.bridge._objects.values()]
        with ExitStack() as stack:
            for handle, record in self.bridge._objects.items():
                original = record[0].close
                def close(handle=handle, original=original):
                    order.append(handle)
                    original()
                stack.enter_context(patch.object(record[0], 'close', side_effect=close))
            self.bridge.close()
            self.bridge.close()
        self.assertEqual(order[:2], sessions)
        self.assertEqual(set(order[2:]), {engine, edge})
        self.assertEqual(self.bridge._objects, {})
        self.assertTrue(all(backend._closed for backend in backends))
        with self.assertRaisesRegex(RuntimeError, 'bridge is closed'):
            self.bridge.__enter__()
        backend = OlmoeEdge(self.root)
        try:
            with self.assertRaisesRegex(RuntimeError, 'bridge is closed'):
                self.bridge.register(backend)
            self.assertFalse(backend._closed)
        finally:
            backend.close()

    def test_close_failure_retains_children_and_continues_unrelated_cleanup(self):
        first = self.engine()
        failing = self.session(first)
        sibling = self.session(first)
        second = self.engine()
        other = self.session(second)
        edge = self.register(OlmoeEdge(self.root))
        record = self.bridge._objects[failing]
        parent = self.bridge._objects[first][0]
        failure = RuntimeError('session close failed')
        with patch.object(record[0], 'close', side_effect=failure), \
                patch.object(parent, 'close', wraps=parent.close) as parent_close:
            with self.assertRaises(SegmentABICloseError) as caught:
                self.bridge.close()
            parent_close.assert_not_called()
        error = caught.exception
        self.assertEqual(error.remaining_handles, (first, failing))
        self.assertEqual([(h, e.status) for h, e in error.failures],
                         [(failing, CALLBACK_ERROR), (first, BUSY)])
        self.assertIs(error.failures[0][1].__cause__, failure)
        self.assertIs(self.bridge._objects[failing], record)
        gc.collect()
        self.bridge.reset(failing)
        for handle in (sibling, second, other, edge):
            self.assertEqual(self.bridge.lib.oa_segment_release(handle), BAD_HANDLE)
        self.bridge.close()
        self.assertEqual(self.bridge._objects, {})

    def test_close_reports_multiple_root_failures_and_can_retry(self):
        engine = self.engine()
        edge = self.register(OlmoeEdge(self.root))
        with patch.object(self.bridge._objects[engine][0], 'close', side_effect=RuntimeError('engine')), \
                patch.object(self.bridge._objects[edge][0], 'close', side_effect=RuntimeError('edge')):
            with self.assertRaises(SegmentABICloseError) as caught:
                self.bridge.close()
        self.assertEqual([h for h, e in caught.exception.failures], [engine, edge])
        self.bridge.close()

    def test_context_manager_success_and_body_exception(self):
        with self.bridge as bridge:
            self.assertIs(bridge, self.bridge)
            self.session(self.engine())
            self.register(OlmoeEdge(self.root))
        self.assertEqual(bridge._objects, {})
        bridge = SegmentBridge(self.library)
        original = KeyboardInterrupt('body interrupted')
        with self.assertRaises(KeyboardInterrupt) as caught:
            with bridge:
                bridge.register(OlmoeEdge(self.root))
                raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(bridge._objects, {})

    def test_context_cleanup_failure_preserves_original_and_adds_note(self):
        engine = self.engine()
        session = self.session(engine)
        backend = self.bridge._objects[session][0]
        original = ValueError('body failure')
        with patch.object(backend, 'close', side_effect=RuntimeError('cleanup failure')):
            with self.assertRaises(ValueError) as caught:
                with self.bridge:
                    raise original
        self.assertIs(caught.exception, original)
        self.assertIn('SegmentBridge cleanup failed', original.__notes__[0])
        self.assertIn(str(session), original.__notes__[0])
        self.assertIn('retry bridge.close()', original.__notes__[0])
        self.assertEqual(set(self.bridge._objects), {engine, session})
        self.bridge.close()

    def test_context_cleanup_failure_without_body_error_is_raised(self):
        edge = self.register(OlmoeEdge(self.root))
        with patch.object(self.bridge._objects[edge][0], 'close', side_effect=RuntimeError('cleanup')):
            with self.assertRaises(SegmentABICloseError):
                with self.bridge:
                    pass
        self.bridge.close()

    def test_context_unexpected_cleanup_error_does_not_mask_body(self):
        original = RuntimeError('body failure')
        with patch.object(self.bridge, 'close', side_effect=KeyboardInterrupt('cleanup interrupted')):
            with self.assertRaises(RuntimeError) as caught:
                with self.bridge:
                    raise original
        self.assertIs(caught.exception, original)
        self.assertIn('cleanup interrupted', original.__notes__[0])

    def test_close_during_callback_is_busy_and_preserves_handles(self):
        session = self.session(self.engine())
        def cancelled():
            self.status(BUSY, self.bridge.close)
            return True
        self.status(CANCELLED, self.bridge.hidden, session, np.ones((1, 8), np.float32), cancelled=cancelled)
        self.bridge.reset(session)

    def test_duplicate_backend_across_adapters_does_not_transfer_ownership(self):
        backend = OlmoeEdge(self.root)
        handle = self.register(backend)
        with SegmentBridge(self.library) as other:
            with self.assertRaisesRegex(ValueError, 'already registered'):
                other.register(backend)
            self.assertEqual(other._objects, {})
            with patch.object(backend, 'close', side_effect=RuntimeError('retry')):
                self.status(CALLBACK_ERROR, self.bridge.release, handle)
                with self.assertRaisesRegex(ValueError, 'already registered'):
                    other.register(backend)
        self.assertFalse(backend._closed)
        self.assertEqual(self.bridge.embeddings(handle, [1]).shape, (1, 8))

    def test_native_rmsnorm_differential_and_strided_inputs(self):
        native = NativeSegmentMath(self.library, max_bytes=65536)
        rng = np.random.default_rng(817)
        for rows, width in ((1, 1), (3, 8), (7, 33), (2, 1024)):
            hidden = rng.standard_normal((rows, width * 2)).astype(np.float32)[:, ::2]
            weight = rng.standard_normal(width * 2).astype(np.float32)[::2]
            before = hidden.copy()
            for epsilon in (1e-5, 0.1):
                reference = hidden * (1.0 / np.sqrt(
                    np.mean(hidden.astype(np.float64) ** 2, axis=1, keepdims=True) + epsilon)) * weight
                actual = native.rmsnorm(hidden, weight, epsilon=epsilon)
                np.testing.assert_allclose(actual, reference.astype(np.float32), rtol=2e-6, atol=2e-7)
                fp32_reference = hidden / np.sqrt(np.mean(hidden * hidden, axis=1, keepdims=True) + epsilon) * weight
                np.testing.assert_allclose(actual, fp32_reference, rtol=2e-6, atol=2e-7)
            np.testing.assert_array_equal(hidden, before)
        zeros = np.zeros((2, 8), np.float32)
        np.testing.assert_array_equal(native.rmsnorm(zeros, np.ones(8, np.float32)), zeros)
        extreme = np.array([[np.finfo(np.float32).max, 0]], np.float32)
        np.testing.assert_allclose(native.rmsnorm(extreme, np.ones(2, np.float32)),
                                   [[np.sqrt(2), 0]], rtol=2e-6)

    def test_native_rmsnorm_raw_errors_preserve_output(self):
        native = NativeSegmentMath(self.library)
        hidden = np.ones((2, 2), np.float32)
        weight = np.ones(2, np.float32)
        output = np.full((2, 2), 17, np.float32)
        def raw(rows=2, width=2, epsilon=1e-5, limit=16, src=None):
            return native._rmsnorm(hidden.ctypes.data if src is None else src,
                weight.ctypes.data, output.ctypes.data, rows, width, epsilon, limit)
        for status, kwargs in ((INVALID, {'rows': 0}), (INVALID, {'width': 0}),
                               (INVALID, {'epsilon': 0}), (INVALID, {'epsilon': float('nan')}),
                               (INVALID, {'src': 0}), (LIMIT, {'limit': 15}),
                               (LIMIT, {'rows': (1 << 64) - 1}),
                               (LIMIT, {'width': 1 << 63})):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(raw(**kwargs), status)
                np.testing.assert_array_equal(output, 17)
        hidden[-1, -1] = np.inf
        self.assertEqual(raw(), INVALID)
        np.testing.assert_array_equal(output, 17)
        hidden[:] = [[1, 0], [0, 1]]
        weight[-1] = np.finfo(np.float32).max
        self.assertEqual(raw(), INVALID)
        np.testing.assert_array_equal(output, 17)
        weight[-1] = np.nan
        self.assertEqual(raw(), INVALID)
        np.testing.assert_array_equal(output, 17)
        with self.assertRaises(ValueError):
            native.rmsnorm(hidden.astype(np.float64), weight)
        self.status(LIMIT, NativeSegmentMath(self.library, max_bytes=4).rmsnorm,
                    hidden, weight)

    def test_native_operation_envelopes_rejected_without_callback(self):
        lib = self.bridge.lib
        calls = []
        def callback(ctx, op, rows, src, size, dst, cap, written):
            calls.append(op)
            written[0] = 0
            return OK
        cb = CALLBACK(callback)
        engine, session, edge = U64(), U64(), U64()
        buffer = C.create_string_buffer(8)
        with self.bridge.lock:
            self.assertEqual(lib.oa_segment_register(VERSION, 1, 0, 128, cb, None, C.byref(engine)), OK)
            self.assertEqual(lib.oa_segment_register(VERSION, SESSION, engine, 128, cb, None, C.byref(session)), OK)
            self.assertEqual(lib.oa_segment_register(VERSION, EDGE, 0, 128, cb, None, C.byref(edge)), OK)
            try:
                cases = [(session, FORWARD, 0, 8, 8), (edge, EMBED, 1, 0, 8),
                         (edge, 4, 1, 8, 0), (session, RESET, 1, 0, 0),
                         (session, RESET, 0, 8, 0), (session, RESET, 0, 0, 8),
                         (session, SNAPSHOT, 1, 0, 8), (session, SNAPSHOT, 0, 8, 8),
                         (session, 7, 0, 0, 0), (session, 7, 1, 8, 0),
                         (session, 7, 0, 8, 8)]
                for handle, op, rows, size, capacity in cases:
                    with self.subTest(op=op, rows=rows, size=size, capacity=capacity):
                        written = U64(999)
                        self.assertEqual(lib.oa_segment_call(handle, op, rows, buffer,
                            size, buffer, capacity, C.byref(written)), INVALID)
                        self.assertEqual(written.value, 0)
                self.assertEqual(calls, [])
            finally:
                for handle in (session, engine, edge):
                    self.assertEqual(lib.oa_segment_release(handle), OK)


if __name__ == '__main__':
    unittest.main()
