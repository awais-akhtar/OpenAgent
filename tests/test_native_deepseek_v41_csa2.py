"""Compiled ownership and equation fixtures, not decoder certification."""
import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


class CSA2Ownership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/native_deepseek_v41_csa2.c'
        output = Path(cls.directory.name) / ('v41.dll' if os.name == 'nt' else 'v41.so')
        if os.name == 'nt':
            vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
            location = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
                'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
            env = Path(location) / 'VC/Auxiliary/Build/vcvars64.bat'
            args = ['cl', '/nologo', '/W4', '/WX', '/O2', '/LD', str(source), '/link', '/OUT:' + str(output)]
            command = 'cmd.exe /d /s /c ""' + str(env) + '" && ' + subprocess.list2cmdline(args) + '"'
        else:
            command = [shutil.which('cc') or 'cc', '-std=c99', '-Wall', '-Wextra', '-Werror',
                       '-shared', '-fPIC', str(source), '-o', str(output), '-lm']
        result = subprocess.run(command, cwd=cls.directory.name, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        cls.lib = C.CDLL(str(output))
        if os.name == 'nt':
            import _ctypes
            cls.addClassCleanup(lambda: _ctypes.FreeLibrary(cls.lib._handle))
        cls.lib.oa_v41_csa2_size.restype = C.c_size_t
        cls.lib.oa_v41_ced_project.argtypes = [C.c_void_p, C.c_size_t] * 3 + [
            C.c_uint32] * 4 + [C.c_void_p, C.c_size_t] * 2
        cls.lib.oa_v41_csa2_reset.argtypes = [C.c_void_p, C.c_size_t]
        cls.lib.oa_v41_csa2_publish.argtypes = [C.c_void_p, C.c_size_t, C.c_int, C.c_int32,
            C.c_int32, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
            C.c_uint32, C.c_uint32, C.c_uint32, C.c_void_p, C.c_size_t]
        cls.lib.oa_v41_csa2_read.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p,
            C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]

    def setUp(self):
        self.size = self.lib.oa_v41_csa2_size()
        self.buffer = C.create_string_buffer(self.size)
        self.kv = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.keys = np.arange(8, dtype=np.float32).reshape(4, 2) * -0.5

    def project(self, hidden, w_kv, weights_z, kv, z, **overrides):
        args = dict(hidden=hidden.ctypes.data, hidden_count=hidden.size,
                    w_kv=w_kv.ctypes.data, kv_weight_count=w_kv.size,
                    w_z=weights_z.ctypes.data, z_weight_count=weights_z.size,
                    rows=hidden.shape[0], hidden_width=hidden.shape[1],
                    kv_width=w_kv.shape[1], z_width=weights_z.shape[1],
                    kv=kv.ctypes.data, kv_count=kv.size, z=z.ctypes.data, z_count=z.size)
        args.update(overrides)
        return self.lib.oa_v41_ced_project(*args.values())

    def test_ced_equation_one_full_and_chunked(self):
        rng = np.random.default_rng(4101)
        for rows, width, kv_width, z_width in [(1, 1, 1, 1), (4, 3, 3, 2), (32, 8, 8, 8)]:
            hidden = rng.normal(size=(rows, width)).astype(np.float32)
            w_kv = rng.normal(size=(width, kv_width)).astype(np.float32)
            w_z = rng.normal(size=(width, z_width)).astype(np.float32)
            kv = np.empty((rows, kv_width), np.float32)
            z = np.empty((rows, z_width), np.float32)
            self.assertEqual(self.project(hidden, w_kv, w_z, kv, z), 0)
            np.testing.assert_allclose(kv, (hidden.astype(np.float64) @ w_kv).astype(np.float32),
                                       rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(z, (hidden.astype(np.float64) @ w_z).astype(np.float32),
                                       rtol=1e-6, atol=1e-6)
            for r in range(rows):
                chunk_kv = np.empty((1, kv_width), np.float32)
                chunk_z = np.empty((1, z_width), np.float32)
                self.assertEqual(self.project(hidden[r:r+1], w_kv, w_z, chunk_kv, chunk_z), 0)
                np.testing.assert_array_equal(chunk_kv[0], kv[r])
                np.testing.assert_array_equal(chunk_z[0], z[r])

    def test_ced_projection_failures_preserve_both_outputs(self):
        hidden = np.ones((4, 3), np.float32)
        w_kv = np.ones((3, 3), np.float32)
        w_z = np.ones((3, 2), np.float32)
        kv = np.full((4, 3), -123, np.float32)
        z = np.full((4, 2), -456, np.float32)
        for overrides in [dict(rows=33), dict(hidden_count=11), dict(z_count=7),
                          dict(w_z=None), dict(hidden_width=0)]:
            self.assertEqual(self.project(hidden, w_kv, w_z, kv, z, **overrides), 1)
            np.testing.assert_array_equal(kv, -123)
            np.testing.assert_array_equal(z, -456)

        for bad in [np.nan, np.inf, np.finfo(np.float32).max]:
            broken_z = w_z.copy()
            broken_z[-1, -1] = bad
            broken_hidden = hidden * (2 if np.isfinite(bad) else 1)
            self.assertEqual(self.project(broken_hidden, w_kv, broken_z, kv, z), 3)
            np.testing.assert_array_equal(kv, -123)
            np.testing.assert_array_equal(z, -456)

    def test_ced_overlapping_outputs_reject_before_publication(self):
        hidden = np.ones((2, 3), np.float32)
        weights = np.ones((3, 3), np.float32)
        shared = np.full(12, -99, np.float32)
        before = shared.copy()
        self.assertEqual(self.project(hidden, weights, weights,
            shared[:6].reshape(2, 3), shared[3:9].reshape(2, 3)), 1)
        np.testing.assert_array_equal(shared, before)

    def test_corrupted_state_rejected_without_publication(self):
        self.assertEqual(self.publish(0, 2, 2, [1, 3], self.kv, self.keys), 0)
        good = self.buffer.raw
        index_offset = 8 * 4 + 2 * 32 * 8 * 4
        mutations = [(4 * 4, C.c_uint32, 100), (2 * 4, C.c_uint32, 99),
                     (5 * 4, C.c_int32, -1), (6 * 4, C.c_int32, 1),
                     (7 * 4, C.c_int32, 1), (index_offset, C.c_int32, 99),
                     (index_offset + 4, C.c_int32, 1), (8 * 4, C.c_float, np.nan)]
        for offset, kind, value in mutations:
            C.memmove(self.buffer, good, self.size)
            kind.from_buffer(self.buffer, offset).value = value
            corrupt = self.buffer.raw
            with self.subTest(offset=offset, value=value):
                self.assertNotEqual(self.publish(2, 3, 2), 0)
                self.assertEqual(self.buffer.raw, corrupt)
                kv, keys = np.full(12, -11, np.float32), np.full(8, -22, np.float32)
                ids, metadata = np.full(2, -33, np.int32), np.full(3, -44, np.int32)
                self.assertEqual(self.lib.oa_v41_csa2_read(self.buffer, self.size,
                    kv.ctypes.data, kv.size, keys.ctypes.data, keys.size,
                    ids.ctypes.data, ids.size, metadata.ctypes.data, metadata.size), 2)
                np.testing.assert_array_equal(kv, -11)
                np.testing.assert_array_equal(keys, -22)
                np.testing.assert_array_equal(ids, -33)
                np.testing.assert_array_equal(metadata, -44)

    def test_read_rejects_state_and_output_aliases(self):
        self.assertEqual(self.publish(0, 2, 2, [1, 3], self.kv, self.keys), 0)
        before = self.buffer.raw
        kv, keys = np.full(12, -11, np.float32), np.full(8, -22, np.float32)
        ids, metadata = np.full(2, -33, np.int32), np.full(3, -44, np.int32)
        for kv_pointer, key_pointer in [(C.addressof(self.buffer), keys.ctypes.data),
                                         (kv.ctypes.data, kv.ctypes.data + 4)]:
            self.assertEqual(self.lib.oa_v41_csa2_read(self.buffer, self.size,
                kv_pointer, kv.size, key_pointer, keys.size,
                ids.ctypes.data, ids.size, metadata.ctypes.data, metadata.size), 1)
            self.assertEqual(self.buffer.raw, before)
            np.testing.assert_array_equal(kv, -11)
            np.testing.assert_array_equal(keys, -22)
            np.testing.assert_array_equal(ids, -33)
            np.testing.assert_array_equal(metadata, -44)
    def test_projected_payload_cache_reindex_reuse_and_rollback(self):
        hidden = np.arange(12, dtype=np.float32).reshape(4, 3)
        weights = np.eye(3, dtype=np.float32)
        index_weights = np.array([[1, 0], [0, 1], [1, -1]], np.float32)
        kv, keys = np.empty_like(self.kv), np.empty_like(self.keys)
        self.assertEqual(self.project(hidden, weights, index_weights, kv, keys), 0)
        # The second projection is Z mathematically, not certified indexer K.
        self.assertEqual(self.publish(0, 2, 2, [0, 1], kv, keys), 0)
        before = self.buffer.raw
        self.assertEqual(self.publish(1, 3, 2, [4]), 1)
        self.assertEqual(self.buffer.raw, before)
        self.assertEqual(self.publish(1, 3, 2, [3, 2]), 0)
        self.assertEqual(self.publish(2, 4, 2), 0)
        actual_kv, actual_keys, ids, metadata = self.read()
        np.testing.assert_array_equal(actual_kv, hidden)
        np.testing.assert_array_equal(actual_keys, hidden.astype(np.float64) @ index_weights)
        np.testing.assert_array_equal(ids, [3, 2])
        np.testing.assert_array_equal(metadata, [2, 3, 4])

    def publish(self, mode, layer, source, selection=None, kv=None, keys=None, **overrides):
        ids = None if selection is None else np.asarray(selection, dtype=np.int32)
        args = dict(buffer=self.buffer, size=self.size, mode=mode, layer=layer, source=source,
                    kv=None if kv is None else kv.ctypes.data, kv_count=0 if kv is None else kv.size,
                    keys=None if keys is None else keys.ctypes.data, key_count=0 if keys is None else keys.size,
                    rows=0 if kv is None else kv.shape[0], width=0 if kv is None else kv.shape[1],
                    index_width=0 if keys is None else keys.shape[1],
                    indices=None if ids is None else ids.ctypes.data, count=0 if ids is None else ids.size)
        args.update(overrides)
        return self.lib.oa_v41_csa2_publish(*args.values())

    def read(self, count=2):
        kv, keys = np.empty_like(self.kv), np.empty_like(self.keys)
        ids, metadata = np.empty(count, np.int32), np.empty(3, np.int32)
        code = self.lib.oa_v41_csa2_read(self.buffer, self.size, kv.ctypes.data, kv.size,
            keys.ctypes.data, keys.size, ids.ctypes.data, ids.size, metadata.ctypes.data, metadata.size)
        self.assertEqual(code, 0)
        return kv, keys, ids, metadata

    def test_full_reindex_reuse_analytical_fixture(self):
        # Independent score/ranking fixture; native code owns state, not scoring.
        query = np.array([1, -2], dtype=np.float32)
        selected = np.argsort(-(self.keys @ query))[:2]
        self.assertEqual(self.publish(0, 2, 2, [0, 1], self.kv, self.keys), 0)
        self.assertEqual(self.publish(1, 3, 2, selected), 0)
        self.assertEqual(self.publish(2, 4, 2), 0)
        kv, keys, ids, metadata = self.read()
        np.testing.assert_array_equal(kv, self.kv)
        np.testing.assert_array_equal(keys, self.keys)
        np.testing.assert_array_equal(ids, selected)
        np.testing.assert_array_equal(metadata, [2, 3, 4])
        self.kv[:] = -99
        self.keys[:] = -99
        kv, keys, _, _ = self.read()
        np.testing.assert_array_equal(kv, np.arange(12, dtype=np.float32).reshape(4, 3))
        np.testing.assert_array_equal(keys, np.arange(8, dtype=np.float32).reshape(4, 2) * -0.5)

    def test_new_full_invalidates_old_owner_and_selection(self):
        self.assertEqual(self.publish(0, 2, 2, [0, 1], self.kv, self.keys), 0)
        self.assertEqual(self.publish(0, 8, 8, [3, 2], self.kv + 10, self.keys + 10), 0)
        before = self.buffer.raw
        self.assertEqual(self.publish(2, 9, 2), 2)
        self.assertEqual(self.buffer.raw, before)
        self.assertEqual(self.publish(2, 9, 8), 0)
        np.testing.assert_array_equal(self.read()[2], [3, 2])

    def test_rejected_publications_rollback_byte_for_byte(self):
        self.assertEqual(self.publish(0, 2, 2, [0, 1], self.kv, self.keys), 0)
        before = self.buffer.raw
        calls = [lambda: self.publish(1, 3, 2, [0, 0]),
                 lambda: self.publish(1, 3, 2, [-1, 0]),
                 lambda: self.publish(1, 3, 2, [0, 4]),
                 lambda: self.publish(2, 2, 2),
                 lambda: self.publish(2, 3, 8),
                 lambda: self.publish(2, 3, 2, [0]),
                 lambda: self.publish(0, 8, 8, [0, 1], self.kv, self.keys, rows=33),
                 lambda: self.publish(0, 8, 8, [0, 1], self.kv, self.keys, kv_count=11),
                 lambda: self.publish(0, 8, 8, [0, 1], self.kv * np.nan, self.keys),
                 lambda: self.publish(2, 3, 2, size=self.size - 1)]
        for call in calls:
            self.assertNotEqual(call(), 0)
            self.assertEqual(self.buffer.raw, before)
        self.assertEqual(self.publish(1, 3, 2, [2, 3]), 0)

    def test_output_bounds_and_reset(self):
        self.assertEqual(self.publish(2, 1, 0), 2)
        self.assertEqual(self.publish(0, 2, 2, [0, 1], self.kv, self.keys), 0)
        guard = C.create_string_buffer(b'Z' * 100)
        before = guard.raw
        self.assertEqual(self.lib.oa_v41_csa2_read(self.buffer, self.size,
            guard, 11, guard, 8, guard, 2, guard, 3), 1)
        self.assertEqual(guard.raw, before)
        self.assertEqual(self.lib.oa_v41_csa2_reset(self.buffer, self.size), 0)
        self.assertEqual(self.buffer.raw, bytes(self.size))
        self.assertEqual(self.publish(1, 3, 2, [0]), 2)

    def test_state_canaries_and_independent_query_lifetimes(self):
        guarded = C.create_string_buffer(b'A' * 16 + bytes(self.size) + b'B' * 16,
                                         self.size + 32)
        pointer = C.c_void_p(C.addressof(guarded) + 16)
        self.assertEqual(self.publish(0, 2, 2, [0, 1], self.kv, self.keys,
                                      buffer=pointer), 0)
        self.assertEqual(guarded.raw[:16], b'A' * 16)
        self.assertEqual(guarded.raw[-16:], b'B' * 16)
        self.assertEqual(self.buffer.raw, bytes(self.size))
        before = guarded.raw
        self.assertNotEqual(self.publish(1, 3, 2, list(range(9)), buffer=pointer), 0)
        self.assertEqual(guarded.raw, before)
        self.assertEqual(self.lib.oa_v41_csa2_reset(pointer, self.size), 0)
        self.assertEqual(guarded.raw, b'A' * 16 + bytes(self.size) + b'B' * 16)


if __name__ == '__main__':
    unittest.main()
