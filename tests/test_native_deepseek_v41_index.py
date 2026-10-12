"""Analytical equations and native rollback, not decoder certification."""
import ctypes as C
import importlib.util
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v41_index_fixture", ROOT / "openagent/adaptive_engine/native_deepseek_v41_index.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IndexEquations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        source = ROOT / "openagent/adaptive_engine/native_deepseek_v41_index.c"
        cls.output = Path(cls.directory.name) / ("index.dll" if os.name == "nt" else "index.so")
        if os.name == "nt":
            vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
            location = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
            env = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
            args = ["cl", "/nologo", "/W4", "/WX", "/O2", "/fp:strict", "/LD", str(source), "/link", "/OUT:" + str(cls.output)]
            command = 'cmd.exe /d /s /c ""' + str(env) + '" && ' + subprocess.list2cmdline(args) + '"'
        else:
            command = [shutil.which("cc") or "cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
                       "-shared", "-fPIC", str(source), "-o", str(cls.output), "-lm"]
        result = subprocess.run(command, cwd=cls.directory.name, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        cls.index = module.NativeDeepSeekV41Index(cls.output)
        cls.addClassCleanup(cls.index.close)
        cls.lib = cls.index.library

    def test_analytical_multihead_relu_and_signed_weights(self):
        # Dot pairs: (2,-3), (-2,3), (4,6). ReLU precedes signed mixing.
        scores = self.index.score([1, 0, 0, 1], [2, 9, 8, -3, -2, 0, 0, 3, 4, 0, 0, 6],
                                  [2, -1], 3, 2, 2)
        self.assertEqual(scores, (4, -3, 2))
        self.assertEqual(self.index.score([1], [-2], [1], 1, 1, 1), (0,))

    def test_maximum_geometry_and_row_chunk_equation(self):
        q = [float(i % 5 - 2) for i in range(64)]
        k = [float(i % 7 - 3) for i in range(2048)]
        w = [float(i - 3) for i in range(8)]
        actual = self.index.score(q, k, w, 32, 8, 8)
        expected = tuple(sum(w[h] * max(0, sum(q[h*8+d] * k[r*64+h*8+d]
                        for d in range(8))) for h in range(8)) for r in range(32))
        self.assertEqual(actual, expected)
        for r in range(32):
            self.assertEqual(self.index.score(q, k[r*64:(r+1)*64], w, 1, 8, 8), (actual[r],))

    def test_block_max_admission_tail_and_cutoff_ties(self):
        s = [100, -2, 5, 5, 5, 4, 5]
        a = [0, 1, 1, 1, 1, 1, 1]
        self.assertEqual(self.index.select(s, a, 7, 2, 2), (1, 2))
        self.assertEqual(self.index.select(s, a, 7, 2, 32), (1, 2, 3, 0))
        self.assertEqual(self.index.select(s, [0]*7, 7, 2, 2, capacity=0), ())

    def test_deeper_selection_recomputes_only_candidate_membership(self):
        blocks = self.index.select([1, 9, 3, 2, 8], [1]*5, 5, 2, 2)
        self.assertEqual(blocks, (0, 2))
        # A deeper query ranks the same key rows differently. Excluded row 2 wins globally.
        scores = self.index.score([1], [5, 5, 100, 90, 6], [1], 5, 1, 1)
        for candidates in (blocks, tuple(reversed(blocks))):
            self.assertEqual(self.index.select(scores, [1]*5, 5, 2, 2, candidates=candidates), (4, 0))
        self.assertEqual(self.index.select(scores, [1]*5, 5, 2, 2, candidates=[]), ())
        self.assertEqual(self.index.select(scores, [0, 1, 1, 1, 1], 5, 2, 3, candidates=blocks), (4, 1))

    def test_selection_against_independent_sort_equation(self):
        scores = [-3, -2, -2, 0, -1, 0, -4]
        for block_size in (1, 2, 3, 7, 32):
            nb = (7 + block_size - 1) // block_size
            for admitted in itertools.product((0, 1), repeat=7):
                maxima = {b: max(scores[r] for r in range(7)
                            if admitted[r] and r // block_size == b)
                          for b in range(nb) if any(admitted[r] and r // block_size == b
                                                   for r in range(7))}
                ranked = sorted(maxima, key=lambda b: (-maxima[b], b))
                for limit in (1, 2, 32):
                    self.assertEqual(self.index.select(scores, admitted, 7, block_size, limit),
                                     tuple(ranked[:limit]))
                candidates = tuple(reversed(ranked[:2]))
                rows = sorted((r for r in range(7) if admitted[r] and r // block_size in candidates),
                              key=lambda r: (-scores[r], r))
                self.assertEqual(self.index.select(scores, admitted, 7, block_size, 2,
                                                   candidates=candidates), tuple(rows[:2]))
        scores = [float(r % 3 - 2) for r in range(32)]
        expected = tuple(sorted(range(32), key=lambda r: (-scores[r], r)))
        self.assertEqual(self.index.select(scores, [1]*32, 32, 1, 32), expected)
        self.assertEqual(self.index.select(scores, [1]*32, 32, 32, 32, candidates=[0]), expected)

    def select_raw(self, **changes):
        scores = (C.c_float * 5)(1, 9, 3, 2, 8)
        admitted = (C.c_uint8 * 5)(1, 1, 1, 1, 1)
        out = (C.c_int32 * 4)(-77, -77, -77, -77)
        count = C.c_uint32(99)
        args = dict(scores=scores, ns=5, admitted=admitted, na=5, rows=5, block_size=2,
                    mode=0, candidates=None, nc=0, limit=2, out=out, capacity=4,
                    count=C.byref(count), budget=63)
        args.update(changes)
        before = bytes(out), bytes(count)
        status = self.lib.oa_v41_index_select(*args.values())
        return status, before, (bytes(out), bytes(count))

    def test_selection_errors_are_transactional(self):
        for changes, expected in [({"rows":33}, 1), ({"ns":4}, 1), ({"block_size":0}, 1),
            ({"limit":0}, 1), ({"budget":62}, 2), ({"capacity":1}, 4),
            ({"scores":(C.c_float*5)(1, 9, 3, 2, float("nan"))}, 3),
            ({"admitted":(C.c_uint8*5)(1, 1, 1, 1, 2)}, 1),
            ({"mode":1, "candidates":(C.c_int32*2)(0, 0), "nc":2}, 1),
            ({"mode":1, "candidates":(C.c_int32*1)(3), "nc":1}, 1),
            ({"mode":1, "candidates":(C.c_int32*1)(-1), "nc":1}, 1),
            ({"scores":None}, 1), ({"count":None}, 1)]:
            with self.subTest(changes=changes):
                status, before, after = self.select_raw(**changes)
                self.assertEqual(status, expected)
                self.assertEqual(before, after)
        self.assertEqual(self.select_raw()[0], 0)

    def test_score_numeric_and_extent_rollback(self):
        out = (C.c_float*2)(-77, -88)
        q, k, w = (C.c_float*1)(2), (C.c_float*2)(1, 2), (C.c_float*1)(1)
        base = [q, 1, k, 2, w, 1, 2, 1, 1, out, 2, 8]
        for position, value, expected in [(1, 2, 1), (6, 33, 1), (11, 7, 2),
                                         (0, None, 1), (10, 1, 1)]:
            args = base.copy()
            args[position] = value
            before = bytes(out)
            self.assertEqual(self.lib.oa_v41_index_score(*args), expected)
            self.assertEqual(bytes(out), before)
        for bad in [float("nan"), float("inf"), 3.4028234663852886e38]:
            k[1] = bad
            before = bytes(out)
            self.assertEqual(self.lib.oa_v41_index_score(*base), 3)
            self.assertEqual(bytes(out), before)

    def test_alias_canaries_alignment_and_output_overlap(self):
        q, w = (C.c_float*1)(1), (C.c_float*1)(1)
        keys = (C.c_float*2)(3, 2)
        self.assertEqual(self.lib.oa_v41_index_score(q, 1, keys, 2, w, 1, 2, 1, 1, keys, 2, 8), 0)
        self.assertEqual(tuple(keys), (3, 2))
        guard = (C.c_int32*6)(111, -1, -1, -1, -1, 222)
        s, a = (C.c_float*3)(4, 4, 2), (C.c_uint8*3)(1, 1, 1)
        count = C.c_uint32(99)
        ptr = C.c_void_p(C.addressof(guard)+4)
        args = [s, 3, a, 3, 3, 1, 0, None, 0, 2, ptr, 4, C.byref(count), 39]
        self.assertEqual(self.lib.oa_v41_index_select(*args), 0)
        self.assertEqual(tuple(guard), (111, 0, 1, -1, -1, 222))
        before = bytes(guard)
        args[12] = ptr
        self.assertEqual(self.lib.oa_v41_index_select(*args), 1)
        self.assertEqual(bytes(guard), before)
        args[12] = C.byref(count)
        args[10] = C.c_void_p(C.addressof(guard)+1)
        self.assertEqual(self.lib.oa_v41_index_select(*args), 1)
        args[10] = C.c_void_p(C.c_size_t(-4).value)
        self.assertEqual(self.lib.oa_v41_index_select(*args), 1)
        # Selection may publish over scores only after the complete ranking is staged.
        aliased = (C.c_float*3)(4, 4, 2)
        args[0] = aliased
        args[10] = aliased
        args[11] = 3
        self.assertEqual(self.lib.oa_v41_index_select(*args), 0)
        self.assertEqual(tuple(C.cast(aliased, C.POINTER(C.c_int32))[:2]), (0, 1))

    def test_wrapper_bounds_capacity_and_close(self):
        for call in [lambda: self.index.score(itertools.repeat(1), [1], [1], 1, 1, 1),
                     lambda: self.index.score([1], [1], [1], True, 1, 1),
                     lambda: self.index.select([1], [1], 1, 1, 1, capacity=0),
                     lambda: self.index.select([1], [1], 1, 1, 1, candidates=[0, 0]),
                     lambda: self.index.select([1], [256], 1, 1, 1)]:
            with self.assertRaises(ValueError):
                call()
        local = module.NativeDeepSeekV41Index(self.output)
        local.close()
        local.close()
        with self.assertRaises(RuntimeError):
            local.score([1], [1], [1], 1, 1, 1)
        local = module.NativeDeepSeekV41Index(self.output)
        self.addCleanup(local.close)
        def closes():
            local.close()
            yield 1
        with self.assertRaises(RuntimeError):
            local.score(closes(), [1], [1], 1, 1, 1)
        local = module.NativeDeepSeekV41Index(self.output)
        self.addCleanup(local.close)
        with self.assertRaises(RuntimeError):
            local.select([1], [1], 1, 1, 1, candidates=closes())

    def test_concurrent_close_waits_for_score_and_select(self):
        for name, invoke, expected in [
            ("score", lambda x: x.score([2], [3], [1], 1, 1, 1), (6,)),
            ("select", lambda x: x.select([3, 4], [1, 1], 2, 1, 1), (1,)),
        ]:
            with self.subTest(operation=name):
                instance = module.NativeDeepSeekV41Index(self.output)
                entered, finish, closing = threading.Event(), threading.Event(), threading.Event()
                original = getattr(instance.library, "oa_v41_index_" + name)

                def held(*args):
                    entered.set()
                    if not finish.wait(5):
                        raise RuntimeError("test worker timed out")
                    return original(*args)

                def close():
                    closing.set()
                    instance.close()

                try:
                    with mock.patch.object(instance.library, "oa_v41_index_" + name, side_effect=held):
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            operation = pool.submit(invoke, instance)
                            try:
                                self.assertTrue(entered.wait(5))
                                closure = pool.submit(close)
                                self.assertTrue(closing.wait(5))
                                self.assertFalse(closure.done())
                                self.assertTrue(instance._finalizer.alive)
                                self.assertEqual(instance._active, 1)
                            finally:
                                finish.set()
                            self.assertEqual(operation.result(timeout=5), expected)
                            closure.result(timeout=5)
                    self.assertFalse(instance._finalizer.alive)
                    self.assertEqual(instance._active, 0)
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        invoke(instance)
                finally:
                    finish.set()
                    instance.close()

    def test_iterator_close_refused_and_operation_unwinds(self):
        instance = module.NativeDeepSeekV41Index(self.output)
        try:
            def values(value):
                with self.assertRaisesRegex(RuntimeError, "active operation"):
                    instance.close()
                self.assertTrue(instance._finalizer.alive)
                yield value

            self.assertEqual(instance.score(values(2), values(3), values(1), 1, 1, 1), (6,))
            self.assertEqual(instance.select(values(3), values(1), 1, 1, 1,
                                             candidates=values(0)), (0,))
            self.assertEqual(instance._active, 0)

            def unhandled():
                instance.close()
                yield 1

            for invoke in (
                lambda: instance.score(unhandled(), [1], [1], 1, 1, 1),
                lambda: instance.select([1], [1], 1, 1, 1, candidates=unhandled()),
            ):
                with self.assertRaisesRegex(RuntimeError, "active operation"):
                    invoke()
                self.assertEqual(instance._active, 0)
                self.assertTrue(instance._finalizer.alive)
            self.assertEqual(instance.score([2], [3], [1], 1, 1, 1), (6,))
        finally:
            instance.close()
        instance.close()
        self.assertFalse(instance._finalizer.alive)


if __name__ == "__main__":
    unittest.main()
