"""Compiled primitive versus independent analytical equations, not decoder parity."""
import ctypes as C
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest


class KimiK3AttnResTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='kimi_k3_attnres_')
        source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/native_kimi_k3_attnres.c'
        output = Path(cls.temp.name) / ('kimi_k3.dll' if os.name == 'nt' else 'kimi_k3.so')
        if os.name == 'nt':
            vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
            location = subprocess.check_output([str(vswhere), '-latest', '-products', '*',
                '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                '-property', 'installationPath'], text=True).strip()
            if not location:
                raise RuntimeError('MSVC required; no skip counts as verification')
            batch = Path(location) / 'VC/Auxiliary/Build/vcvars64.bat'
            args = ['cl', '/nologo', '/W4', '/WX', '/O2', '/fp:precise', '/LD', str(source),
                    '/link', '/OUT:' + str(output)]
            command = 'cmd.exe /d /s /c ""' + str(batch) + '" && ' + subprocess.list2cmdline(args) + '"'
        else:
            cc = shutil.which('cc')
            if not cc:
                raise RuntimeError('C compiler required')
            command = [cc, '-std=c99', '-Wall', '-Wextra', '-Werror', '-O2', '-shared',
                       '-fPIC', str(source), '-o', str(output), '-lm']
        subprocess.run(command, cwd=cls.temp.name, check=True, capture_output=True, timeout=120)
        cls.library = C.CDLL(str(output))
        cls.function = cls.library.oa_kimi_k3_attnres
        cls.function.argtypes = [C.POINTER(C.c_float)] * 3 + [C.c_int, C.c_int, C.c_double, C.POINTER(C.c_float)]
        cls.function.restype = C.c_int

    @classmethod
    def tearDownClass(cls):
        if os.name == 'nt':
            free = C.windll.kernel32.FreeLibrary
            free.argtypes = [C.c_void_p]
            free(cls.library._handle)
        cls.temp.cleanup()

    def run_native(self, rows, query, gain=None, epsilon=1e-5, count=None, width=None):
        def array(values):
            return (C.c_float * len(values))(*values)

        sources = array([x for row in rows for x in row])
        q = array(query)
        g = array(gain) if gain is not None else None
        out = array([123.0] * len(query))
        code = self.function(sources, q, g, len(rows) if count is None else count,
                             len(query) if width is None else width, epsilon, out)
        return code, list(out)

    def test_uniform_and_single_source(self):
        self.assertEqual(self.run_native([[2, 4], [6, 8]], [0, 0]), (0, [4, 6]))
        self.assertEqual(self.run_native([[2, -7]], [99, -31]), (0, [2, -7]))

    def test_closed_form_two_sources(self):
        code, out = self.run_native([[1], [-1]], [2], epsilon=3)
        self.assertEqual(code, 0)
        self.assertAlmostEqual(out[0], math.tanh(1), places=6)

    def test_random_analytical_reference_and_permutation(self):
        rng = random.Random(713)
        for count in (1, 2, 9, 128):
            rows = [[rng.uniform(-4, 4) for _ in range(7)] for _ in range(count)]
            q = [rng.uniform(-2, 2) for _ in range(7)]
            gain = [rng.uniform(-1, 2) for _ in range(7)]
            def f32(x):
                return C.c_float(x).value

            rows = [[f32(x) for x in row] for row in rows]
            q, gain = [f32(x) for x in q], [f32(x) for x in gain]
            scores = [sum(a*b*c for a,b,c in zip(q, gain, row)) /
                      math.sqrt(sum(x*x for x in row)/7 + 1e-5) for row in rows]
            weights = [math.exp(x-max(scores)) for x in scores]
            expected = [sum(w*row[j] for w,row in zip(weights, rows))/sum(weights) for j in range(7)]
            code, out = self.run_native(rows, q, gain)
            self.assertEqual(code, 0)
            for actual, wanted in zip(out, expected):
                self.assertAlmostEqual(actual, wanted, delta=5e-7)
            self.assertEqual(self.run_native(rows[::-1], q, gain)[1], out)

    def test_extreme_scores_zero_source_and_width_bound(self):
        self.assertEqual(self.run_native([[1], [-1], [0]], [1e30]), (0, [1]))
        self.assertEqual(self.run_native([[0]*4096], [1]*4096), (0, [0]*4096))

    def test_block_partial_sum_is_a_separate_source(self):
        # Embedding, completed block, and current block's partial sum (Eq. 10).
        rows = [[1, 2], [4, 6], [3+7, 5+9]]
        self.assertEqual(self.run_native(rows, [0, 0]), (0, [5, C.c_float(22/3).value]))
        self.assertEqual(self.run_native(rows[:2], [0, 0]), (0, [2.5, 4]))

    def test_rejection_preserves_output(self):
        for kwargs in ({'epsilon': 0}, {'epsilon': float('nan')}, {'count': 0},
                       {'count': 129}, {'width': 4097}, {'width': 0}):
            code, out = self.run_native([[1, 2]], [1, 2], **kwargs)
            self.assertNotEqual(code, 0)
            self.assertEqual(out, [123, 123])
        for rows, q, gain in (([[float('nan'), 1]], [1, 2], None),
                              ([[1, 2]], [float('inf'), 1], None),
                              ([[1, 2]], [1, 2], [1, float('nan')])):
            code, out = self.run_native(rows, q, gain)
            self.assertEqual(code, 2)
            self.assertEqual(out, [123, 123])


if __name__ == '__main__':
    unittest.main()
