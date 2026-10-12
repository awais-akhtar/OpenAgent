"""Original independent equation fixtures, not publisher checkpoint parity."""

import ctypes as C
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mhc_fixture", ROOT / "openagent/adaptive_engine/native_deepseek_v41_mhc.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def f32(v):
    return C.c_float(v).value


def oracle(x, previous, rows, n, w, hw, hb, fw, fb, eps=1e-6, sweeps=8):
    q, h = n * (n + 2), n * w
    coeff, mixed, result = [], [], []
    hw, hb, fw, fb = [list(map(f32, a)) for a in (hw, hb, fw, fb)]
    for r in range(rows):
        xr = x[r * h : (r + 1) * h]
        scale = math.sqrt(eps + sum(v * v / h for v in xr))
        z = [
            hb[k] + sum(hw[k * h + j] * (xr[j] / scale) for j in range(h))
            for k in range(q)
        ]
        def sigmoid(v):
            return 1 / (1 + math.exp(-v))
        a = [f32(sigmoid(v)) for v in z[:n]]
        c = [f32(2 * sigmoid(v)) for v in z[n + n * n :]]
        maximum = max(z[n : n + n * n])
        b = [
            [math.exp(z[n + i * n + j] - maximum) + 1e-12 for j in range(n)]
            for i in range(n)
        ]
        for _ in range(sweeps):
            b = [[v / sum(row) for v in row] for row in b]
            sums = [sum(b[i][j] for i in range(n)) for j in range(n)]
            b = [[b[i][j] / sums[j] for j in range(n)] for i in range(n)]
        b = [[f32(v) for v in row] for row in b]
        inp = [
            f32(sum(previous[r * n + j] * xr[j * w + d] for j in range(n)))
            for d in range(w)
        ]
        f = [
            f32(fb[i] + sum(fw[i * w + j] * inp[j] for j in range(w))) for i in range(w)
        ]
        coeff.extend(a + [v for row in b for v in row] + c)
        mixed.extend(inp)
        result.extend(
            f32(c[i] * f[d] + sum(b[i][j] * xr[j * w + d] for j in range(n)))
            for i in range(n)
            for d in range(w)
        )
    return (
        result,
        [coeff[r * q + j] for r in range(rows) for j in range(n)],
        coeff,
        mixed,
    )


class MHC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        source = ROOT / "openagent/adaptive_engine/native_deepseek_v41_mhc.c"
        cls.output = Path(cls.directory.name) / (
            "mhc.dll" if os.name == "nt" else "mhc.so"
        )
        if os.name == "nt":
            vswhere = (
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
                / "Microsoft Visual Studio/Installer/vswhere.exe"
            )
            location = subprocess.check_output(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                text=True,
            ).strip()
            env = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
            args = [
                "cl",
                "/nologo",
                "/W4",
                "/WX",
                "/O2",
                "/fp:strict",
                "/LD",
                str(source),
                "/link",
                "/OUT:" + str(cls.output),
            ]
            command = (
                'cmd.exe /d /s /c ""'
                + str(env)
                + '" && '
                + subprocess.list2cmdline(args)
                + '"'
            )
        else:
            command = [
                shutil.which("cc") or "cc",
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-O2",
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(cls.output),
                "-lm",
            ]
        result = subprocess.run(
            command, cwd=cls.directory.name, capture_output=True, text=True, timeout=120
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def component(self, rows=1, n=2, w=2, capacity=64):
        x = [f32((i % 11 - 5) * 0.2) for i in range(rows * n * w)]
        initial = [f32((i % n + 1) / (n + 1)) for i in range(rows * n)]
        obj = m.NativeDeepSeekV41MHC(
            self.output, rows, n, w, x, initial, capacity=capacity
        )
        self.addCleanup(obj.close)
        return obj

    def weights(self, blocks, n, w):
        q = n * (n + 2)
        return [
            [f32((i % 13 - 6) * 0.03) for i in range(blocks * q * n * w)],
            [f32((i % 7 - 3) * 0.1) for i in range(blocks * q)],
            [f32((i % 9 - 4) * 0.05) for i in range(blocks * w * w)],
            [f32((i % 5 - 2) * 0.02) for i in range(blocks * w)],
        ]

    def assertFloats(self, a, b):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, delta=2e-6 * max(1, abs(y)))

    def test_previous_not_current_input_and_current_post_residual(self):
        obj = m.NativeDeepSeekV41MHC(self.output, 1, 2, 1, [2, 10], [1, 0])
        self.addCleanup(obj.close)
        self.assertEqual(obj.mix(), (2,))
        # Current A deliberately differs, B swaps, C differs by output stream.
        obj.commit([0, 1, 0, 1, 1, 0, 1, 2], [3], expected_depth=0)
        self.assertEqual(obj.residual, (13, 8))
        self.assertEqual(obj.mix(), (8,))
        self.assertEqual(obj.previous_mix, (0, 1))
        with self.assertRaises(ValueError):
            obj.commit([0] * 8, [0], expected_depth=0)

    def test_independent_dynamic_predict_and_affine_depth_equations(self):
        obj = self.component(rows=3, n=3, w=4)
        weights = self.weights(5, 3, 4)
        x, previous = list(obj.residual), list(obj.previous_mix)
        strides = [15 * 12, 15, 16, 4]
        for layer in range(5):
            tensors = [a[layer * k : (layer + 1) * k] for a, k in zip(weights, strides)]
            xnew, anew, coefficients, mixed = oracle(x, previous, 3, 3, 4, *tensors)
            self.assertFloats(obj.mix(), mixed)
            self.assertFloats(obj.predict(*tensors[:2]), coefficients)
            obj.run(1, *tensors)
            self.assertFloats(obj.residual, xnew)
            self.assertFloats(obj.previous_mix, anew)
            x, previous = xnew, anew
        self.assertEqual(obj.depth, 5)

    def test_full_chunk_manual_composition_and_replay(self):
        full, chunk, manual = [self.component(rows=2, n=2, w=3) for _ in range(3)]
        weights = self.weights(7, 2, 3)
        full.run(7, *weights)
        strides = [8 * 6, 8, 9, 3]
        chunk.run(3, *[a[: 3 * k] for a, k in zip(weights, strides)])
        snap = chunk.snapshot()
        tail = [a[3 * k :] for a, k in zip(weights, strides)]
        chunk.run(4, *tail)
        self.assertEqual(full.snapshot(), chunk.snapshot())
        chunk.restore(snap)
        chunk.run(4, *tail)
        self.assertEqual(full.snapshot(), chunk.snapshot())
        for layer in range(7):
            hw, hb, fw, fb = [
                a[layer * k : (layer + 1) * k] for a, k in zip(weights, strides)
            ]
            coeff, inp = manual.predict(hw, hb), manual.mix()
            f = [
                f32(fb[i] + sum(fw[i * 3 + j] * inp[r * 3 + j] for j in range(3)))
                for r in range(2)
                for i in range(3)
            ]
            manual.commit(coeff, f, expected_depth=layer)
        self.assertEqual(full.snapshot(), manual.snapshot())

    def test_reset_and_restore_reject_corruption_atomically(self):
        obj = self.component()
        initial = obj.snapshot()
        obj.run(1, *self.weights(1, 2, 2))
        before = obj.snapshot()
        for field, value in [("magic", 0), ("depth", 65), ("rows", 2)]:
            bad = m._State.from_buffer_copy(before)
            setattr(bad, field, value)
            with self.assertRaises(ValueError):
                obj.restore(bytes(bad))
            self.assertEqual(obj.snapshot(), before)
        bad = m._State.from_buffer_copy(before)
        bad.previous[0] = float("nan")
        with self.assertRaises(ValueError):
            obj.restore(bytes(bad))
        bad = m._State.from_buffer_copy(before)
        bad.x[511] = 1
        with self.assertRaises(ValueError):
            obj.restore(bytes(bad))
        for bad in (before[:-1], bytearray(before), before + b"x"):
            with self.assertRaises(ValueError):
                obj.restore(bad)
        obj.reset()
        self.assertEqual(obj.snapshot(), initial)

    def test_late_overflow_and_nonfinite_chunk_rollback(self):
        obj = self.component(n=1, w=1)
        before = obj.snapshot()
        # First block succeeds; second affine output exceeds float32.
        with self.assertRaises(ValueError):
            obj.run(2, [0] * 6, [0] * 6, [0, 3.4e38], [3.4e38, 3.4e38])
        self.assertEqual(obj.snapshot(), before)
        weights = self.weights(2, 1, 1)
        weights[0][-1] = float("nan")
        with self.assertRaises(ValueError):
            obj.run(2, *weights)
        self.assertEqual(obj.snapshot(), before)

    def test_budget_capacity_and_exact_extents(self):
        obj = self.component(capacity=2)
        weights = self.weights(2, 2, 2)
        budget = obj.library.oa_v41_mhc_scratch_bytes()
        before = obj.snapshot()
        with self.assertRaises(ValueError):
            obj.run(2, *weights, budget=budget - 1)
        self.assertEqual(obj.snapshot(), before)
        for i in range(4):
            bad = list(weights)
            bad[i] = bad[i][:-1]
            with self.assertRaises(ValueError):
                obj.run(2, *bad)
        obj.run(2, *weights, budget=budget)
        before = obj.snapshot()
        with self.assertRaises(ValueError):
            obj.run(1, *self.weights(1, 2, 2))
        self.assertEqual(obj.snapshot(), before)

    def test_maximum_geometry_depth_and_independent_rows(self):
        obj = self.component(rows=8, n=4, w=16)
        initial = obj.snapshot()
        weights = self.weights(16, 4, 16)
        for _ in range(4):
            obj.run(16, *weights)
        self.assertEqual(obj.depth, 64)
        before = obj.snapshot()
        with self.assertRaises(ValueError):
            obj.run(1, *self.weights(1, 4, 16))
        self.assertEqual(obj.snapshot(), before)
        obj.restore(initial)
        hw, hb, fw, fb = self.weights(1, 4, 16)
        expected, previous, _, _ = oracle(
            list(obj.residual), list(obj.previous_mix), 8, 4, 16, hw, hb, fw, fb
        )
        obj.run(1, hw, hb, fw, fb)
        self.assertFloats(obj.residual, expected)
        self.assertFloats(obj.previous_mix, previous)

    def test_final_head_is_explicit_readonly_and_not_pending_mix(self):
        obj = self.component(rows=2, n=2, w=2)
        before = obj.snapshot()
        a, norm, weight, bias = [1, 0, 0, 1], [2, -1], [1, 3, -2, 4], [0.5, -0.25]
        x = obj.residual
        expected = []
        for r in range(2):
            mixed = [
                sum(a[r * 2 + j] * x[(r * 2 + j) * 2 + d] for j in range(2))
                for d in range(2)
            ]
            rms = math.sqrt(1e-6 + sum(v * v / 2 for v in mixed))
            expected.extend(
                bias[i]
                + sum(weight[i * 2 + d] * mixed[d] / rms * norm[d] for d in range(2))
                for i in range(2)
            )
        self.assertFloats(obj.head(a, norm, weight, bias, 2), expected)
        self.assertEqual(obj.snapshot(), before)
        with self.assertRaises(ValueError):
            obj.head(a, norm, weight, bias, 2, capacity=3)
        self.assertEqual(obj.snapshot(), before)

    def test_native_raw_alignment_overlap_extents_and_output_sentinels(self):
        obj = self.component()
        lib, state, size = obj.library, C.byref(obj._state), C.sizeof(m._State)
        out = (C.c_float * 2)(-77, -77)
        before = obj.snapshot()
        for ptr, extent in [
            (None, size),
            (C.c_void_p(C.addressof(obj._state) + 1), size),
            (state, size - 1),
            (C.c_void_p(C.c_size_t(-1).value - 3), size),
        ]:
            self.assertEqual(lib.oa_v41_mhc_mix(ptr, extent, out, 2), 1)
            self.assertEqual(tuple(out), (-77, -77))
        for ptr, count in [
            (None, 2),
            (out, 1),
            (out, C.c_size_t(-1).value),
            (C.c_void_p(C.addressof(out) + 1), 2),
            (state, 2),
        ]:
            self.assertEqual(lib.oa_v41_mhc_mix(state, size, ptr, count), 1)
        coeff, f = (C.c_float * 8)(0, 1, 0, 1, 1, 0, 1, 2), (C.c_float * 2)(1, 2)
        for ca, nc, fa, nf in [
            (coeff, 7, f, 2),
            (state, 8, f, 2),
            (coeff, 8, state, 2),
            (coeff, 8, None, 2),
            (coeff, 8, f, C.c_size_t(-1).value),
        ]:
            self.assertEqual(lib.oa_v41_mhc_commit(state, size, 0, ca, nc, fa, nf), 1)
            self.assertEqual(obj.snapshot(), before)

    def test_raw_predict_and_head_overlap_and_capacity(self):
        obj = self.component()
        lib, state, size = obj.library, C.byref(obj._state), C.sizeof(m._State)
        hw, hb = (C.c_float * 32)(), (C.c_float * 8)()
        out = (C.c_float * 8)(*([-77] * 8))
        for output in (hw, hb, state, None):
            self.assertEqual(
                lib.oa_v41_mhc_predict(state, size, hw, 32, hb, 8, 1e-6, 8, output, 8),
                1,
            )
        self.assertEqual(
            lib.oa_v41_mhc_predict(state, size, hw, 31, hb, 8, 1e-6, 8, out, 8), 1
        )
        hb[7] = float("inf")
        self.assertEqual(
            lib.oa_v41_mhc_predict(state, size, hw, 32, hb, 8, 1e-6, 8, out, 8), 3
        )
        self.assertEqual(tuple(out), (-77,) * 8)
        a, norm, weight, bias = (
            (C.c_float * 2)(1, 0),
            (C.c_float * 2)(1, 1),
            (C.c_float * 4)(1, 0, 0, 1),
            (C.c_float * 2)(),
        )
        for output in (a, norm, weight, bias, state):
            self.assertEqual(
                lib.oa_v41_mhc_head(
                    state, size, a, 2, norm, 2, weight, 4, bias, 2, 2, 1e-6, output, 2
                ),
                1,
            )
        self.assertEqual(
            lib.oa_v41_mhc_head(
                state, size, a, 2, norm, 2, weight, 4, bias, 2, 2, 1e-6, out, 1
            ),
            4,
        )
        self.assertEqual(tuple(out), (-77,) * 8)

    def test_invalid_coefficients_and_overflow_commit_are_atomic(self):
        obj = self.component()
        before = obj.snapshot()
        for index, value in [(0, -1), (0, 2), (2, -0.1), (6, 3), (7, float("nan"))]:
            coeff = [0.5, 0.5, 1, 0, 0, 1, 1, 1]
            coeff[index] = value
            with self.assertRaises(ValueError):
                obj.commit(coeff, [1, 2], expected_depth=0)
            self.assertEqual(obj.snapshot(), before)
        with self.assertRaises(ValueError):
            obj.commit([0.5, 0.5, 1, 0, 0, 1, 2, 2], [3.4e38] * 2, expected_depth=0)
        self.assertEqual(obj.snapshot(), before)

    def test_wrapper_bounds_close_and_bounded_iterables(self):
        obj = self.component()
        for bad in (True, 0, 17, -1):
            with self.assertRaises(ValueError):
                obj.run(bad, [], [], [], [])

        def infinite():
            while True:
                yield 0

        with self.assertRaises(ValueError):
            obj.predict(infinite(), [0] * 8)
        for epsilon in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                obj.predict([0] * 32, [0] * 8, epsilon=epsilon)
        obj.close()
        obj.close()
        for operation in (obj.snapshot, obj.reset, obj.mix):
            with self.assertRaises(RuntimeError):
                operation()

    def test_raw_init_and_run_failures_leave_state_unchanged(self):
        obj = self.component()
        lib, state, size = obj.library, C.byref(obj._state), C.sizeof(m._State)
        x, initial = (C.c_float * 4)(1, 2, 3, 4), (C.c_float * 2)(0.5, 0.5)
        before = obj.snapshot()
        for changes in (
            {"nx": 3},
            {"x": state},
            {"initial": None},
            {"rows": 9},
            {"width": 0},
            {"capacity": 65},
            {"bytes": size - 1},
        ):
            args = dict(
                state=state,
                bytes=size,
                rows=1,
                streams=2,
                width=2,
                capacity=64,
                x=x,
                nx=4,
                initial=initial,
                na=2,
            )
            args.update(changes)
            self.assertEqual(lib.oa_v41_mhc_init(*args.values()), 1)
            self.assertEqual(obj.snapshot(), before)
        hw, hb, fw, fb = [(C.c_float * len(a))(*a) for a in self.weights(1, 2, 2)]
        for changes, expected in [
            ({"nhw": 31}, 1),
            ({"hw": state}, 1),
            ({"hb": None}, 1),
            ({"expected_depth": 1}, 1),
            ({"sweeps": 0}, 1),
            ({"budget": 0}, 2),
            ({"blocks": 0}, 1),
            ({"nfw": C.c_size_t(-1).value}, 1),
        ]:
            args = dict(
                state=state,
                bytes=size,
                expected_depth=0,
                blocks=1,
                hw=hw,
                nhw=32,
                hb=hb,
                nhb=8,
                fw=fw,
                nfw=4,
                fb=fb,
                nfb=2,
                eps=1e-6,
                sweeps=8,
                budget=65536,
            )
            args.update(changes)
            self.assertEqual(lib.oa_v41_mhc_run(*args.values()), expected)
            self.assertEqual(obj.snapshot(), before)

    def test_reentrant_close_and_mutation_refused_during_ingestion(self):
        obj = self.component()
        before = obj.snapshot()

        def callback(action):
            action()
            yield 0

        for action in (obj.close, obj.reset, lambda: obj.restore(before)):
            with self.assertRaises(RuntimeError):
                obj.predict(callback(action), [0] * 8)
            self.assertEqual(obj.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
