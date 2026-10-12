"""Compiled K3 equation fixtures, cache continuity and transactional failures."""

import ctypes as C
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.native_kimi_k3_kda import (
    KimiK3KDAConfig,
    NativeKimiK3KDA,
    kda_tensor_shapes,
)


def build(directory):
    root = Path(__file__).resolve().parents[1] / "openagent/adaptive_engine"
    sources = [root / "native_kimi.c", root / "native_kimi_k3_kda.c"]
    output = Path(directory) / ("k3_kda.dll" if os.name == "nt" else "k3_kda.so")
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
        if not location:
            raise RuntimeError("MSVC required")
        batch = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
        args = [
            "cl",
            "/nologo",
            "/W4",
            "/WX",
            "/O2",
            "/fp:strict",
            "/TC",
            "/LD",
            *map(str, sources),
            "/link",
            "/OUT:" + str(output),
        ]
        command = (
            'cmd.exe /d /s /c ""'
            + str(batch)
            + '" && '
            + subprocess.list2cmdline(args)
            + '"'
        )
    else:
        cc = shutil.which("cc")
        if not cc:
            raise RuntimeError("C compiler required")
        command = [
            cc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-shared",
            "-fPIC",
            *map(str, sources),
            "-o",
            str(output),
            "-lm",
        ]
    result = subprocess.run(
        command, cwd=directory, capture_output=True, text=True, timeout=120
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return output


def fixture(c):
    rng = np.random.default_rng(813)
    result = {
        n: rng.normal(0, 0.3, s).astype(np.float32)
        for n, s in kda_tensor_shapes(c).items()
    }
    result["out_norm.weight"] = np.linspace(0.6, 1.2, c.value, dtype=np.float32)
    x = rng.normal(0, 0.6, (7, c.hidden)).astype(np.float32)
    return result, x


def analytic(c, w, x):
    """Direct float64 equation evaluation, separate from the shared runtime kernel."""
    a = {n: v.astype(np.float64) for n, v in w.items()}
    x = x.astype(np.float64)

    def sigmoid(z):
        return np.exp(-np.logaddexp(0, -z))

    q, k, v = [x @ a[n + "_proj.weight"].T for n in ("q", "k", "v")]
    raw = x @ a["decay_down.weight"].T @ a["decay_up.weight"].T + a["decay_bias"]
    decay = -5 * sigmoid(
        raw.reshape(-1, c.heads, c.key) * np.exp(a["A_log"])[None, :, None]
    )
    beta = sigmoid(x @ a["beta_proj.weight"].T)
    gate = sigmoid(x @ a["gate_proj.weight"].T).reshape(-1, c.heads, c.value)
    # Native matrix projections publish FP32 streams before the equation kernel.
    streams = [
        z.astype(np.float32).astype(np.float64).reshape(len(x), c.heads, d)
        for z, d in zip((q, k, v), (c.key, c.key, c.value))
    ]
    state = np.zeros((c.heads, c.key, c.value))
    histories = [np.zeros((c.kernel - 1, c.heads, d)) for d in (c.key, c.key, c.value)]
    output = []
    for t in range(len(x)):
        filtered = []
        for i, name in enumerate(("q", "k", "v")):
            window = np.concatenate([histories[i], streams[i][t : t + 1]], axis=0)
            z = np.einsum("thd,hdt->hd", window, a[name + "_conv.weight"])
            filtered.append(z * sigmoid(z))
            histories[i] = window[1:].copy()
        qf, kf, vf = filtered
        qf = (
            qf
            / np.sqrt(np.sum(qf * qf, axis=-1, keepdims=True) + c.l2_epsilon)
            * c.query_scale
        )
        kf = kf / np.sqrt(np.sum(kf * kf, axis=-1, keepdims=True) + c.l2_epsilon)
        state = (
            (state * np.exp(decay[t].astype(np.float32).astype(np.float64))[..., None])
            .astype(np.float32)
            .astype(np.float64)
        )
        error = beta[t].astype(np.float32).astype(np.float64)[:, None] * (
            vf - np.einsum("hk,hkv->hv", kf, state)
        )
        state = (
            (state + kf[..., None] * error[:, None, :])
            .astype(np.float32)
            .astype(np.float64)
        )
        value = np.einsum("hk,hkv->hv", qf, state)
        value = value / np.sqrt(
            np.mean(value * value, axis=-1, keepdims=True) + c.epsilon
        )
        value *= a["out_norm.weight"] * gate[t].astype(np.float32).astype(np.float64)
        output.append(
            value.astype(np.float32).astype(np.float64).reshape(-1)
            @ a["out_proj.weight"].T
        )
    return (
        np.array(output, np.float32),
        state.astype(np.float32),
        tuple(z.astype(np.float32) for z in histories),
    )


class KimiK3KDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="native_k3_kda_")
        cls.path = build(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        cls.temp.cleanup()

    def setUp(self):
        self.c = KimiK3KDAConfig()
        self.w, self.x = fixture(self.c)
        self.model = NativeKimiK3KDA(self.c, self.w, library=self.path)

    def tearDown(self):
        self.model = None
        gc.collect()

    def same_state(self, a, b):
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1][0], b[1][0])
        for x, y in zip((a[1][1], *a[1][2]), (b[1][1], *b[1][2])):
            np.testing.assert_array_equal(x, y)

    def test_equations_outputs_and_convolution_recurrent_state(self):
        expected, state, history = analytic(self.c, self.w, self.x)
        actual = self.model.forward(self.x)
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
        saved = self.model.snapshot()[1]
        np.testing.assert_allclose(saved[1], state, atol=1e-6, rtol=3e-5)
        for x, y in zip(saved[2], history):
            np.testing.assert_allclose(x, y, atol=2e-7, rtol=2e-6)

    def test_chunk_continuity_restore_and_detached_snapshots(self):
        full = self.model.forward(self.x)
        state = self.model.snapshot()
        for lengths in ([1] * 7, [2, 1, 4], [4, 3]):
            self.model.reset()
            offset = 0
            rows = []
            for n in lengths:
                rows.append(self.model.forward(self.x[offset : offset + n]))
                offset += n
            np.testing.assert_array_equal(np.concatenate(rows), full)
            self.same_state(state, self.model.snapshot())
        self.model.reset()
        self.model.forward(self.x[:2])
        saved = self.model.snapshot()
        tail = self.model.forward(self.x[2:])
        self.model.restore(saved)
        saved[1][1].fill(11)
        np.testing.assert_array_equal(self.model.forward(self.x[2:]), tail)

    def test_failure_after_recurrence_does_not_commit(self):
        self.model.forward(self.x[:2])
        before = self.model.snapshot()
        with patch.object(self.model, "_output", return_value=3):
            with self.assertRaisesRegex(ValueError, "state unchanged"):
                self.model.forward(self.x[2:])
        self.same_state(before, self.model.snapshot())
        with patch.object(self.model, "_project", return_value=3):
            with self.assertRaisesRegex(ValueError, "state unchanged"):
                self.model.forward(self.x[2:])
        self.same_state(before, self.model.snapshot())

    def test_admission_capacity_budget_and_snapshot_rejection(self):
        before = self.model.snapshot()
        for x in (
            self.x.astype(np.float64),
            self.x[:0],
            self.x[:, ::-1],
            np.full_like(self.x, np.nan),
        ):
            with self.assertRaises(ValueError):
                self.model.forward(x)
            self.same_state(before, self.model.snapshot())
        self.model.max_bytes = 1
        with self.assertRaisesRegex(ValueError, "memory budget"):
            self.model.forward(self.x)
        self.same_state(before, self.model.snapshot())
        self.model.max_bytes = 64 * 1024**2
        with self.assertRaises(ValueError):
            self.model.restore(("foreign", before[1]))
        with self.assertRaises(ValueError):
            self.model.restore((before[0], (129, *before[1][1:])))
        c = replace(self.c, capacity=1)
        small = NativeKimiK3KDA(c, self.w, library=self.path)
        small.forward(self.x[:1])
        saved = small.snapshot()
        with self.assertRaisesRegex(ValueError, "capacity"):
            small.forward(self.x[:1])
        self.same_state(saved, small.snapshot())

    def test_extreme_decay_limits_and_full_gate(self):
        c = replace(self.c, query_scale=0.5, kernel=1)
        w, x = fixture(c)
        w["decay_bias"][: c.key] = -1e4
        w["decay_bias"][c.key :] = 1e4
        model = NativeKimiK3KDA(c, w, library=self.path)
        expected, _, _ = analytic(c, w, x)
        np.testing.assert_allclose(model.forward(x), expected, atol=3e-6, rtol=3e-5)
        w["gate_proj.weight"] *= 3
        alternate = NativeKimiK3KDA(c, w, library=self.path)
        self.assertGreater(np.max(np.abs(alternate.forward(x) - expected)), 1e-4)

    def test_c_projection_invalid_counts_and_overflow_preserve_all_outputs(self):
        m = self.model
        c = self.c
        n = 2
        shapes = [(n, c.heads, c.key)] * 2 + [
            (n, c.heads, c.value),
            (n, c.heads, c.key),
            (n, c.heads),
            (n, c.heads, c.value),
        ]
        arrays = tuple(np.full(s, 117, np.float32) for s in shapes)

        def ptr(a):
            return a.ctypes.data_as(C.POINTER(C.c_float))

        counts = (C.c_size_t * 6)(*(a.size for a in arrays))
        bad = m._project_weights.copy()
        # A finite weight with an unusable exponential log-scale is rejected.
        offset = sum(np.prod(s) for s in list(kda_tensor_shapes(c).values())[:6])
        bad[int(offset)] = 1000
        for w, nw in (
            (m._project_weights, m._project_weights.size - 1),
            (bad, bad.size),
        ):
            status = m._project(
                C.byref(m._abi),
                n,
                ptr(w),
                nw,
                ptr(self.x),
                n * c.hidden,
                *(ptr(a) for a in arrays),
                counts,
            )
            self.assertNotEqual(status, 0)
            for a in arrays:
                np.testing.assert_array_equal(a, 117)

    def test_runtime_does_not_use_numpy_matrix_math(self):
        expected = self.model.forward(self.x)
        self.model.reset()
        with (
            patch("numpy.einsum", side_effect=AssertionError("NumPy math")),
            patch("numpy.matmul", side_effect=AssertionError("NumPy math")),
        ):
            actual = self.model.forward(self.x)
        np.testing.assert_array_equal(actual, expected)

    def test_native_projection_overlap_rejected_and_config_is_fixed(self):
        m, c = self.model, self.c
        fp = C.POINTER(C.c_float)
        arrays = [np.full((1, c.heads, d), 117, np.float32)
                  for d in (c.key, c.key, c.value, c.key, 1, c.value)]
        arrays[1] = arrays[0]
        counts = (C.c_size_t * 6)(*(a.size for a in arrays))
        status = m._project(
            C.byref(m._abi), 1, m._project_weights.ctypes.data_as(fp),
            m._project_weights.size, self.x.ctypes.data_as(fp), c.hidden,
            *(a.ctypes.data_as(fp) for a in arrays), counts,
        )
        self.assertEqual(status, 1)
        for a in arrays:
            np.testing.assert_array_equal(a, 117)
        with self.assertRaises(AttributeError):
            m.config = replace(c, hidden=1)

    def test_restore_bad_cache_and_reset_budget_are_atomic(self):
        self.model.forward(self.x[:2])
        before = self.model.snapshot()
        for value in (np.nan, np.inf):
            state = before[1][1].copy()
            state.flat[0] = value
            with self.assertRaises(ValueError):
                self.model.restore((before[0], (2, state, before[1][2])))
            self.same_state(before, self.model.snapshot())
        self.model.max_bytes = 1
        with self.assertRaises(ValueError):
            self.model.reset()
        self.same_state(before, self.model.snapshot())

    def test_large_finite_logits_and_latent_do_not_require_fp32_range(self):
        w = {n: a.copy() for n, a in self.w.items()}
        for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
            w[name].fill(0)
        for name in ("beta_proj.weight", "gate_proj.weight", "decay_down.weight"):
            w[name].fill(np.finfo(np.float32).max)
        w["decay_up.weight"].fill(0)
        model = NativeKimiK3KDA(self.c, w, library=self.path)
        x = np.full((1, self.c.hidden), 2, np.float32)
        np.testing.assert_array_equal(model.forward(x), 0)
        self.assertEqual(model.position, 1)

    def test_weights_are_owned_and_immutable(self):
        expected = self.model.forward(self.x)
        self.model.reset()
        for a in self.w.values():
            a.fill(100)
        np.testing.assert_array_equal(self.model.forward(self.x), expected)
        for a in (
            self.model._project_weights,
            self.model._output_weights,
            *self.model._mixer._taps,
            self.model._mixer._norm,
        ):
            with self.assertRaises(ValueError):
                a.setflags(write=True)

    def test_concurrent_identical_calls_are_serialized(self):
        row = self.x[:1]
        expected = [self.model.forward(row) for _ in range(12)]
        saved = self.model.snapshot()
        self.model.reset()
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda _: self.model.forward(row), range(12)))
        # Completion order may differ from submission order.
        self.assertEqual(
            sorted(a.tobytes() for a in actual), sorted(a.tobytes() for a in expected)
        )
        self.same_state(saved, self.model.snapshot())

    def test_real_output_overflow_rolls_back_recurrence(self):
        w = {n: a.copy() for n, a in self.w.items()}
        w["out_proj.weight"].fill(np.finfo(np.float32).max)
        w["out_norm.weight"].fill(100)
        model = NativeKimiK3KDA(self.c, w, library=self.path)
        before = model.snapshot()
        with self.assertRaisesRegex(ValueError, "output failed"):
            model.forward(self.x)
        self.same_state(before, model.snapshot())

    def test_temporary_dll_unloads_after_last_function_reference(self):
        with tempfile.TemporaryDirectory(prefix="k3_lifetime_") as directory:
            path = Path(directory) / self.path.name
            shutil.copy2(self.path, path)
            model = NativeKimiK3KDA(self.c, self.w, library=path)
            function = model._project
            del model
            gc.collect()
            self.assertTrue(path.exists())
            del function
            gc.collect()
            path.unlink()

    def test_invalid_mutated_budget_rejected_before_allocation(self):
        before = self.model.snapshot()
        for budget in (True, float("nan"), -1):
            self.model.max_bytes = budget
            with self.assertRaisesRegex(ValueError, "memory budget"):
                self.model.forward(self.x)
            self.same_state(before, self.model.snapshot())

    def test_output_abi_bounds_counts_null_and_nonfinite_are_atomic(self):
        m = self.model
        c = self.c

        def ptr(a):
            return a.ctypes.data_as(C.POINTER(C.c_float))

        x = np.ones((2, c.heads * c.value), np.float32)
        out = np.full((2, c.hidden), 117, np.float32)
        for field, limit in (
            ("hidden", 64),
            ("heads", 8),
            ("key", 32),
            ("value", 32),
            ("rank", 64),
        ):
            original = getattr(m._abi, field)
            for invalid in (0, limit + 1):
                setattr(m._abi, field, invalid)
                self.assertNotEqual(
                    m._output(
                        C.byref(m._abi),
                        2,
                        ptr(m._output_weights),
                        m._output_weights.size,
                        ptr(x),
                        x.size,
                        ptr(out),
                        out.size,
                    ),
                    0,
                )
                np.testing.assert_array_equal(out, 117)
            setattr(m._abi, field, original)
        for n, nw, nx, no in (
            (0, m._output_weights.size, x.size, out.size),
            (129, m._output_weights.size, x.size, out.size),
            (2, m._output_weights.size - 1, x.size, out.size),
            (2, m._output_weights.size, x.size - 1, out.size),
            (2, m._output_weights.size, x.size, out.size - 1),
        ):
            self.assertNotEqual(
                m._output(
                    C.byref(m._abi),
                    n,
                    ptr(m._output_weights),
                    nw,
                    ptr(x),
                    nx,
                    ptr(out),
                    no,
                ),
                0,
            )
            np.testing.assert_array_equal(out, 117)
        for source in (None, ptr(x)):
            x.fill(np.nan)
            self.assertNotEqual(
                m._output(
                    C.byref(m._abi),
                    2,
                    ptr(m._output_weights),
                    m._output_weights.size,
                    source,
                    x.size,
                    ptr(out),
                    out.size,
                ),
                0,
            )
            np.testing.assert_array_equal(out, 117)
