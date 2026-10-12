"""Native hybrid logits and state against separately evaluated equations."""

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

from openagent.adaptive_engine.native_kimi_k3_hybrid import (
    KimiK3HybridConfig,
    NativeKimiK3Hybrid,
    hybrid_tensor_shapes,
)
from openagent.adaptive_engine.native_kimi_k3_decoder import KimiK3DenseMLAConfig
from openagent.adaptive_engine.native_kimi_k3_kda import (
    KimiK3KDAConfig,
    NativeKimiK3KDA,
    kda_tensor_shapes,
)
from openagent.adaptive_engine.native_kimi_k3_moe import (
    KimiK3MoEConfig,
    tensor_shapes as moe_shapes,
)
from tests import test_native_kimi_k3_kda as kda_reference
from tests import test_native_kimi_k3_moe as moe_reference


def build(directory):
    root = Path(__file__).resolve().parents[1] / "openagent/adaptive_engine"
    sources = [
        root / n
        for n in (
            "native_kimi.c",
            "native_kimi_k3_attnres.c",
            "native_kimi_k3_kda.c",
            "native_kimi_k3_moe.c",
            "native_kimi_k3_blocks.c",
        )
    ]
    output = Path(directory) / ("hybrid.dll" if os.name == "nt" else "hybrid.so")
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
            "/fp:precise",
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


def configuration(types=("kda", "kda", "kda", "mla")):
    return KimiK3HybridConfig(
        KimiK3DenseMLAConfig(layers=len(types)),
        KimiK3KDAConfig(capacity=16),
        KimiK3MoEConfig(),
        types,
    )


def fixture(c):
    rng = np.random.default_rng(812)
    w = {
        n: rng.normal(0, 0.2, s).astype(np.float32)
        for n, s in hybrid_tensor_shapes(c).items()
    }
    for name in w:
        if "norm.weight" in name:
            w[name] = np.linspace(0.75, 1.25, w[name].size, dtype=np.float32).reshape(
                w[name].shape
            )
        if name.endswith("router.bias"):
            w[name][:] = np.linspace(-0.3, 0.3, c.moe.experts)
    return w


def analytical(config, w, ids):
    c = config.mla

    def norm(x, gain, epsilon=c.epsilon):
        z = x.astype(np.float64)
        return (
            z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + epsilon) * gain
        ).astype(np.float32)

    def mat(x, w):
        return (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)

    def combine(rows, q, gain):
        data = np.stack(rows, axis=1).astype(np.float64)
        score = np.sum(
            data
            / np.sqrt(np.mean(data * data, axis=-1, keepdims=True) + c.epsilon)
            * gain
            * q,
            axis=-1,
        )
        p = np.exp(score - score.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        return np.sum(data * p[..., None], axis=1).astype(np.float32)

    prefix = w["model.embed_tokens.weight"][ids].copy()
    blocks = []
    caches = []
    for i, kind in enumerate(config.layer_types):
        root = f"model.layers.{i}."
        mixed = (
            prefix
            if not blocks
            else combine(
                blocks + [prefix],
                w[root + "self_attention_res_proj.weight"],
                w[root + "self_attention_res_norm.weight"],
            )
        )
        if i % c.block == 0:
            blocks.append(prefix.copy())
            prefix = np.zeros_like(prefix)
        x = norm(mixed, w[root + "input_layernorm.weight"])
        if kind == "kda":
            tensors = {
                n: w[root + "attention." + n] for n in kda_tensor_shapes(config.kda)
            }
            branch, state, history = kda_reference.analytic(config.kda, tensors, x)
            caches.append(("kda", state, history))
        else:

            def a(n):
                return w[root + "attention." + n]

            q = mat(
                norm(mat(x, a("q_a_proj.weight")), a("q_a_layernorm.weight"), 1e-6),
                a("q_b_proj.weight"),
            ).reshape(-1, c.heads, c.nope + c.rope)
            latent = mat(x, a("kv_a_proj_with_mqa.weight"))
            expanded = mat(
                norm(latent[:, : c.kvrank], a("kv_a_layernorm.weight"), 1e-6),
                a("kv_b_proj.weight"),
            ).reshape(-1, c.heads, c.nope + c.value)
            keys = np.concatenate(
                [
                    expanded[..., : c.nope],
                    np.broadcast_to(
                        latent[:, None, c.kvrank :], (len(ids), c.heads, c.rope)
                    ),
                ],
                axis=-1,
            )
            values = expanded[..., c.nope :]
            scores = np.einsum(
                "thd,shd->hts", q.astype(np.float64), keys.astype(np.float64)
            ) / np.sqrt(c.nope + c.rope)
            upper = np.triu_indices(len(ids), 1)
            scores[:, upper[0], upper[1]] = -np.inf
            p = np.exp(scores - scores.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            read = np.einsum("hts,shv->thv", p, values.astype(np.float64)).reshape(
                len(ids), -1
            )
            gate = np.exp(
                -np.logaddexp(0, -mat(x, a("g_proj.weight")).astype(np.float64))
            )
            branch = mat((read * gate).astype(np.float32), a("o_proj.weight"))
            caches.append(("mla", keys, values))
        prefix = (prefix + branch).astype(np.float32)
        x = norm(
            combine(
                blocks + [prefix],
                w[root + "mlp_res_proj.weight"],
                w[root + "mlp_res_norm.weight"],
            ),
            w[root + "post_attention_layernorm.weight"],
        )
        branch, _ = moe_reference.analytical(
            config.moe, {n: w[root + "moe." + n] for n in moe_shapes(config.moe)}, x
        )
        prefix = (prefix + branch.astype(np.float32)).astype(np.float32)
    x = norm(
        combine(
            blocks + [prefix],
            w["model.output_attn_res_proj.weight"],
            w["model.output_attn_res_norm.weight"],
        ),
        w["model.norm.weight"],
    )
    return mat(x, w["lm_head.weight"]), caches


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="native_k3_hybrid_")
        cls.path = build(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        cls.temp.cleanup()

    def setUp(self):
        self.c = configuration()
        self.w = fixture(self.c)
        self.ids = [1, 4, 2, 9, 3, 8, 5]
        self.model = NativeKimiK3Hybrid(self.c, self.w, library=self.path)

    def tearDown(self):
        self.model = None
        gc.collect()

    def same_state(self, a, b):
        self.assertEqual(a.identity, b.identity)
        self.assertEqual(a.length, b.length)
        for x, y in zip(a.keys + a.values, b.keys + b.values):
            if x is None:
                self.assertIsNone(y)
            else:
                np.testing.assert_array_equal(x, y)
        for x, y in zip(a.recurrent, b.recurrent):
            if x is None:
                self.assertIsNone(y)
            else:
                self.assertEqual(x[0], y[0])
                self.assertEqual(x[1][0], y[1][0])
                for p, q in zip((x[1][1], *x[1][2]), (y[1][1], *y[1][2])):
                    np.testing.assert_array_equal(p, q)

    def test_full_logits_all_layer_cache_equations(self):
        expected, caches = analytical(self.c, self.w, self.ids)
        actual = self.model.forward(self.ids)
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=1e-4)
        state = self.model.snapshot()
        for i, (kind, x, y) in enumerate(caches):
            if kind == "mla":
                np.testing.assert_allclose(
                    state.keys[i][: len(self.ids)], x, atol=2e-5, rtol=1e-4
                )
                np.testing.assert_allclose(
                    state.values[i][: len(self.ids)], y, atol=2e-5, rtol=1e-4
                )
            else:
                np.testing.assert_allclose(
                    state.recurrent[i][1][1], x, atol=2e-5, rtol=1e-4
                )
                for p, q in zip(state.recurrent[i][1][2], y):
                    np.testing.assert_allclose(p, q, atol=2e-5, rtol=1e-4)

    def test_tokenwise_irregular_chunks_and_restore(self):
        full = self.model.forward(self.ids)
        state = self.model.snapshot()
        for lengths in ([1] * 7, [2, 1, 4], [4, 3]):
            self.model.reset()
            offset = 0
            rows = []
            for n in lengths:
                rows.append(self.model.forward(self.ids[offset : offset + n]))
                offset += n
            np.testing.assert_array_equal(np.concatenate(rows), full)
            self.same_state(state, self.model.snapshot())
        self.model.reset()
        self.model.forward(self.ids[:2])
        saved = self.model.snapshot()
        tail = self.model.forward(self.ids[2:])
        self.model.restore(saved)
        np.testing.assert_array_equal(self.model.forward(self.ids[2:]), tail)

    def test_every_cache_for_mixed_schedules_and_chunk_prefixes(self):
        for kinds in (("mla", "kda", "mla"), ("kda", "mla", "kda", "mla")):
            c = configuration(kinds)
            w = fixture(c)
            model = NativeKimiK3Hybrid(c, w, library=self.path)
            rows = []
            for end, start in ((1, 0), (4, 1), (7, 4)):
                rows.append(model.forward(self.ids[start:end]))
                expected, caches = analytical(c, w, self.ids[:end])
                np.testing.assert_allclose(np.concatenate(rows), expected,
                                           atol=5e-5, rtol=1e-4)
                state = model.snapshot()
                self.assertEqual(state.length, end)
                for i, (kind, x, y) in enumerate(caches):
                    if kind == "mla":
                        for actual, reference in ((state.keys[i], x), (state.values[i], y)):
                            np.testing.assert_allclose(actual[:end], reference,
                                                       atol=2e-5, rtol=1e-4)
                            np.testing.assert_array_equal(actual[end:], 0)
                    else:
                        self.assertEqual(state.recurrent[i][1][0], end)
                        for actual, reference in zip(
                            (state.recurrent[i][1][1], *state.recurrent[i][1][2]),
                            (x, *y),
                        ):
                            np.testing.assert_allclose(actual, reference,
                                                       atol=2e-5, rtol=1e-4)

    def test_snapshot_identity_and_all_buffer_detachment(self):
        self.model.forward(self.ids[:2])
        saved = self.model.snapshot()
        twin = NativeKimiK3Hybrid(self.c, self.w, library=self.path)
        twin.restore(saved)
        self.same_state(saved, twin.snapshot())
        altered = {n: a.copy() for n, a in self.w.items()}
        altered["lm_head.weight"][0, 0] += np.float32(0.01)
        foreign = NativeKimiK3Hybrid(self.c, altered, library=self.path)
        with self.assertRaisesRegex(ValueError, "foreign"):
            foreign.restore(saved)
        changed = replace(self.c, mla=replace(self.c.mla, epsilon=2e-6))
        foreign_config = NativeKimiK3Hybrid(changed, self.w, library=self.path)
        with self.assertRaisesRegex(ValueError, "foreign"):
            foreign_config.restore(saved)
        before = twin.snapshot()
        for a in saved.keys + saved.values:
            if a is not None:
                a.fill(13)
        for item in saved.recurrent:
            if item is not None:
                for a in (item[1][1], *item[1][2]):
                    a.fill(13)
        self.same_state(before, twin.snapshot())
        self.same_state(before, self.model.snapshot())

    def test_snapshot_subclass_cannot_defeat_detachment(self):
        class AliasingArray(np.ndarray):
            def copy(self, *args, **kwargs):
                return self

        self.model.forward([1])
        saved = self.model.snapshot()
        keys = list(saved.keys)
        keys[-1] = keys[-1].view(AliasingArray)
        recurrent = list(saved.recurrent)
        identity, (position, state, history) = recurrent[0]
        recurrent[0] = (identity, (position, state.view(AliasingArray), history))
        for bad in (replace(saved, keys=tuple(keys)),
                    replace(saved, recurrent=tuple(recurrent))):
            with self.assertRaises(ValueError):
                self.model.restore(bad)
            self.same_state(saved, self.model.snapshot())

    def test_budget_boundary_rejects_before_native_execution(self):
        m = self.model
        m.forward([1])
        before = m.snapshot()
        base = 5 * m._weights_bytes + 5 * m._cache_bytes + 1024 * 1024
        required = base + 4 * len(self.ids) * (self.c.mla.vocab + 16 * self.c.mla.hidden)
        m.max_bytes = required - 1
        with patch.object(m, "_norm", side_effect=AssertionError("native entered")):
            with self.assertRaisesRegex(ValueError, "memory budget"):
                m.forward(self.ids)
        self.same_state(before, m.snapshot())
        m.max_bytes = base - 1
        with self.assertRaisesRegex(ValueError, "memory budget"):
            m.restore(before)
        m.max_bytes = required
        self.same_state(before, m.snapshot())
        self.assertEqual(m.forward(self.ids).shape, (len(self.ids), self.c.mla.vocab))

    def test_later_token_final_head_failure_rolls_back_and_retries(self):
        m = self.model
        m.forward(self.ids[:2])
        before = m.snapshot()
        original = m._matrix
        calls = 0

        def fail(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                w = np.full((self.c.mla.vocab, self.c.mla.hidden),
                            np.finfo(np.float32).max, np.float32)
                x = np.ones(self.c.mla.hidden, np.float32)
                return original(m._ptr(w), w.size, m._ptr(x), *args[3:])
            return original(*args)

        with patch.object(m, "_matrix", side_effect=fail):
            with self.assertRaisesRegex(ValueError, "block failed"):
                m.forward(self.ids[2:])
        self.same_state(before, m.snapshot())
        actual = m.forward(self.ids[2:])
        m.restore(before)
        np.testing.assert_array_equal(actual, m.forward(self.ids[2:]))

    def test_raw_mla_rejects_mutable_aliases_of_inputs(self):
        m = self.model
        for source, target in ((s, t) for s in ("weights", "input")
                               for t in ("keys", "values", "output")):
            w = m._mla_weights[-1].copy()
            keys, values = m._keys[-1].copy(), m._values[-1].copy()
            size = max(w.size, keys.size, values.size, self.c.mla.hidden)
            storage = np.ones(size, np.float32)
            x = storage[:self.c.mla.hidden] if source == "input" else np.ones(
                self.c.mla.hidden, np.float32)
            if source == "weights":
                storage[:w.size] = w
                w = storage[:w.size]
            out = np.full(x.size, 119, np.float32)
            if target == "keys":
                keys = storage[:keys.size].reshape(keys.shape)
            elif target == "values":
                values = storage[:values.size].reshape(values.shape)
            else:
                out = storage[:out.size]
            before = tuple(a.copy() for a in (w, x, keys, values, out))
            rc = m._mla(C.byref(m._c), m._ptr(w), w.size,
                        m._ptr(x), x.size, m._ptr(keys), keys.size,
                        m._ptr(values), values.size, 0, m._ptr(out), out.size)
            self.assertEqual(rc, 1)
            for actual, expected in zip((w, x, keys, values, out), before):
                np.testing.assert_array_equal(actual, expected)

    def test_native_blocks_late_overflow_never_publishes_partial_output(self):
        m = self.model
        largest = np.finfo(np.float32).max
        x = np.array([1, largest], np.float32)
        y = x.copy()
        out = np.full(2, 119, np.float32)
        self.assertEqual(m._add(m._ptr(x), m._ptr(y), 2, m._ptr(out)), 2)
        np.testing.assert_array_equal(out, 119)
        w = np.array([[1, 0], [0, largest]], np.float32)
        self.assertEqual(m._matrix(m._ptr(w), w.size, m._ptr(x),
                                   2, 2, m._ptr(out)), 2)
        np.testing.assert_array_equal(out, 119)
        x = np.array([0, 2], np.float32)
        gain = np.array([1, largest], np.float32)
        self.assertEqual(m._norm(m._ptr(x), m._ptr(gain), 2,
                                 1e-6, m._ptr(out)), 2)
        np.testing.assert_array_equal(out, 119)

    def test_constructor_budget_precedes_library_loading(self):
        with patch("ctypes.CDLL", side_effect=AssertionError("library entered")):
            with self.assertRaisesRegex(ValueError, "memory budget"):
                NativeKimiK3Hybrid(self.c, self.w, library=self.path, max_bytes=1)

    def test_whole_call_failure_keeps_all_layer_caches(self):
        self.model.forward(self.ids[:2])
        before = self.model.snapshot()
        with self.assertRaisesRegex(ValueError, "injected"):
            self.model.forward(self.ids[2:], _fail_after_token=2)
        self.same_state(before, self.model.snapshot())
        with patch.object(
            self.model._experts[-1],
            "forward",
            side_effect=RuntimeError("late expert failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "late expert"):
                self.model.forward(self.ids[2:])
        self.same_state(before, self.model.snapshot())

    def test_snapshot_validation_and_detachment(self):
        self.model.forward(self.ids[:2])
        saved = self.model.snapshot()
        for bad in (
            replace(saved, identity="foreign"),
            replace(saved, length=-1),
            replace(saved, keys=()),
            replace(saved, recurrent=(None,) * self.c.mla.layers),
        ):
            with self.assertRaises(ValueError):
                self.model.restore(bad)
            self.same_state(saved, self.model.snapshot())
        self.model.restore(saved)
        saved.keys[-1].fill(3)
        self.assertFalse(np.all(self.model.snapshot().keys[-1] == 3))

    def test_tokens_budget_configuration_and_no_numpy_math(self):
        for ids in ([], [True], [-1], [self.c.mla.vocab], [1.2], [[1]]):
            with self.assertRaises(ValueError):
                self.model.forward(ids)
        for value in (True, 0, -1, "unbounded"):
            self.model.max_bytes = value
            with self.assertRaises(ValueError):
                self.model.forward([1])
        self.model.max_bytes = 128 * 1024**2
        expected = self.model.forward(self.ids)
        self.model.reset()
        with (
            patch("numpy.matmul", side_effect=AssertionError("matrix fallback")),
            patch("numpy.einsum", side_effect=AssertionError("equation fallback")),
        ):
            np.testing.assert_array_equal(self.model.forward(self.ids), expected)
        with self.assertRaises(ValueError):
            replace(self.c, layer_types=("mla", "mla", "mla", "kda"))

    def test_mixed_schedule_and_block_boundaries(self):
        for kinds, block in (
            (("mla",), 1),
            (("kda", "mla"), 1),
            (("mla", "kda", "mla"), 2),
        ):
            c = configuration(kinds)
            c = replace(c, mla=replace(c.mla, block=block))
            w = fixture(c)
            model = NativeKimiK3Hybrid(c, w, library=self.path)
            expected, _ = analytical(c, w, self.ids)
            np.testing.assert_allclose(
                model.forward(self.ids), expected, atol=5e-5, rtol=1e-4
            )

    def test_native_mla_rejection_does_not_publish(self):
        m = self.model
        i = 3
        w = m._mla_weights[i]
        h = self.c.mla.hidden
        x = np.ones(h, np.float32)
        out = np.full(h, 119, np.float32)
        keys = m._keys[i].copy()
        values = m._values[i].copy()
        rc = m._mla(
            C.byref(m._c),
            m._ptr(w),
            w.size - 1,
            m._ptr(x),
            x.size,
            m._ptr(keys),
            keys.size,
            m._ptr(values),
            values.size,
            0,
            m._ptr(out),
            out.size,
        )
        self.assertNotEqual(rc, 0)
        np.testing.assert_array_equal(out, 119)
        np.testing.assert_array_equal(keys, 0)
        np.testing.assert_array_equal(values, 0)

    def test_reset_allocation_failure_and_budget_are_transactional(self):
        self.model.forward(self.ids[:2])
        before = self.model.snapshot()
        original = np.zeros
        calls = 0

        def failing_zeros(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise MemoryError("reset allocation")
            return original(*args, **kwargs)

        with patch("numpy.zeros", side_effect=failing_zeros):
            with self.assertRaises(MemoryError):
                self.model.reset()
        self.same_state(before, self.model.snapshot())
        self.model.max_bytes = 1
        for operation in (self.model.reset, self.model.snapshot):
            with self.assertRaisesRegex(ValueError, "memory budget"):
                operation()
        self.model.max_bytes = 128 * 1024**2
        self.same_state(before, self.model.snapshot())

    def test_late_restore_failure_never_changes_earlier_recurrence(self):
        self.model.forward(self.ids[:2])
        saved = self.model.snapshot()
        self.model.forward(self.ids[2:])
        before = self.model.snapshot()
        keys = list(saved.keys)
        keys[-1] = keys[-1].copy()
        keys[-1][-1].fill(1)
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            self.model.restore(replace(saved, keys=tuple(keys)))
        self.same_state(before, self.model.snapshot())
        recurrent = list(saved.recurrent)
        identity, item = recurrent[1]
        recurrent[1] = (identity, (item[0] + 1, *item[1:]))
        with self.assertRaisesRegex(ValueError, "positions"):
            self.model.restore(replace(saved, recurrent=tuple(recurrent)))
        self.same_state(before, self.model.snapshot())

    def test_capacity_last_token_and_overflow(self):
        ids = [1] * self.c.mla.capacity
        expected, _ = analytical(self.c, self.w, ids)
        np.testing.assert_allclose(
            self.model.forward(ids), expected, atol=5e-5, rtol=1e-4
        )
        before = self.model.snapshot()
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.model.forward([1])
        self.same_state(before, self.model.snapshot())
        self.model.restore(before)
        self.assertEqual(self.model.position, self.c.mla.capacity)

    def test_restore_allocation_failure_preserves_all_live_caches(self):
        self.model.forward(self.ids[:2])
        saved = self.model.snapshot()
        self.model.forward(self.ids[2:])
        before = self.model.snapshot()
        original = NativeKimiK3KDA.restore
        calls = 0

        def failing_restore(candidate, snapshot):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise MemoryError("restore allocation")
            return original(candidate, snapshot)

        with patch.object(NativeKimiK3KDA, "restore", failing_restore):
            with self.assertRaises(MemoryError):
                self.model.restore(saved)
        self.same_state(before, self.model.snapshot())

    def test_owned_weights_and_concurrent_calls(self):
        expected = [self.model.forward([1]) for _ in range(12)]
        before = self.model.snapshot()
        self.model.reset()
        for a in self.w.values():
            a.fill(100)
        for a in self.model._weights.values():
            with self.assertRaises(ValueError):
                a.setflags(write=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda _: self.model.forward([1]), range(12)))
        self.assertEqual(
            sorted(a.tobytes() for a in expected), sorted(a.tobytes() for a in actual)
        )
        self.same_state(before, self.model.snapshot())

    def test_mla_malformed_config_and_overlapping_outputs(self):
        m = self.model
        w = m._mla_weights[-1]
        x = np.ones(self.c.mla.hidden, np.float32)
        out = np.full_like(x, 119)
        keys = m._keys[-1].copy()
        values = m._values[-1].copy()

        def run(length=0, nw=None, nx=None, nk=None, nv=None, no=None, output=None):
            return m._mla(
                C.byref(m._c),
                m._ptr(w),
                w.size if nw is None else nw,
                m._ptr(x),
                x.size if nx is None else nx,
                m._ptr(keys),
                keys.size if nk is None else nk,
                m._ptr(values),
                values.size if nv is None else nv,
                length,
                m._ptr(out) if output is None else output,
                out.size if no is None else no,
            )

        for field, limit in (
            ("hidden", 64),
            ("layers", 8),
            ("heads", 4),
            ("nope", 16),
            ("rope", 16),
            ("value", 16),
            ("qrank", 64),
            ("kvrank", 64),
            ("intermediate", 128),
            ("block", 8),
            ("vocab", 256),
            ("capacity", 128),
        ):
            original = getattr(m._c, field)
            for invalid in (-1, limit + 1) if field == "rope" else (0, limit + 1):
                setattr(m._c, field, invalid)
                self.assertNotEqual(run(), 0)
            setattr(m._c, field, original)
        for field in ("epsilon", "beta", "linear_beta"):
            original = getattr(m._c, field)
            for invalid in (0, -1, float("nan"), float("inf")):
                setattr(m._c, field, invalid)
                self.assertNotEqual(run(), 0)
            setattr(m._c, field, original)
        for arguments in (
            {"length": -1},
            {"length": self.c.mla.capacity},
            {"nw": w.size - 1},
            {"nx": x.size - 1},
            {"nk": keys.size - 1},
            {"nv": values.size - 1},
            {"no": out.size - 1},
            {"output": m._ptr(keys)},
        ):
            self.assertNotEqual(run(**arguments), 0)
        np.testing.assert_array_equal(keys, 0)
        np.testing.assert_array_equal(values, 0)
        np.testing.assert_array_equal(out, 119)

    def test_mla_large_gate_logits_are_valid_probabilities(self):
        m = self.model
        w = m._mla_weights[-1].copy()
        c = self.c.mla
        width = c.heads * c.value
        w[-2 * width * c.hidden : -width * c.hidden] = np.finfo(np.float32).max
        x = np.full(c.hidden, 2, np.float32)
        out = np.empty(c.hidden, np.float32)
        keys = m._keys[-1].copy()
        values = m._values[-1].copy()
        self.assertEqual(
            m._mla(
                C.byref(m._c),
                m._ptr(w),
                w.size,
                m._ptr(x),
                x.size,
                m._ptr(keys),
                keys.size,
                m._ptr(values),
                values.size,
                0,
                m._ptr(out),
                out.size,
            ),
            0,
        )
        self.assertTrue(np.isfinite(out).all())
