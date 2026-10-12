"""Analytical K3 expert tests; exact publisher oracle is separately opt-in."""

import ctypes as C
import hashlib
import importlib
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.native_kimi_k3_moe import (
    KimiK3MoEConfig,
    NativeKimiK3MoE,
    budgets,
    tensor_shapes,
)


def compile_library(directory):
    source = (
        Path(__file__).resolve().parents[1]
        / "openagent/adaptive_engine/native_kimi_k3_moe.c"
    )
    output = Path(directory) / ("k3_moe.dll" if os.name == "nt" else "k3_moe.so")
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
            raise RuntimeError("MSVC required, not a passing skip")
        batch = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
        args = [
            "cl",
            "/nologo",
            "/TC",
            "/W4",
            "/WX",
            "/O2",
            "/fp:strict",
            "/LD",
            str(source),
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
            str(source),
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
    rng = np.random.default_rng(832)
    w = {
        name: rng.normal(0, 0.2, shape).astype(np.float32)
        for name, shape in tensor_shapes(c).items()
    }
    w["latent.norm.weight"][:] = np.linspace(0.7, 1.3, c.latent)
    w["router.bias"][:] = np.linspace(-0.15, 0.15, c.experts)
    return w


def analytical(c, w, x):
    # Independent float64 matrix formulation of report equations 11-13.
    w = {n: a.astype(np.float64) for n, a in w.items()}
    x = x.astype(np.float64)
    scores = 1 / (1 + np.exp(-(x @ w["router.weight"].T)))
    order = np.argsort(-(scores + w["router.bias"]), axis=1, kind="stable")[
        :, : c.selected
    ]

    def expert(prefix, index, value):
        g = value @ w[prefix + ".gate.weight"][index].T
        u = value @ w[prefix + ".up.weight"][index].T
        act = (
            c.beta
            * np.tanh(g / c.beta)
            / (1 + np.exp(-g))
            * c.linear_beta
            * np.tanh(u / c.linear_beta)
        )
        return act @ w[prefix + ".down.weight"][index].T

    result = []
    for t, value in enumerate(x):
        z = value @ w["latent.down.weight"].T
        p = scores[t, order[t]]
        p /= p.sum()
        aggregate = sum(
            p[j] * expert("routed", int(e), z) for j, e in enumerate(order[t])
        )
        normalized = (
            aggregate
            / np.sqrt(np.mean(aggregate**2) + c.epsilon)
            * w["latent.norm.weight"]
        )
        result.append(
            normalized @ w["latent.up.weight"].T
            + sum(expert("shared", j, value) for j in range(c.shared))
        )
    return np.array(result), order


class K3MoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(
            prefix="k3_moe_", ignore_cleanup_errors=True
        )
        cls.library = compile_library(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.c = KimiK3MoEConfig()
        self.w = fixture(self.c)
        self.x = (
            np.random.default_rng(15).normal(size=(7, self.c.hidden)).astype(np.float32)
        )
        self.model = NativeKimiK3MoE(self.library, self.c, self.w)

    def test_analytical_and_token_chunks(self):
        expected, routes = analytical(self.c, self.w, self.x)
        actual, chosen = self.model.forward(self.x)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=5e-6)
        np.testing.assert_array_equal(chosen, routes)
        chunks = [self.model.forward(a) for a in (self.x[:2], self.x[2:3], self.x[3:])]
        np.testing.assert_array_equal(np.concatenate([a[0] for a in chunks]), actual)

    def test_one_dimensional_closed_form(self):
        c = KimiK3MoEConfig(hidden=1, latent=1, intermediate=1, experts=3, selected=1)
        w = {n: np.ones(s, np.float32) for n, s in tensor_shapes(c).items()}
        w["router.bias"][:] = [0, 1, 2]
        model = NativeKimiK3MoE(self.library, c, w)
        for value in (0.0, 1.0, -2.0, 1000.0):
            act = (
                4 * np.tanh(value / 4) / (1 + np.exp(-value)) * 25 * np.tanh(value / 25)
            )
            expected = 2 * act + act / np.sqrt(act * act + c.epsilon)
            out, ids = model.forward(np.array([[value]], np.float32))
            np.testing.assert_allclose(out, expected, atol=3e-5, rtol=2e-6)
            np.testing.assert_array_equal(ids, [[2]])

    def test_shared_full_width_and_bias_dispatch_only(self):
        w = {n: a.copy() for n, a in self.w.items()}
        w["latent.up.weight"].fill(0)
        expected, _ = analytical(self.c, w, self.x)
        np.testing.assert_allclose(
            NativeKimiK3MoE(self.library, self.c, w).forward(self.x)[0],
            expected,
            atol=2e-6,
            rtol=5e-6,
        )
        w = {n: a.copy() for n, a in self.w.items()}
        baseline = self.model.forward(self.x)
        w["router.bias"] += np.float32(1)
        actual = NativeKimiK3MoE(self.library, self.c, w).forward(self.x)
        np.testing.assert_array_equal(actual[0], baseline[0])
        np.testing.assert_array_equal(actual[1], baseline[1])

    def test_cutoff_tie_fails_closed_internal_ties_allowed(self):
        w = {n: a.copy() for n, a in self.w.items()}
        w["router.weight"].fill(0)
        w["router.bias"].fill(0)
        model = NativeKimiK3MoE(self.library, self.c, w)
        with self.assertRaisesRegex(RuntimeError, r"\(4\)"):
            model.forward(self.x)
        w["router.bias"][:2] = 1
        ids = NativeKimiK3MoE(self.library, self.c, w).forward(self.x)[1]
        np.testing.assert_array_equal(ids, np.tile([0, 1], (7, 1)))

    def test_inventory_config_and_input_rejection(self):
        for name, value in (
            ("shared", 1),
            ("selected", 5),
            ("hidden", True),
            ("epsilon", np.nan),
            ("beta", 0),
            ("experts", 65),
        ):
            with self.assertRaises(ValueError):
                replace(self.c, **{name: value})
        for w in (
            {**self.w, "extra": np.ones(1, np.float32)},
            {},
            {**self.w, "router.weight": self.w["router.weight"].astype(np.float64)},
            {**self.w, "router.bias": np.full(self.c.experts, np.inf, np.float32)},
        ):
            with self.assertRaises(ValueError):
                NativeKimiK3MoE(self.library, self.c, w)
        for x in (
            self.x.astype(np.float64),
            self.x[:, :-1],
            self.x[:0],
            np.full_like(self.x, np.nan),
            self.x[0],
            np.zeros((129, 12), np.float32),
        ):
            with self.assertRaises(ValueError):
                self.model.forward(x)

    def test_budgets_and_detached_weights(self):
        mem, work = budgets(self.c, len(self.x))
        model = NativeKimiK3MoE(
            self.library, self.c, self.w, memory_budget=mem, mac_budget=work
        )
        expected = model.forward(self.x)[0]
        self.w["latent.norm.weight"].fill(99)
        np.testing.assert_array_equal(model.forward(self.x)[0], expected)
        with self.assertRaises(ValueError):
            model._weights.flags.writeable = True
        for kw in ({"memory_budget": mem - 1}, {"mac_budget": work - 1}):
            with self.assertRaises(ValueError):
                NativeKimiK3MoE(self.library, self.c, fixture(self.c), **kw).forward(
                    self.x
                )

    def test_library_ownership(self):
        peer = NativeKimiK3MoE(self.library, self.c, self.w)
        self.model.close()
        self.model.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.model.forward(self.x)
        peer.forward(self.x)
        peer.close()

    def tearDown(self):
        self.model.close()

    def test_mutated_budgets_and_configuration_fail_before_native(self):
        for field in ("memory_budget", "mac_budget"):
            original = getattr(self.model, field)
            for invalid in (True, -1, float("nan"), 2**64, 1.5):
                setattr(self.model, field, invalid)
                with self.assertRaises(ValueError):
                    self.model.forward(self.x)
            setattr(self.model, field, original)
        with self.assertRaises(AttributeError):
            self.model.config = replace(self.c, hidden=1)

    def test_large_finite_aggregate_normalizes_without_square_overflow(self):
        c = KimiK3MoEConfig(hidden=1, latent=1, intermediate=1, experts=3, selected=1)
        w = {n: np.ones(s, np.float32) for n, s in tensor_shapes(c).items()}
        w["router.bias"][:] = [0, 1, 2]
        w["routed.down.weight"].fill(1e20)
        x = np.ones((1, 1), np.float32)
        with NativeKimiK3MoE(self.library, c, w) as model:
            out, ids = model.forward(x)
        expected, routes = analytical(c, w, x)
        np.testing.assert_allclose(out, expected, rtol=2e-6)
        np.testing.assert_array_equal(ids, routes)

    def test_native_overlapping_output_routes_rejected_atomically(self):
        m = self.model
        storage = np.full(self.x.size + 2, 123, np.float32)
        fp = C.POINTER(C.c_float)
        status = m._function(
            C.byref(m._c), m._weights.ctypes.data_as(fp), m._weights.size,
            self.x.ctypes.data_as(fp), self.x.size, len(self.x),
            storage.ctypes.data_as(fp), self.x.size,
            storage[1:].ctypes.data_as(C.POINTER(C.c_int)), len(self.x)*self.c.selected,
            m.memory_budget, m.mac_budget, -1,
        )
        self.assertEqual(status, 1)
        np.testing.assert_array_equal(storage, 123)

    def test_dot_product_cancellation_does_not_overflow_intermediately(self):
        c = replace(self.c, hidden=2)
        w = fixture(c)
        w["router.weight"][:] = [np.finfo(np.float32).max, -np.finfo(np.float32).max]
        x = np.full((1, 2), 2, np.float32)
        expected, routes = analytical(c, w, x)
        with NativeKimiK3MoE(self.library, c, w) as model:
            out, ids = model.forward(x)
        np.testing.assert_allclose(out, expected, atol=2e-6, rtol=5e-6)
        np.testing.assert_array_equal(ids, routes)

    def test_normalized_fp32_overflow_is_rejected(self):
        c = replace(self.c, latent=2, selected=1)
        w = fixture(c)
        w["routed.down.weight"][:, 1].fill(0)
        w["latent.norm.weight"].fill(np.finfo(np.float32).max)
        with NativeKimiK3MoE(self.library, c, w) as model:
            with self.assertRaisesRegex(RuntimeError, r"\(3\)"):
                model.forward(self.x)

    def test_direct_native_transactional_faults(self):
        m = self.model

        def ptr(a):
            return a.ctypes.data_as(C.POINTER(C.c_float))

        out = np.full_like(self.x, 123)
        ids = np.full((7, self.c.selected), 123, np.int32)

        def call(nw=None, fault=-1, mem=16 * 1024 * 1024, work=100_000_000):
            return m._function(
                C.byref(m._c),
                ptr(m._weights),
                m._weights.size if nw is None else nw,
                ptr(self.x),
                self.x.size,
                7,
                ptr(out),
                out.size,
                ids.ctypes.data_as(C.POINTER(C.c_int)),
                ids.size,
                mem,
                work,
                fault,
            )

        for kw in ({"nw": m._weights.size - 1}, {"fault": 5}, {"mem": 1}, {"work": 1}):
            self.assertNotEqual(call(**kw), 0)
            np.testing.assert_array_equal(out, 123)
            np.testing.assert_array_equal(ids, 123)
        self.x[-1].fill(np.finfo(np.float32).max)
        self.assertEqual(call(), 3)
        np.testing.assert_array_equal(out, 123)
        np.testing.assert_array_equal(ids, 123)

    def test_geometry_variants_and_zero_aggregate(self):
        for c in (
            replace(self.c, hidden=3, latent=5, intermediate=2, selected=4),
            replace(
                self.c, hidden=64, latent=64, intermediate=128, experts=64, selected=16
            ),
        ):
            w = fixture(c)
            x = np.random.default_rng(13).normal(size=(2, c.hidden)).astype(np.float32)
            expected, ids = analytical(c, w, x)
            out, actual_ids = NativeKimiK3MoE(
                self.library, c, w, memory_budget=32 * 1024 * 1024
            ).forward(x)
            np.testing.assert_allclose(out, expected, atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(actual_ids, ids)
        self.w["routed.down.weight"].fill(0)
        expected, _ = analytical(self.c, self.w, self.x)
        out, _ = NativeKimiK3MoE(self.library, self.c, self.w).forward(self.x)
        np.testing.assert_allclose(out, expected, atol=2e-6, rtol=5e-6)


def verify_vendor(directory, dependencies):
    """Only public state_dict/forward; no source extraction, patching or substitution."""
    from tests.probe_kimi_k3_oracle import HASHES, REVISION

    root = Path(directory)
    for name, digest in HASHES.items():
        if (
            hashlib.sha256(
                (root / "kimi_k3_vendor_probe" / name).read_bytes()
            ).hexdigest()
            != digest
        ):
            raise ValueError("exact publisher hash mismatch: " + name)
    sys.path[:0] = [str(root), str(dependencies)]
    import torch

    vendor = importlib.import_module("kimi_k3_vendor_probe.modeling_kimi_linear")
    c = KimiK3MoEConfig()
    w = fixture(c)
    x = np.random.default_rng(15).normal(size=(7, c.hidden)).astype(np.float32)
    cfg = vendor.KimiLinearConfig(
        hidden_size=c.hidden,
        moe_intermediate_size=c.intermediate,
        num_experts=c.experts,
        num_experts_per_token=c.selected,
        num_shared_experts=2,
        routed_expert_hidden_size=c.latent,
        latent_moe_use_norm=True,
        hidden_act="situ",
        activation_situ_beta=c.beta,
        activation_situ_linear_beta=c.linear_beta,
        rms_norm_eps=c.epsilon,
        use_grouped_topk=False,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0,
    )
    oracle = vendor.KimiSparseMoeBlock(cfg).float().eval()
    state = {
        "gate.weight": w["router.weight"],
        "gate.e_score_correction_bias": w["router.bias"],
        "routed_expert_down_proj.weight": w["latent.down.weight"],
        "routed_expert_up_proj.weight": w["latent.up.weight"],
        "routed_expert_norm.weight": w["latent.norm.weight"],
        "shared_experts.gate_proj.weight": w["shared.gate.weight"].reshape(
            2 * c.intermediate, c.hidden
        ),
        "shared_experts.up_proj.weight": w["shared.up.weight"].reshape(
            2 * c.intermediate, c.hidden
        ),
        "shared_experts.down_proj.weight": np.concatenate(
            list(w["shared.down.weight"]), axis=1
        ),
    }
    for e in range(c.experts):
        for vendor_name, native_name in (("w1", "gate"), ("w2", "down"), ("w3", "up")):
            state[f"experts.{e}.{vendor_name}.weight"] = w[
                f"routed.{native_name}.weight"
            ][e]
    oracle.load_state_dict(
        {n: torch.from_numpy(a.copy()) for n, a in state.items()}, strict=True
    )
    inputs = torch.from_numpy(x[None].copy())
    with tempfile.TemporaryDirectory(prefix="k3_moe_vendor_") as build:
        with NativeKimiK3MoE(compile_library(build), c, w) as native, torch.no_grad():
            actual, ids = native.forward(x)
            expected = oracle(inputs).numpy()[0]
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            vi, vp = oracle.gate(inputs)
            # Compare unordered dispatch: Torch topk may return a different within-set order.
            np.testing.assert_array_equal(
                np.sort(ids, axis=1), np.sort(vi.numpy(), axis=1)
            )
            raw = torch.sigmoid(inputs.reshape(-1, c.hidden) @ oracle.gate.weight.T)
            selected = raw.gather(1, vi)
            np.testing.assert_allclose(
                vp.numpy(),
                (selected / selected.sum(-1, keepdim=True)).numpy(),
                atol=2e-7,
                rtol=2e-7,
            )
            baseline = vp.clone()
            oracle.gate.e_score_correction_bias.add_(1.0)
            shifted_ids, shifted_p = oracle.gate(inputs)
            np.testing.assert_array_equal(shifted_ids.numpy(), vi.numpy())
            np.testing.assert_allclose(
                shifted_p.numpy(), baseline.numpy(), atol=2e-7, rtol=2e-7
            )
            oracle.gate.routed_scaling_factor = 2.5
            _, scaled = oracle.gate(inputs)
            np.testing.assert_allclose(
                scaled.numpy(), baseline.numpy() * 2.5, atol=2e-7, rtol=2e-7
            )
            oracle.gate.routed_scaling_factor = 1.0
            oracle.gate.moe_renormalize = False
            raw_ids, unnormalized = oracle.gate(inputs)
            np.testing.assert_allclose(
                unnormalized.numpy(),
                raw.gather(1, raw_ids).numpy(),
                atol=2e-7,
                rtol=2e-7,
            )
            print(
                "EXACT K3 publisher revision",
                REVISION,
                "MoE output + routes + unbiased renormalized weights verified; max abs error",
                float(np.max(np.abs(actual - expected))),
            )
            print(
                "Router scale 2.5 and renormalize=False observed separately; native admits only scale=1 and renormalize=True."
            )


if __name__ == "__main__":
    if "--vendor-directory" in sys.argv:
        verify_vendor(
            sys.argv[sys.argv.index("--vendor-directory") + 1],
            sys.argv[sys.argv.index("--vendor-dependencies") + 1],
        )
    else:
        unittest.main()
