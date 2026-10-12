"""Optional dense branches against independent SwiGLU equations."""

from dataclasses import replace
import gc
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.native_kimi_k3_checkpoint import (
    hybrid_fixture_config,
    hybrid_fixture_document,
    load_native_kimi_k3_fixture,
)
from openagent.adaptive_engine.native_kimi_k3_hybrid import NativeKimiK3Hybrid
from tests import test_native_kimi_k3_hybrid as reference
from tests import test_native_checkpoint as checkpoint


class DenseHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        source = (
            Path(__file__).resolve().parents[1]
            / "openagent/adaptive_engine/native_kimi_k3_dense.c"
        )
        run = subprocess.run

        def compile_dense(command, **kwargs):
            if isinstance(command, str):
                command = command.replace(" /link", f' "{source}" /link')
            elif "-shared" in command:
                command = [*command, str(source)]
            return run(command, **kwargs)

        with patch.object(reference.subprocess, "run", side_effect=compile_dense):
            cls.library = reference.build(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        cls.temp.cleanup()

    def test_config_and_old_document(self):
        old = reference.configuration()
        doc = hybrid_fixture_document(old)
        self.assertNotIn("dense_layers", doc)
        self.assertEqual(hybrid_fixture_config(doc), old)
        for indices in ([0], (True,), (-1,), (4,), (0, 0)):
            with self.assertRaises(ValueError):
                replace(old, dense_layers=indices)
        for width in (True, 0, 129, 1.5):
            with self.assertRaises(ValueError):
                replace(old, dense_intermediate=width)
        dense = replace(old, dense_layers=(0, 2), dense_intermediate=13)
        self.assertEqual(hybrid_fixture_config(hybrid_fixture_document(dense)), dense)
        self.assertEqual(
            replace(old, dense_layers=(0,)).dense_width, old.moe.intermediate
        )

    def test_reentrant_input_mutation_is_rejected(self):
        c = replace(reference.configuration(), dense_layers=(0,))
        model = NativeKimiK3Hybrid(c, reference.fixture(c), library=self.library)
        model.forward([1])
        before = model.snapshot()
        for operation in (
            model.reset,
            lambda: model.restore(before),
            lambda: model.forward([2]),
        ):

            class Input:
                def __array__(self, dtype=None, copy=None):
                    operation()
                    return np.array([2], dtype=np.int64)

            with self.assertRaisesRegex(ValueError, "reentrant"):
                model.forward(Input())
            reference.HybridTests.same_state(self, before, model.snapshot())
        model.forward([2])
        self.assertEqual(model.position, 2)

    def test_dense_checkpoint_roundtrip_and_metadata_rejection(self):
        c = replace(reference.configuration(), dense_layers=(0, 1))
        w = reference.fixture(c)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint.NativeCheckpointExecutionTests.write_checkpoint(
                root, hybrid_fixture_document(c), w
            )
            loaded = load_native_kimi_k3_fixture(
                root, library=self.library, max_weight_bytes=64 * 1024**2
            )
            direct = NativeKimiK3Hybrid(c, w, library=self.library)
            np.testing.assert_array_equal(
                loaded.forward([1, 2, 3]), direct.forward([1, 2, 3])
            )
            del loaded, direct
            gc.collect()
            bad = dict(w)
            bad["model.layers.0.mlp.gate_proj.weight"] = np.zeros((1, 1), np.float32)
            checkpoint.NativeCheckpointExecutionTests.write_checkpoint(
                root, hybrid_fixture_document(c), bad
            )
            with patch(
                "openagent.adaptive_engine.native_checkpoint._read_tensor",
                side_effect=AssertionError("payload read"),
            ):
                with self.assertRaisesRegex(ValueError, "metadata"):
                    load_native_kimi_k3_fixture(
                        root, library=self.library, max_weight_bytes=64 * 1024**2
                    )
        doc = hybrid_fixture_document(c)
        for name, value in (
            ("dense_layers", (0,)),
            ("unknown", 1),
            ("dense_intermediate", True),
        ):
            with self.assertRaises(ValueError):
                hybrid_fixture_config({**doc, name: value})

    def test_equations_chunks_restore_and_native_failure(self):
        for indices in ((0,), (0, 1), (0, 3), (0, 1, 2, 3)):
            c = replace(
                reference.configuration(), dense_layers=indices, dense_intermediate=13
            )
            w = reference.fixture(c)
            model = NativeKimiK3Hybrid(c, w, library=self.library)
            # Reuse only the independently evaluated mixer/AttnRes equations.
            rw = reference.fixture(replace(c, dense_layers=()))
            rw.update(w)
            moe_equations = reference.moe_reference.analytical
            layer = 0

            def feedforward(config, weights, x):
                nonlocal layer
                i = layer
                layer += 1
                if i not in indices:
                    return moe_equations(config, weights, x)
                p = f"model.layers.{i}.mlp."
                gate = (x.astype(np.float64) @ w[p + "gate_proj.weight"].T).astype(
                    np.float32
                )
                up = (x.astype(np.float64) @ w[p + "up_proj.weight"].T).astype(
                    np.float32
                )
                g = gate.astype(np.float64)
                activation = (g * np.exp(-np.logaddexp(0, -g)) * up).astype(np.float32)
                return (
                    activation.astype(np.float64) @ w[p + "down_proj.weight"].T
                ).astype(np.float32), None

            ids = [1, 4, 2, 9, 3, 8, 5]
            with patch.object(
                reference.moe_reference, "analytical", side_effect=feedforward
            ):
                expected, _ = reference.analytical(c, rw, ids)
            full = model.forward(ids)
            np.testing.assert_allclose(full, expected, atol=5e-5, rtol=1e-4)
            state = model.snapshot()
            model.reset()
            with (
                patch("numpy.exp", side_effect=AssertionError("Python math")),
                patch("numpy.matmul", side_effect=AssertionError("Python math")),
            ):
                chunks = [
                    model.forward(ids[:2]),
                    model.forward(ids[2:3]),
                    model.forward(ids[3:]),
                ]
            np.testing.assert_array_equal(np.concatenate(chunks), full)
            reference.HybridTests.same_state(self, state, model.snapshot())
            model.reset()
            model.forward(ids[:2])
            before = model.snapshot()
            native = model._dense
            ptr = model._ptr
            calls = 0

            def fail(*args):
                nonlocal calls
                calls += 1
                if calls == len(indices) + 1:
                    bad = np.full_like(
                        w["model.layers.0.mlp.gate_proj.weight"],
                        np.finfo(np.float32).max,
                    )
                    x = np.full(c.mla.hidden, 2, np.float32)
                    return native(ptr(bad), args[1], args[2], ptr(x), *args[4:])
                return native(*args)

            with patch.object(model, "_dense", side_effect=fail):
                with self.assertRaisesRegex(ValueError, "block failed"):
                    model.forward(ids[2:])
            reference.HybridTests.same_state(self, before, model.snapshot())
            tail = model.forward(ids[2:])
            model.restore(before)
            np.testing.assert_array_equal(model.forward(ids[2:]), tail)
            np.testing.assert_array_equal(tail, full[2:])
            del model
            gc.collect()
