import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.olmoe_validation import validate_olmoe
from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts


IDENTITY = {"model_id": "tiny-fixture", "revision": "test-only",
            "implementation": "independent-prefix-formula", "device": "cpu", "dtype": "float64"}


def reference(ids):
    # Closed-form reference does not call the stateful decoder.
    totals = np.cumsum(ids)
    return np.stack((totals, -totals, np.ones(len(ids))), axis=1).astype(float)


class TinyDecoder:
    config = SimpleNamespace(vocab_size=3, num_hidden_layers=1)
    manifest = SimpleNamespace(model_id="tiny-fixture", fingerprint="fixture-only")
    root = Path("tiny-fixture")
    context_tokens = 4

    def __init__(self, offset=0):
        self.offset = offset
        self.calls = []
        self.reset()

    def reset(self):
        self.total = 0

    def forward(self, ids):
        self.calls.append(list(ids))
        self.total += ids[0]
        return np.array([self.total, -self.total, 1.0]) + self.offset


class OlmoeValidationTests(unittest.TestCase):
    def test_real_tiny_checkpoint_against_closed_form_reference(self):
        # Zero attention/expert weights yield an independently calculable decoder.
        # This checks wiring, not nontrivial attention/routing or public weights.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(model_type="olmoe", hidden_size=2, intermediate_size=2,
                          num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
                          num_experts=1, num_experts_per_tok=1, vocab_size=3,
                          max_position_embeddings=4, rms_norm_eps=1e-5,
                          rope_theta=10000.0, norm_topk_prob=True, eos_token_id=2)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            embeddings = np.array([[1, 2], [2, -1], [-1, 3]], dtype="<f4")
            head = np.array([[1, 0], [0, 1], [-1, -1]], dtype="<f4")
            tensors = {"model.embed_tokens.weight": embeddings, "lm_head.weight": head,
                       "model.norm.weight": np.ones(2, dtype="<f4")}
            prefix = "model.layers.0."
            for name in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
                tensors[prefix + name + ".weight"] = np.ones(2, dtype="<f4")
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                tensors[prefix + "self_attn." + name + ".weight"] = np.zeros((2, 2), dtype="<f4")
            tensors[prefix + "mlp.gate.weight"] = np.zeros((1, 2), dtype="<f4")
            for name in ("gate_proj", "up_proj", "down_proj"):
                tensors[prefix + "mlp.experts.0." + name + ".weight"] = np.zeros((2, 2), dtype="<f4")
            header, data = {}, bytearray()
            for name, tensor in tensors.items():
                start = len(data)
                data.extend(tensor.tobytes())
                header[name] = dict(dtype="F32", shape=list(tensor.shape), data_offsets=[start, len(data)])
            encoded = json.dumps(header).encode("utf-8")
            encoded += b" " * (-len(encoded) % 8)
            (root / "model.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)
            manifest = pack_olmoe_experts(root, root / "experts.i8", model_id="tiny-test-only")
            capacity = max(region.size for region in manifest.layers[0].regions.values())
            class InvalidRegistry:
                def resolve(self, **kwargs):
                    raise RuntimeError("backend initialization failed")

            with patch("openagent.adaptive_engine.olmoe.SparseExpertRuntime") as runtime:
                with self.assertRaisesRegex(RuntimeError, "backend initialization failed"):
                    OlmoeModel(root, manifest, expert_cache_bytes=capacity,
                               kernels=InvalidRegistry())
                runtime.assert_not_called()
            def independent(ids):
                x = embeddings[list(ids)].astype(np.float64)
                return (x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-5)) @ head.T
            with OlmoeModel(root, manifest, expert_cache_bytes=capacity) as model:
                report = self.run_validation(model, ref=independent, sequences=((0, 1, 2), (2, 0)))
                self.assertTrue(report["passed"])
                self.assertEqual(model.position, 0)
                self.assertFalse(report["public_checkpoint_validated"])

    def run_validation(self, model=None, ref=reference, sequences=((1, 2), (0, 1)), **kwargs):
        return validate_olmoe(model or TinyDecoder(), ref, sequences,
                              reference_identity=IDENTITY, **kwargs)

    def test_teacher_forcing_resets_and_report(self):
        model = TinyDecoder()
        model.total = 99
        report = self.run_validation(model)
        self.assertTrue(report["passed"])
        self.assertEqual(report["positions"], 4)
        self.assertEqual(model.calls, [[1], [2], [0], [1]])
        self.assertEqual(model.total, 0)
        self.assertEqual(report["metrics"]["argmax_agreement"], 1)
        self.assertEqual(report["metrics"]["root_mean_square_error"], 0)
        self.assertFalse(report["public_checkpoint_validated"])
        self.assertEqual(report["reference"], IDENTITY)
        self.assertEqual(report["model"]["metadata_fingerprint"], "fixture-only")
        self.assertIn("machine", report["hardware"])
        self.assertGreaterEqual(report["timing"]["decoder_seconds"], 0)
        self.assertEqual(len(report["sequences"][0]["decoder_step_seconds"]), 2)
        json.dumps(report, allow_nan=False)

    def test_numerical_thresholds_and_argmax_are_separate(self):
        report = self.run_validation(TinyDecoder(0.25), atol=0, rtol=0)
        self.assertFalse(report["passed"])
        self.assertEqual(report["metrics"]["argmax_agreement"], 1)
        self.assertEqual(report["metrics"]["mean_absolute_error"], 0.25)
        self.assertEqual(report["metrics"]["root_mean_square_error"], 0.25)
        self.assertTrue(self.run_validation(TinyDecoder(0.25), atol=0.3, rtol=0)["passed"])
        report = self.run_validation(ref=lambda ids: reference(ids)[:, ::-1], atol=100)
        self.assertTrue(report["metrics"]["allclose"])
        self.assertFalse(report["passed"])

    def test_strict_reference_and_decoder_output_validation(self):
        for value in (np.zeros((1, 2, 3)), np.zeros(3), np.zeros((2, 3), dtype=int),
                      np.full((2, 3), np.nan), np.full((2, 3), np.inf),
                      np.zeros((2, 3), dtype=complex)):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                model = TinyDecoder()
                with self.assertRaises(ValueError):
                    self.run_validation(model, ref=lambda ids: value, sequences=((1, 2),))
                self.assertEqual(model.total, 0)
        for value in (np.zeros((1, 3)), np.array([1, 2, np.inf]), np.array([1, 2, 3])):
            model = TinyDecoder()
            model.forward = lambda ids: value
            with self.assertRaises(ValueError):
                self.run_validation(model)

    def test_bad_inputs_and_thresholds(self):
        for sequences in ([], [[]], [[True]], [[1.5]], [[-1]], [[3]], [[0] * 5], [[[1]]]):
            with self.subTest(sequences=sequences), self.assertRaises(ValueError):
                self.run_validation(sequences=sequences)
        for kwargs in ({"atol": -1}, {"rtol": float("nan")}, {"atol": True},
                       {"minimum_argmax_agreement": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_validation(**kwargs)
        with self.assertRaises(ValueError):
            validate_olmoe(TinyDecoder(), reference, [[1]], reference_identity={})

    def test_exception_cleanup_and_synchronization(self):
        model = TinyDecoder()
        def broken(ids):
            raise RuntimeError("reference failed")
        with self.assertRaisesRegex(RuntimeError, "reference failed"):
            self.run_validation(model, ref=broken)
        self.assertEqual(model.total, 0)
        calls = []
        report = self.run_validation(reference_synchronize=lambda: calls.append("sync"))
        self.assertEqual(calls, ["sync"] * 4)
        self.assertTrue(report["timing"]["reference_synchronized"])


if __name__ == "__main__":
    unittest.main()
