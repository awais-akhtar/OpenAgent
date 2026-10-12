import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_batch import OlmoeBatch


def checkpoint(root, norm_topk):
    config = dict(model_type="olmoe", hidden_size=8, intermediate_size=6,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  num_experts=3, num_experts_per_tok=2, vocab_size=13,
                  max_position_embeddings=10, rms_norm_eps=1e-5,
                  rope_theta=10000.0, norm_topk_prob=norm_topk, eos_token_id=12)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rng = np.random.default_rng(198)
    tensors = {}

    def matrix(name, shape):
        tensors[name] = rng.normal(0, 0.2, shape).astype("<f4")

    matrix("model.embed_tokens.weight", (13, 8))
    matrix("lm_head.weight", (13, 8))
    tensors["model.norm.weight"] = np.ones(8, dtype="<f4")
    for index in range(2):
        prefix = f"model.layers.{index}."
        for name, width in (("input_layernorm", 8), ("post_attention_layernorm", 8),
                            ("self_attn.q_norm", 8), ("self_attn.k_norm", 4)):
            tensors[prefix + name + ".weight"] = rng.uniform(0.8, 1.2, width).astype("<f4")
        for name, width in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4), ("o_proj", 8)):
            matrix(prefix + "self_attn." + name + ".weight", (width, 8))
        matrix(prefix + "mlp.gate.weight", (3, 8))
        # Exercise stable router ties in one layer and nontrivial routes in the other.
        if index == 0:
            tensors[prefix + "mlp.gate.weight"].fill(0)
        for expert in range(3):
            for name, shape in (("gate_proj", (6, 8)), ("up_proj", (6, 8)), ("down_proj", (8, 6))):
                matrix(prefix + f"mlp.experts.{expert}.{name}.weight", shape)
    data = bytearray()
    header = {}
    for name, tensor in tensors.items():
        begin = len(data)
        data.extend(tensor.tobytes())
        header[name] = dict(dtype="F32", shape=list(tensor.shape), data_offsets=[begin, len(data)])
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)
    return pack_olmoe_experts(root, root / "experts.i8", model_id="batch-tiny")


class OlmoeBatchTests(unittest.TestCase):
    def assert_state(self, left, right):
        self.assertEqual(left["position"], right["position"])
        self.assertEqual(left["fingerprint"], right["fingerprint"])
        for field in ("keys", "values"):
            for a, b in zip(left[field], right[field]):
                np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)

    def test_ragged_sessions_match_serial_across_cache_budgets(self):
        for norm_topk in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = checkpoint(root, norm_topk)
                region_bytes = max(r.size for layer in manifest.layers for r in layer.regions.values())
                for budget in (region_bytes, region_bytes * 6):
                    with self.subTest(norm_topk=norm_topk, budget=budget), ExitStack() as stack:
                        def load():
                            return stack.enter_context(OlmoeModel(root, manifest, expert_cache_bytes=budget))
                        model = load()
                        serial = [load(), load()]
                        engine = OlmoeBatch(model)
                        sessions = [engine.create_session(context_tokens=7), engine.create_session(context_tokens=10)]
                        original = model.snapshot_state()
                        for chunks in (([1, 2, 3], [4]), ([5], [6, 7]), ([8, 9], [2])):
                            calls = []
                            execute = model.experts.execute_batch

                            def record(layer, hidden, routes):
                                calls.append((layer, hidden.shape, routes))
                                return execute(layer, hidden, routes)

                            with patch.object(model.experts, "execute_batch", side_effect=record):
                                actual = engine.forward(sessions, chunks)
                            self.assertEqual(len(calls), 2)
                            for _, shape, routes in calls:
                                self.assertEqual(shape, (sum(map(len, chunks)), 8))
                                self.assertEqual(len(routes), shape[0])
                            self.assertTrue(all([expert for expert, _ in row] == ["0", "1"]
                                                for row in calls[0][2]))
                            expected = np.stack([reference.forward(ids) for reference, ids in zip(serial, chunks)])
                            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
                            for session, reference in zip(sessions, serial):
                                self.assert_state(session.snapshot_state(), reference.snapshot_state())
                        snapshots = [session.snapshot_state() for session in sessions]
                        for session, snapshot in zip(sessions, snapshots):
                            session.reset()
                            session.restore_state(snapshot)
                        actual = engine.step(sessions, [3, 4])
                        expected = np.stack([reference.forward([token]) for reference, token in zip(serial, [3, 4])])
                        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
                        self.assert_state(model.snapshot_state(), original)
                        # Compatible with the serial state API in both directions.
                        serial[1].restore_state(sessions[1].snapshot_state())
                        resumed = engine.create_session()
                        resumed.restore_state(serial[1].snapshot_state())
                        np.testing.assert_allclose(engine.step([resumed], [5])[0], serial[1].forward([5]),
                                                   atol=2e-6, rtol=2e-5)
                        snapshots[1]["keys"][0].fill(999)
                        self.assertFalse(np.any(sessions[1]._k[0] == 999))

    def test_invalid_batches_and_execution_failure_are_transactional(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            manifest = checkpoint(root, True)
            model = stack.enter_context(OlmoeModel(root, manifest, expert_cache_bytes=4096))
            engine = OlmoeBatch(model)
            a, b = engine.create_session(context_tokens=2), engine.create_session()
            engine.step([a, b], [1, 2])
            before = [s.snapshot_state() for s in (a, b)]
            foreign = OlmoeBatch(model).create_session()
            invalid = (([], []), ([a, b], [[1]]), ([a, a], [[1], [2]]),
                       ([a, foreign], [[1], [2]]), ([a, b], [[1], []]),
                       ([a, b], [[1], [True]]), ([a, b], [[1], [1.5]]),
                       ([a, b], [[1], [1, True]]),
                       ([a, b], [[1], [-1]]), ([a, b], [[1], [13]]),
                       ([a, b], [[1, 2], [3]]), ([a, b], [[1], [[2]]]),
                       ([a, b], [[1], ["2"]]))
            with patch.object(model.experts, "execute_batch", wraps=model.experts.execute_batch) as execute:
                for sessions, chunks in invalid:
                    with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                        engine.forward(sessions, chunks)
                    for session, snapshot in zip((a, b), before):
                        self.assert_state(session.snapshot_state(), snapshot)
                execute.assert_not_called()
            for tokens in ([1, True], [1, 2.0], [[1], [2]], [1], []):
                with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                    engine.step([a, b], tokens)
            for session, snapshot in zip((a, b), before):
                self.assert_state(session.snapshot_state(), snapshot)
            execute = model.experts.execute_batch

            def fail(layer, hidden, routes):
                if layer.endswith(".1"):
                    raise RuntimeError("expert failure")
                return execute(layer, hidden, routes)

            with patch.object(model.experts, "execute_batch", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "expert failure"):
                    engine.step([a, b], [3, 4])
            for session, snapshot in zip((a, b), before):
                self.assert_state(session.snapshot_state(), snapshot)
            bad = b.snapshot_state()
            bad["values"][-1] = np.zeros((1, 1, 1))
            with self.assertRaises(ValueError):
                b.restore_state(bad)
            self.assert_state(b.snapshot_state(), before[1])
            for capacity in (0, -1, 11, True, 1.5):
                with self.assertRaises(ValueError):
                    engine.create_session(context_tokens=capacity)
            engine.step([a, b], [3, 4])
            with self.assertRaises(ValueError):
                engine.step([a, b], [5, 6])
            self.assertEqual(b.position, 2)


if __name__ == "__main__":
    unittest.main()
