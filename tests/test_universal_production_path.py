"""End-to-end production universal path on a real compiled native checkpoint.

A tiny OLMoE checkpoint is built, its native executor is compiled, and the
model is opened through ``AdaptiveExecutionEngine.open_universal_model`` with
the production adapter registry.  Generated tokens are compared with the
independent NumPy model; rollback, receipts, Phase 1B telemetry, the learning
cache and cleanup are checked against real runtime state.
"""
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.olmoe import OlmoeModel
from openagent.adaptive_engine.runtime import AdaptiveExecutionEngine
from openagent.adaptive_engine.universal_adapters import (
    admitted_universal_model_types, resolve_checkpoint_route)
from openagent.adaptive_engine.universal_engine import (
    GenerationCancelled, ModelRoute, PublicationFailed)
from openagent.storage import OpenAgentStore
from tests.test_native_segment_executor import build_executor
from tests.test_olmoe_batch import checkpoint


def _oracle(root, manifest, prompt, count):
    with OlmoeModel(root, manifest, context_tokens=8, expert_cache_bytes=1024 ** 2) as model:
        pending, expected = prompt, []
        for _ in range(count):
            token = int(np.argmax(model.forward(pending)))
            expected.append(token)
            pending = [token]
    return expected


class UniversalProductionPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = tempfile.mkdtemp(prefix="oa-universal-native-")
        cls.library = build_executor(cls.compiled)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.compiled, ignore_errors=True)

    def setUp(self):
        self.threads_before = {thread.ident for thread in threading.enumerate()}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root = base / "checkpoint"
        self.root.mkdir()
        self.manifest = checkpoint(self.root, True)
        self.manifest.write(self.root / "manifest.json")
        data = base / "data"
        data.mkdir()
        self.engine = AdaptiveExecutionEngine(OpenAgentStore(data))
        self.options = dict(library=str(self.library), context_tokens=8, cache_mib=1,
                            resident_mib=1, transfer_mib=1, staging_mib=1)

    def tearDown(self):
        self.engine.close()
        leaked = [thread for thread in threading.enumerate()
                  if thread.ident not in self.threads_before and thread.is_alive()]
        self.assertEqual(leaked, [], "universal path leaked threads")

    def open(self, **extra):
        return self.engine.open_universal_model(self.root, **self.options, **extra)

    def test_registered_types_and_resolvers_equal_admitted_identities(self):
        admitted = admitted_universal_model_types()
        self.assertEqual(len(admitted), 14)
        self.assertEqual(self.engine.universal.registered_model_types(), admitted)
        self.assertEqual(self.engine.universal.resolver_model_types(), admitted)
        self.assertIn("kimi_k3", admitted)
        self.assertIn("kimi_linear", admitted)

    def test_phase1b_facade_construction_failure_closes_base_owner(self):
        class Session:
            def __init__(self):
                self.close_attempts = 0
                self.closed = False

            def close(self):
                self.close_attempts += 1
                if self.close_attempts == 1:
                    raise OSError("injected close failure")
                self.closed = True

        session = Session()
        with patch.object(self.engine.universal, "open", return_value=session), \
             patch("openagent.adaptive_engine.runtime.wrap_universal_session_with_phase1b",
                   side_effect=ValueError("injected facade failure")):
            with self.assertRaisesRegex(ValueError, "injected facade failure") as raised:
                self.engine.open_universal_model(self.root, **self.options)

        self.assertEqual(session.close_attempts, 1)
        retry_cleanup = getattr(raised.exception, "cleanup", None)
        self.assertTrue(callable(retry_cleanup))
        retry_cleanup()
        self.assertEqual(session.close_attempts, 2)
        self.assertTrue(session.closed)

    def test_resolver_reads_checkpoint_dtype_and_packed_manifest(self):
        route = resolve_checkpoint_route({"model_type": "olmoe"}, self.root)
        self.assertEqual(route, ModelRoute("olmoe", "olmoe", "native", "fp32", "F32", "int8-row"))
        second = self.root / "second.json"
        self.manifest.write(second)
        with self.assertRaisesRegex(ValueError, "exactly one sparse-expert manifest"):
            resolve_checkpoint_route({"model_type": "olmoe"}, self.root)

    def test_generation_matches_independent_numpy_model(self):
        expected = _oracle(self.root, self.manifest, [1, 3], 2)
        session = self.open()
        with patch.object(OlmoeModel, "forward", side_effect=AssertionError("no NumPy forward")):
            result = session.generate([1, 3], max_tokens=2)
        self.assertEqual(result["generated_ids"], expected[:len(result["generated_ids"])])
        self.assertTrue(result["generated_ids"])
        self.assertFalse(result["numerical_parity_verified"])

    def test_unleased_admitted_format_uses_direct_explicit_kernel_without_library(self):
        from openagent.adaptive_engine.olmoe import pack_olmoe_experts

        manifest = pack_olmoe_experts(self.root, self.root / "experts.i2", dtype="int2-row",
                                      model_id="universal-direct-int2")
        manifest_path = self.root / "manifest.i2.json"
        manifest.write(manifest_path)
        route = ModelRoute("olmoe", "olmoe", "native", "fp32", "F32", "int2-row")
        options = dict(manifest=str(manifest_path), context_tokens=8, cache_mib=1)

        session = self.engine.open_universal_model(self.root, route=route, **options)
        self.assertEqual(session.owner.persisted_format, "int2-row")
        self.assertEqual(session.owner._provider.preferred_kernel, "int2-row-numba")
        result = session.generate([1, 3], max_tokens=2)
        self.assertTrue(result["generated_ids"])
        self.assertEqual(session.owner.position, result["context_tokens"])

        # F32 tensors also have a canonical BF16 route row, so the route gate
        # alone cannot detect a caller that lies about this checkpoint's dtype.
        wrong_source_route = ModelRoute("olmoe", "olmoe", "native", "fp32", "BF16", "int2-row")
        with self.assertRaisesRegex(ValueError, "explicit route source format 'BF16'.*inventory 'F32'"):
            self.engine.open_universal_model(self.root, route=wrong_source_route, **options)

        before_cancel = session.owner.position
        calls = 0

        def cancel_after_prefill():
            nonlocal calls
            calls += 1
            return calls >= 3

        with self.assertRaises(GenerationCancelled):
            session.generate([2], max_tokens=2, cancelled=cancel_after_prefill)
        self.assertEqual(session.owner.position, before_cancel)

    def test_phase1b_metrics_are_computed_from_runtime_counters(self):
        session = self.open()
        before = session.metrics()
        self.assertTrue(before["expert_streaming"])
        self.assertIsNone(before["batch_union_reduction"])
        session.generate([1, 3, 4], max_tokens=2)
        metrics = session.metrics()
        runtime = session.owner.expert_telemetry_sources()[0]
        compute, store = runtime.status()["compute"], runtime.store.status()
        self.assertGreater(compute["expert_calls"], 0)
        self.assertEqual(metrics["expert_route_requests"], compute["expert_calls"])
        self.assertEqual(metrics["unique_expert_loads"], compute["unique_expert_loads"])
        self.assertGreater(compute["batched_expert_calls"], 0)
        self.assertAlmostEqual(metrics["batch_union_reduction"],
                               1 - compute["batched_unique_expert_loads"] / compute["batched_expert_calls"])
        self.assertGreater(metrics["batch_union_reduction"], 0.0)
        self.assertEqual(metrics["cache_misses"], store["misses"])
        self.assertEqual(metrics["async_reads_submitted"], store["prefetch_submitted"])
        self.assertEqual(metrics["bytes_read"], store["bytes_read"])
        # The native leased loop now feeds the route predictor (learning cache).
        observations = sum(layer["observations"] for layer in runtime.status()["routes"].values())
        self.assertEqual(observations, compute["expert_calls"])

    def test_cancellation_and_publication_failure_roll_back_decoder_state(self):
        session = self.open()
        session.generate([1, 3], max_tokens=1)
        decoder = session.owner._decoder
        position = decoder.position
        state = decoder.snapshot_state()
        calls = {"count": 0}

        def cancel_after_first_token():
            calls["count"] += 1
            return calls["count"] > 2

        with self.assertRaises(GenerationCancelled):
            session.generate([5], max_tokens=4, continue_session=True, cancelled=cancel_after_first_token)
        self.assertEqual(decoder.position, position)
        np.testing.assert_array_equal(decoder.snapshot_state()["range_state"]["keys"][0],
                                      state["range_state"]["keys"][0])

        def reject(result):
            raise OSError("sink closed")

        with self.assertRaises(PublicationFailed):
            session.generate([5], max_tokens=2, continue_session=True, publish=reject)
        self.assertEqual(decoder.position, position)
        with self.assertRaises(GenerationCancelled):
            session.generate([5], max_tokens=1, cancelled=lambda: True)
        self.assertEqual(decoder.position, position)

    def test_stream_publishes_tokens_and_rolls_back_when_abandoned(self):
        session = self.open()
        events = list(session.stream([1, 3], max_tokens=2))
        self.assertEqual(events[-1]["event"], "result")
        tokens = [event["token_ids"][0] for event in events if event["event"] == "tokens"]
        self.assertEqual(tokens, events[-1]["result"]["generated_ids"])
        position = session.owner._decoder.position
        stream = session.stream([4], max_tokens=3, continue_session=True)
        next(stream)
        stream.close()
        self.assertEqual(session.owner._decoder.position, position)

    def test_receipts_restore_owned_state_and_reject_foreign_receipts(self):
        session = self.open()
        session.generate([1, 3], max_tokens=1)
        receipt = session.snapshot()
        position = session.owner._decoder.position
        session.generate([4], max_tokens=1, continue_session=True)
        self.assertNotEqual(session.owner._decoder.position, position)
        session.restore(receipt)
        self.assertEqual(session.owner._decoder.position, position)
        for foreign in (receipt.as_dict(), replace(receipt), None, object()):
            with self.subTest(foreign=type(foreign).__name__), self.assertRaises(ValueError):
                session.restore(foreign)
        session.release_snapshot(receipt)
        with self.assertRaisesRegex(ValueError, "unknown, released, or forged"):
            session.restore(receipt)

    def test_warm_reuse_uses_canonical_options_and_one_facade(self):
        first = self.open()
        self.assertIs(self.open(), first)
        different = self.engine.open_universal_model(self.root, **{**self.options, "cache_mib": 2})
        self.assertIsNot(different, first)
        with self.assertRaisesRegex(ValueError, "undeclared adapter options"):
            self.engine.open_universal_model(self.root, **self.options, model_family="olmoe")

    def test_phase1b_warm_reuse_requires_identical_facade_options(self):
        budget = 1024 ** 2
        first = self.open(hot_budget_bytes=budget)
        self.assertIs(self.open(hot_budget_bytes=budget), first)
        with self.assertRaisesRegex(ValueError, "different hot_budget_bytes"):
            self.open(hot_budget_bytes=budget * 2)
        with self.assertRaisesRegex(ValueError, "requires phase1b=True"):
            self.engine.open_universal_model(self.root, phase1b=False,
                hot_budget_bytes=budget, **self.options)
        with self.assertRaisesRegex(TypeError, "phase1b must be a boolean"):
            self.engine.open_universal_model(self.root, phase1b=1, **self.options)
        for invalid in (-1, True):
            with self.subTest(hot_budget_bytes=invalid), self.assertRaisesRegex(
                    ValueError, "hot_budget_bytes must be"):
                self.engine.open_universal_model(self.root, hot_budget_bytes=invalid,
                    **self.options)

    def test_unadmitted_explicit_route_is_rejected_before_opening(self):
        route = ModelRoute("olmoe", "olmoe", "device", "fp32", "F32", "int8-row")
        with patch("openagent.adaptive_engine.universal_adapters.OlmoeNativeOwner") as owner:
            with self.assertRaises(ValueError):
                self.engine.open_universal_model(self.root, route=route, **self.options)
            owner.assert_not_called()

    def test_learning_cache_is_isolated_per_checkpoint_and_reloaded(self):
        session = self.open()
        session.generate([1, 3, 4], max_tokens=2)
        state = Path(session.metrics()["state_directory"])
        self.assertNotIn(self.root.resolve(), state.parents)
        self.assertEqual(state.parent, self.engine.phase1b_root)
        session.close()
        self.assertTrue(list(state.glob("route-profile-*.json")))
        reopened = self.open()
        self.assertEqual(reopened.metrics()["route_profiles_loaded"], 1)
        self.assertEqual(Path(reopened.metrics()["state_directory"]), state)

        other = self.root.parent / "other-checkpoint"
        other.mkdir()
        checkpoint(other, False).write(other / "manifest.json")
        second = self.engine.open_universal_model(other, **self.options)
        self.assertNotEqual(second.metrics()["state_directory"], str(state))

    def test_dense_matrix_decode_budget_is_explicit_and_honoured(self):
        from openagent.adaptive_engine.expert_runtime import SparseExpertManifest
        from openagent.adaptive_engine.native_olmoe_segments import NativeOlmoeEdge
        with self.assertRaisesRegex(MemoryError, "source matrix decoding exceeds admission"):
            OlmoeModel(self.root, self.manifest, context_tokens=8, expert_cache_bytes=1024 ** 2,
                       max_matrix_decode_bytes=64)
        with self.assertRaisesRegex(ValueError, "max_matrix_decode_bytes"):
            OlmoeModel(self.root, self.manifest, context_tokens=8, expert_cache_bytes=1024 ** 2,
                       max_matrix_decode_bytes=0)
        with NativeOlmoeEdge(self.root, self.library, 1024 ** 2, max_bytes=1024 ** 2,
                             max_conversion_bytes=64 * 1024 ** 2) as edge:
            self.assertEqual(edge._matrix_decode_bytes, 64 * 1024 ** 2)
        session = self.open(conversion_mib=64)
        self.assertEqual(session.owner._edge._matrix_decode_bytes, 64 * 1024 ** 2)
        self.assertTrue(session.generate([1, 3], max_tokens=1)["generated_ids"])
        with self.assertRaisesRegex(ValueError, "conversion_mib"):
            self.engine.open_universal_model(self.root, **{**self.options, "conversion_mib": 0})

    def test_hot_budget_pins_learned_experts_from_real_route_history(self):
        session = self.open(hot_budget_bytes=1024 ** 2)
        session.generate([1, 3, 4], max_tokens=2)
        self.assertGreater(session.metrics()["pinned_experts"], 0)


if __name__ == "__main__":
    unittest.main()
