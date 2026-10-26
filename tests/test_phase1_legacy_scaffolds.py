"""Retired Phase 1 scaffolds must not claim or synthesize model execution."""
from pathlib import Path
import tempfile
import unittest

from openagent.adaptive_engine.a12_unified_inference import (
    TieredExpertStore, UnifiedInferenceExecution)
from openagent.adaptive_engine.a14_lookahead_prefetch import PrefetchAndLearnExecutor
from openagent.adaptive_engine.a15_model_integration_api import (
    AdaptiveEngine, AdaptiveEngineAPI)


class LegacyScaffoldTests(unittest.TestCase):
    def test_a12_forward_fails_before_router_or_state_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TieredExpertStore(Path(directory))
            execution = UnifiedInferenceExecution({'vocab_size': 5}, store)
            calls = []
            execution.router = lambda *_args: calls.append(True)
            before_history = dict(execution.routing_history)
            with self.assertRaisesRegex(RuntimeError, "retired routing/placement scaffold"):
                execution.forward([1, 2])
            self.assertEqual(calls, [])
            self.assertEqual(execution.routing_history, before_history)
            self.assertEqual(execution.batch_count, 0)
            with self.assertRaisesRegex(RuntimeError, "no model tensor executor"):
                execution._compute_forward({}, [], [1])

    def test_a12_placement_hit_metrics_are_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TieredExpertStore(Path(directory))
            store.record_placement_hit("expert-1", "ram")
            store.record_placement_hit("expert-2", "disk")
            self.assertEqual(store.placement_hits, {'vram': 0, 'ram': 1, 'disk': 1})
            with self.assertRaisesRegex(ValueError, "placement tier"):
                store.record_placement_hit("expert-3", "remote")

    def test_a14_prefetch_facade_cannot_return_placeholder_logits_or_record_empty_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = PrefetchAndLearnExecutor(None, TieredExpertStore(Path(directory)))
            with self.assertRaisesRegex(RuntimeError, "does not execute model tensors"):
                executor.forward([1], 0)
            with self.assertRaisesRegex(RuntimeError, "production route trace"):
                executor._get_routing(0)
            self.assertEqual(executor.learning_cache.routing_history, {})

    def test_a15_reports_unobserved_state_instead_of_fabricated_success(self):
        api = AdaptiveEngineAPI()
        self.assertFalse(api.get_status()['phase1_complete'])
        self.assertEqual(api.get_status()['status'], 'unavailable')
        self.assertEqual(api.list_models()['models'], [])
        self.assertFalse(api.get_placement()['observed'])
        self.assertFalse(api.get_routing_heat()['observed'])
        self.assertFalse(api.get_learning_stats()['observed'])
        self.assertIsNone(api.get_hardware_plan()['recommendation'])

    def test_a15_placeholder_generation_cannot_return_fake_text_or_metrics(self):
        engine = AdaptiveEngine({'olmoe': {'vocab_size': 10}})
        with self.assertRaisesRegex(RuntimeError, "placeholder tokenization"):
            engine.generate('olmoe', 'hello')
        with self.assertRaisesRegex(RuntimeError, "configuration shells"):
            engine.start()
        self.assertEqual(engine.metrics.metrics_history, [])


if __name__ == '__main__':
    unittest.main()
