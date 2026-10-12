import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.olmoe import OlmoeModel
from openagent.adaptive_engine.olmoe_autotune import tune_olmoe_threads, save_profile, load_profile
from tests.test_olmoe_batch import checkpoint


@unittest.skipUnless(importlib.util.find_spec("threadpoolctl"),
                     "optional autotune dependency threadpoolctl unavailable")
class AutotuneTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        manifest = checkpoint(self.root, True)
        self.model = OlmoeModel(self.root, manifest, expert_cache_bytes=4096)
        self.addCleanup(self.model.close)

    def tune(self, **kwargs):
        return tune_olmoe_threads(self.model, [1, 2], no_active_external_users=True, **kwargs)

    def test_real_checkpoint_all_candidates_match(self):
        expected = self.model.generate_ids([1, 2], max_tokens=2)
        self.model.reset()
        profile = self.tune()
        status = self.model.experts.status()
        runtime = profile["identity"]["runtime"]
        for name in ("io_mode", "capacity_bytes", "prefetch_workers", "prefetch_depth"):
            self.assertEqual(runtime[name], status["store"][name])
        self.assertEqual(runtime["preferred_kernel"], status["preferred_kernel"])
        self.assertEqual([item["name"] for item in runtime["selected_backends"]],
                         ["numpy-reference"] * 2)
        self.assertEqual(profile["baseline_token_ids"], expected)
        for result in profile["results"]:
            self.assertTrue(result["accepted"])
            self.assertEqual(len(result["measurements"]), 2)
            for run in result["measurements"]:
                self.assertEqual(run["token_ids"], expected)
                self.assertGreater(run["elapsed_seconds"], 0)
                self.assertGreaterEqual(run["ttft_seconds"], 0)
                self.assertAlmostEqual(run["tokens_per_second"], len(expected) / run["elapsed_seconds"])
        self.assertEqual(self.model.position, 0)
        for array in self.model._k + self.model._v:
            self.assertFalse(np.any(array))

    def test_invalid_parameters_and_session(self):
        for options in ({"candidates": ()}, {"candidates": (1, 1)}, {"candidates": (True,)},
                        {"candidates": (0,)}, {"max_tokens": 0}, {"max_tokens": 1.5},
                        {"warmup": -1}, {"warmup": True}, {"runs": 0}, {"runs": False},
                        {"max_tokens": 10}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.tune(**options)
        for ids in ([], [True], [1.0], [-1], [13], [[1]], "12"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                tune_olmoe_threads(self.model, ids, no_active_external_users=True)
        with self.assertRaisesRegex(ValueError, "external"):
            tune_olmoe_threads(self.model, [1])
        self.model.forward([1])
        with self.assertRaisesRegex(ValueError, "idle"):
            self.tune()
        self.assertEqual(self.model.position, 1)

    def test_changed_output_rejected_including_warmup(self):
        calls = []

        def stream(*args, **kwargs):
            self.assertEqual(self.model.position, 0)
            self.model.position = 1
            calls.append(1)
            yield 3 if len(calls) == 3 else 2

        with patch.object(self.model, "generate_id_stream", side_effect=stream):
            profile = self.tune(candidates=(1, 2), warmup=1, runs=1)
        self.assertTrue(profile["results"][0]["accepted"])
        self.assertFalse(profile["results"][1]["accepted"])
        self.assertEqual(profile["recommended_threads"], 1)
        self.assertEqual(self.model.position, 0)

    def test_reset_finally_on_inference_and_scope_failure(self):
        from threadpoolctl import threadpool_info

        before = threadpool_info()
        def broken(*args, **kwargs):
            self.model.forward([1])
            raise RuntimeError("inference failure")
            yield 1

        with patch.object(self.model, "generate_id_stream", side_effect=broken):
            with self.assertRaisesRegex(RuntimeError, "inference failure"):
                self.tune()
        self.assertEqual(self.model.position, 0)
        self.assertEqual(threadpool_info(), before)
        for array in self.model._k + self.model._v:
            self.assertFalse(np.any(array))
        with patch("openagent.adaptive_engine.olmoe_autotune.blas_thread_scope", side_effect=RuntimeError("scope")):
            with self.assertRaisesRegex(RuntimeError, "scope"):
                self.tune()
        self.assertEqual(self.model.position, 0)

    def test_timing_uses_counter_without_rounding(self):
        with patch.object(self.model, "generate_id_stream", side_effect=lambda *a, **k: iter([3, 4])), patch(
                "openagent.adaptive_engine.olmoe_autotune.perf_counter", side_effect=[10.0, 10.25, 11.0]):
            profile = self.tune(candidates=(1,), warmup=0, runs=1)
        run = profile["results"][0]["measurements"][0]
        self.assertEqual(run["ttft_seconds"], 0.25)
        self.assertEqual(run["elapsed_seconds"], 1.0)
        self.assertEqual(run["tokens_per_second"], 2.0)

    def test_cache_validation_and_atomic_failure(self):
        profile = self.tune(candidates=(1,), warmup=0, runs=1)
        path = self.root / "profile.json"
        save_profile(profile, path)
        self.assertEqual(load_profile(path, expected_identity=profile["identity"]), profile)
        changed = copy.deepcopy(profile["identity"])
        changed["environment"]["numpy_version"] = "other"
        with self.assertRaisesRegex(ValueError, "identity"):
            load_profile(path, expected_identity=changed)
        original = path.read_bytes()
        with patch("openagent.adaptive_engine.olmoe_autotune.os.replace", side_effect=OSError("replace")):
            with self.assertRaises(OSError):
                save_profile(profile, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        for mutate in (
                lambda p: p.update(scope="global-best"),
                lambda p: p.update(schema_version=True),
                lambda p: p.update(cache_key="wrong"),
                lambda p: p.update(recommended_threads=2),
                lambda p: p["results"][0]["measurements"][0].update(elapsed_seconds=float("nan")),
                lambda p: p["results"][0]["measurements"][0].update(tokens_per_second=0),
                lambda p: p["identity"]["configuration"].update(runs=True),
                lambda p: p.update(schema_version=1),
                lambda p: p["identity"].pop("runtime"),
                lambda p: p["identity"]["runtime"].update(capacity_bytes=True),
                lambda p: p["identity"]["runtime"].update(prefetch_depth=0),
                lambda p: p["identity"]["runtime"].update(selected_backends=[]),
                lambda p: p.update(extra=1)):
            bad = copy.deepcopy(profile)
            mutate(bad)
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_profile(path, expected_identity=profile["identity"])
        path.write_text('{"scope": 1, "scope": 2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_profile(path, expected_identity=profile["identity"])

    def test_cache_rejects_other_execution_strategies(self):
        profile = self.tune(candidates=(1,), warmup=0, runs=1)
        path = self.root / "profile.json"
        save_profile(profile, path)
        for field, value in (("io_mode", "buffered"), ("capacity_bytes", 8192),
                             ("prefetch_workers", 1), ("prefetch_depth", 16),
                             ("preferred_kernel", "numpy-reference")):
            expected = copy.deepcopy(profile["identity"])
            expected["runtime"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                load_profile(path, expected_identity=expected)
        expected = copy.deepcopy(profile["identity"])
        expected["runtime"]["selected_backends"][0]["name"] = "other-backend"
        with self.assertRaisesRegex(ValueError, "identity"):
            load_profile(path, expected_identity=expected)
        with OlmoeModel(self.root, self.model.manifest, expert_cache_bytes=8192,
                        io_mode="buffered", prefetch_workers=1, prefetch_depth=3,
                        preferred_kernel="numpy-reference") as other:
            other_profile = tune_olmoe_threads(other, [1, 2], candidates=(1,),
                                               warmup=0, runs=1, no_active_external_users=True)
        self.assertNotEqual(profile["cache_key"], other_profile["cache_key"])
        with self.assertRaisesRegex(ValueError, "identity"):
            load_profile(path, expected_identity=other_profile["identity"])

    def test_missing_runtime_capability_errors_before_inference(self):
        status = self.model.experts.status()
        for field in ("io_mode", "capacity_bytes", "prefetch_workers", "prefetch_depth"):
            bad = copy.deepcopy(status)
            del bad["store"][field]
            with self.subTest(field=field), patch.object(self.model.experts, "status", return_value=bad), patch.object(
                    self.model, "generate_id_stream") as generate:
                with self.assertRaisesRegex(ValueError, "capability"):
                    self.tune()
                generate.assert_not_called()
        with patch.object(self.model, "experts", object()):
            with self.assertRaisesRegex(ValueError, "capability"):
                self.tune()
