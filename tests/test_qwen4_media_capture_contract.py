"""Source contracts for exact Qwen4 publisher-media capture routes."""
from pathlib import Path
import unittest
import numpy as np

from openagent.adaptive_engine.device_capture import CaptureError
from openagent.adaptive_engine.qwen4_media_capture import (
    ROUTE_FIELDS,
    _canonical_route,
    _required_source_paths,
    _validate_media_configuration,
    load_qwen4_publisher_candidate,
)
from openagent.adaptive_engine.independent_qwen4_vision import (
    IndependentQwen4VisionEncoder,
    _shapes as _independent_vision_shapes,
)
from openagent.adaptive_engine.source_scope import source_scope_matrix


class Qwen4MediaCaptureContractTests(unittest.TestCase):
    def test_exact_target_contains_twelve_publisher_media_rows(self):
        rows = [row for row in source_scope_matrix()["execution_cases"]
                if row["family"] == "qwen4" and row["model_type"] == "qwen4_exp"
                and row.get("combination_status") == "verified-admitted"
                and row.get("admission") == "verified-admitted"]
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["backend"] for row in rows},
                         {"native", "device", "public-range"})
        self.assertEqual({row["source_format"] for row in rows},
                         {"F32", "F16", "BF16", "F8_E4M3"})
        for row in rows:
            route = {key: row[key] for key in ROUTE_FIELDS}
            self.assertEqual(_canonical_route(route)["case_id"], row["case_id"])

    def test_source_format_persisted_format_pairs_are_exact(self):
        rows = [row for row in source_scope_matrix()["execution_cases"]
                if row["family"] == "qwen4" and row["model_type"] == "qwen4_exp"
                and row.get("combination_status") == "verified-admitted"]
        pairs = {(row["source_format"], row["persisted_format"]) for row in rows}
        self.assertEqual(pairs, {
            ("F32", "float32"), ("F16", "float32"), ("BF16", "float32"),
            ("F8_E4M3", "fp8-e4m3-b128-f32"),
        })

    def test_route_rejects_text_aliases_and_unadmitted_format_pairs(self):
        with self.assertRaisesRegex(CaptureError, "unsupported Qwen4 publisher"):
            _canonical_route({"family": "qwen4", "model_type": "qwen4_exp_text",
                              "backend": "native", "arithmetic": "fp32",
                              "source_format": "F32", "persisted_format": "float32"})
        with self.assertRaisesRegex(CaptureError, "unsupported Qwen4 publisher"):
            _canonical_route({"family": "qwen4", "model_type": "qwen4_exp",
                              "backend": "device", "arithmetic": "fp32",
                              "source_format": "F8_E4M3", "persisted_format": "float32"})

    def test_candidate_source_pins_cover_actual_media_and_backend_owners(self):
        for backend, expected in {
            "native": {"qwen4_multimodal.py", "qwen4_vision.py", "native_qwen4.c",
                       "native_media_ops.c"},
            "device": {"qwen4_multimodal.py", "qwen4_vision.py", "torch_qwen4_decoder.py",
                       "device_qwen4_forward.py", "native_media_ops.c"},
            "public-range": {"qwen4_multimodal.py", "qwen4_vision.py",
                             "native_qwen4_public_text.py", "native/qwen4_public_model.c",
                             "native_media_ops.c"},
        }.items():
            paths = _required_source_paths(backend)
            self.assertEqual(len(paths), len(set(paths)))
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue(expected <= {path.relative_to(
                Path(__file__).resolve().parents[1] / "openagent" / "adaptive_engine").as_posix()
                for path in paths})

    def test_candidate_loader_fails_closed_before_checkpoint_access(self):
        with self.assertRaisesRegex(CaptureError, "exact bounded Qwen4 publisher loader options"):
            load_qwen4_publisher_candidate(offline_root="missing", options={})

    def test_comparison_config_requires_exact_route_and_production_adapter(self):
        row = next(row for row in source_scope_matrix()["execution_cases"]
                   if row["family"] == "qwen4" and row["model_type"] == "qwen4_exp"
                   and row["backend"] == "native" and row["source_format"] == "F32"
                   and row["persisted_format"] == "float32"
                   and row.get("combination_status") == "verified-admitted")
        route = {key: row[key] for key in ROUTE_FIELDS}
        comparison = {
            "schema": "openagent.qwen4-media-comparison-configuration.v1",
            "artifacts": [], "candidate": {
                "factory": "openagent.adaptive_engine.qwen4_media_capture:create_qwen4_media_candidate",
                "options": {"route": dict(route), "metadata": {}, "config": {},
                            "max_cache_export_bytes": 1, "max_routing_export_bytes": 1,
                            "max_observation_bytes": 1,
                            "decoder_options": {"route": dict(route)}},
            },
            "oracle": {"factory": "independent:publisher_media_oracle", "options": {}},
            "fixture_config": "fixture.json", "max_observation_bytes": 1,
            "seed": 0, "generation_steps": 1, "tolerance": {"atol": 0.0, "rtol": 0.0},
        }
        configuration = {"schema": "openagent.phase1.qwen4-media-capture.v1",
                          "route": route, "comparison": comparison}
        with self.assertRaisesRegex(CaptureError, "source closure is not pinned"):
            _validate_media_configuration(configuration)
        comparison["candidate"]["options"]["decoder_options"]["route"]["backend"] = "device"
        with self.assertRaisesRegex(CaptureError, "exactly match"):
            _validate_media_configuration(configuration)

    def test_independent_vision_forward_and_frame_local_video_layout(self):
        config = {
            "hidden_size": 4, "depth": 1, "num_heads": 1,
            "intermediate_size": 6, "out_hidden_size": 4,
            "patch_size": 1, "temporal_patch_size": 1,
            "spatial_merge_size": 1, "in_channels": 1,
            "num_position_embeddings": 4, "hidden_act": "gelu",
            "deepstack_visual_indexes": [],
        }
        _, shapes = _independent_vision_shapes(config)
        tensors = {"model.visual." + name: np.ones(shape, dtype="float32") * 0.01
                   for name, shape in shapes.items()}
        encoder = IndependentQwen4VisionEncoder(config, tensors, max_bytes=1_000_000,
                                                 max_patches=32)
        try:
            result = encoder.forward(
                np.ones((4, 1), dtype="float32"), np.array([[2, 2, 1]], dtype="int64"))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].shape, (4, 4))
            self.assertTrue(np.isfinite(result[0]).all())
            with self.assertRaises(InterruptedError):
                encoder.forward(np.ones((4, 1), dtype="float32"),
                                np.array([[2, 2, 1]], dtype="int64"),
                                cancelled=lambda: True)
            # Cancellation and completed inference do not leave execution ownership held.
            again = encoder.forward(
                np.ones((4, 1), dtype="float32"), np.array([[2, 2, 1]], dtype="int64"))
            self.assertTrue(np.array_equal(result[0], again[0]))
        finally:
            encoder.close()

    def test_independent_vision_rejects_missing_tensor_and_bad_geometry(self):
        config = {
            "hidden_size": 4, "depth": 1, "num_heads": 1,
            "intermediate_size": 4, "out_hidden_size": 4,
            "patch_size": 1, "temporal_patch_size": 1,
            "spatial_merge_size": 1, "in_channels": 1,
            "num_position_embeddings": 4, "hidden_act": "gelu",
            "deepstack_visual_indexes": [],
        }
        _, shapes = _independent_vision_shapes(config)
        tensors = {"model.visual." + name: np.zeros(shape, dtype="float32")
                   for name, shape in shapes.items()}
        tensors.pop("model.visual.pos_embed.weight")
        with self.assertRaisesRegex(ValueError, "complete exact"):
            IndependentQwen4VisionEncoder(config, tensors, max_bytes=1_000_000)


if __name__ == "__main__":
    unittest.main()
