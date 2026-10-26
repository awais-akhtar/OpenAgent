from __future__ import annotations

import unittest

from openagent.adaptive_engine.phase1_a07_format_loader_placement import (
    get_format_spec,
    report_format_placement_matrix,
    require_format_placement,
)
from openagent.adaptive_engine.source_scope import (
    require_format_execution_consumer,
    require_source_matrix_admission,
)


class Phase1A07FormatPlacementTests(unittest.TestCase):
    def test_native_registry_exposes_exact_geometry_and_scale_contract(self) -> None:
        spec = require_format_placement("mxfp4-e2m1-g32-ue8m0", "native")

        self.assertEqual(spec.block_size, 32)
        self.assertEqual(spec.scale_dtype, "ue8m0")
        self.assertEqual(spec.loader_module, "native_mxfp4_expert")

    def test_restricted_device_placement_fails_before_consumer_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "no admitted CUDA kernel"):
            require_format_placement("iq3-format6-98-v1", "device")

    def test_unknown_formats_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in the P07 format registry"):
            get_format_spec("invented-format")

    def test_source_consumer_path_uses_format_placement_gate(self) -> None:
        admission = require_format_execution_consumer(
            family="olmoe", backend="native", persisted_format="float32"
        )

        self.assertEqual(admission["placement_contract"]["loader_module"], "native_checkpoint")
        self.assertEqual(
            admission["placement_contract"]["admitted_placements"],
            ("device", "native", "public-range"),
        )

    def test_public_olmoe_int8_resolves_to_its_real_segment_edge_loader(self) -> None:
        spec = require_format_placement("int8-row", "public-range")
        self.assertEqual(
            spec.loader_for("public-range"),
            ("segment_edge_adapters", "NativeOlmoePublicAdapters"),
        )
        contract = require_format_execution_consumer(
            family="olmoe", backend="public-range", persisted_format="int8-row"
        )
        self.assertEqual(
            contract["placement_contract"]["placement_loader"],
            ("segment_edge_adapters", "NativeOlmoePublicAdapters"),
        )

    def test_k3_public_mxfp4_resolves_to_checkpoint_bound_publisher_loader(self) -> None:
        spec = require_format_placement("mxfp4-e2m1-g32-ue8m0", "public-range")
        self.assertEqual(
            spec.loader_for("public-range"),
            ("publisher_checkpoint.k3_adapter", "load_k3"),
        )
        contract = require_format_execution_consumer(
            family="kimi-k3",
            backend="public-range",
            persisted_format="mxfp4-e2m1-g32-ue8m0",
        )
        self.assertEqual(
            contract["consumer"],
            ("publisher_public_adapters.NativePublisherPublicAdapters",),
        )

    def test_exact_fp8_checkpoint_route_is_distinct_from_unbound_sparse_manifest(self) -> None:
        self.assertTrue(require_source_matrix_admission(
            family="qwen4",
            model_type="qwen4_exp",
            backend="device",
            arithmetic="fp32",
            source_format="F8_E4M3",
            persisted_format="fp8-e4m3-b128-f32",
        ))
        with self.assertRaisesRegex(ValueError, "packed expert manifests carry no source-dtype"):
            require_format_execution_consumer(
                family="qwen4", backend="device", persisted_format="fp8-e4m3-b128-f32"
            )

    def test_report_contains_format_and_placement_dimensions(self) -> None:
        report = report_format_placement_matrix()

        self.assertIn("e8-lattice-caller-v1", report)
        self.assertIn("iq3-format6-98-v1", report)
        self.assertIn("REJECT", report)


if __name__ == "__main__":
    unittest.main()
