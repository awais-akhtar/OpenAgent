"""Capture-adapter contracts that can be checked without a model runtime."""
import unittest

from openagent.adaptive_engine.device_capture import CaptureError
from openagent.adaptive_engine.inkling_capture import (
    ADMITTED_BACKENDS as INKLING_BACKENDS,
    capture_inkling_route,
)
from openagent.adaptive_engine.qwen4_capture import (
    ADMITTED_BACKENDS as QWEN4_BACKENDS,
    ADMITTED_MODEL_TYPES as QWEN4_TYPES,
    capture_qwen4_route,
)
from openagent.adaptive_engine.qwen35_capture import (
    ADMITTED_BACKENDS as QWEN35_BACKENDS,
    ADMITTED_MODEL_TYPES as QWEN35_TYPES,
    capture_qwen35_route,
)
from openagent.adaptive_engine.native_family_capture import _candidate_source_paths
from openagent.adaptive_engine.source_scope import source_scope_matrix


class NativeFamilyCaptureContractTests(unittest.TestCase):
    def test_qwen_capture_advertises_only_implemented_exact_text_paths(self):
        self.assertEqual(QWEN4_TYPES, ("qwen4_exp_text",))
        self.assertEqual(QWEN35_TYPES, ("qwen3_5_moe_text",))
        self.assertEqual(QWEN4_BACKENDS, ("native", "device", "public-range"))
        self.assertEqual(QWEN35_BACKENDS, ("native",))
        self.assertEqual(INKLING_BACKENDS, ("native", "device", "public-range"))

    def test_multimodal_qwen_cannot_enter_text_capture(self):
        with self.assertRaisesRegex(CaptureError, "multimodal media evidence"):
            capture_qwen4_route("unused", model_type="qwen4_exp", source_format="F32",
                                library="unused", prompt_ids=[1])
        with self.assertRaisesRegex(CaptureError, "multimodal media evidence"):
            capture_qwen35_route("unused", model_type="qwen3_5_moe", source_format="F32",
                                 library="unused", prompt_ids=[1])

    def test_qwen4_public_range_requires_both_real_runtime_libraries(self):
        with self.assertRaisesRegex(CaptureError, "both compiled Qwen4 and Segment/Edge"):
            capture_qwen4_route("unused", source_format="F32", backend="public-range",
                                prompt_ids=[1])

    def test_qwen4_public_candidate_artifact_pin_closure_is_complete_and_unique(self):
        paths = _candidate_source_paths("qwen4", "public-range")
        names = {path.name for path in paths}
        self.assertIn("native_qwen4_public_text.py", names)
        self.assertIn("native_qwen4_public_adapters.py", names)
        self.assertIn("qwen4_public_adapters.c", names)
        self.assertIn("native_qwen4.c", names)
        self.assertIn("qwen4_public_model.c", names)
        self.assertIn("original_experts.c", names)
        self.assertIn("segment_executor.c", names)
        self.assertIn("segment_bridge.c", names)
        self.assertIn("segment_math.c", names)
        self.assertEqual(len(paths), len(set(paths)))

    def test_inkling_public_range_requires_segment_runtime_before_checkpoint_access(self):
        with self.assertRaisesRegex(CaptureError, "Segment/Edge runtime"):
            capture_inkling_route("unused", source_format="F32", backend="public-range",
                                  library="unused", prompt_ids=[1])

    def test_inkling_public_candidate_pins_executed_route_observer_and_unique_sources(self):
        paths = _candidate_source_paths("inkling", "public-range")
        names = {path.name for path in paths}
        self.assertTrue({
            "native_family_capture.py", "native_family_public_text.py",
            "native_family_public_adapters.py", "native_inkling.py", "native_inkling.c",
            "segment_edge_runtime.py", "segment_edge_compat.h", "segment_edge_compat.c",
            "segment_family_models.c", "independent_inkling.py",
        }.issubset(names))
        self.assertEqual(len(paths), len(set(paths)))

    def test_inkling_public_candidate_is_bound_to_exact_execution_and_abi_owners(self):
        rows = source_scope_matrix()["execution_cases"]
        row = next(row for row in rows if row["family"] == "inkling" and
                   row["model_type"] == "inkling_text" and row["backend"] == "public-range" and
                   row["source_format"] == "F32" and row["persisted_format"] == "float32" and
                   row["combination_status"] == "verified-admitted")
        owner = "native_family_public_text.NativeFamilyPublicTextDecoder"
        self.assertIn(owner, row["callable_chain"]["execution"])
        self.assertIn(owner, row["callable_chain"]["native_abi"])

    def test_glm_public_candidate_is_bound_and_pins_native_observer(self):
        row = next(row for row in source_scope_matrix()['execution_cases']
                   if row['family'] == 'glm5-next' and row['model_type'] == 'glm5_next_text'
                   and row['backend'] == 'public-range' and row['source_format'] == 'F32'
                   and row['combination_status'] == 'verified-admitted')
        owner = 'native_family_public_text.NativeFamilyPublicTextDecoder'
        self.assertIn(owner, row['callable_chain']['execution'])
        self.assertIn(owner, row['callable_chain']['native_abi'])
        names = {path.name for path in _candidate_source_paths('glm5-next', 'public-range')}
        self.assertTrue({'native_glm5_next.c', 'segment_family_models.c',
                         'segment_family_models.h', 'native_family_public_adapters.py'}.issubset(names))

    def test_inkling_multimodal_cannot_enter_text_capture(self):
        with self.assertRaisesRegex(CaptureError, "actual media plus target/draft"):
            capture_inkling_route("unused", source_format="F32", prompt_ids=[1],
                                  model_type="inkling_mm_model")

    def test_mtp_is_not_applicable_to_the_standalone_inkling_text_identity(self):
        from openagent.adaptive_engine.device_observations import coverage_applicability
        row = next(row for row in source_scope_matrix()["execution_cases"]
                   if row["family"] == "inkling" and row["model_type"] == "inkling_text" and
                   row["backend"] == "native" and row["source_format"] == "F32" and
                   row["persisted_format"] == "float32" and
                   row["combination_status"] == "verified-admitted")
        coverage = coverage_applicability(row)
        self.assertFalse(coverage["speculative_state"])
        self.assertFalse(coverage["media_lifecycle"])

    def test_inkling_multimodal_keeps_its_mtp_and_media_requirements(self):
        from openagent.adaptive_engine.device_observations import coverage_applicability
        row = next(row for row in source_scope_matrix()["execution_cases"]
                   if row["family"] == "inkling" and row["model_type"] == "inkling_mm_model" and
                   row["backend"] == "native" and row["source_format"] == "F32" and
                   row["persisted_format"] == "float32" and
                   row["combination_status"] == "verified-admitted")
        coverage = coverage_applicability(row)
        self.assertTrue(coverage["speculative_state"])
        self.assertTrue(coverage["media_lifecycle"])

    def test_native_text_decoder_is_bound_to_execution_and_abi_chains(self):
        rows = source_scope_matrix()["execution_cases"]
        expected = {
            ("qwen4", "qwen4_exp_text"): "native_qwen4.NativeQwen4TextDecoder",
            ("qwen35", "qwen3_5_moe_text"): "native_qwen35.NativeQwen35TextDecoder",
            ("glm5-next", "glm5_next_text"): "native_glm5_next.NativeGlm5NextTextDecoder",
        }
        for (family, model_type), owner in expected.items():
            row = next(row for row in rows if row["family"] == family and
                       row["model_type"] == model_type and row["backend"] == "native" and
                       row["source_format"] == "F32" and row["persisted_format"] == "float32" and
                       row["combination_status"] == "verified-admitted")
            self.assertIn(owner, row["callable_chain"]["execution"])
            self.assertIn(owner, row["callable_chain"]["native_abi"])

    def test_qwen4_device_text_decoder_is_bound_to_exact_route(self):
        rows = source_scope_matrix()["execution_cases"]
        row = next(row for row in rows if row["family"] == "qwen4" and
                   row["model_type"] == "qwen4_exp_text" and row["backend"] == "device" and
                   row["source_format"] == "F32" and row["persisted_format"] == "float32" and
                   row["combination_status"] == "verified-admitted")
        owner = "torch_qwen4_decoder.TorchQwen4Decoder"
        self.assertIn(owner, row["callable_chain"]["execution"])
        self.assertIn(owner, row["callable_chain"]["native_abi"])

    def test_qwen4_device_receipt_pins_candidate_source_closure(self):
        paths = {path.name for path in _candidate_source_paths("qwen4", "device")}
        self.assertTrue({
            "torch_qwen4_decoder.py", "torch_qwen35_decoder.py",
            "device_qwen4_forward.py", "device_execution_policy.py",
            "native_family_capture.py", "device_observations.py", "source_scope.py",
        }.issubset(paths))

    def test_inkling_device_route_binds_candidate_and_sources(self):
        rows = source_scope_matrix()["execution_cases"]
        row = next(row for row in rows if row["family"] == "inkling" and
                   row["model_type"] == "inkling_text" and row["backend"] == "device" and
                   row["source_format"] == "F32" and row["persisted_format"] == "float32" and
                   row["combination_status"] == "verified-admitted")
        owner = "torch_inkling_decoder.TorchInklingDecoder"
        self.assertIn(owner, row["callable_chain"]["execution"])
        self.assertIn(owner, row["callable_chain"]["native_abi"])
        paths = {path.name for path in _candidate_source_paths("inkling", "device")}
        self.assertTrue({"torch_inkling_decoder.py", "device_inkling_forward.py",
                         "device_execution_policy.py"}.issubset(paths))

    def test_inkling_device_requires_explicit_device_before_loading(self):
        with self.assertRaisesRegex(CaptureError, "explicit CUDA/MPS device"):
            capture_inkling_route("unused", source_format="F32", backend="device",
                                  prompt_ids=[1])


if __name__ == "__main__":
    unittest.main()
