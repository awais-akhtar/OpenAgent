"""P13 reference eligibility pins every reviewed local equation dependency."""
import unittest
from copy import deepcopy

from openagent.adaptive_engine.phase1_observations import (
    ObservationError, REFERENCE_IMPLEMENTATIONS, REFERENCE_SOURCES, reference_source_pins,
    reference_implementation, verify_publisher_config_contract,
    verify_reference_source_binding)


class Phase1ReferenceRegistryTests(unittest.TestCase):
    def test_registered_references_have_existing_unique_pinned_sources(self):
        for family, modules in REFERENCE_SOURCES.items():
            with self.subTest(family=family):
                pins = reference_source_pins(family)
                self.assertEqual([pin["module"] for pin in pins], list(modules))
                self.assertEqual(len({pin["module"] for pin in pins}), len(pins))
                self.assertTrue(all(len(pin["sha256"]) == 64 for pin in pins))

    def test_references_without_an_independent_decoder_remain_ineligible(self):
        for family in ("kimi-k3", "deepseek-v41"):
            with self.subTest(family=family):
                with self.assertRaisesRegex(ObservationError, "no reviewed reference"):
                    reference_source_pins(family)

    def test_shared_numerical_helpers_are_part_of_each_reference_closure(self):
        self.assertIn("openagent/adaptive_engine/native_mixtral.py",
                      REFERENCE_SOURCES["qwen35"])
        self.assertIn("openagent/adaptive_engine/independent_qwen35.py",
                      REFERENCE_SOURCES["qwen4"])
        self.assertIn("openagent/adaptive_engine/independent_qwen35.py",
                      REFERENCE_SOURCES["glm-dsa"])
        self.assertIn("openagent/adaptive_engine/device_glm5_forward.py",
                      REFERENCE_SOURCES["glm5-next"])
        self.assertIn("openagent/adaptive_engine/torch_glm5_decoder.py",
                      REFERENCE_SOURCES["glm5-next"])

    def test_multi_file_reference_requires_every_current_dependency_hash(self):
        pins = reference_source_pins("qwen4")
        reference = {"implementation": REFERENCE_IMPLEMENTATIONS["qwen4"], "sources": pins}
        verify_reference_source_binding("qwen4", reference)

        changed = deepcopy(pins)
        changed[-1]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ObservationError, "changed since capture"):
            verify_reference_source_binding("qwen4", {**reference, "sources": changed})

        incomplete = {**reference, "sources": pins[:-1]}
        with self.assertRaisesRegex(ObservationError, "reviewed in-repository"):
            verify_reference_source_binding("qwen4", incomplete)

        wrong_decoder = {**reference, "implementation": REFERENCE_IMPLEMENTATIONS["mixtral"]}
        with self.assertRaisesRegex(ObservationError, "registered family implementation"):
            verify_reference_source_binding("qwen4", wrong_decoder)

    def test_glm5_device_route_uses_independent_native_reference(self):
        pins = reference_source_pins("glm5-next", "device", "glm5_next_text")
        self.assertEqual(reference_implementation("glm5-next", "device", "glm5_next_text"),
                         "openagent.adaptive_engine.native_glm5_next.NativeGlm5NextTextDecoder (independent C reference)")
        self.assertIn("openagent/adaptive_engine/native_glm5_next.c",
                      [pin["module"] for pin in pins])
        reference = {"implementation": reference_implementation("glm5-next", "device", "glm5_next_text"),
                     "sources": pins}
        verify_reference_source_binding("glm5-next", reference, "device", "glm5_next_text")
        with self.assertRaisesRegex(ObservationError, "registered family implementation"):
            verify_reference_source_binding("glm5-next", {
                "implementation": REFERENCE_IMPLEMENTATIONS["glm5-next"],
                "sources": pins,
            }, "device", "glm5_next_text")

        # A publisher comparison additionally executes the independent vision
        # encoder; a text-only C receipt cannot stand in for that composition.
        publisher = reference_implementation("glm5-next", "device", "glm5_next")
        self.assertIn('IndependentGlm5VisionEncoder', publisher)
        publisher_pins = reference_source_pins('glm5-next', 'device', 'glm5_next')
        self.assertIn('openagent/adaptive_engine/independent_media_math.py',
                      [pin['module'] for pin in publisher_pins])
        with self.assertRaisesRegex(ObservationError, 'registered family implementation'):
            verify_reference_source_binding('glm5-next', reference, 'device', 'glm5_next')

    def test_single_file_reference_retains_olmoe_receipt_shape(self):
        pin = reference_source_pins("olmoe")[0]
        reference = {"implementation": REFERENCE_IMPLEMENTATIONS["olmoe"], "source": pin}
        verify_reference_source_binding("olmoe", reference)
        with self.assertRaisesRegex(ObservationError, "reviewed in-repository"):
            verify_reference_source_binding("olmoe", {"implementation": REFERENCE_IMPLEMENTATIONS["olmoe"],
                                                       "sources": [pin]})

    def test_publisher_geometry_rejects_config_from_another_mixtral_checkpoint(self):
        from openagent.adaptive_engine.source_scope import source_scope_matrix

        row = next(case for case in source_scope_matrix()["execution_cases"]
                   if case["family"] == "mixtral" and
                   case["combination_status"] == "verified-admitted")
        config = dict(row["publisher"], model_type="mixtral")
        verify_publisher_config_contract(row, config)

        wrong = dict(config, hidden_size=config["hidden_size"] // 2)
        with self.assertRaisesRegex(ObservationError, "publisher field hidden_size"):
            verify_publisher_config_contract(row, wrong)

        missing = dict(config)
        del missing["intermediate_size"]
        with self.assertRaisesRegex(ObservationError, "lacks publisher field intermediate_size"):
            verify_publisher_config_contract(row, missing)


if __name__ == "__main__":
    unittest.main()
