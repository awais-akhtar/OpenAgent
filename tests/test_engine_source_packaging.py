"""Native sources must remain available in source and wheel distributions."""
from pathlib import Path
import tomllib
import unittest


class EngineSourcePackagingTests(unittest.TestCase):
    def test_gated_delta_source_is_in_both_distribution_contracts(self):
        root = Path(__file__).resolve().parents[1]
        configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        patterns = configuration["tool"]["setuptools"]["package-data"]["openagent.adaptive_engine"]
        package = root / "openagent/adaptive_engine"
        declared = {path.resolve() for pattern in patterns for path in package.glob(pattern)}
        manifest = (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        recursive_sources = "recursive-include openagent/adaptive_engine *.c *.h" in manifest
        for filename in ("native_mixtral_delta.c", "native_qwen35.c", "native_glm_dsa.c",
                         "native_qwen4.c", "native_qwen4_hyper.c", "native_glm5_next.c",
                         "native_deepseek_v4.c", "native_inkling.c", "native_kimi.c",
                         "native_mixtral_decoder.c", "native_qwen35_vision.c",
                         "native_deepseek_v41_csa2.c", "native_kimi_k3_attnres.c"):
            with self.subTest(filename=filename):
                source = package / filename
                self.assertTrue(source.is_file())
                self.assertIn(source.resolve(), declared)
                self.assertTrue(recursive_sources or f"include openagent/adaptive_engine/{filename}" in manifest)

    def test_validation_workflow_includes_packed_and_native_execution(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/expert-engine.yml").read_text(encoding="utf-8")
        for module in ("test_q4_0_olmoe", "test_q4_1_olmoe", "test_q8_0_olmoe", "test_q5_0_olmoe",
                       "test_q5_1_olmoe", "test_q5_0_integration_adversarial",
                       "test_native_segment_executor", "test_native_olmoe_segments",
                       "test_native_segment_adversarial", "test_family_conformance",
                       "test_native_checkpoint", "test_native_glm5_next",
                       "test_native_deepseek_v4", "test_native_inkling", "test_native_kimi",
                       "test_native_mixtral_decoder", "test_native_qwen35_vision",
                       "test_native_deepseek_v41_csa2", "test_native_kimi_k3_attnres"):
            with self.subTest(module=module):
                self.assertIn("tests." + module, workflow)
        self.assertIn('python -m tests.prepare_kimi_oracle .tmp-kimi-oracle', workflow)
        self.assertIn('KIMI_LINEAR_ORACLE_DIR: .tmp-kimi-oracle', workflow)


if __name__ == "__main__":
    unittest.main()
