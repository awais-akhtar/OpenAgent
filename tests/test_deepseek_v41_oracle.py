"""Isolated V4.1 oracle prerequisite probe, not decoder certification.

Run this file directly to print evidence. No downloads, source patches, V4
fallback, or changes to the application's model registry occur here.
"""

import importlib.util
import json
import unittest


def probe_oracle():
    evidence = {"numerical_oracle_verified": False}
    if importlib.util.find_spec("transformers") is None:
        evidence["transformers"] = None
    else:
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        evidence.update(
            transformers=transformers.__version__,
            transformers_path=transformers.__file__,
            registered_identities={
                name: name in CONFIG_MAPPING
                for name in ("deepseek_v41", "deepseek_v41_text")
            },
            modeling_package_present=importlib.util.find_spec(
                "transformers.models.deepseek_v41"
            ) is not None,
        )
    if importlib.util.find_spec("torch") is None:
        evidence["torch"] = None
    else:
        import torch

        evidence.update(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                        cuda_devices=torch.cuda.device_count())
    for name in ("triton", "tilelang"):
        evidence[name + "_present"] = importlib.util.find_spec(name) is not None
    return evidence


class DeepseekV41EvidenceTests(unittest.TestCase):
    def test_probe_does_not_claim_numerical_certification(self):
        evidence = probe_oracle()
        self.assertIs(evidence["numerical_oracle_verified"], False)
        json.dumps(evidence)

    def test_probe_has_no_native_or_v4_dependency(self):
        import ast
        from pathlib import Path

        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("openagent"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith("openagent"))


if __name__ == "__main__":
    print(json.dumps(probe_oracle(), indent=2, sort_keys=True))
    unittest.main()
