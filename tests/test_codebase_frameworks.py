from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openagent.codebase.frameworks import framework_catalog, toolchain_preflight
from openagent.codebase.repository import RepositoryIntelligence
from openagent.codebase.profiles import check_artifact_contract, classify_request
from openagent.codebase.sdk import CodeQSDK, DeclarativeFrameworkAdapter, GenerationRequest, GeneratorRegistry
from openagent.codebase.service import CodebaseService
from openagent.codebase.frameworks import FrameworkSpec
from openagent.codebase.cli import run as run_cli
from openagent.routes.codebase_routes import contract
from openagent.storage import OpenAgentStore


class FrameworkCatalogTests(unittest.TestCase):
    def test_catalog_covers_web_native_mobile_game_and_contract_stacks(self) -> None:
        identifiers = {item["id"] for item in framework_catalog()}
        self.assertTrue({
            "react", "django", "go", "rust", "dotnet", "flutter", "android", "ios",
            "solidity", "godot", "unity", "unreal", "cmake", "zig",
        }.issubset(identifiers))

    def test_unspecified_product_remains_model_selected(self) -> None:
        with patch("openagent.codebase.frameworks.shutil.which", return_value=None):
            result = toolchain_preflight("Build an inventory application")
        self.assertEqual("model-selected", result["framework"]["id"])
        self.assertEqual("available", result["source_generation"])
        self.assertEqual("not_evaluated", result["production_readiness"])

    def test_ios_reports_real_host_and_signing_boundaries(self) -> None:
        with patch("openagent.codebase.frameworks.platform.system", return_value="Windows"), patch(
            "openagent.codebase.frameworks.platform.release", return_value="11"
        ), patch("openagent.codebase.frameworks.shutil.which", return_value=None):
            result = toolchain_preflight("Create a native iOS application and submit it to the App Store")
        self.assertEqual("ios", result["framework"]["id"])
        self.assertFalse(result["host"]["supported"])
        self.assertEqual("requires_setup", result["local_validation"])
        self.assertIn("Apple signing identity and provisioning profile for signed distribution", result["external_requirements"])
        self.assertIn("app_store_submission", {item["boundary"] for item in result["external_boundaries"]})

    def test_external_operations_are_declared_without_secret_values(self) -> None:
        result = toolchain_preflight(
            "Deploy a Stripe service to Kubernetes, configure DNS, and connect enterprise SSO"
        )
        boundaries = {item["boundary"]: item["status"] for item in result["external_boundaries"]}
        self.assertEqual("approval_required", boundaries["deployment"])
        self.assertEqual("approval_required", boundaries["payments"])
        self.assertEqual("approval_required", boundaries["dns"])
        self.assertEqual("external_environment_required", boundaries["cluster"])
        self.assertEqual("provider_configuration_required", boundaries["enterprise_integration"])
        self.assertNotIn("secret", str(result).casefold().replace("secret vault", ""))

    def test_route_contract_exposes_catalog_and_preflight(self) -> None:
        routes = {(item["method"], item["path"]) for item in contract()["routes"]}
        self.assertIn(("GET", "/api/codebase/frameworks"), routes)
        self.assertIn(("POST", "/api/codebase/preflight"), routes)

    def test_service_preflight_preserves_acceptance_criteria(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CodebaseService(OpenAgentStore(Path(temp_dir) / "data"), None)  # type: ignore[arg-type]
            result = service.framework_preflight(
                "Build a Rust command-line package",
                acceptance_criteria=("cargo test passes", "invalid input is rejected"),
            )

        self.assertEqual(
            ["cargo test passes", "invalid input is rejected"],
            result["generation_contract"]["acceptance_criteria"],
        )


class CrossFrameworkRepositoryTests(unittest.TestCase):
    def test_scanner_detects_solidity_flutter_and_zig_projects(self) -> None:
        cases = (
            ({"foundry.toml": "[profile.default]\n", "src/Counter.sol": "contract Counter {}\n"}, "solidity", "forge test"),
            ({"pubspec.yaml": "name: demo\n", "lib/main.dart": "void main() {}\n"}, "flutter", "flutter test"),
            ({"build.zig": "pub fn build() void {}\n", "src/main.zig": "pub fn main() void {}\n"}, "zig", "zig build test"),
        )
        for files, framework, test_command in cases:
            with self.subTest(framework=framework), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for relative, content in files.items():
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                project_map = RepositoryIntelligence(root).scan()
                self.assertEqual(framework, project_map["framework"])
                self.assertEqual(test_command, project_map["commands"]["test"])

    def test_native_artifact_contracts_require_native_project_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = classify_request("Create a native Android app", {"file_count": 0, "languages": [], "framework": ""})
            missing = check_artifact_contract(root, profile)
            self.assertIn("one of: settings.gradle, settings.gradle.kts", missing)
            self.assertTrue(any(".kt" in item for item in missing))


class GeneratorSDKTests(unittest.TestCase):
    def test_custom_framework_adapter_controls_the_complete_contract(self) -> None:
        registry = GeneratorRegistry(load_plugins=False)
        registry.register(DeclarativeFrameworkAdapter(
            FrameworkSpec(
                "quantum-kit",
                "Quantum Kit",
                ("qsharp",),
                ("quantum.toml",),
                (("qtool",),),
                external_requirements=("quantum simulator or target hardware",),
            ),
            keywords=("quantum kit", "qsharp"),
            artifacts={"all": ["quantum.toml"], "any_suffix": [".qs"], "minimum_files": 3},
            commands={"test": "qtool test", "build": "qtool build"},
        ))
        sdk = CodeQSDK(registry)
        with patch("openagent.codebase.frameworks.shutil.which", return_value=None):
            prepared = sdk.prepare("Build a Quantum Kit application")
        contract = prepared["generation_contract"]
        self.assertEqual("quantum-kit", prepared["framework"]["id"])
        self.assertEqual(["qtool"], prepared["missing_toolchains"][0])
        self.assertEqual("qtool test", contract["validation"]["commands"]["test"])
        self.assertEqual(["quantum.toml"], contract["artifacts"]["all"])
        self.assertIn("quantum simulator or target hardware", prepared["external_requirements"])

    def test_generation_contract_separates_model_output_from_completion(self) -> None:
        prepared = CodeQSDK(GeneratorRegistry(load_plugins=False)).prepare(GenerationRequest(
            goal="Create a production Rust command-line package",
            acceptance_criteria=("handles invalid input", "documents the public API"),
        ))
        contract = prepared["generation_contract"]
        self.assertEqual("rust", contract["profile"]["framework"])
        self.assertIn("structured_json", contract["model_contract"]["required_capabilities"])
        self.assertTrue(contract["validation"]["completion_requires_all_configured_gates"])
        self.assertEqual(["handles invalid input", "documents the public API"], contract["acceptance_criteria"])
        self.assertEqual("not_evaluated", prepared["production_readiness"])

    def test_cli_exposes_catalog_inspection_and_preflight(self) -> None:
        sdk = CodeQSDK(GeneratorRegistry(load_plugins=False))
        catalog = run_cli(["catalog"], sdk=sdk)
        self.assertGreaterEqual(len(catalog), 30)
        prepared = run_cli(["preflight", "Build a Zig library", "--accept", "zig build test passes"], sdk=sdk)
        self.assertEqual("zig", prepared["framework"]["id"])
        self.assertEqual(["zig build test passes"], prepared["generation_contract"]["acceptance_criteria"])


if __name__ == "__main__":
    unittest.main()
