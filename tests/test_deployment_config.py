from __future__ import annotations

from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
KUBERNETES = ROOT / "deploy" / "kubernetes"


def load_documents(path: Path) -> list[dict[str, object]]:
    return [document for document in yaml.safe_load_all(path.read_text(encoding="utf-8")) if document]


class ContainerConfigurationTests(unittest.TestCase):
    def test_runtime_image_is_non_root_and_contains_only_built_artifacts(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        runtime = dockerfile.split(" AS runtime", maxsplit=1)[1]
        self.assertIn("USER 10001:10001", runtime)
        self.assertIn("HEALTHCHECK", runtime)
        self.assertIn("STOPSIGNAL SIGTERM", runtime)
        self.assertNotRegex(runtime, r"(?m)^COPY\s+\.\s")
        self.assertIn("COPY --from=builder /wheels /wheels", runtime)
        self.assertIn("[media,security]", runtime)
        self.assertIn("chown 10001:10001 /data", runtime)

    def test_build_context_excludes_private_and_generated_files(self) -> None:
        patterns = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
        for required in (".git", ".venv", "node_modules", ".env", ".tmp*", "*.sqlite3", "tests"):
            self.assertIn(required, patterns)

    def test_compose_defaults_to_loopback_team_mode_and_drops_privileges(self) -> None:
        compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        service = compose["services"]["openagent"]
        self.assertEqual(service["ports"], ["127.0.0.1:8765:8765"])
        self.assertEqual(service["user"], "10001:10001")
        self.assertTrue(service["read_only"])
        self.assertTrue(service["init"])
        self.assertEqual(service["restart"], "no")
        self.assertEqual(service["cap_drop"], ["ALL"])
        self.assertIn("no-new-privileges:true", service["security_opt"])
        self.assertEqual(service["environment"]["OPENAGENT_DEPLOYMENT_MODE"], "team")
        self.assertEqual(str(service["environment"]["OPENAGENT_AUTH_ENFORCED"]), "1")
        self.assertIn(":?Set OPENAGENT_BOOTSTRAP_TOKEN", service["environment"]["OPENAGENT_BOOTSTRAP_TOKEN"])


class KubernetesConfigurationTests(unittest.TestCase):
    def resources(self) -> list[dict[str, object]]:
        resources: list[dict[str, object]] = []
        for path in sorted(KUBERNETES.glob("*.yaml")):
            if path.name != "secret.example.yaml" and path.name != "kustomization.yaml":
                resources.extend(load_documents(path))
        return resources

    def resource(self, kind: str, name: str) -> dict[str, object]:
        for resource in self.resources():
            if resource.get("kind") == kind and resource.get("metadata", {}).get("name") == name:
                return resource
        self.fail(f"missing {kind}/{name}")

    def test_runtime_workload_has_restricted_security_context_and_versioned_image(self) -> None:
        deployment = self.resource("Deployment", "openagent")
        pod = deployment["spec"]["template"]["spec"]
        container = pod["containers"][0]
        self.assertTrue(pod["securityContext"]["runAsNonRoot"])
        self.assertEqual(pod["securityContext"]["seccompProfile"]["type"], "RuntimeDefault")
        self.assertFalse(container["securityContext"]["allowPrivilegeEscalation"])
        self.assertTrue(container["securityContext"]["readOnlyRootFilesystem"])
        self.assertEqual(container["securityContext"]["capabilities"]["drop"], ["ALL"])
        self.assertNotIn(":latest", container["image"])
        self.assertIn("startupProbe", container)
        self.assertIn("readinessProbe", container)
        self.assertIn("livenessProbe", container)
        self.assertEqual(deployment["spec"]["strategy"]["type"], "Recreate")

    def test_auth_defaults_and_service_accounts_fail_closed(self) -> None:
        config = self.resource("ConfigMap", "openagent-config")["data"]
        self.assertEqual(config["OPENAGENT_DEPLOYMENT_MODE"], "team")
        self.assertEqual(str(config["OPENAGENT_AUTH_ENFORCED"]), "1")
        for account in ("openagent", "openagent-backup"):
            service_account = self.resource("ServiceAccount", account)
            self.assertFalse(service_account["automountServiceAccountToken"])

    def test_network_policy_denies_by_default_and_scopes_backup_traffic(self) -> None:
        policies = [item for item in self.resources() if item.get("kind") == "NetworkPolicy"]
        by_name = {item["metadata"]["name"]: item for item in policies}
        default_deny = by_name["openagent-default-deny"]
        self.assertEqual(default_deny["spec"]["podSelector"], {})
        self.assertEqual(set(default_deny["spec"]["policyTypes"]), {"Ingress", "Egress"})
        backup = by_name["openagent-backup-trigger"]
        self.assertEqual(backup["spec"]["podSelector"]["matchLabels"]["app"], "openagent-backup")
        ports = {
            port["port"]
            for rule in backup["spec"]["egress"]
            for port in rule.get("ports", [])
        }
        self.assertEqual(ports, {53, 8765})

    def test_backup_uses_a_secret_header_file_without_shell_interpolation(self) -> None:
        cronjob_path = KUBERNETES / "cronjob-backup.yaml"
        text = cronjob_path.read_text(encoding="utf-8")
        cronjob = load_documents(cronjob_path)[0]
        pod = cronjob["spec"]["jobTemplate"]["spec"]["template"]
        container = pod["spec"]["containers"][0]
        self.assertTrue(cronjob["spec"]["suspend"])
        self.assertEqual(pod["metadata"]["labels"]["app"], "openagent-backup")
        self.assertIn("@/var/run/secrets/openagent/authorization", container["args"])
        self.assertNotIn("$(", text)
        self.assertFalse(container["securityContext"]["allowPrivilegeEscalation"])

    def test_development_and_production_overlays_are_declared(self) -> None:
        for profile in ("development", "production"):
            path = KUBERNETES / "overlays" / profile / "kustomization.yaml"
            self.assertTrue(path.is_file())
            overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(overlay["resources"], ["../.."])
        production = (KUBERNETES / "overlays" / "production" / "kustomization.yaml").read_text(encoding="utf-8")
        self.assertNotIn("newTag: latest", production)


class WorkflowConfigurationTests(unittest.TestCase):
    def test_ci_has_named_release_acceptance_gates(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        required = (
            "static-quality:",
            "security-contracts:",
            "browser-e2e:",
            "package:",
            "deployment:",
            "tests.test_deployment_config",
            "kubeconform",
            "Verify non-root authenticated runtime",
            "trivy-action",
            "sbom-action",
        )
        for marker in required:
            self.assertIn(marker, workflow)

    def test_pypi_release_builds_once_and_uses_oidc_exact_artifacts(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text(encoding="utf-8")
        self.assertEqual(workflow.count("python -m build"), 1)
        self.assertIn("actions/upload-artifact@v4", workflow)
        self.assertIn("actions/download-artifact@v4", workflow)
        self.assertIn("sha256sum dist/* > SHA256SUMS", workflow)
        self.assertIn("sha256sum --check SHA256SUMS", workflow)
        self.assertIn("actions/attest-build-provenance@v2", workflow)
        self.assertIn("id-token: write", workflow)
        self.assertNotIn("password:", workflow)
        self.assertNotIn("PYPI_API_TOKEN", workflow)

    def test_security_workflow_has_dependency_secret_and_sast_gates(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
        for marker in ("pip-audit", "npm audit", "dependency-review-action", "gitleaks-action", "codeql-action"):
            self.assertIn(marker, workflow)

    def test_workflow_yaml_is_well_formed(self) -> None:
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            with self.subTest(path=path.name):
                parsed = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
                self.assertIsInstance(parsed, dict)
                self.assertIn("jobs", parsed)


if __name__ == "__main__":
    unittest.main()
