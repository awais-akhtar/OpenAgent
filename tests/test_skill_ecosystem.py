from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
import zipfile

from openagent.routes.skill_routes import ROUTES
from openagent.skill_ecosystem import SkillEcosystem
from openagent.storage import OpenAgentStore


class SkillEcosystemTests(unittest.TestCase):
    def test_edit_test_permission_scan_export_and_signed_package(self) -> None:
        with TemporaryDirectory() as temp:
            skills = SkillEcosystem(OpenAgentStore(temp))
            skill = skills.save(
                name="Local researcher", markdown="# Local researcher\n\nUse local sources.", project="core",
                manifest={"permissions": ["filesystem:read"], "dependencies": {"example": "==1.0.0"}},
            )
            self.assertEqual(skills.test(skill["id"])["status"], "passed")
            self.assertFalse(skills.permission_preview(skill["id"])["approval_required"])
            self.assertTrue(skills.export_markdown(skill["id"])["path"].endswith(".md"))
            package = skills.package(skill["id"], signing_key="0123456789abcdef")
            with zipfile.ZipFile(package["path"]) as archive:
                self.assertSetEqual(set(archive.namelist()), {"SKILL.md", "manifest.json"})

    def test_routes_cover_skill_lifecycle(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/skills/{id}/package", paths)
        self.assertIn("/api/skills/{id}/dependency-scan", paths)
        self.assertIn("/api/skills/{id}/sandbox", paths)
        self.assertIn("/api/skills/marketplace/{id}/install", paths)

    def test_declarative_sandbox_and_signed_marketplace_lifecycle(self) -> None:
        with TemporaryDirectory() as temp:
            skills = SkillEcosystem(OpenAgentStore(temp))
            skill = skills.save(
                name="Reviewed writer", markdown="# Reviewed writer\n\nProduce a bounded local draft.", project="core",
                manifest={"permissions": ["filesystem:read"], "dependencies": {}},
            )
            blocked = skills.run_sandbox(skill["id"], input_data={"topic": "local AI"})
            self.assertEqual(blocked["status"], "blocked")
            completed = skills.run_sandbox(
                skill["id"], input_data={"topic": "local AI"}, approved_permissions=["filesystem:read"]
            )
            self.assertEqual(completed["status"], "completed")
            key = "0123456789abcdef"
            listing = skills.publish(skill["id"], signing_key=key)
            installation = skills.install(listing["id"], signing_key=key, project="installed")
            self.assertEqual(installation["status"], "installed")
            removed = skills.uninstall(installation["id"])
            self.assertEqual(removed["status"], "uninstalled")


if __name__ == "__main__":
    unittest.main()
