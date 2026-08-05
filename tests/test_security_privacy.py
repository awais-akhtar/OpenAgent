from __future__ import annotations

from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
import unittest

from openagent.models import AuditEvent, ContextEvent
from openagent.security_privacy import SecurityPrivacyError, SecurityPrivacyManager
from openagent.storage import OpenAgentStore


class SecurityPrivacyManagerTests(unittest.TestCase):
    def test_secret_retention_audit_privacy_and_sandbox_lifecycle(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            old = (datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=10)).isoformat()
            store.add_event(
                ContextEvent(
                    title="Old context",
                    content="This record should be removed by retention.",
                    project="core",
                    created_at=old,
                )
            )
            store.add_audit_event(
                AuditEvent(actor="test", action="stage9.start", resource="security", project="security")
            )
            manager = SecurityPrivacyManager(store)

            secret = manager.store_secret(
                name="openai-key",
                value="sk-local-placeholder",
                passphrase="correct horse battery staple",
                description="Local test secret",
            )
            self.assertTrue(secret["encrypted"])
            self.assertNotIn("value", secret)
            revealed = manager.reveal_secret(
                secret_id=secret["id"],
                passphrase="correct horse battery staple",
            )
            self.assertEqual(revealed["value"], "sk-local-placeholder")
            with self.assertRaises(SecurityPrivacyError):
                manager.reveal_secret(secret_id=secret["id"], passphrase="wrong")
            manager.rotate_secret(secret_id=secret["id"], value="rotated", passphrase="correct horse battery staple")
            self.assertEqual(
                manager.reveal_secret(secret_id=secret["id"], passphrase="correct horse battery staple")["value"],
                "rotated",
            )

            policy = manager.create_retention_policy(resource_type="context_events", retention_days=1)
            planned = manager.apply_retention(policy_id=policy["id"], dry_run=True)
            self.assertEqual(planned["planned"], 1)
            applied = manager.apply_retention(policy_id=policy["id"], dry_run=False)
            self.assertEqual(applied["deleted"], 1)
            self.assertEqual(store.list_events(project="core"), [])

            grant = manager.create_share_grant(
                resource_type="document",
                resource_id="doc-1",
                grantee="team@example.com",
                scope="read",
            )
            self.assertEqual(grant["status"], "active")
            revoked = manager.resolve_share_grant(grant_id=grant["id"], status="revoked")
            self.assertEqual(revoked["status"], "revoked")

            egress = manager.request_egress(
                requester="agent",
                purpose="external model call",
                destination="https://api.example.com",
                resource_type="conversation",
                resource_id="conv-1",
                risk_level="high",
            )
            self.assertEqual(egress["status"], "pending")
            resolved = manager.resolve_egress_request(
                request_id=egress["id"],
                approved=False,
                approved_by="owner",
            )
            self.assertEqual(resolved["status"], "denied")

            sealed = manager.seal_audit_events()
            self.assertGreaterEqual(sealed["sealed"], 1)
            self.assertTrue(manager.verify_audit_chain()["ok"])

            profile = manager.create_sandbox_profile(
                name="read-only-tool",
                permissions=["context.read"],
                network_allowed=False,
            )
            allowed = manager.evaluate_sandbox_request(
                profile_id=profile["id"],
                permissions=["context.read"],
                network=False,
            )
            denied = manager.evaluate_sandbox_request(
                profile_id=profile["id"],
                permissions=["context.read"],
                network=True,
            )
            self.assertTrue(allowed["allowed"])
            self.assertFalse(denied["allowed"])

            report = manager.create_privacy_report()
            self.assertGreaterEqual(report["report"]["data_inventory"]["security_secrets"], 1)
            self.assertGreaterEqual(manager.status()["privacy_reports"], 1)


if __name__ == "__main__":
    unittest.main()
