from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from openagent.enterprise import EnterpriseManager
from openagent.models import AuditEvent
from openagent.storage import OpenAgentStore


class EnterpriseManagerTests(unittest.TestCase):
    def test_enterprise_lifecycle(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = EnterpriseManager(store)

            user = manager.upsert_user(email="owner@example.com", name="Owner")
            workspace = manager.create_workspace(name="Engineering", owner_user_id=user["id"])
            manager.add_member(workspace_id=workspace["id"], user_id=user["id"], role_id="admin")

            access = manager.check_access(
                user_id=user["id"],
                workspace_id=workspace["id"],
                permission="audit.export",
            )
            self.assertTrue(access["allowed"])
            self.assertIn("owner", {role["id"] for role in access["roles"]})

            memory = manager.add_org_memory(
                workspace_id=workspace["id"],
                title="Local data boundary",
                content="Data remains inside the user's infrastructure.",
            )
            self.assertEqual(memory["workspace_id"], workspace["id"])
            self.assertEqual(len(manager.org_memory(workspace["id"])), 1)

            manager.upsert_policy(
                name="Deny external network",
                rules=[{"scope": "network.external", "effect": "deny", "unless": "allow_external"}],
            )
            denied = manager.evaluate_policy(scope="network.external", context={})
            allowed = manager.evaluate_policy(scope="network.external", context={"allow_external": True})
            self.assertEqual(denied["decision"], "deny")
            self.assertEqual(allowed["decision"], "allow")

            sso = manager.configure_sso(
                provider="oidc",
                issuer="https://idp.example",
                client_id="openagent",
                metadata={"jwks_uri": "https://idp.example/.well-known/jwks.json", "client_secret": "local-secret"},
            )
            self.assertTrue(sso["enabled"])
            self.assertEqual(sso["provider"], "oidc")
            self.assertEqual(sso["metadata"]["client_secret"], "***")
            token = make_hs256_jwt(
                {"iss": "https://idp.example", "aud": "openagent", "sub": "owner@example.com", "exp": int(time.time()) + 3600},
                "local-secret",
            )
            validation = manager.validate_sso_token(provider="oidc", token=token)
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["claims"]["sub"], "owner@example.com")
            wrong_audience = manager.validate_sso_token(provider="oidc", token=token, audience="other-client")
            self.assertFalse(wrong_audience["valid"])
            self.assertEqual(wrong_audience["reason"], "audience_mismatch")
            malformed = manager.validate_sso_token(provider="oidc", token="not.a.jwt")
            self.assertFalse(malformed["valid"])
            self.assertEqual(malformed["reason"], "invalid_jwt_payload")
            bad_time_claim = make_hs256_jwt(
                {"iss": "https://idp.example", "aud": "openagent", "sub": "owner@example.com", "exp": "soon"},
                "local-secret",
            )
            invalid_time = manager.validate_sso_token(provider="oidc", token=bad_time_claim)
            self.assertFalse(invalid_time["valid"])
            self.assertEqual(invalid_time["reason"], "invalid_time_claim")

            store.add_audit_event(
                AuditEvent(
                    actor="owner@example.com",
                    action="workspace.read",
                    resource=workspace["id"],
                    project=workspace["id"],
                )
            )
            export = manager.export_audit(project=workspace["id"])
            self.assertGreaterEqual(export["event_count"], 1)
            self.assertTrue(Path(export["path"]).exists())

            backup = manager.create_backup()
            self.assertEqual(backup["status"], "completed")
            self.assertTrue(Path(backup["manifest"]["database"]).exists())
            self.assertGreaterEqual(len(manager.backups()), 1)

            dr = manager.create_disaster_recovery_plan(
                name="Default recovery",
                rpo_minutes=60,
                rto_minutes=240,
            )
            self.assertEqual(dr["plan"]["rpo_minutes"], 60)
            self.assertEqual(len(manager.disaster_recovery_plans()), 1)
            self.assertEqual(manager.status()["disaster_recovery_plans"], 1)

    def test_membership_rejects_unknown_records(self) -> None:
        with TemporaryDirectory() as tmp:
            manager = EnterpriseManager(OpenAgentStore(tmp))

            with self.assertRaises(ValueError):
                manager.add_member(workspace_id="missing", user_id="missing", role_id="member")

def make_hs256_jwt(claims: dict[str, object], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join([base64url_json(header), base64url_json(claims)])
    signature = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{base64url(signature)}"


def base64url_json(payload: dict[str, object]) -> str:
    return base64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def base64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


if __name__ == "__main__":
    unittest.main()
