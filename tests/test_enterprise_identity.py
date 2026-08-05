from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.auth import AuthError, AuthManager, _totp_code
from openagent.enterprise_identity import EnterpriseIdentity
from openagent.routes.identity_routes import ROUTES
from openagent.storage import OpenAgentStore


class EnterpriseIdentityTests(unittest.TestCase):
    def test_passkey_challenge_is_origin_bound_expiring_and_single_use(self) -> None:
        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            auth = AuthManager(store)
            user = auth.register("owner", "owner@example.com", "strong-password")
            identity = EnterpriseIdentity(store, auth)

            options = identity.registration_options(user["user_id"], rp_id="127.0.0.1", origin="http://127.0.0.1:8765")
            self.assertIn("challenge", options["publicKey"])
            consumed = identity._consume_challenge(options["ceremony_id"], "registration", user_id=user["user_id"])
            self.assertEqual(consumed["origin"], "http://127.0.0.1:8765")
            with self.assertRaises(AuthError):
                identity._consume_challenge(options["ceremony_id"], "registration", user_id=user["user_id"])
            with self.assertRaises(AuthError):
                identity.registration_options(user["user_id"], rp_id="example.com", origin="http://example.com")

    def test_recovery_device_and_federation_denial_paths(self) -> None:
        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            auth = AuthManager(store)
            owner = auth.register("owner", "owner@example.com", "strong-password")
            other = auth.register("other", "other@example.com", "strong-password")
            setup = auth.setup_two_factor(owner["user_id"])
            auth.verify_two_factor(owner["user_id"], _totp_code(setup["secret"]))
            session = auth.login("owner", "strong-password", otp_code=setup["backup_codes"][0])
            self.assertIn("access_token", session)
            with self.assertRaises(AuthError):
                auth.login("owner", "strong-password", otp_code=setup["backup_codes"][0])

            identity = EnterpriseIdentity(store, auth)
            trusted = identity.trust_device(owner["user_id"], name="Laptop", fingerprint="hardware-fingerprint")
            self.assertIsNotNone(identity.verify_device(trusted["device_token"], "hardware-fingerprint"))
            self.assertIsNone(identity.verify_device(trusted["device_token"], "wrong-device"))
            with self.assertRaises(KeyError):
                identity.revoke_device(other["user_id"], trusted["id"])
            identity.revoke_device(owner["user_id"], trusted["id"])
            self.assertIsNone(identity.verify_device(trusted["device_token"], "hardware-fingerprint"))

            identity.save_federation_policy(
                provider="corp", issuer="https://login.example.com", audience="openagent",
                allowed_domains=["example.com"], required_groups=["ai-users"],
            )
            allowed = identity.evaluate_federation("corp", {
                "iss": "https://login.example.com", "aud": ["openagent"],
                "email": "person@example.com", "groups": ["ai-users"],
            })
            self.assertTrue(allowed["allowed"])
            denied = identity.evaluate_federation("corp", {
                "iss": "https://attacker.invalid", "aud": ["openagent"],
                "email": "person@attacker.invalid", "groups": [],
            })
            self.assertFalse(denied["allowed"])
            self.assertIn("issuer_mismatch", denied["reasons"])
            self.assertIn("domain_denied", denied["reasons"])
            self.assertIn("required_group_missing", denied["reasons"])

    def test_route_contract_exposes_identity_lifecycle(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/auth/passkeys/registration/options", paths)
        self.assertIn("/api/auth/passkeys/authentication/verify", paths)
        self.assertIn("/api/auth/recovery/rotate", paths)
        self.assertIn("/api/auth/federation/policies", paths)


if __name__ == "__main__":
    unittest.main()
