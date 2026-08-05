from __future__ import annotations

import unittest

from openagent.auth import required_scope_for_path


class AuthPolicyContractTests(unittest.TestCase):
    def test_sensitive_route_groups_have_required_scopes(self) -> None:
        self.assertEqual(required_scope_for_path("/api/developer/status", "GET"), "developer:read")
        self.assertEqual(required_scope_for_path("/api/local-ai/models", "POST"), "local_ai:admin")
        self.assertEqual(required_scope_for_path("/api/security/audit-chain", "GET"), "security:read")


if __name__ == "__main__":
    unittest.main()
