from __future__ import annotations

import unittest

from openagent.auth import required_scope_for_path


class AuthPolicyContractTests(unittest.TestCase):
    def test_sensitive_route_groups_have_required_scopes(self) -> None:
        self.assertEqual(required_scope_for_path("/api/developer/status", "GET"), "developer:read")
        self.assertEqual(required_scope_for_path("/api/local-ai/models", "POST"), "local_ai:admin")
        self.assertEqual(required_scope_for_path("/api/local-ai/benchmarks", "GET"), "local_ai:read")
        self.assertEqual(required_scope_for_path("/api/security/audit-chain", "GET"), "security:read")
        self.assertEqual(required_scope_for_path("/api/audit", "GET"), "security:admin")
        self.assertEqual(required_scope_for_path("/api/agent-runs", "GET"), "agents:read")
        self.assertEqual(required_scope_for_path("/api/agents/run", "POST"), "agents:run")
        self.assertEqual(required_scope_for_path("/api/agents/run-checked", "POST"), "agents:run")
        self.assertEqual(required_scope_for_path("/api/agents/delegate", "POST"), "agents:run")
        self.assertEqual(required_scope_for_path("/api/agents/approvals", "GET"), "agents:admin")
        self.assertEqual(required_scope_for_path("/api/media/assets/asset-1", "DELETE"), "media:write")
        self.assertEqual(required_scope_for_path("/api/media/assets", "GET"), "media:read")


if __name__ == "__main__":
    unittest.main()
