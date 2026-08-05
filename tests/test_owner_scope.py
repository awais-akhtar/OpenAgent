from __future__ import annotations

import unittest

from openagent.auth import auth_enforced_owner


class OwnerScopeContractTests(unittest.TestCase):
    def test_owner_scope_uses_authenticated_user_when_enforced(self) -> None:
        user = {"user_id": "usr_a", "role": "member"}

        self.assertEqual(auth_enforced_owner(user, fallback="fallback-owner"), "usr_a")
        self.assertEqual(auth_enforced_owner(None, fallback="fallback-owner"), "fallback-owner")


if __name__ == "__main__":
    unittest.main()
