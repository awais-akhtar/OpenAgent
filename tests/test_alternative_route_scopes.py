"""A capability must not be gated behind an unrelated permission.

The gate is single-scope: one route, one permission. That made the API-keys tab
of the Developer screen unusable. Creating a key needs ``developer:admin``, but
the tab populates its permission checkboxes from ``/api/auth/scopes``, which
required ``settings:read`` -- and the Create button is disabled while the
catalogue is empty. A credential holding exactly the Developer screen's own
scopes therefore rendered the form and could never submit it.

The catalogue is 61 static ``{scope, description}`` pairs: the product's own
vocabulary, with no account state and no user data in it. Letting a developer
credential read it grants access to nothing else, so the fix is an explicit
alternative rather than a looser gate.
"""

from __future__ import annotations

import unittest

from openagent.auth import (
    ALTERNATIVE_ROUTE_SCOPES,
    SCOPE_CATALOG,
    acceptable_scopes_for_path,
    required_scope_for_path,
)
from openagent.routes import route_contracts


class AcceptableScopesTests(unittest.TestCase):
    def test_the_catalogue_is_reachable_with_either_permission(self) -> None:
        accepted = acceptable_scopes_for_path("/api/auth/scopes", "GET")
        self.assertIn("settings:read", accepted)
        self.assertIn("developer:read", accepted)

    def test_the_primary_scope_stays_first_so_the_inventory_is_unchanged(self) -> None:
        # The route inventory and the gating invariant both report one
        # permission per route, and that reported permission must not move.
        self.assertEqual(
            acceptable_scopes_for_path("/api/auth/scopes", "GET")[0],
            required_scope_for_path("/api/auth/scopes", "GET"),
        )

    def test_a_route_without_an_alternative_accepts_only_its_own_scope(self) -> None:
        self.assertEqual(acceptable_scopes_for_path("/api/vault/entries", "GET"), ("vault:read",))
        self.assertEqual(acceptable_scopes_for_path("/api/auth/policy", "GET"), ("settings:read",))

    def test_an_ungated_route_accepts_nothing_by_scope(self) -> None:
        # /api/health is public, so there is no scope to check membership in.
        self.assertEqual(acceptable_scopes_for_path("/api/health", "GET"), ())

    def test_the_alternative_does_not_apply_to_another_method(self) -> None:
        # Only the catalogue read is widened, not anything that writes.
        self.assertNotIn(("POST", "/api/auth/scopes"), ALTERNATIVE_ROUTE_SCOPES)

    def test_every_alternative_names_real_scopes(self) -> None:
        catalogue = {entry["scope"] for entry in SCOPE_CATALOG}
        for (method, path), scopes in ALTERNATIVE_ROUTE_SCOPES.items():
            for scope in scopes:
                with self.subTest(route=f"{method} {path}", scope=scope):
                    self.assertIn(scope, catalogue)

    def test_every_alternative_points_at_a_registered_route(self) -> None:
        registered = {
            (entry["method"], entry["path"])
            for group in route_contracts()
            for entry in group["routes"]
        }
        for route in ALTERNATIVE_ROUTE_SCOPES:
            with self.subTest(route=route):
                self.assertIn(route, registered)

    def test_the_mechanism_stays_a_short_explicit_list(self) -> None:
        # This exists to fix named couplings, not to become a general way round
        # the gate. A growing list is a sign the scope model needs rethinking
        # rather than another entry here.
        self.assertLessEqual(len(ALTERNATIVE_ROUTE_SCOPES), 5)

    def test_no_alternative_widens_a_write_or_admin_route(self) -> None:
        for (method, path), _scopes in ALTERNATIVE_ROUTE_SCOPES.items():
            with self.subTest(route=f"{method} {path}"):
                self.assertEqual(method, "GET", "only reads may have an alternative scope")
                primary = required_scope_for_path(path, method)
                self.assertIsNotNone(primary)
                self.assertFalse(
                    str(primary).endswith((":admin", ":write")),
                    f"{path} is gated by {primary}; an administrative route must not be widened",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
