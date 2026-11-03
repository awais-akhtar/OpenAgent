"""Legacy attribution is dispatched, so it must also be registered and gated.

The runtime has dispatched these five routes since d8a0b2b, but the module was
left out of ``DOMAIN_ROUTE_MODULES``. Two things followed: the routes were
absent from the route contract registry that the gap inventory is generated
from, and ``required_scope_for_path`` returned ``None`` for all five, so any
authenticated token reached them whatever its scopes. Each handler already
demanded an owner, so this was an under-reported surface rather than an open
door -- but a validation verdict about record ownership belongs behind the same
scope as the rest of the ownership surface.
"""

from __future__ import annotations

import unittest

from openagent.auth import required_scope_for_path
from openagent.routes import DOMAIN_ROUTE_MODULES, route_contracts
from openagent.routes.legacy_attribution_routes import ROUTE_GROUP, ROUTES


class LegacyAttributionRegistrationTests(unittest.TestCase):
    def test_the_module_is_in_the_domain_list(self) -> None:
        self.assertIn("legacy_attribution_routes", DOMAIN_ROUTE_MODULES)

    def test_the_group_is_in_the_contract_registry(self) -> None:
        groups = {group["group"] for group in route_contracts()}
        self.assertIn(ROUTE_GROUP, groups)

    def test_every_declared_route_is_in_the_contract(self) -> None:
        group = next(entry for entry in route_contracts() if entry["group"] == ROUTE_GROUP)
        declared = {(method, path) for method, path in ROUTES}
        registered = {(route["method"], route["path"]) for route in group["routes"]}
        self.assertEqual(declared, registered)

    def test_reads_require_the_security_read_scope(self) -> None:
        for method, path in ROUTES:
            if method != "GET":
                continue
            with self.subTest(path=path):
                self.assertEqual(required_scope_for_path(path, method), "security:read")

    def test_writes_require_the_security_admin_scope(self) -> None:
        for method, path in ROUTES:
            if method != "POST":
                continue
            with self.subTest(path=path):
                self.assertEqual(required_scope_for_path(path, method), "security:admin")

    def test_no_declared_route_is_left_ungated(self) -> None:
        ungated = [(m, p) for m, p in ROUTES if required_scope_for_path(p, m) is None]
        self.assertEqual(ungated, [])


class RegistryGatingTests(unittest.TestCase):
    """The only ungated routes are the deliberate public ones."""

    PUBLIC = {
        ("GET", "/api/health"),
        ("POST", "/api/auth/register"),
        ("POST", "/api/auth/login"),
        ("POST", "/api/auth/refresh"),
    }

    def test_every_other_registered_route_has_a_required_scope(self) -> None:
        ungated = set()
        for group in route_contracts():
            for entry in group["routes"]:
                method, path = entry["method"], entry["path"]
                if required_scope_for_path(path, method) is None:
                    ungated.add((method, path))
        self.assertEqual(ungated, self.PUBLIC)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
