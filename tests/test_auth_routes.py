from __future__ import annotations

import unittest

from openagent.routes import auth_routes


class AuthRouteContractTests(unittest.TestCase):
    def test_blueprint_auth_refresh_contract_is_declared(self) -> None:
        contract = auth_routes.contract()
        routes = {(route["method"], route["path"]) for route in contract["routes"]}

        self.assertIn(("POST", "/api/auth/refresh"), routes)
        self.assertIn(("GET", "/api/auth/me"), routes)
        self.assertGreaterEqual(contract["route_count"], 10)


if __name__ == "__main__":
    unittest.main()
