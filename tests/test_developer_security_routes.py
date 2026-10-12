from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openagent.developer import DeveloperPlatform
from openagent.routes.developer_routes import (
    DeveloperRouteDependencies,
    contract as developer_contract,
    handle_get as handle_developer_get,
    handle_post as handle_developer_post,
)
from openagent.routes.security_routes import (
    SecurityRouteDependencies,
    contract as security_contract,
    handle_get as handle_security_get,
    handle_post as handle_security_post,
)
from openagent.security_privacy import SecurityPrivacyError, SecurityPrivacyManager
from openagent.storage import OpenAgentStore


def _first(params: dict[str, list[str]], name: str) -> str | None:
    return params.get(name, [None])[0]


def _parse_bool(value: object, default: bool = False) -> bool:
    return default if value is None else str(value).lower() in {"1", "true", "yes", "on"}


def _parse_int(value: object, default: int = 0) -> int:
    return default if value in {None, ""} else int(str(value))


class DeveloperAndSecurityRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.store = OpenAgentStore(self.temp.name)
        self.developer = DeveloperPlatform(self.store)
        self.security = SecurityPrivacyManager(self.store)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def developer_deps(self, payload: dict[str, object] | None = None) -> DeveloperRouteDependencies:
        return DeveloperRouteDependencies(
            developer=self.developer,
            parse_bool=_parse_bool,
            read_json=lambda: dict(payload or {}),
        )

    def security_deps(self, payload: dict[str, object] | None = None) -> SecurityRouteDependencies:
        return SecurityRouteDependencies(
            security=self.security,
            security_error=SecurityPrivacyError,
            store=self.store,
            first=_first,
            parse_bool=_parse_bool,
            parse_int=_parse_int,
            read_json=lambda: dict(payload or {}),
        )

    def test_developer_get_and_api_key_post_preserve_response_contracts(self) -> None:
        status = handle_developer_get("/api/developer/status", "", self.developer_deps())
        self.assertIsNotNone(status)
        self.assertIn("api_keys", status.payload)

        created = handle_developer_post(
            "/api/developer/api-keys",
            self.developer_deps({"label": "route test", "scopes": ["read"]}),
        )
        self.assertEqual(created.status, HTTPStatus.CREATED)
        self.assertIn("key", created.payload["api_key"])

        listing = handle_developer_get("/api/developer/api-keys", "", self.developer_deps())
        self.assertEqual(len(listing.payload["api_keys"]), 1)

    def test_developer_graphql_and_validation_errors_are_mapped(self) -> None:
        graphql = handle_developer_post(
            "/api/graphql",
            self.developer_deps({"query": "{ status }"}),
        )
        self.assertIsNotNone(graphql)
        self.assertEqual(graphql.status, HTTPStatus.OK)

        invalid = handle_developer_post(
            "/api/developer/webhooks",
            self.developer_deps({"target_url": "file:///tmp/not-http"}),
        )
        self.assertEqual(invalid.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("error", invalid.payload)

    def test_security_audit_query_and_secret_round_trip_preserve_contracts(self) -> None:
        audit = handle_security_get("/api/audit", "project=core&limit=7", self.security_deps())
        self.assertEqual(audit.payload, {"events": []})

        created = handle_security_post(
            "/api/security/secrets",
            self.security_deps({"name": "route-secret", "value": "value", "passphrase": "passphrase"}),
        )
        self.assertEqual(created.status, HTTPStatus.CREATED)
        secret_id = created.payload["secret"]["id"]

        revealed = handle_security_post(
            "/api/security/secrets/reveal",
            self.security_deps({"secret_id": secret_id, "passphrase": "passphrase"}),
        )
        self.assertEqual(revealed.payload["secret"]["value"], "value")

    def test_declared_routes_are_owned_by_the_extracted_handlers(self) -> None:
        for route in developer_contract()["routes"]:
            result = (
                handle_developer_get(route["path"], "", self.developer_deps())
                if route["method"] == "GET"
                else handle_developer_post(route["path"], self.developer_deps())
            )
            self.assertIsNotNone(result, route)

        for route in security_contract()["routes"]:
            result = (
                handle_security_get(route["path"], "", self.security_deps())
                if route["method"] == "GET"
                else handle_security_post(route["path"], self.security_deps())
            )
            self.assertIsNotNone(result, route)

    def test_runtime_no_longer_contains_migrated_inline_dispatch_branches(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('if parsed.path == "/api/developer/', runtime)
        self.assertNotIn('if parsed.path in {"/api/graphql", "/api/developer/graphql"}', runtime)
        self.assertNotIn('if parsed.path == "/api/security/', runtime)
        self.assertNotIn('if parsed.path == "/api/audit"', runtime)


if __name__ == "__main__":
    unittest.main()
