from __future__ import annotations

from unittest import TestCase
from unittest.mock import Mock

from openagent.routes.legacy_attribution_routes import (
    LegacyAttributionRouteDependencies,
    handle_get,
    handle_post,
)


class LegacyAttributionDispatchTests(TestCase):
    def test_unrelated_get_is_not_claimed_without_an_owner(self):
        deps = LegacyAttributionRouteDependencies(
            legacy_attribution=Mock(),
            first=lambda params, key: None,
            read_json=lambda: {},
            auth_scope_owner=lambda: "",
        )
        self.assertIsNone(handle_get("/api/health", "", deps))

    def test_unrelated_post_is_not_claimed_without_an_owner(self):
        deps = LegacyAttributionRouteDependencies(
            legacy_attribution=Mock(),
            first=lambda params, key: None,
            read_json=lambda: {},
            auth_scope_owner=lambda: "",
        )
        self.assertIsNone(handle_post("/api/chat", deps))

