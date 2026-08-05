"""Security domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="security",
        routes_module="openagent.routes.security_routes",
        owned_prefixes=("/api/security", "/api/auth", "/api/vault"),
        notes="Auth policy and security/privacy surfaces are grouped by this boundary.",
    )
