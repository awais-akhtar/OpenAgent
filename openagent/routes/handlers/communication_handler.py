"""Communication domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="communication",
        routes_module="openagent.routes.communication_routes",
        owned_prefixes=("/api/communication",),
        notes="Provider certification, OAuth callback, sync, and attachment routes live here.",
    )
