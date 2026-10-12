"""Agent route ownership metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="agents",
        routes_module="openagent.routes.agents_routes",
        owned_prefixes=("/api/agents",),
    )
