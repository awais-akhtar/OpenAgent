"""Digital twin route ownership metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="twin",
        routes_module="openagent.routes.twin_routes",
        owned_prefixes=("/api/twin",),
    )
