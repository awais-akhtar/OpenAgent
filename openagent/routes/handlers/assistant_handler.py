"""Assistant route ownership metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="assistant",
        routes_module="openagent.routes.assistant_routes",
        owned_prefixes=("/api/assistant",),
    )
