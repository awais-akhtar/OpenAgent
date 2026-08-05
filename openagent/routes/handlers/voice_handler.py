"""Voice domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="voice",
        routes_module="openagent.routes.media_routes",
        owned_prefixes=("/api/voice",),
        notes="Voice upload and session routes are isolated by this handler boundary.",
    )
