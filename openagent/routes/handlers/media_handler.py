"""Media route ownership metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="media",
        routes_module="openagent.routes.media_routes",
        owned_prefixes=("/api/media",),
        notes="Media ingestion, asset analysis, and voice session routes share the media service boundary.",
    )
