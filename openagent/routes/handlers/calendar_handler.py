"""Calendar domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="calendar",
        routes_module="openagent.routes.calendar_routes",
        owned_prefixes=("/api/calendar",),
        notes="Calendar and CalDAV sync routes are delegated through calendar_routes.",
    )
