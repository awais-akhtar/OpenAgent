"""Scraper route ownership."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="scraper",
        routes_module="openagent.routes.scraper_routes",
        owned_prefixes=("/api/scraper",),
        notes="Approved scraping and scheduled job operations.",
    )
