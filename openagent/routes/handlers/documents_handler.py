"""Documents domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="documents",
        routes_module="openagent.routes.document_routes",
        owned_prefixes=("/api/document", "/api/documents", "/api/drafts"),
        notes="Editor documents plus deep import/export processing boundaries.",
    )
