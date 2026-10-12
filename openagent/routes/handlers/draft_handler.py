"""Editor draft route ownership."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="drafts",
        routes_module="openagent.routes.draft_routes",
        owned_prefixes=("/api/editor-drafts",),
        notes="Owner-scoped draft reads, writes, updates, and deletion.",
    )
