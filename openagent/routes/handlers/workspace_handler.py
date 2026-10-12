"""Workspace route ownership metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="workspace",
        routes_module="openagent.routes.workspace_routes",
        owned_prefixes=("/api/workspace",),
    )
