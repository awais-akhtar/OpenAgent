"""Developer platform domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="developer",
        routes_module="openagent.routes.developer_routes",
        owned_prefixes=("/api/developer", "/api/graphql", "/api/webhooks", "/api/mcp", "/api/terminal"),
        notes="Developer APIs now include MCP runtime and safe terminal surfaces.",
    )
