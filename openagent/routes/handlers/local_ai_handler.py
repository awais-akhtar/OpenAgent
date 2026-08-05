"""Local AI domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="local_ai",
        routes_module="openagent.routes.local_ai_routes",
        owned_prefixes=("/api/local-ai", "/api/model-cookbook", "/api/models"),
        notes="Local model operations, setup jobs, and cookbook serve sessions.",
    )
