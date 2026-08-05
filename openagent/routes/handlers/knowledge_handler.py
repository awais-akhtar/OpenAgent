"""Knowledge domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="knowledge",
        routes_module="openagent.routes.knowledge_routes",
        owned_prefixes=("/api/knowledge", "/api/semantic", "/api/vector"),
        notes="Knowledge, semantic memory, and vector routes share local RAG state.",
    )
