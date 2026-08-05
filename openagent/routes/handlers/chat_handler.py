"""Chat domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="chat",
        routes_module="openagent.routes.chat_routes",
        owned_prefixes=("/api/chat", "/api/conversations"),
        notes="Chat route blocks are ready for extraction behind this handler boundary.",
    )
