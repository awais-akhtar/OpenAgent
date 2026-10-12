"""Semantic memory route ownership metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="semantic",
        routes_module="openagent.routes.semantic_routes",
        owned_prefixes=("/api/semantic",),
    )
