"""Workflow domain handler metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="workflows",
        routes_module="openagent.routes.workflows_routes",
        owned_prefixes=("/api/workflows", "/api/worker", "/api/tasks"),
        notes="Workflow, durable queue, and scheduler paths are grouped here.",
    )
