"""Fine-tuning route ownership."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="finetune",
        routes_module="openagent.routes.finetune_routes",
        owned_prefixes=("/api/finetune",),
        notes="Dataset, training job, template, and adapter operations.",
    )
