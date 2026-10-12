"""Physical domain handler registry for the stdlib runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any


HANDLER_MODULES: tuple[str, ...] = (
    "chat_handler",
    "agents_handler",
    "workspace_handler",
    "assistant_handler",
    "twin_handler",
    "semantic_handler",
    "vault_handler",
    "voice_handler",
    "knowledge_handler",
    "finetune_handler",
    "scraper_handler",
    "draft_handler",
    "media_handler",
    "calendar_handler",
    "communication_handler",
    "local_ai_handler",
    "security_handler",
    "developer_handler",
    "documents_handler",
    "workflows_handler",
)


def handler_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for module_name in HANDLER_MODULES:
        module = import_module(f"{__name__}.{module_name}")
        spec = module.handler_spec()
        specs.append(
            {
                "domain": spec.domain,
                "routes_module": spec.routes_module,
                "owned_prefixes": list(spec.owned_prefixes),
                "notes": spec.notes,
            }
        )
    return specs
