"""HTTP route helpers for the local OpenAgent server.

The stdlib server remains the runtime entrypoint, but domain route modules can
now own URL contracts incrementally. This lets server.py become a facade without
forcing a framework migration or breaking existing desktop/web URLs.
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class JsonRouteResult:
    """A JSON payload plus status code returned by a route module."""

    payload: dict[str, Any]
    status: HTTPStatus = HTTPStatus.OK


@dataclass(frozen=True)
class StreamRouteResult:
    """An event iterator returned by a streaming domain route."""

    events: Any


DOMAIN_ROUTE_MODULES: tuple[str, ...] = (
    "health_routes",
    "core_routes",
    "integrations_routes",
    "chat_routes",
    "agents_routes",
    "workflows_routes",
    "knowledge_routes",
    "semantic_routes",
    "media_routes",
    "local_ai_routes",
    "enterprise_routes",
    "developer_routes",
    "security_routes",
    "calendar_routes",
    "communication_routes",
    "assistant_routes",
    "personal_os_routes",
    "production_routes",
    "twin_routes",
    "vault_routes",
    "scraper_routes",
    "research_routes",
    "finetune_routes",
    "workspace_routes",
    "auth_routes",
    "document_routes",
    "draft_routes",
    "worker_routes",
    "vector_routes",
    "advanced_rag_routes",
    "inference_routes",
    "ultra_model_routes",
    "adaptive_engine_routes",
    "contacts_routes",
    "backup_routes",
    "model_cookbook_routes",
    "document_processing_routes",
    "image_studio_routes",
    "scheduler_routes",
    "terminal_routes",
    "mcp_runtime_routes",
    "tool_runtime_routes",
    "compute_routes",
    "agent_bridge_routes",
    "skill_routes",
    "conversation_routes",
    "identity_routes",
    "phd_routes",
    "science_routes",
    "codebase_routes",
)


def route_contracts() -> list[dict[str, Any]]:
    """Return the declared HTTP contract for every domain route module."""

    contracts: list[dict[str, Any]] = []
    for module_name in DOMAIN_ROUTE_MODULES:
        module = import_module(f"{__name__}.{module_name}")
        contracts.append(module.contract())
    return contracts
