"""Advanced local RAG route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs

from ..advanced_rag import AdvancedRAGEngine
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "advanced_rag"
ROUTES = (
    ("GET", "/api/rag/status"),
    ("POST", "/api/rag/query"),
    ("POST", "/api/rag/sync"),
    ("POST", "/api/rag/benchmark"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class AdvancedRAGRouteDependencies:
    rag: AdvancedRAGEngine
    read_json: Any


def handle_get(path: str, query: str, deps: AdvancedRAGRouteDependencies) -> JsonRouteResult | None:
    if path != "/api/rag/status":
        return None
    params = parse_qs(query)
    project = _first(params, "project") or "default"
    return JsonRouteResult(deps.rag.status(project=project))


def handle_post(path: str, deps: AdvancedRAGRouteDependencies) -> JsonRouteResult | None:
    if path not in {"/api/rag/query", "/api/rag/sync", "/api/rag/benchmark"}:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/rag/query":
            result = deps.rag.query(
                str(payload.get("query", "")),
                project=project,
                limit=_as_int(payload.get("limit"), 8),
                context_token_budget=_as_int(payload.get("context_token_budget"), 1200),
                generate=_as_bool(payload.get("generate"), False),
                provider=str(payload.get("provider", "auto")).strip() or "auto",
                model=str(payload.get("model", "")).strip() or None,
                allow_external=_as_bool(payload.get("allow_external"), False),
                use_cache=_as_bool(payload.get("use_cache"), True),
                embedding_provider=str(payload.get("embedding_provider", "")).strip(),
            )
            return JsonRouteResult(result)
        if path == "/api/rag/sync":
            result = deps.rag.knowledge.sync_vector_index(
                project=project,
                force=_as_bool(payload.get("force"), False),
                embedding_provider=str(payload.get("embedding_provider", "")).strip(),
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        result = deps.rag.benchmark(
            project=project or "rag-benchmark",
            iterations=_as_int(payload.get("iterations"), 5),
        )
        return JsonRouteResult(result, HTTPStatus.CREATED)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
