"""Semantic memory route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs

from ..semantic_memory import SemanticMemoryManager
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "semantic"
ROUTES = (
    ("GET", "/api/semantic/status"),
    ("GET", "/api/semantic/memories"),
    ("POST", "/api/semantic/memories"),
    ("GET", "/api/semantic/collections"),
    ("POST", "/api/semantic/collections"),
    ("GET", "/api/semantic/graph-analytics"),
    ("POST", "/api/semantic/consolidate"),
    ("POST", "/api/semantic/vectors"),
    ("POST", "/api/semantic/vector-search"),
    ("POST", "/api/semantic/rerank"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class SemanticRouteDependencies:
    semantic_memory: SemanticMemoryManager
    read_json: Any


def _first(params: dict[str, list[str]], key: str) -> str | None:
    value = params.get(key, [None])[0]
    return value or None


def _parse_int(value: str | None, *, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def handle_get(path: str, query: str, deps: SemanticRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/semantic/status":
        params = parse_qs(query)
        project = _first(params, "project") or "default"
        return JsonRouteResult(deps.semantic_memory.status(project=project))
    if path == "/api/semantic/memories":
        params = parse_qs(query)
        project = _first(params, "project") or "default"
        search_query = _first(params, "q") or ""
        memory_type = _first(params, "memory_type") or ""
        limit = _parse_int(_first(params, "limit"), default=50)
        memories = (
            deps.semantic_memory.search_memories(search_query, project=project, limit=limit)
            if search_query
            else deps.semantic_memory.list_memories(project=project, memory_type=memory_type, limit=limit)
        )
        return JsonRouteResult({"memories": memories})
    if path == "/api/semantic/collections":
        params = parse_qs(query)
        project = _first(params, "project") or "default"
        return JsonRouteResult({"collections": deps.semantic_memory.collections(project=project)})
    if path == "/api/semantic/graph-analytics":
        params = parse_qs(query)
        project = _first(params, "project") or "default"
        return JsonRouteResult(deps.semantic_memory.graph_analytics(project=project))
    return None


def handle_post(path: str, deps: SemanticRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/semantic/memories":
        payload = deps.read_json()
        try:
            memory = deps.semantic_memory.add_memory(
                title=str(payload.get("title", "")).strip(),
                content=str(payload.get("content", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                memory_type=str(payload.get("memory_type", "semantic")).strip() or "semantic",
                profile_id=str(payload.get("profile_id", "")).strip(),
                tags=[str(item) for item in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else [],
                source=str(payload.get("source", "manual")).strip() or "manual",
                confidence=float(payload.get("confidence", 1.0) or 1.0),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
        except (TypeError, ValueError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"memory": memory}, HTTPStatus.CREATED)
    if path == "/api/semantic/consolidate":
        payload = deps.read_json()
        result = deps.semantic_memory.consolidate(
            project=str(payload.get("project", "default")).strip() or "default",
            limit=_parse_int(str(payload.get("limit", "100")), default=100),
        )
        return JsonRouteResult(result, HTTPStatus.CREATED)
    if path == "/api/semantic/collections":
        payload = deps.read_json()
        collection = deps.semantic_memory.create_collection(
            str(payload.get("name", "default")).strip() or "default",
            project=str(payload.get("project", "default")).strip() or "default",
            dimensions=_parse_int(str(payload.get("dimensions", "64")), default=64),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )
        return JsonRouteResult({"collection": collection}, HTTPStatus.CREATED)
    if path == "/api/semantic/vectors":
        payload = deps.read_json()
        vector = payload.get("vector")
        try:
            record = deps.semantic_memory.upsert_vector_record(
                str(payload.get("collection_id", "")).strip(),
                content=str(payload.get("content", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                external_id=str(payload.get("external_id", "")).strip(),
                vector=[float(item) for item in vector] if isinstance(vector, list) else None,
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
        except (TypeError, ValueError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"record": record}, HTTPStatus.CREATED)
    if path == "/api/semantic/vector-search":
        payload = deps.read_json()
        try:
            results = deps.semantic_memory.vector_search(
                str(payload.get("collection_id", "")).strip(),
                str(payload.get("query", "")).strip(),
                limit=_parse_int(str(payload.get("limit", "10")), default=10),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"results": results})
    if path == "/api/semantic/rerank":
        payload = deps.read_json()
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            return JsonRouteResult({"error": "candidates must be a list."}, HTTPStatus.BAD_REQUEST)
        result = deps.semantic_memory.rerank(
            str(payload.get("query", "")).strip(),
            [item for item in candidates if isinstance(item, dict)],
            project=str(payload.get("project", "default")).strip() or "default",
            limit=_parse_int(str(payload.get("limit", "10")), default=10),
        )
        return JsonRouteResult(result, HTTPStatus.CREATED)
    return None
