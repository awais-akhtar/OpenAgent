"""Vector store route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs

from ..vector_store import VectorStore
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "vector_store"
ROUTES = (
    ("GET", "/api/vector-store/status"),
    ("GET", "/api/vector-store/collections"),
    ("POST", "/api/vector-store/upsert"),
    ("POST", "/api/vector-store/search"),
    ("POST", "/api/vector-store/benchmark"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class VectorRouteDependencies:
    vector_store: VectorStore
    read_json: Any


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def _parse_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def handle_get(path: str, query: str, deps: VectorRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/vector-store/status":
        params = parse_qs(query)
        project = _first(params, "project") or "default"
        return JsonRouteResult(deps.vector_store.status(project=project))
    if path == "/api/vector-store/collections":
        params = parse_qs(query)
        project = _first(params, "project") or "default"
        return JsonRouteResult({"collections": deps.vector_store.collections(project=project)})
    return None


def handle_post(path: str, deps: VectorRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/vector-store/upsert":
        payload = deps.read_json()
        try:
            record = deps.vector_store.upsert_text(
                collection=str(payload.get("collection", "default")).strip() or "default",
                content=str(payload.get("content", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                source_id=str(payload.get("source_id", "")).strip(),
                chunk_id=str(payload.get("chunk_id", "")).strip(),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                dimensions=_parse_int(payload.get("dimensions"), 128),
                embedding_provider=str(payload.get("embedding_provider", "local-hash")).strip() or "local-hash",
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"record": record}, HTTPStatus.CREATED)
    if path == "/api/vector-store/search":
        payload = deps.read_json()
        try:
            result = deps.vector_store.search(
                str(payload.get("query", "")).strip(),
                collection=str(payload.get("collection", "default")).strip() or "default",
                project=str(payload.get("project", "default")).strip() or "default",
                limit=_parse_int(payload.get("limit"), 10),
                embedding_provider=str(payload.get("embedding_provider", "local-hash")).strip() or "local-hash",
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)
    if path == "/api/vector-store/benchmark":
        payload = deps.read_json()
        result = deps.vector_store.benchmark(
            project=str(payload.get("project", "default")).strip() or "default",
            collection=str(payload.get("collection", "benchmark")).strip() or "benchmark",
        )
        return JsonRouteResult(result, HTTPStatus.CREATED)
    return None
