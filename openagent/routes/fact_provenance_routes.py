"""Fact provenance HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "fact_provenance"
ROUTES = (
    ("GET", "/api/fact-provenance/facts"),
    ("GET", "/api/fact-provenance/facts/{id}"),
    ("GET", "/api/fact-provenance/search"),
    ("GET", "/api/fact-provenance/statistics"),
    ("POST", "/api/fact-provenance/facts"),
    ("POST", "/api/fact-provenance/facts/{id}/retract"),
    ("POST", "/api/fact-provenance/facts/{id}/restore"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class FactProvenanceRouteDependencies:
    fact_provenance: Any
    first: Callable[..., str | None]
    parse_int: Callable[..., int]
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, query: str, deps: FactProvenanceRouteDependencies) -> JsonRouteResult | None:
    params = parse_qs(query)
    project = (deps.first(params, "project") or "default").strip()

    if path == "/api/fact-provenance/facts":
        limit = deps.parse_int(deps.first(params, "limit"), default=100)
        active_only = deps.first(params, "active_only") != "false"
        facts = deps.fact_provenance.list_facts(project, limit=limit, active_only=active_only)
        return JsonRouteResult({"facts": facts})

    if path == "/api/fact-provenance/search":
        search_query = (deps.first(params, "q") or "").strip()
        if not search_query:
            return JsonRouteResult({"error": "Search query (q) is required."}, HTTPStatus.BAD_REQUEST)
        limit = deps.parse_int(deps.first(params, "limit"), default=50)
        results = deps.fact_provenance.search_facts(project, search_query, limit=limit)
        return JsonRouteResult({"results": results})

    if path == "/api/fact-provenance/statistics":
        stats = deps.fact_provenance.get_fact_statistics(project)
        return JsonRouteResult(stats)

    if path.startswith("/api/fact-provenance/facts/"):
        fact_id = path.removeprefix("/api/fact-provenance/facts/").strip("/")
        if not fact_id or "/" in fact_id:
            return JsonRouteResult({"error": "Fact ID is required."}, HTTPStatus.BAD_REQUEST)
        fact = deps.fact_provenance.get_fact(unquote(fact_id), project=project)
        if not fact:
            return JsonRouteResult({"error": "Fact not found."}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult({"fact": fact})

    return None


def handle_post(path: str, deps: FactProvenanceRouteDependencies) -> JsonRouteResult | None:
    payload = deps.read_json()
    project = (str(payload.get("project", "default")) or "default").strip()

    if path == "/api/fact-provenance/facts":
        fact = str(payload.get("fact", "")).strip()
        if not fact:
            return JsonRouteResult({"error": "Fact is required."}, HTTPStatus.BAD_REQUEST)
        try:
            result = deps.fact_provenance.record_fact(
                fact=fact,
                project=project,
                fact_type=str(payload.get("fact_type", "statement")).strip() or "statement",
                confidence=float(payload.get("confidence", 1.0)),
                sources=payload.get("sources") if isinstance(payload.get("sources"), list) else None,
            )
            return JsonRouteResult({"fact": result}, HTTPStatus.CREATED)
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    if path.startswith("/api/fact-provenance/facts/") and path.endswith("/retract"):
        fact_id = path.removeprefix("/api/fact-provenance/facts/").removesuffix("/retract").strip("/")
        if not fact_id or "/" in fact_id:
            return JsonRouteResult({"error": "Fact ID is required."}, HTTPStatus.BAD_REQUEST)
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            return JsonRouteResult({"error": "Retraction reason is required."}, HTTPStatus.BAD_REQUEST)
        try:
            result = deps.fact_provenance.retract_fact(
                unquote(fact_id),
                project,
                retracted_by=str(payload.get("retracted_by", "system")).strip() or "system",
                reason=reason,
                evidence=str(payload.get("evidence", "")).strip(),
            )
            return JsonRouteResult({"fact": result})
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST if "not found" not in str(exc).lower() else HTTPStatus.NOT_FOUND)

    if path.startswith("/api/fact-provenance/facts/") and path.endswith("/restore"):
        fact_id = path.removeprefix("/api/fact-provenance/facts/").removesuffix("/restore").strip("/")
        if not fact_id or "/" in fact_id:
            return JsonRouteResult({"error": "Fact ID is required."}, HTTPStatus.BAD_REQUEST)
        try:
            result = deps.fact_provenance.restore_fact(unquote(fact_id), project)
            return JsonRouteResult({"fact": result})
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST if "not found" not in str(exc).lower() else HTTPStatus.NOT_FOUND)

    return None
