"""Research engine and lifecycle HTTP route handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "research"
ROUTES = (
    ("GET", "/api/research/status"),
    ("GET", "/api/research/sessions"),
    ("GET", "/api/research/sessions/{id}"),
    ("GET", "/api/research/sessions/{id}/report"),
    ("GET", "/api/research/sessions/{id}/checkpoints"),
    ("GET", "/api/research/sessions/{id}/source-graph"),
    ("GET", "/api/research/sessions/{id}/source-notes"),
    ("GET", "/api/research/sessions/{id}/stream"),
    ("POST", "/api/research/run"),
    ("POST", "/api/research/sessions"),
    ("POST", "/api/research/sessions/{id}/cancel"),
    ("POST", "/api/research/sessions/{id}/retry"),
    ("POST", "/api/research/sessions/{id}/branch"),
    ("POST", "/api/research/sessions/{id}/archive"),
    ("POST", "/api/research/sessions/{id}/delete"),
    ("POST", "/api/research/sessions/{id}/source-notes/{source_id}"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ResearchRouteDependencies:
    research_engine: Any
    research_lifecycle: Any
    read_json: Callable[[], dict[str, object]]
    auth_scope_owner: Callable[[], str]
    query_owner: Callable
    first: Callable
    parse_int: Callable
    parse_bool: Callable


def handle_get(path: str, query: str, deps: ResearchRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/research/"):
        return None

    params = parse_qs(query)
    first = deps.first

    if path == "/api/research/status":
        return JsonRouteResult(deps.research_engine.status(
            project=first(params, "project") or "default"
        ))

    if path == "/api/research/sessions":
        sessions = deps.research_lifecycle.list(
            project=first(params, "project") or "default",
            owner=deps.auth_scope_owner() or None,
            status=first(params, "status") or "",
            limit=deps.parse_int(first(params, "limit"), default=100),
        )
        return JsonRouteResult({"sessions": sessions})

    # Per-session GETs (/id, /id/report, /id/stream, /id/checkpoints, etc.)
    # are all handled inline in runtime.py with proper owner scoping and SSE
    # support; return None here so the inline handler runs.
    return None


def handle_post(path: str, deps: ResearchRouteDependencies) -> JsonRouteResult | None:
    # All research POST paths are handled inline in runtime.py with proper
    # owner isolation, worker queueing, and 202-ACCEPTED semantics.
    return None
