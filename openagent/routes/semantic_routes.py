"""Semantic memory route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

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


handle_get = not_handled
handle_post = not_handled
