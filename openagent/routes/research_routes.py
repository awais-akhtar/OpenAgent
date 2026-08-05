"""Research engine route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "research"
ROUTES = (
    ("GET", "/api/research/status"),
    ("GET", "/api/research/sessions"),
    ("GET", "/api/research/sessions/{id}"),
    ("GET", "/api/research/sessions/{id}/checkpoints"),
    ("GET", "/api/research/sessions/{id}/source-graph"),
    ("GET", "/api/research/sessions/{id}/stream"),
    ("POST", "/api/research/run"),
    ("POST", "/api/research/sessions"),
    ("POST", "/api/research/sessions/{id}/cancel"),
    ("POST", "/api/research/sessions/{id}/retry"),
    ("POST", "/api/research/sessions/{id}/branch"),
    ("POST", "/api/research/sessions/{id}/archive"),
    ("POST", "/api/research/sessions/{id}/delete"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
