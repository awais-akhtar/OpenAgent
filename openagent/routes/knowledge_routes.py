"""Knowledge and RAG route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "knowledge"
ROUTES = (
    ("GET", "/api/knowledge/status"),
    ("GET", "/api/knowledge/search"),
    ("GET", "/api/knowledge/graph"),
    ("GET", "/api/knowledge/tags"),
    ("GET", "/api/knowledge/timeline"),
    ("POST", "/api/knowledge/ingest"),
    ("POST", "/api/knowledge/upload"),
    ("POST", "/api/knowledge/semantic-search"),
    ("POST", "/api/knowledge/hybrid-search"),
    ("POST", "/api/knowledge/synthesize"),
    ("POST", "/api/knowledge/compress"),
    ("POST", "/api/knowledge/replay"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
