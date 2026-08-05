"""Document lifecycle route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "document"
ROUTES = (
    ("POST", "/api/document"),
    ("GET", "/api/documents/library"),
    ("GET", "/api/document/{doc_id}"),
    ("PUT", "/api/document/{doc_id}"),
    ("PATCH", "/api/document/{doc_id}"),
    ("POST", "/api/document/{doc_id}/archive"),
    ("DELETE", "/api/document/{doc_id}"),
    ("GET", "/api/document/{doc_id}/versions"),
    ("GET", "/api/document/{doc_id}/version/{num}"),
    ("POST", "/api/document/{doc_id}/restore/{num}"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
