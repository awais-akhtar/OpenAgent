"""Editor draft route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "draft"
ROUTES = (
    ("GET", "/api/editor-drafts"),
    ("POST", "/api/editor-drafts"),
    ("GET", "/api/editor-drafts/{draft_id}"),
    ("PUT", "/api/editor-drafts/{draft_id}"),
    ("DELETE", "/api/editor-drafts/{draft_id}"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
