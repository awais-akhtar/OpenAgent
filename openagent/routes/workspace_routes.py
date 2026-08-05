"""Workspace item route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "workspace"
ROUTES = (
    ("GET", "/api/workspace/items"),
    ("POST", "/api/workspace/items"),
    ("POST", "/api/workspace/items/update"),
    ("POST", "/api/workspace/templates/instantiate"),
    ("GET", "/api/workspace/dashboard"),
    ("GET", "/api/workspace/canvas"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
