"""Digital twin route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract


ROUTE_GROUP = "twin"
ROUTES = (
    ("GET", "/api/twin/status"),
    ("GET", "/api/twin/profile"),
    ("GET", "/api/twin/patterns"),
    ("GET", "/api/twin/suggestions"),
    ("GET", "/api/twin/goals"),
    ("POST", "/api/twin/profile"),
    ("POST", "/api/twin/interaction"),
    ("POST", "/api/twin/predict"),
    ("POST", "/api/twin/goals"),
    ("POST", "/api/twin/consolidate"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
