"""Personal assistant route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract


ROUTE_GROUP = "assistant"
ROUTES = (
    ("GET", "/api/assistant/tasks"),
    ("GET", "/api/assistant/goals"),
    ("GET", "/api/assistant/daily-brief"),
    ("GET", "/api/assistant/coach"),
    ("GET", "/api/assistant/crm"),
    ("POST", "/api/assistant/profile"),
    ("POST", "/api/assistant/goals"),
    ("POST", "/api/assistant/tasks"),
    ("POST", "/api/assistant/tasks/complete"),
    ("POST", "/api/assistant/meeting/brief"),
    ("POST", "/api/assistant/meeting/summary"),
    ("POST", "/api/assistant/travel/plan"),
    ("POST", "/api/assistant/crm"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
