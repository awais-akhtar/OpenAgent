"""Agent runtime route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "agents"
ROUTES = (
    ("GET", "/api/agents"),
    ("POST", "/api/agents/run"),
    ("POST", "/api/agents/run-checked"),
    ("POST", "/api/agents/delegate"),
    ("POST", "/api/agents/parallel"),
    ("GET", "/api/agents/marketplace"),
    ("GET", "/api/agents/permissions"),
    ("GET", "/api/agents/approvals"),
    ("POST", "/api/agents/permissions/grant"),
    ("POST", "/api/agents/approvals/request"),
    ("POST", "/api/agents/approvals/resolve"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
