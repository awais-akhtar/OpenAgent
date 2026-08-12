"""Developer platform route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "developer"
ROUTES = (
    ("GET", "/api/developer/status"),
    ("GET", "/api/developer/schema"),
    ("GET", "/api/developer/webhooks"),
    ("GET", "/api/developer/webhook-deliveries"),
    ("GET", "/api/developer/plugins"),
    ("GET", "/api/developer/tools"),
    ("GET", "/api/developer/mcp"),
    ("GET", "/api/developer/mcp/servers"),
    ("GET", "/api/developer/extensions"),
    ("GET", "/api/developer/sdk"),
    ("GET", "/api/developer/api-keys"),
    ("POST", "/api/developer/api-keys/verify"),
    ("POST", "/api/developer/tools/invoke"),
    ("POST", "/api/developer/webhooks/dispatch"),
    ("POST", "/api/graphql"),
    ("POST", "/api/developer/graphql"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
