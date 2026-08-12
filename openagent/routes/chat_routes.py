"""Chat and conversation route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "chat"
ROUTES = (
    ("GET", "/api/conversations"),
    ("GET", "/api/messages"),
    ("GET", "/api/conversation/messages"),
    ("GET", "/api/chat/search"),
    ("GET", "/api/chat/export"),
    ("GET", "/api/chat/replay"),
    ("POST", "/api/chat"),
    ("POST", "/api/chat/stream"),
    ("POST", "/api/chat/branch"),
    ("POST", "/api/chat/import"),
    ("POST", "/api/chat/artifact"),
    ("POST", "/api/chat/research-record"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
