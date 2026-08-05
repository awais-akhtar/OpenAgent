"""Core runtime, context, capability, and event route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract


ROUTE_GROUP = "core"
ROUTES = (
    ("GET", "/api/capabilities"),
    ("GET", "/api/core/runtime"),
    ("GET", "/api/mission/features"),
    ("GET", "/api/context"),
    ("GET", "/api/events"),
    ("GET", "/api/chat/diagnose"),
    ("POST", "/api/events"),
    ("POST", "/api/capabilities/hard-audit"),
    ("POST", "/api/mission/run-check"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
