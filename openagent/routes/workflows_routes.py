"""Workspace automation and workflow route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "workflows"
ROUTES = (
    ("GET", "/api/workflows"),
    ("POST", "/api/workflows"),
    ("GET", "/api/workflows/builder"),
    ("GET", "/api/workflows/visual"),
    ("POST", "/api/workflows/run"),
    ("POST", "/api/workflows/{id}/run"),
    ("POST", "/api/workflows/run-due"),
    ("POST", "/api/workflows/event"),
    ("GET", "/api/workflows/layout"),
    ("POST", "/api/workflows/layout"),
    ("GET", "/api/workflow-runs"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
