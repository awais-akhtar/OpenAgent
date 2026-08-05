"""Enterprise administration route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "enterprise"
ROUTES = (
    ("GET", "/api/enterprise/status"),
    ("GET", "/api/enterprise/users"),
    ("GET", "/api/enterprise/roles"),
    ("GET", "/api/enterprise/workspaces"),
    ("GET", "/api/enterprise/policies"),
    ("GET", "/api/enterprise/sso"),
    ("GET", "/api/enterprise/backups"),
    ("GET", "/api/enterprise/disaster-recovery"),
    ("GET", "/api/enterprise/workspace/members"),
    ("GET", "/api/enterprise/org-memory"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
