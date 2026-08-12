"""Security and privacy route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "security"
ROUTES = (
    ("GET", "/api/audit"),
    ("GET", "/api/security/status"),
    ("GET", "/api/security/secrets"),
    ("GET", "/api/security/retention"),
    ("GET", "/api/security/shares"),
    ("GET", "/api/security/egress"),
    ("GET", "/api/security/audit-chain"),
    ("GET", "/api/security/audit-chain/verify"),
    ("GET", "/api/security/privacy-reports"),
    ("GET", "/api/security/privacy-report"),
    ("GET", "/api/security/sandbox-profiles"),
    ("POST", "/api/security/audit-chain/seal"),
    ("POST", "/api/security/egress/request"),
    ("POST", "/api/security/egress/resolve"),
    ("POST", "/api/security/retention/apply"),
    ("POST", "/api/security/sandbox/evaluate"),
    ("POST", "/api/security/secrets/reveal"),
    ("POST", "/api/security/secrets/rotate"),
    ("POST", "/api/security/shares/resolve"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
