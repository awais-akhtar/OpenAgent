"""Integration hub route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract


ROUTE_GROUP = "integrations"
ROUTES = (
    ("GET", "/api/integrations/status"),
    ("GET", "/api/integrations/pairing-codes"),
    ("POST", "/api/integrations/pairing-codes"),
    ("POST", "/api/integrations/pairing-codes/redeem"),
    ("GET", "/api/integrations/preferences"),
    ("POST", "/api/integrations/preferences"),
    ("GET", "/api/integrations/diagnostics"),
    ("POST", "/api/integrations/diagnostics"),
    ("GET", "/api/integrations/backups"),
    ("POST", "/api/integrations/backups"),
    ("GET", "/api/integrations/safe-shell"),
    ("POST", "/api/integrations/safe-shell"),
    ("GET", "/api/integrations/parity"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
