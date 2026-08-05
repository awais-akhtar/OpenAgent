"""Authentication and token route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "auth"
ROUTES = (
    ("POST", "/api/auth/register"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/refresh"),
    ("POST", "/api/auth/logout"),
    ("GET", "/api/auth/me"),
    ("GET", "/api/auth/sessions"),
    ("POST", "/api/auth/sessions/revoke"),
    ("POST", "/api/auth/2fa/setup"),
    ("POST", "/api/auth/2fa/verify"),
    ("GET", "/api/auth/scopes"),
    ("GET", "/api/auth/users"),
    ("GET", "/api/auth/roles"),
    ("GET", "/api/auth/policy"),
    ("GET", "/api/auth/audit"),
    ("GET", "/api/auth/api-tokens"),
    ("POST", "/api/auth/api-tokens"),
    ("DELETE", "/api/auth/api-tokens/{token_id}"),
    ("GET", "/api/auth/identity/status"),
    ("GET", "/api/auth/passkeys"),
    ("GET", "/api/auth/devices"),
    ("POST", "/api/auth/passkeys/registration/options"),
    ("POST", "/api/auth/passkeys/registration/verify"),
    ("POST", "/api/auth/passkeys/authentication/options"),
    ("POST", "/api/auth/passkeys/authentication/verify"),
    ("POST", "/api/auth/recovery/rotate"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
