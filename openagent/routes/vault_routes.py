"""Encrypted vault route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "vault"
ROUTES = (
    ("GET", "/api/vault/status"),
    ("GET", "/api/vault/entries"),
    ("POST", "/api/vault/entries"),
    ("POST", "/api/vault/store"),
    ("POST", "/api/vault/retrieve"),
    ("POST", "/api/vault/rotate-key"),
    ("POST", "/api/vault/export"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
