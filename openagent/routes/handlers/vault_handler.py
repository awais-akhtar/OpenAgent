"""Vault route ownership metadata."""

from __future__ import annotations

from ..runtime_context import DomainHandlerSpec


def handler_spec() -> DomainHandlerSpec:
    return DomainHandlerSpec(
        domain="vault",
        routes_module="openagent.routes.vault_routes",
        owned_prefixes=("/api/vault",),
    )
