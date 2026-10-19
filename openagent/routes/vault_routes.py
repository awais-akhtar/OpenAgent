"""Encrypted vault HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "vault"
ROUTES = (
    ("GET", "/api/vault/status"),
    ("GET", "/api/vault/entries"),
    ("GET", "/api/vault/access-log"),
    ("POST", "/api/vault/entries"),
    ("POST", "/api/vault/store"),
    ("POST", "/api/vault/retrieve"),
    ("POST", "/api/vault/rotate-key"),
    ("POST", "/api/vault/export"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class VaultRouteDependencies:
    vault: Any
    first: Callable[..., str | None]
    parse_int: Callable[..., int]
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, query: str, deps: VaultRouteDependencies) -> JsonRouteResult | None:
    if path not in {"/api/vault/status", "/api/vault/entries", "/api/vault/access-log"}:
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    if path == "/api/vault/status":
        return JsonRouteResult(deps.vault.status(project=project))
    if path == "/api/vault/access-log":
        return JsonRouteResult({"access_log": deps.vault.list_access_log(
            project=project,
            limit=deps.parse_int(deps.first(params, "limit"), default=100),
        )})
    return JsonRouteResult({"entries": deps.vault.list_entries(
        project=project,
        category=deps.first(params, "category") or "",
        tag=deps.first(params, "tag") or "",
        limit=deps.parse_int(deps.first(params, "limit"), default=100),
    )})


def handle_post(path: str, deps: VaultRouteDependencies) -> JsonRouteResult | None:
    if path not in {route for method, route in ROUTES if method == "POST"}:
        return None
    payload = deps.read_json()
    passphrase = str(payload.get("passphrase", "")).strip()
    project = str(payload.get("project", "default")).strip() or "default"
    if path in {"/api/vault/store", "/api/vault/entries"}:
        secret = str(payload.get("secret", "")).strip()
        label = str(payload.get("label", "")).strip()
        if not passphrase or not secret or not label:
            return JsonRouteResult({"error": "label, secret, and passphrase are required"}, HTTPStatus.BAD_REQUEST)
        try:
            entry = deps.vault.store_secret(
                label, secret, passphrase, project=project,
                category=str(payload.get("category", "secret")),
                tags=[str(item) for item in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else None,
                notes=str(payload.get("notes", "")),
            )
        except (ValueError, PermissionError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"entry": entry}, HTTPStatus.CREATED)
    if path == "/api/vault/retrieve":
        entry_id = str(payload.get("entry_id", "")).strip()
        if not entry_id or not passphrase:
            return JsonRouteResult({"error": "entry_id and passphrase are required"}, HTTPStatus.BAD_REQUEST)
        try:
            secret_data = deps.vault.retrieve_secret(entry_id, passphrase, project=project)
        except (KeyError, ValueError, PermissionError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"secret": secret_data})
    if path == "/api/vault/rotate-key":
        old_passphrase = str(payload.get("old_passphrase", "")).strip()
        new_passphrase = str(payload.get("new_passphrase", "")).strip()
        if not old_passphrase or not new_passphrase:
            return JsonRouteResult({"error": "old_passphrase and new_passphrase are required"}, HTTPStatus.BAD_REQUEST)
        try:
            result = deps.vault.rotate_key(old_passphrase, new_passphrase, project=project)
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)
    if not passphrase:
        return JsonRouteResult({"error": "passphrase required"}, HTTPStatus.BAD_REQUEST)
    return JsonRouteResult(deps.vault.export_vault(passphrase, project=project))
