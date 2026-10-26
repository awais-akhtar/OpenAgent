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
    ("GET", "/api/vault/access-requests"),
    ("POST", "/api/vault/entries"),
    ("POST", "/api/vault/store"),
    ("POST", "/api/vault/retrieve"),
    ("POST", "/api/vault/rotate-key"),
    ("POST", "/api/vault/export"),
    ("POST", "/api/vault/access-requests"),
    ("PATCH", "/api/vault/access-requests/{request_id}"),
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
    if path not in {"/api/vault/status", "/api/vault/entries", "/api/vault/access-log", "/api/vault/access-requests"}:
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
    if path == "/api/vault/access-requests":
        try:
            requests = deps.vault.list_access_requests(
                project=project,
                limit=deps.parse_int(deps.first(params, "limit"), default=50),
            )
            return JsonRouteResult({"access_requests": requests})
        except (AttributeError, NotImplementedError):
            # If vault service doesn't implement access requests yet, return empty
            return JsonRouteResult({"access_requests": []})
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
    if path == "/api/vault/access-requests":
        # Create a new access request
        reason = str(payload.get("reason", "")).strip()
        requested_entries = payload.get("requested_entries", []) if isinstance(payload.get("requested_entries"), list) else []
        if not reason or not requested_entries:
            return JsonRouteResult({"error": "reason and requested_entries are required"}, HTTPStatus.BAD_REQUEST)
        try:
            request = deps.vault.create_access_request(
                reason=reason,
                requested_entries=requested_entries,
                project=project,
                duration_minutes=int(payload.get("duration_minutes", 0)) or 60,
            )
            return JsonRouteResult({"access_request": request}, HTTPStatus.CREATED)
        except (AttributeError, NotImplementedError, ValueError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
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


def handle_patch(path: str, deps: VaultRouteDependencies) -> JsonRouteResult | None:
    """Handle PATCH requests for vault access request review/approval."""
    prefix = "/api/vault/access-requests/"
    if not path.startswith(prefix) or path == prefix:
        return None

    request_id = path.removeprefix(prefix).strip("/")
    if not request_id or "/" in request_id:
        return JsonRouteResult({"error": "request_id is required"}, HTTPStatus.BAD_REQUEST)

    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    action = str(payload.get("action", "")).strip().lower()

    if action not in {"approve", "deny"}:
        return JsonRouteResult({"error": "action must be 'approve' or 'deny'"}, HTTPStatus.BAD_REQUEST)

    try:
        if action == "approve":
            result = deps.vault.approve_access_request(request_id, project=project)
        else:  # deny
            result = deps.vault.deny_access_request(request_id, project=project)
        return JsonRouteResult({"access_request": result})
    except (AttributeError, NotImplementedError, KeyError, PermissionError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
