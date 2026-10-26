"""Integration hub HTTP route handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


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


@dataclass(frozen=True)
class IntegrationsRouteDependencies:
    integrations: Any
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_bool: Callable[..., bool]
    parse_int: Callable[..., int]
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, query: str, deps: IntegrationsRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/integrations/"):
        return None
    params = parse_qs(query)
    first = deps.first
    integrations = deps.integrations

    if path == "/api/integrations/status":
        project = first(params, "project") or "default"
        return JsonRouteResult(integrations.status(project=project))

    if path == "/api/integrations/pairing-codes":
        project = first(params, "project") or "default"
        include_expired = deps.parse_bool(first(params, "include_expired"), default=True)
        return JsonRouteResult({"codes": integrations.pairing_codes(project=project, include_expired=include_expired)})

    if path == "/api/integrations/preferences":
        project = first(params, "project") or "default"
        scope = first(params, "scope") or ""
        return JsonRouteResult({"preferences": integrations.preferences(project=project, scope=scope)})

    if path == "/api/integrations/diagnostics":
        project = first(params, "project") or "default"
        limit = deps.parse_int(first(params, "limit"), default=20)
        latest = deps.parse_bool(first(params, "latest"), default=False)
        if latest:
            return JsonRouteResult(integrations.diagnostic_snapshot(project=project))
        return JsonRouteResult({"diagnostics": integrations.diagnostics(project=project, limit=limit)})

    if path == "/api/integrations/backups":
        return JsonRouteResult({"backups": integrations.backups_provider()})

    if path == "/api/integrations/safe-shell":
        project = first(params, "project") or "default"
        limit = deps.parse_int(first(params, "limit"), default=20)
        return JsonRouteResult(integrations.safe_commands(project=project, limit=limit))

    if path == "/api/integrations/parity":
        project = first(params, "project") or "default"
        return JsonRouteResult(integrations.feature_matrix(project=project))

    return None


def handle_post(path: str, deps: IntegrationsRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/integrations/"):
        return None
    payload = deps.read_json()
    integrations = deps.integrations

    try:
        if path == "/api/integrations/pairing-codes":
            code = integrations.create_pairing_code(
                label=str(payload.get("label", "Local companion")).strip(),
                scopes=[str(item) for item in payload["scopes"]] if isinstance(payload.get("scopes"), list) else None,
                project=str(payload.get("project", "default")).strip() or "default",
                ttl_minutes=deps.parse_int(str(payload.get("ttl_minutes", "15")), default=15),
            )
            return JsonRouteResult({"code": code}, HTTPStatus.CREATED)

        if path == "/api/integrations/pairing-codes/redeem":
            paired = integrations.redeem_pairing_code(
                code=str(payload.get("code", "")).strip(),
                device_name=str(payload.get("device_name", "Local device")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
            )
            return JsonRouteResult(paired, HTTPStatus.CREATED)

        if path == "/api/integrations/preferences":
            project = str(payload.get("project", "default")).strip() or "default"
            scope = str(payload.get("scope", "ui")).strip() or "ui"
            key = str(payload.get("key", "")).strip()
            if str(payload.get("action", "")).strip().lower() == "delete":
                result = integrations.delete_preference(project=project, scope=scope, key=key)
                return JsonRouteResult(result)
            pref = integrations.set_preference(
                project=project,
                scope=scope,
                key=key,
                value=payload.get("value"),
            )
            return JsonRouteResult({"preference": pref}, HTTPStatus.CREATED)

        if path == "/api/integrations/diagnostics":
            project = str(payload.get("project", "default")).strip() or "default"
            return JsonRouteResult({"diagnostic": integrations.create_diagnostic_report(project=project)}, HTTPStatus.CREATED)

        if path == "/api/integrations/backups":
            project = str(payload.get("project", "default")).strip() or "default"
            return JsonRouteResult(integrations.create_backup(project=project), HTTPStatus.CREATED)

        if path == "/api/integrations/safe-shell":
            result = integrations.run_safe_command(
                command_key=str(payload.get("command_key", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                timeout_seconds=deps.parse_int(str(payload.get("timeout_seconds", "15")), default=15),
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)

    except Exception as exc:
        from openagent.integration_hub import IntegrationHubError
        if isinstance(exc, (IntegrationHubError, ValueError)):
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        raise

    return None
