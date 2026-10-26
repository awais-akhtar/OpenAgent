"""Legacy attribution policy routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "legacy_attribution"
ROUTES = (
    ("GET", "/api/legacy-attribution/policy"),
    ("GET", "/api/legacy-attribution/verify"),
    ("POST", "/api/legacy-attribution/validate-ownership"),
    ("POST", "/api/legacy-attribution/validate-creation"),
    ("POST", "/api/legacy-attribution/validate-modification"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class LegacyAttributionRouteDependencies:
    legacy_attribution: Any
    first: Callable[..., str | None]
    read_json: Callable[[], dict[str, Any]]
    auth_scope_owner: Callable[[], str | None]


def handle_get(path: str, query: str, deps: LegacyAttributionRouteDependencies) -> JsonRouteResult | None:
    current_owner = deps.auth_scope_owner()
    if not current_owner:
        return JsonRouteResult({"error": "Authentication required."}, HTTPStatus.UNAUTHORIZED)

    if path == "/api/legacy-attribution/policy":
        policy = deps.legacy_attribution.get_policy_summary()
        return JsonRouteResult({"policy": policy})

    if path == "/api/legacy-attribution/verify":
        return JsonRouteResult({
            "authenticated": True,
            "owner": current_owner,
            "policy_enabled": True,
            "enforcement": "strict_ownership_boundary",
        })

    return None


def handle_post(path: str, deps: LegacyAttributionRouteDependencies) -> JsonRouteResult | None:
    current_owner = deps.auth_scope_owner()
    if not current_owner:
        return JsonRouteResult({"error": "Authentication required."}, HTTPStatus.UNAUTHORIZED)

    payload = deps.read_json()

    if path == "/api/legacy-attribution/validate-ownership":
        record_id = str(payload.get("record_id", "")).strip()
        resource_type = str(payload.get("resource_type", "")).strip()
        requested_owner = str(payload.get("requested_owner", "")).strip() or None

        if not record_id or not resource_type:
            return JsonRouteResult(
                {"error": "record_id and resource_type are required."},
                HTTPStatus.BAD_REQUEST,
            )

        try:
            result = deps.legacy_attribution.validate_ownership(
                record_id,
                resource_type,
                current_owner,
                requested_owner=requested_owner,
            )
            return JsonRouteResult(result)
        except Exception as exc:
            deps.legacy_attribution.audit_cross_owner_denial(
                "read",
                resource_type,
                current_owner,
                requested_owner or "unknown",
            )
            return JsonRouteResult(
                {"error": str(exc)},
                HTTPStatus.FORBIDDEN,
            )

    if path == "/api/legacy-attribution/validate-creation":
        resource_type = str(payload.get("resource_type", "")).strip()
        if not resource_type:
            return JsonRouteResult(
                {"error": "resource_type is required."},
                HTTPStatus.BAD_REQUEST,
            )

        try:
            result = deps.legacy_attribution.validate_creation(
                resource_type,
                current_owner,
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        except ValueError as exc:
            return JsonRouteResult(
                {"error": str(exc)},
                HTTPStatus.BAD_REQUEST,
            )

    if path == "/api/legacy-attribution/validate-modification":
        record_id = str(payload.get("record_id", "")).strip()
        resource_type = str(payload.get("resource_type", "")).strip()
        record_owner = str(payload.get("record_owner", "")).strip()
        modification_type = str(payload.get("modification_type", "update")).strip() or "update"

        if not record_id or not resource_type or not record_owner:
            return JsonRouteResult(
                {"error": "record_id, resource_type, and record_owner are required."},
                HTTPStatus.BAD_REQUEST,
            )

        try:
            result = deps.legacy_attribution.validate_modification(
                record_id,
                resource_type,
                record_owner,
                current_owner,
                modification_type=modification_type,
            )
            return JsonRouteResult(result)
        except Exception as exc:
            deps.legacy_attribution.audit_cross_owner_denial(
                modification_type,
                resource_type,
                current_owner,
                record_owner,
                context=f"record:{record_id}",
            )
            return JsonRouteResult(
                {"error": str(exc)},
                HTTPStatus.FORBIDDEN,
            )

    return None
