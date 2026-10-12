"""Security and privacy HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


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
    ("POST", "/api/security/secrets"),
    ("POST", "/api/security/audit-chain/seal"),
    ("POST", "/api/security/audit-chain/verify"),
    ("POST", "/api/security/privacy-report"),
    ("POST", "/api/security/egress/request"),
    ("POST", "/api/security/egress/resolve"),
    ("POST", "/api/security/retention"),
    ("POST", "/api/security/retention/apply"),
    ("POST", "/api/security/sandbox-profiles"),
    ("POST", "/api/security/sandbox/evaluate"),
    ("POST", "/api/security/secrets/reveal"),
    ("POST", "/api/security/secrets/rotate"),
    ("POST", "/api/security/shares"),
    ("POST", "/api/security/shares/resolve"),
)
POST_PATHS = frozenset(path for method, path in ROUTES if method == "POST")


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class SecurityRouteDependencies:
    security: Any
    security_error: type[Exception]
    store: Any
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_bool: Callable[..., bool]
    parse_int: Callable[..., int]
    read_json: Callable[[], dict[str, object]]


def handle_get(path: str, query: str, deps: SecurityRouteDependencies) -> JsonRouteResult | None:
    security = deps.security
    if path == "/api/audit":
        params = parse_qs(query)
        return JsonRouteResult({
            "events": deps.store.list_audit_events(
                project=deps.first(params, "project"),
                limit=deps.parse_int(deps.first(params, "limit"), default=100),
            )
        })
    if path == "/api/security/status":
        return JsonRouteResult(security.status())
    if path == "/api/security/secrets":
        return JsonRouteResult({"secrets": security.secrets()})
    if path == "/api/security/retention":
        return JsonRouteResult({"policies": security.retention_policies()})
    if path == "/api/security/shares":
        return JsonRouteResult({"grants": security.share_grants()})
    if path == "/api/security/egress":
        return JsonRouteResult({"requests": security.egress_requests()})
    if path == "/api/security/audit-chain":
        return JsonRouteResult({"chain": security.audit_chain()})
    if path == "/api/security/audit-chain/verify":
        return JsonRouteResult(security.verify_audit_chain())
    if path in {"/api/security/privacy-reports", "/api/security/privacy-report"}:
        params = parse_qs(query)
        return JsonRouteResult({
            "reports": security.privacy_reports(),
            "project": deps.first(params, "project") or "default",
        })
    if path == "/api/security/sandbox-profiles":
        return JsonRouteResult({"profiles": security.sandbox_profiles()})
    return None


def handle_post(path: str, deps: SecurityRouteDependencies) -> JsonRouteResult | None:
    if path not in POST_PATHS:
        return None

    payload = deps.read_json()
    security = deps.security
    parse_bool = deps.parse_bool
    try:
        if path == "/api/security/secrets":
            value = security.store_secret(
                name=str(payload.get("name", "")).strip(),
                value=str(payload.get("value", "")),
                passphrase=str(payload.get("passphrase", "")),
                description=str(payload.get("description", "")),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"secret": value}, HTTPStatus.CREATED)
        if path == "/api/security/secrets/reveal":
            try:
                value = security.reveal_secret(
                    secret_id=str(payload.get("secret_id", "")).strip(),
                    passphrase=str(payload.get("passphrase", "")),
                )
            except (deps.security_error, ValueError) as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return JsonRouteResult({"secret": value})
        if path == "/api/security/secrets/rotate":
            value = security.rotate_secret(
                secret_id=str(payload.get("secret_id", "")).strip(),
                value=str(payload.get("value", "")),
                passphrase=str(payload.get("passphrase", "")),
            )
            return JsonRouteResult({"secret": value})
        if path == "/api/security/retention":
            value = security.create_retention_policy(
                resource_type=str(payload.get("resource_type", "")).strip(),
                retention_days=deps.parse_int(str(payload.get("retention_days", "365")), default=365),
                action=str(payload.get("action", "delete")).strip() or "delete",
                status=str(payload.get("status", "active")).strip() or "active",
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"policy": value}, HTTPStatus.CREATED)
        if path == "/api/security/retention/apply":
            return JsonRouteResult(security.apply_retention(
                policy_id=str(payload.get("policy_id", "")).strip(),
                dry_run=parse_bool(payload.get("dry_run"), default=True),
            ))
        if path == "/api/security/shares":
            value = security.create_share_grant(
                resource_type=str(payload.get("resource_type", "")).strip(),
                resource_id=str(payload.get("resource_id", "")).strip(),
                grantee=str(payload.get("grantee", "")).strip(),
                scope=str(payload.get("scope", "read")).strip() or "read",
                expires_at=str(payload.get("expires_at", "")).strip(),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"grant": value}, HTTPStatus.CREATED)
        if path == "/api/security/shares/resolve":
            value = security.resolve_share_grant(
                grant_id=str(payload.get("grant_id", "")).strip(),
                status=str(payload.get("status", "revoked")).strip() or "revoked",
            )
            return JsonRouteResult({"grant": value})
        if path == "/api/security/egress/request":
            value = security.request_egress(
                requester=str(payload.get("requester", "user")).strip() or "user",
                purpose=str(payload.get("purpose", "")).strip(),
                destination=str(payload.get("destination", "")).strip(),
                resource_type=str(payload.get("resource_type", "unknown")).strip() or "unknown",
                resource_id=str(payload.get("resource_id", "")).strip(),
                risk_level=str(payload.get("risk_level", "medium")).strip() or "medium",
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"request": value}, HTTPStatus.CREATED)
        if path == "/api/security/egress/resolve":
            value = security.resolve_egress_request(
                request_id=str(payload.get("request_id", "")).strip(),
                approved=parse_bool(payload.get("approved"), default=False),
                approved_by=str(payload.get("approved_by", "user")).strip() or "user",
            )
            return JsonRouteResult({"request": value})
        if path == "/api/security/audit-chain/seal":
            return JsonRouteResult(security.seal_audit_events(), HTTPStatus.CREATED)
        if path == "/api/security/audit-chain/verify":
            return JsonRouteResult(security.verify_audit_chain())
        if path == "/api/security/privacy-report":
            return JsonRouteResult({"report": security.create_privacy_report()}, HTTPStatus.CREATED)
        if path == "/api/security/sandbox-profiles":
            value = security.create_sandbox_profile(
                name=str(payload.get("name", "")).strip(),
                permissions=[str(item) for item in payload.get("permissions", [])]
                if isinstance(payload.get("permissions"), list) else [],
                network_allowed=parse_bool(payload.get("network_allowed"), default=False),
                filesystem=payload.get("filesystem") if isinstance(payload.get("filesystem"), dict) else {},
                env={str(key): str(value) for key, value in payload.get("env", {}).items()}
                if isinstance(payload.get("env"), dict) else {},
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"profile": value}, HTTPStatus.CREATED)
        if path == "/api/security/sandbox/evaluate":
            return JsonRouteResult(security.evaluate_sandbox_request(
                profile_id=str(payload.get("profile_id", "")).strip(),
                permissions=[str(item) for item in payload.get("permissions", [])]
                if isinstance(payload.get("permissions"), list) else [],
                network=parse_bool(payload.get("network"), default=False),
            ))
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None
