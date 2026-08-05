"""Enterprise identity and passkey routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import unquote, urlparse

from ..auth import AuthError
from ..enterprise_identity import EnterpriseIdentity
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "enterprise-identity"
ROUTES = (
    ("GET", "/api/auth/identity/status"), ("GET", "/api/auth/passkeys"), ("GET", "/api/auth/devices"),
    ("GET", "/api/auth/federation/policies"),
    ("POST", "/api/auth/passkeys/registration/options"), ("POST", "/api/auth/passkeys/registration/verify"),
    ("POST", "/api/auth/passkeys/authentication/options"), ("POST", "/api/auth/passkeys/authentication/verify"),
    ("POST", "/api/auth/recovery/rotate"), ("POST", "/api/auth/devices/trust"),
    ("POST", "/api/auth/devices/{id}/revoke"), ("POST", "/api/auth/devices/verify"),
    ("POST", "/api/auth/federation/policies"), ("POST", "/api/auth/federation/evaluate"),
    ("DELETE", "/api/auth/passkeys/{id}"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class IdentityRouteDependencies:
    identity: EnterpriseIdentity
    auth: Any
    read_json: Any
    resolve_user: Any
    request_origin: Any
    request_ip: Any
    user_agent: Any


def _require_user(deps: IdentityRouteDependencies, *, admin: bool = False) -> tuple[dict[str, Any] | None, JsonRouteResult | None]:
    user = deps.resolve_user()
    if user is None:
        return None, JsonRouteResult({"error": "authentication required"}, HTTPStatus.UNAUTHORIZED)
    if admin and user.get("role") not in {"owner", "admin"}:
        return None, JsonRouteResult({"error": "administrator role required"}, HTTPStatus.FORBIDDEN)
    return user, None


def _rp_context(deps: IdentityRouteDependencies) -> tuple[str, str]:
    origin = deps.request_origin()
    return str(urlparse(origin).hostname or ""), origin


def handle_get(path: str, deps: IdentityRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/auth/"):
        return None
    if path == "/api/auth/identity/status":
        user = deps.resolve_user()
        return JsonRouteResult(deps.identity.status(user_id=str(user["user_id"]) if user else ""))
    if path == "/api/auth/passkeys":
        user, error = _require_user(deps)
        return error or JsonRouteResult({"passkeys": deps.identity.passkeys(str(user["user_id"]))})
    if path == "/api/auth/devices":
        user, error = _require_user(deps)
        return error or JsonRouteResult({"devices": deps.identity.devices(str(user["user_id"]))})
    if path == "/api/auth/federation/policies":
        _, error = _require_user(deps, admin=True)
        return error or JsonRouteResult({"policies": deps.identity.federation_policies()})
    return None


def handle_post(path: str, deps: IdentityRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/auth/"):
        return None
    supported_paths = {
        "/api/auth/passkeys/authentication/options",
        "/api/auth/passkeys/authentication/verify",
        "/api/auth/devices/verify",
        "/api/auth/passkeys/registration/options",
        "/api/auth/passkeys/registration/verify",
        "/api/auth/recovery/rotate",
        "/api/auth/devices/trust",
        "/api/auth/federation/policies",
        "/api/auth/federation/evaluate",
    }
    if path not in supported_paths and not (path.startswith("/api/auth/devices/") and path.endswith("/revoke")):
        return None
    payload = deps.read_json()
    try:
        if path == "/api/auth/passkeys/authentication/options":
            rp_id, origin = _rp_context(deps)
            return JsonRouteResult(deps.identity.authentication_options(str(payload.get("username_or_email", "")), rp_id=rp_id, origin=origin))
        if path == "/api/auth/passkeys/authentication/verify":
            credential = payload.get("credential") if isinstance(payload.get("credential"), dict) else {}
            return JsonRouteResult(deps.identity.finish_authentication(str(payload.get("ceremony_id", "")), credential,
                ip=deps.request_ip(), user_agent=deps.user_agent()))
        if path == "/api/auth/devices/verify":
            device = deps.identity.verify_device(str(payload.get("device_token", "")), str(payload.get("fingerprint", "")))
            return JsonRouteResult({"trusted": device is not None, "device": device or {}})

        user, error = _require_user(deps)
        if error:
            return error
        user_id = str(user["user_id"])
        if path == "/api/auth/passkeys/registration/options":
            rp_id, origin = _rp_context(deps)
            return JsonRouteResult(deps.identity.registration_options(user_id, rp_id=rp_id, origin=origin))
        if path == "/api/auth/passkeys/registration/verify":
            credential = payload.get("credential") if isinstance(payload.get("credential"), dict) else {}
            return JsonRouteResult({"passkey": deps.identity.finish_registration(user_id, str(payload.get("ceremony_id", "")), credential, label=str(payload.get("label", "")))}, HTTPStatus.CREATED)
        if path == "/api/auth/recovery/rotate":
            return JsonRouteResult(deps.auth.rotate_recovery_codes(user_id, count=int(payload.get("count", 8))))
        if path == "/api/auth/devices/trust":
            return JsonRouteResult({"device": deps.identity.trust_device(user_id, name=str(payload.get("name", "")),
                fingerprint=str(payload.get("fingerprint", "")), trust_level=str(payload.get("trust_level", "standard")), days=int(payload.get("days", 30)))}, HTTPStatus.CREATED)
        if path.startswith("/api/auth/devices/") and path.endswith("/revoke"):
            device_id = unquote(path.removeprefix("/api/auth/devices/").removesuffix("/revoke")).strip("/")
            return JsonRouteResult(deps.identity.revoke_device(user_id, device_id))
        if path == "/api/auth/federation/policies":
            if user.get("role") not in {"owner", "admin"}:
                return JsonRouteResult({"error": "administrator role required"}, HTTPStatus.FORBIDDEN)
            return JsonRouteResult({"policy": deps.identity.save_federation_policy(
                provider=str(payload.get("provider", "")), issuer=str(payload.get("issuer", "")), audience=str(payload.get("audience", "")),
                allowed_domains=[str(item) for item in payload.get("allowed_domains", [])] if isinstance(payload.get("allowed_domains"), list) else [],
                required_groups=[str(item) for item in payload.get("required_groups", [])] if isinstance(payload.get("required_groups"), list) else [],
                default_role=str(payload.get("default_role", "member")), enabled=bool(payload.get("enabled", True)))}, HTTPStatus.CREATED)
        if path == "/api/auth/federation/evaluate":
            if user.get("role") not in {"owner", "admin"}:
                return JsonRouteResult({"error": "administrator role required"}, HTTPStatus.FORBIDDEN)
            claims = payload.get("claims") if isinstance(payload.get("claims"), dict) else {}
            return JsonRouteResult(deps.identity.evaluate_federation(str(payload.get("provider", "")), claims))
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except (AuthError, ValueError, TypeError, RuntimeError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None


def handle_delete(path: str, deps: IdentityRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/auth/passkeys/"):
        return None
    user, error = _require_user(deps)
    if error:
        return error
    passkey_id = unquote(path.removeprefix("/api/auth/passkeys/")).strip("/")
    try:
        return JsonRouteResult(deps.identity.delete_passkey(str(user["user_id"]), passkey_id))
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
