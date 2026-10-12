"""Authentication, session, two-factor, and API-token HTTP handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from ..auth import AuthError
from . import JsonRouteResult
from ._contracts import route_contract


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
    ("POST", "/api/auth/2fa/disable"),
    ("GET", "/api/auth/scopes"),
    ("GET", "/api/auth/users"),
    ("GET", "/api/auth/roles"),
    ("GET", "/api/auth/policy"),
    ("GET", "/api/auth/audit"),
    ("GET", "/api/auth/api-tokens"),
    ("POST", "/api/auth/api-tokens"),
    ("DELETE", "/api/auth/api-tokens/{token_id}"),
)


def contract() -> dict[str, object]:
    result = route_contract(ROUTE_GROUP, ROUTES)
    result["handling"] = "concrete"
    return result


@dataclass(frozen=True)
class AuthRouteDependencies:
    auth: Any
    read_json: Callable[[], dict[str, object]]
    bearer_token: Callable[[], str]
    bootstrap_token: Callable[[], str]
    request_ip: Callable[[], str]
    user_agent: Callable[[], str]


def _required_user(deps: AuthRouteDependencies) -> tuple[dict[str, Any] | None, JsonRouteResult | None]:
    user = deps.auth.resolve_bearer(deps.bearer_token())
    if user is None:
        return None, JsonRouteResult({"error": "authentication required"}, HTTPStatus.UNAUTHORIZED)
    return user, None


def handle_get(path: str, query: str, deps: AuthRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/auth/scopes":
        return JsonRouteResult({"scopes": deps.auth.list_scopes()})
    if path == "/api/auth/policy":
        return JsonRouteResult(deps.auth.policy())
    if path == "/api/auth/roles":
        return JsonRouteResult({"roles": deps.auth.roles()})
    if path not in {
        "/api/auth/users",
        "/api/auth/audit",
        "/api/auth/me",
        "/api/auth/sessions",
        "/api/auth/api-tokens",
    }:
        return None

    user, error = _required_user(deps)
    if error is not None:
        return error
    assert user is not None
    if path in {"/api/auth/users", "/api/auth/audit"} and user.get("role") != "owner":
        return JsonRouteResult({"error": "owner role required"}, HTTPStatus.FORBIDDEN)
    if path == "/api/auth/users":
        return JsonRouteResult({"users": deps.auth.list_users()})
    if path == "/api/auth/audit":
        params = parse_qs(query)
        values = params.get("limit") or []
        try:
            limit = int(values[0]) if values else 50
        except (TypeError, ValueError):
            limit = 50
        return JsonRouteResult(deps.auth.auth_audit_summary(limit=limit))
    if path == "/api/auth/me":
        return JsonRouteResult({"user": user})
    if path == "/api/auth/sessions":
        return JsonRouteResult({"sessions": deps.auth.list_sessions(user["user_id"])})
    return JsonRouteResult({"tokens": deps.auth.list_api_tokens(user["user_id"])})


def handle_post(path: str, deps: AuthRouteDependencies) -> JsonRouteResult | None:
    if path not in {route_path for method, route_path in ROUTES if method == "POST"}:
        return None
    payload = deps.read_json()

    if path == "/api/auth/register":
        registering_user = deps.auth.resolve_bearer(deps.bearer_token())
        try:
            user = deps.auth.register(
                str(payload.get("username", "")),
                str(payload.get("email", "")),
                str(payload.get("password", "")),
                display_name=str(payload.get("display_name", "")),
                bootstrap_token=deps.bootstrap_token() or str(payload.get("bootstrap_token", "")),
                authorized=bool(registering_user and registering_user.get("role") == "owner"),
            )
        except (ValueError, AuthError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"user": user}, HTTPStatus.CREATED)

    if path == "/api/auth/login":
        try:
            result = deps.auth.login(
                str(payload.get("username_or_email", "")),
                str(payload.get("password", "")),
                otp_code=str(payload.get("otp_code", "")),
                ip=deps.request_ip(),
                user_agent=deps.user_agent(),
            )
        except AuthError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.UNAUTHORIZED)
        return JsonRouteResult(result)

    if path == "/api/auth/refresh":
        try:
            return JsonRouteResult(deps.auth.refresh_session(str(payload.get("refresh_token", ""))))
        except AuthError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.UNAUTHORIZED)

    if path == "/api/auth/logout":
        return JsonRouteResult(deps.auth.logout(deps.bearer_token()))

    user, error = _required_user(deps)
    if error is not None:
        return error
    assert user is not None
    user_id = user["user_id"]

    if path == "/api/auth/sessions/revoke":
        try:
            return JsonRouteResult(deps.auth.revoke_session(user_id, str(payload.get("session_id", ""))))
        except KeyError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    if path == "/api/auth/2fa/setup":
        return JsonRouteResult(deps.auth.setup_two_factor(user_id))
    if path == "/api/auth/2fa/verify":
        try:
            return JsonRouteResult(deps.auth.verify_two_factor(user_id, str(payload.get("otp_code", ""))))
        except AuthError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    if path == "/api/auth/2fa/disable":
        try:
            return JsonRouteResult(deps.auth.disable_two_factor(user_id, str(payload.get("otp_code", ""))))
        except AuthError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    if path == "/api/auth/api-tokens":
        try:
            token = deps.auth.create_api_token(
                user_id,
                str(payload.get("name", "")),
                scopes=payload.get("scopes") if isinstance(payload.get("scopes"), list) else None,
                expires_at=str(payload.get("expires_at", "")),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(token, HTTPStatus.CREATED)
    return None


def handle_delete(path: str, deps: AuthRouteDependencies) -> JsonRouteResult | None:
    prefix = "/api/auth/api-tokens/"
    if not path.startswith(prefix):
        return None
    user, error = _required_user(deps)
    if error is not None:
        return error
    assert user is not None
    token_id = unquote(path.removeprefix(prefix)).strip()
    try:
        return JsonRouteResult(deps.auth.revoke_api_token(user["user_id"], token_id))
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
