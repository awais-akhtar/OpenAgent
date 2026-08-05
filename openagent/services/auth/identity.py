"""Identity service for local OpenAgent accounts."""

from __future__ import annotations

from typing import Any

from ...auth import AuthManager


class AuthIdentityService:
    """Small service facade over account registration, login, and lookup."""

    def __init__(self, auth: AuthManager) -> None:
        self.auth = auth

    def register(
        self,
        *,
        username: str,
        email: str,
        password: str,
        display_name: str = "",
    ) -> dict[str, Any]:
        return self.auth.register(username, email, password, display_name)

    def login(
        self,
        *,
        identifier: str,
        password: str,
        totp_code: str = "",
        ip: str = "",
        user_agent: str = "",
    ) -> dict[str, Any]:
        return self.auth.login(
            identifier,
            password,
            otp_code=totp_code,
            ip=ip,
            user_agent=user_agent,
        )

    def me(self, access_token: str) -> dict[str, Any] | None:
        return self.auth.resolve_bearer(access_token)
