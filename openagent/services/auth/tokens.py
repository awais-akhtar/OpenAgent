"""API-token service for local OpenAgent auth."""

from __future__ import annotations

from typing import Any

from ...auth import AuthManager


class AuthTokenService:
    """Creates, lists, revokes, and resolves scoped API tokens."""

    def __init__(self, auth: AuthManager) -> None:
        self.auth = auth

    def create(
        self,
        *,
        user_id: str,
        name: str,
        scopes: list[str] | None = None,
        expires_at: str = "",
    ) -> dict[str, Any]:
        return self.auth.create_api_token(user_id, name, scopes=scopes, expires_at=expires_at)

    def list(self, user_id: str) -> list[dict[str, Any]]:
        return self.auth.list_api_tokens(user_id)

    def revoke(self, user_id: str, token_id: str) -> dict[str, Any]:
        return self.auth.revoke_api_token(user_id, token_id)

    def resolve(self, token: str) -> dict[str, Any] | None:
        return self.auth.resolve_bearer(token)
