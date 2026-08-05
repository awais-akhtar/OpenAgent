"""Session lifecycle service for local OpenAgent auth."""

from __future__ import annotations

from typing import Any

from ...auth import AuthManager


class AuthSessionService:
    """Owns logout and session revocation workflows."""

    def __init__(self, auth: AuthManager) -> None:
        self.auth = auth

    def logout(self, access_token: str) -> dict[str, Any]:
        return self.auth.logout(access_token)

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        return self.auth.list_sessions(user_id)

    def revoke(self, user_id: str, session_id: str) -> dict[str, Any]:
        return self.auth.revoke_session(user_id, session_id)
