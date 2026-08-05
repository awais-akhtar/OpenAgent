"""Scope catalog and route-policy service."""

from __future__ import annotations

from ...auth import AuthManager, required_scope_for_path


class AuthScopeService:
    """Provides the route-scope contract used by auth gates and tests."""

    def __init__(self, auth: AuthManager) -> None:
        self.auth = auth

    def catalog(self) -> list[dict[str, str]]:
        return self.auth.list_scopes()

    def required_scope(self, path: str, method: str = "GET") -> str | None:
        return required_scope_for_path(path, method)

    def is_allowed(self, granted_scopes: list[str], path: str, method: str = "GET") -> bool:
        required = self.required_scope(path, method)
        return required is None or required in set(granted_scopes)
