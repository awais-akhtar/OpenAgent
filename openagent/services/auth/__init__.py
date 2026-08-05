"""Authentication service facades."""

from __future__ import annotations

from .identity import AuthIdentityService
from .scopes import AuthScopeService
from .sessions import AuthSessionService
from .tokens import AuthTokenService

__all__ = [
    "AuthIdentityService",
    "AuthScopeService",
    "AuthSessionService",
    "AuthTokenService",
]
