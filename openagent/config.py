"""Validated configuration for local and server OpenAgent processes."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import sys


class StartupConfigurationError(RuntimeError):
    """Raised when OpenAgent would start with an unsafe network boundary."""


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise StartupConfigurationError(f"{name} must be a boolean value")


def is_loopback_host(host: str) -> bool:
    """Return whether a bind target is restricted to this machine."""

    normalized = host.strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class StartupSettings:
    """Security-sensitive settings resolved once before a server is bound."""

    host: str
    port: int
    deployment_mode: str
    auth_enforced: bool
    unsafe_network_override: bool
    json_body_limit_bytes: int
    upload_limit_bytes: int
    trusted_origins: tuple[str, ...]

    @property
    def loopback_only(self) -> bool:
        return is_loopback_host(self.host)

    def validate(self) -> None:
        if not 0 <= self.port <= 65535:
            raise StartupConfigurationError("port must be between 0 and 65535")
        if self.json_body_limit_bytes < 1024:
            raise StartupConfigurationError("OPENAGENT_JSON_BODY_LIMIT_BYTES must be at least 1024")
        if self.upload_limit_bytes < self.json_body_limit_bytes:
            raise StartupConfigurationError(
                "OPENAGENT_UPLOAD_LIMIT_BYTES cannot be smaller than OPENAGENT_JSON_BODY_LIMIT_BYTES"
            )
        if self.loopback_only:
            return
        if self.deployment_mode in {"team", "enterprise"} and self.auth_enforced:
            return
        if self.unsafe_network_override:
            return
        raise StartupConfigurationError(
            "Refusing a non-loopback bind without team/enterprise mode and authentication. "
            "Set OPENAGENT_DEPLOYMENT_MODE=team and OPENAGENT_AUTH_ENFORCED=1. "
            "For an isolated development environment only, explicitly set "
            "OPENAGENT_UNSAFE_ALLOW_UNAUTHENTICATED_NETWORK=1."
        )


def startup_settings(host: str, port: int) -> StartupSettings:
    """Load and validate process settings for a requested network bind."""

    settings = StartupSettings(
        host=host.strip() or "127.0.0.1",
        port=int(port),
        deployment_mode=deployment_mode(),
        auth_enforced=auth_enforcement_enabled(),
        unsafe_network_override=_env_bool("OPENAGENT_UNSAFE_ALLOW_UNAUTHENTICATED_NETWORK"),
        json_body_limit_bytes=int(os.environ.get("OPENAGENT_JSON_BODY_LIMIT_BYTES", 1_048_576)),
        upload_limit_bytes=int(os.environ.get("OPENAGENT_UPLOAD_LIMIT_BYTES", 52_428_800)),
        trusted_origins=tuple(
            origin.strip()
            for origin in os.environ.get("OPENAGENT_TRUSTED_ORIGINS", "").split(",")
            if origin.strip()
        ),
    )
    settings.validate()
    return settings


def default_home() -> Path:
    configured = os.environ.get("OPENAGENT_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".openagent"


def desktop_home() -> Path:
    """Return the shared desktop data directory used by source and installed launches."""

    configured = os.environ.get("OPENAGENT_HOME")
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / "OpenAgent" / "data"
    return default_home()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def web_root() -> Path:
    candidates = [
        project_root() / "web",
        Path(__file__).resolve().parent / "web",
        Path(sys.prefix) / "web",
    ]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return candidates[0]


def auth_enforcement_enabled() -> bool:
    """Whether bearer-token + scope checks are enforced on protected routes.

    Off by default so existing local-first deployments (and the existing
    test suite) keep working unmodified. Operators opt in via
    OPENAGENT_AUTH_ENFORCED once they have registered an account through
    POST /api/auth/register — see openagent/auth.py for the scope policy
    that applies once this is turned on.
    """
    explicit = os.environ.get("OPENAGENT_AUTH_ENFORCED", "").strip()
    if explicit:
        return _env_bool("OPENAGENT_AUTH_ENFORCED")
    return deployment_mode() in {"team", "enterprise"}


def deployment_mode() -> str:
    """Return the local deployment mode.

    single_user keeps the first-run local app frictionless. team and
    enterprise make protected-route auth default-on unless explicitly disabled
    with OPENAGENT_AUTH_ENFORCED=0 for a controlled local maintenance window.
    """

    mode = os.environ.get("OPENAGENT_DEPLOYMENT_MODE", "single_user").strip().lower().replace("-", "_")
    if mode in {"team", "enterprise"}:
        return mode
    return "single_user"
