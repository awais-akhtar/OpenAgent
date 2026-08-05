"""Configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path
import sys


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
    explicit = os.environ.get("OPENAGENT_AUTH_ENFORCED", "").strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if explicit in {"0", "false", "no", "off"}:
        return False
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
