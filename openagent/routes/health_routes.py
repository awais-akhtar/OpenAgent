"""Health, system telemetry, and UI-build route contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "health"
ROUTES = (
    ("GET", "/api/health"),
    ("GET", "/api/system/metrics"),
    ("GET", "/api/ui/build"),
    ("GET", "/api/modules/health"),
    ("GET", "/api/routes/handlers"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class HealthRouteDependencies:
    """Callables supplied by the server bootstrap layer."""

    health: Callable[[], dict[str, Any]]
    system_metrics: Callable[[], dict[str, Any]]
    ui_build_info: Callable[[], dict[str, Any]]
    module_health: Callable[[str], dict[str, Any]]
    handler_specs: Callable[[], list[dict[str, Any]]] | None = None


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def handle_get(path: str, query: str, deps: HealthRouteDependencies) -> JsonRouteResult | None:
    """Return a health-route response for *path*, or None when not handled."""

    if path == "/api/health":
        return JsonRouteResult(deps.health())
    if path == "/api/system/metrics":
        return JsonRouteResult(deps.system_metrics())
    if path == "/api/ui/build":
        return JsonRouteResult(deps.ui_build_info())
    if path == "/api/modules/health":
        params = parse_qs(query)
        project = _first(params, "project") or "default"
        return JsonRouteResult(deps.module_health(project))
    if path == "/api/routes/handlers":
        specs = deps.handler_specs() if deps.handler_specs is not None else []
        return JsonRouteResult({"handlers": specs, "count": len(specs)})
    return None
