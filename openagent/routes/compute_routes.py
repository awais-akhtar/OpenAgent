"""Capability and plan routes for the unified compute decision layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..compute_router import ComputeRequest, ComputeRouter
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "compute_router"
ROUTES = (("GET", "/api/compute/status"), ("POST", "/api/compute/plan"))


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ComputeRouteDependencies:
    router: ComputeRouter
    read_json: Any


def handle_get(path: str, deps: ComputeRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/compute/status":
        return JsonRouteResult(deps.router.status())
    return None


def handle_post(path: str, deps: ComputeRouteDependencies) -> JsonRouteResult | None:
    if path != "/api/compute/plan":
        return None
    payload = deps.read_json()
    return JsonRouteResult(
        deps.router.plan(
            ComputeRequest(
                task=str(payload.get("task", "chat")),
                model=str(payload.get("model", "")),
                allow_external=_as_bool(payload.get("allow_external"), False),
                target_latency_ms=max(0, _as_int(payload.get("target_latency_ms"), 0)),
                memory_budget_gb=max(0.0, _as_float(payload.get("memory_budget_gb"), 0.0)),
                require_streaming=_as_bool(payload.get("require_streaming"), False),
            )
        )
    )


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
