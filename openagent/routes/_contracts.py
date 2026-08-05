"""Shared route-contract helpers for domain modules."""

from __future__ import annotations

from typing import Any


Route = tuple[str, str]


def route_contract(group: str, routes: tuple[Route, ...]) -> dict[str, Any]:
    return {
        "group": group,
        "routes": [{"method": method, "path": path} for method, path in routes],
        "route_count": len(routes),
    }


def not_handled(*_: Any, **__: Any) -> None:
    return None
