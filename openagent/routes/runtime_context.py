"""Shared route-dispatch metadata for extracted domain handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import JsonRouteResult


RouteCallable = Callable[[str, str, Any], JsonRouteResult | None]


@dataclass(frozen=True)
class DomainHandlerSpec:
    """Describes a physical domain handler module."""

    domain: str
    routes_module: str
    owned_prefixes: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class RuntimeRouteContext:
    """Thin facade object passed to future extracted route handlers."""

    services: dict[str, Any]
    helpers: dict[str, Any]

    def service(self, name: str) -> Any:
        return self.services[name]

    def helper(self, name: str) -> Any:
        return self.helpers[name]
