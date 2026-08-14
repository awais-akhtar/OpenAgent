"""In-process registry for typed tool handlers."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from .context import ExecutionContext
from .schema import ToolSpec


ToolHandler = Callable[[dict[str, Any], ExecutionContext], Any]


@dataclass(frozen=True)
class ToolRegistration:
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    """Thread-safe registry shared by local agents and route handlers."""

    def __init__(self) -> None:
        self._items: dict[str, ToolRegistration] = {}
        self._lock = RLock()

    def register(self, spec: ToolSpec, handler: ToolHandler) -> ToolSpec:
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        with self._lock:
            self._items[spec.id] = ToolRegistration(spec, handler)
        return spec

    def get(self, tool_id: str) -> ToolRegistration:
        with self._lock:
            try:
                return self._items[tool_id]
            except KeyError as exc:
                raise KeyError(f"tool is not registered: {tool_id}") from exc

    def list(self) -> list[ToolSpec]:
        with self._lock:
            return [item.spec for item in sorted(self._items.values(), key=lambda value: value.spec.id)]
