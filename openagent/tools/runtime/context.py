"""Cancellation and tracing state passed to tool handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
import time
from uuid import uuid4

from .schema import ToolRequest


class ToolCancelled(RuntimeError):
    """Raised by a handler that observes a cancelled execution."""


@dataclass(frozen=True)
class TraceContext:
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid4().hex)
    parent_span_id: str = ""

    def child(self) -> "TraceContext":
        return TraceContext(self.trace_id, uuid4().hex, self.span_id)


@dataclass
class ExecutionContext:
    request: ToolRequest
    trace: TraceContext
    cancel_event: Event = field(default_factory=Event)
    started_monotonic: float = field(default_factory=time.monotonic)

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise ToolCancelled("tool execution was cancelled")
        if self.request.timeout_seconds > 0 and time.monotonic() - self.started_monotonic > self.request.timeout_seconds:
            raise TimeoutError("tool execution exceeded its timeout")

    def cancel(self) -> None:
        self.cancel_event.set()
