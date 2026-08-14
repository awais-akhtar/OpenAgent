"""Serializable request and result types used by the tool runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ToolSpec:
    """The stable description of one executable capability."""

    id: str
    description: str
    risk: int = 1
    input_schema: dict[str, Any] = field(default_factory=dict)
    network: str = "none"
    capabilities: tuple[str, ...] = ()
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("tool id is required")
        if self.risk < 0 or self.risk > 3:
            raise ValueError("tool risk must be between 0 and 3")
        if self.network not in {"none", "local", "external"}:
            raise ValueError("tool network must be none, local, or external")

    @property
    def risk_label(self) -> str:
        return ("automatic", "sandbox", "approval", "denied")[self.risk]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "risk": self.risk,
            "risk_label": self.risk_label,
            "input_schema": self.input_schema,
            "network": self.network,
            "capabilities": list(self.capabilities),
            "version": self.version,
        }


@dataclass(frozen=True)
class ToolRequest:
    """A single attempt to run a registered tool."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    project: str = "default"
    owner: str = "local"
    workspace: str = ""
    network: str = "none"
    timeout_seconds: float = 30.0
    approval_id: str = ""
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid4().hex)
    attempt: int = 1

    def __post_init__(self) -> None:
        if not self.tool.strip():
            raise ValueError("tool id is required")
        if self.network not in {"none", "local", "external"}:
            raise ValueError("request network must be none, local, or external")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("request timeout must be a finite positive number")
        if self.attempt < 1:
            raise ValueError("request attempt must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "project": self.project,
            "owner": self.owner,
            "workspace": self.workspace,
            "network": self.network,
            "timeout_seconds": self.timeout_seconds,
            "approval_id": self.approval_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "attempt": self.attempt,
        }


@dataclass
class ToolResult:
    """Bounded, inspectable output from a tool attempt."""

    success: bool
    status: str
    tool: str
    output: Any = None
    error: str = ""
    trace_id: str = ""
    span_id: str = ""
    started_at: str = field(default_factory=_now)
    completed_at: str = field(default_factory=_now)
    duration_ms: float = 0.0
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "tool": self.tool,
            "output": self.output,
            "error": self.error,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": round(float(self.duration_ms), 2),
            "artifacts": self.artifacts,
        }
