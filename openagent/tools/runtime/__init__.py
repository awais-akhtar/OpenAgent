"""Shared execution contracts for local tools and optional integrations.

The runtime is deliberately small. Existing services keep ownership of their
domain behaviour; this package gives them one request, policy, trace, and
result envelope at the orchestration boundary.
"""

from .adapters import register_safe_terminal
from .context import ExecutionContext, ToolCancelled, TraceContext
from .executor import ToolRuntime
from .policy import PermissionPolicy, ToolDecision
from .registry import ToolRegistry
from .schema import ToolRequest, ToolResult, ToolSpec

__all__ = [
    "ExecutionContext",
    "PermissionPolicy",
    "ToolCancelled",
    "ToolDecision",
    "ToolRegistry",
    "ToolRequest",
    "ToolResult",
    "ToolRuntime",
    "ToolSpec",
    "TraceContext",
    "register_safe_terminal",
]
