"""Adapters for existing OpenAgent execution services."""

from __future__ import annotations

from typing import Any

from .context import ExecutionContext
from .executor import ToolRuntime
from .schema import ToolSpec


def register_safe_terminal(runtime: ToolRuntime, terminal: Any) -> ToolSpec:
    """Expose SafeTerminal without replacing its command allowlist or audit log."""

    spec = ToolSpec(
        id="terminal.run",
        description="Run one local command through the audited SafeTerminal boundary.",
        risk=1,
        input_schema={"type": "object", "required": ["command"]},
        network="none",
        capabilities=("terminal", "audit", "bounded-timeout"),
    )

    def handler(arguments: dict[str, Any], context: ExecutionContext) -> Any:
        context.check()
        command = arguments.get("command", [])
        if isinstance(command, str):
            command = [command]
        return terminal.run(
            command=[str(part) for part in command],
            cwd=str(arguments.get("cwd", "")),
            project=context.request.project,
            owner=context.request.owner,
            timeout_seconds=max(1, int(context.request.timeout_seconds)),
            approved=bool(context.request.approval_id),
        )

    runtime.registry.register(spec, handler)
    return spec
