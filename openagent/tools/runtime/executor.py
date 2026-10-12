"""Execution engine shared by local tools and optional adapters."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
import json
import time
from typing import Any

from ...models import AuditEvent
from .context import ExecutionContext, ToolCancelled, TraceContext
from .policy import PermissionPolicy
from .registry import ToolRegistry
from .schema import ToolRequest, ToolResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolRuntime:
    """Run registered tools with one policy and one result envelope."""

    def __init__(self, store: Any = None, *, max_output_bytes: int = 100_000) -> None:
        self.store = store
        self.max_output_bytes = max(1_000, int(max_output_bytes))
        self.registry = ToolRegistry()
        self.policy = PermissionPolicy()

    def describe(self) -> dict[str, Any]:
        return {
            "tools": [spec.to_dict() for spec in self.registry.list()],
            "policy": {
                "local_first": True,
                "external_network": "approval_id required",
                "result_limit_bytes": self.max_output_bytes,
            },
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        started_at = _now()
        started = time.monotonic()
        context: ExecutionContext | None = None
        future: Any = None
        try:
            registration = self.registry.get(request.tool)
        except KeyError as exc:
            return self._finish(request, started_at, started, False, "failed", error=str(exc))
        decision = self.policy.decide(registration.spec, request)
        if not decision.allowed:
            result = self._finish(request, started_at, started, False, decision.status, error=decision.reason)
            self._audit(request, result)
            return result
        try:
            self._validate_arguments(registration.spec.input_schema, request.arguments)
            context = ExecutionContext(request, TraceContext(request.trace_id, request.span_id))
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openagent-tool")
            future = pool.submit(registration.handler, request.arguments, context)
            try:
                output = future.result(timeout=max(0.1, float(request.timeout_seconds)))
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            result = self._finish(request, started_at, started, True, "succeeded", output=self._bound(output))
        except FutureTimeout:
            if context is not None:
                context.cancel()
            if future is not None:
                future.cancel()
            result = self._finish(request, started_at, started, False, "timeout", error="tool execution exceeded its timeout")
        except ToolCancelled as exc:
            result = self._finish(request, started_at, started, False, "cancelled", error=str(exc))
        except Exception as exc:
            result = self._finish(request, started_at, started, False, "failed", error=str(exc))
        self._audit(request, result)
        return result

    def execute_named(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        project: str = "default",
        owner: str = "local",
        approval_id: str = "",
        timeout_seconds: float = 30.0,
        network: str = "none",
    ) -> ToolResult:
        return self.execute(
            ToolRequest(
                tool=tool,
                arguments=arguments or {},
                project=project,
                owner=owner,
                approval_id=approval_id,
                timeout_seconds=timeout_seconds,
                network=network,
            )
        )

    @staticmethod
    def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        required = schema.get("required", []) if isinstance(schema, dict) else []
        missing = [str(name) for name in required if str(name) not in arguments]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")

    def _bound(self, value: Any) -> Any:
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            encoded = json.dumps(str(value))
        if len(encoded.encode("utf-8")) <= self.max_output_bytes:
            return value
        return {
            "truncated": True,
            "preview": encoded[: self.max_output_bytes // 2],
            "limit_bytes": self.max_output_bytes,
        }

    def _finish(
        self,
        request: ToolRequest,
        started_at: str,
        started: float,
        success: bool,
        status: str,
        *,
        output: Any = None,
        error: str = "",
    ) -> ToolResult:
        return ToolResult(
            success=success,
            status=status,
            tool=request.tool,
            output=output,
            error=error,
            trace_id=request.trace_id,
            span_id=request.span_id,
            started_at=started_at,
            completed_at=_now(),
            duration_ms=(time.monotonic() - started) * 1000,
        )

    def _audit(self, request: ToolRequest, result: ToolResult) -> None:
        if self.store is None:
            return
        self.store.add_audit_event(
            AuditEvent(
                actor="tool-runtime",
                action="tool.execute",
                resource=f"tool:{request.tool}",
                project=request.project,
                status="allowed" if result.success else result.status,
                metadata={
                    "owner": request.owner,
                    "trace_id": result.trace_id,
                    "span_id": result.span_id,
                    "status": result.status,
                    "duration_ms": result.duration_ms,
                },
            )
        )
