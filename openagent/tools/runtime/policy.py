"""Policy decisions for the shared tool boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ToolRequest, ToolSpec


@dataclass(frozen=True)
class ToolDecision:
    allowed: bool
    status: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"allowed": self.allowed, "status": self.status, "reason": self.reason}


class PermissionPolicy:
    """Keep automatic execution narrow and make approval explicit."""

    def decide(self, spec: ToolSpec, request: ToolRequest) -> ToolDecision:
        if spec.risk >= 3:
            return ToolDecision(False, "denied", "this operation is outside the autonomous runtime boundary")
        if request.network == "external" or spec.network == "external":
            if not request.approval_id:
                return ToolDecision(False, "needs_approval", "external network access requires an approval id")
        if spec.risk >= 2 and not request.approval_id:
            return ToolDecision(False, "needs_approval", "this tool requires an approval id")
        return ToolDecision(True, "allowed", "tool policy accepted the request")
