"""Shared value objects for local scientific workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScientificFinding:
    """One check result with enough context to be reviewed by a person."""

    check: str
    passed: bool
    message: str
    severity: str = "info"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class ScientificVerification:
    """Aggregate result returned by the scientific verification layer."""

    passed: bool
    findings: tuple[ScientificFinding, ...]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "findings": [item.to_dict() for item in self.findings],
        }


@dataclass(frozen=True)
class ScienceRun:
    """Serializable result for an agent or API invocation."""

    workflow: str
    goal: str
    status: str
    result: dict[str, Any]
    verification: ScientificVerification
    privacy_boundary: str = (
        "Scientific inputs, intermediate calculations, and outputs remain in local OpenAgent storage. "
        "Network access is not used by the scientific runtime."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "goal": self.goal,
            "status": self.status,
            "result": dict(self.result),
            "verification": self.verification.to_dict(),
            "privacy_boundary": self.privacy_boundary,
        }
