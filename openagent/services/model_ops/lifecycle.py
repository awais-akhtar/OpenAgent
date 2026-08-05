"""Model lifecycle and diagnostics service."""

from __future__ import annotations

from typing import Any

from ...local_ai import LocalAIInfrastructure


class ModelOpsLifecycleService:
    """Service facade for model lifecycle state, diagnostics, and hardware fit."""

    def __init__(self, local_ai: LocalAIInfrastructure) -> None:
        self.local_ai = local_ai

    def transition(
        self,
        model_id: str,
        status: str,
        *,
        reason: str = "",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.local_ai.transition_model_lifecycle(
            model_id,
            status,
            reason=reason,
            detail=detail,
        )

    def diagnostics(self, model_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.local_ai.get_model_diagnostics(model_id, limit=limit)

    def hardware_fit(self, model_id: str) -> dict[str, Any]:
        return self.local_ai.hardware_fit_for_model(model_id)
