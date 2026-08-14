"""Capability-aware model and device selection for local-first work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComputeRequest:
    task: str = "chat"
    model: str = ""
    allow_external: bool = False
    target_latency_ms: int = 0
    memory_budget_gb: float = 0.0
    require_streaming: bool = False


class ComputeRouter:
    """Choose a provider using live health and explicit privacy constraints.

    This is a decision layer, not a second inference engine. Actual model
    loading and generation remain owned by ModelGateway and AdaptiveEngine.
    """

    def __init__(self, gateway: Any, adaptive_engine: Any = None) -> None:
        self.gateway = gateway
        self.adaptive_engine = adaptive_engine

    def status(self) -> dict[str, Any]:
        providers = self.gateway.specs()
        hardware: dict[str, Any] = {"status": "not-probed"}
        if self.adaptive_engine is not None:
            try:
                adaptive = self.adaptive_engine.status()
                hardware = {
                    "status": adaptive.get("status", "ready"),
                    "profile": adaptive.get("hardware_profile", {}),
                    "backends": adaptive.get("backends", []),
                    "privacy_boundary": adaptive.get("privacy_boundary", ""),
                }
            except Exception as exc:
                hardware = {"status": "unavailable", "reason": str(exc)}
        return {
            "providers": providers,
            "hardware": hardware,
            "policy": self.gateway.policies(),
            "decision_model": "health + privacy + task fit + explicit constraints",
        }

    def readiness(self, *, project: str = "default") -> dict[str, Any]:
        """Return a fast readiness view without forcing a hardware profile probe."""

        hardware: dict[str, Any] = {"status": "not-refreshed", "backends": []}
        if self.adaptive_engine is not None:
            try:
                hardware = self.adaptive_engine.readiness(project=project)
            except Exception as exc:
                hardware = {"status": "unavailable", "reason": str(exc), "backends": []}
        return {
            "providers": self.gateway.specs(),
            "hardware": hardware,
            "policy": self.gateway.policies(),
            "decision_model": "health + privacy + task fit + explicit constraints",
            "probe_boundary": "Detailed hardware discovery is available from /api/compute/status and is not forced by module health.",
        }

    def plan(self, request: ComputeRequest) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        task = request.task.lower()
        constraint_notes: list[str] = []
        if request.target_latency_ms:
            constraint_notes.append("target latency was recorded but not enforced because providers do not expose a certified latency estimate")
        if request.memory_budget_gb:
            constraint_notes.append("memory budget was recorded but not enforced because provider memory requirements are not certified")
        if request.require_streaming:
            constraint_notes.append("streaming was requested but not independently certified by provider metadata")
        for provider in self.gateway.specs():
            provider_id = str(provider.get("id", ""))
            local = bool(provider.get("local", False))
            available = bool(provider.get("available", False))
            if provider_id == "auto":
                continue
            reasons: list[str] = []
            score = 0.0
            if not available:
                reasons.append("provider health is unavailable")
                score -= 100
            if local:
                score += 30
                reasons.append("local boundary")
            elif not request.allow_external:
                reasons.append("external provider is disabled by this request")
                score -= 80
            else:
                score += 5
                reasons.append("external use explicitly allowed")
            model = str(provider.get("selected_model") or provider.get("default_model") or "")
            requested = request.model.lower().strip()
            if requested and requested in model.lower():
                score += 25
                reasons.append("requested model matches")
            if any(word in task for word in ("code", "debug", "program", "repository")) and any(word in model.lower() for word in ("coder", "code", "qwen")):
                score += 10
                reasons.append("model name fits coding work")
            candidates.append({"provider": provider_id, "model": model, "score": round(score, 2), "available": available, "local": local, "reasons": reasons})
        candidates.sort(key=lambda item: (-float(item["score"]), str(item["provider"])))
        selected = next((item for item in candidates if item["available"] and (item["local"] or request.allow_external)), None)
        return {
            "request": {
                "task": request.task,
                "model": request.model,
                "allow_external": request.allow_external,
                "target_latency_ms": request.target_latency_ms,
                "memory_budget_gb": request.memory_budget_gb,
            },
            "selected": selected,
            "candidates": candidates,
            "status": "ready" if selected else "needs_setup",
            "constraints": constraint_notes,
            "limits": [
                "A plan does not load a model or claim a benchmark.",
                "Hardware acceleration is used only when the selected provider and backend report it available.",
            ],
        }
