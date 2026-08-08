"""Candidate selection strategies for the next local materials experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .predictor import MaterialPropertyPredictor


@dataclass(frozen=True)
class ActiveLearner:
    strategy: str = "uncertainty"

    def select(self, predictor: MaterialPropertyPredictor, candidates: Sequence[Mapping[str, float]], *, count: int = 1) -> list[dict[str, object]]:
        if count < 1:
            return []
        scored = [
            {"features": dict(candidate), "prediction": predictor.predict(candidate), "uncertainty": predictor.uncertainty(candidate)}
            for candidate in candidates
        ]
        if self.strategy == "uncertainty":
            scored.sort(key=lambda item: float(item["uncertainty"]), reverse=True)
        elif self.strategy == "maximum":
            scored.sort(key=lambda item: float(item["prediction"]), reverse=True)
        else:
            raise ValueError("strategy must be uncertainty or maximum")
        return scored[:count]
