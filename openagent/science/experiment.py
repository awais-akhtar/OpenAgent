"""Experiment proposal and ranking helpers for scientific workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ExperimentProposal:
    title: str
    variables: dict[str, Any]
    rationale: str
    priority: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "variables": dict(self.variables),
            "rationale": self.rationale,
            "priority": self.priority,
        }


class ExperimentPlanner:
    """Ranks reproducible candidates using a caller-supplied scoring function."""

    def propose(
        self,
        goal: str,
        candidates: Iterable[dict[str, Any]],
        *,
        score: Callable[[dict[str, Any]], float] | None = None,
    ) -> list[ExperimentProposal]:
        scorer = score or (lambda candidate: float(candidate.get("priority", 0.0)))
        proposals = [
            ExperimentProposal(
                title=f"Experiment {index}",
                variables=dict(candidate),
                rationale=f"Tests a controlled candidate for: {goal.strip() or 'the stated objective'}.",
                priority=float(scorer(candidate)),
            )
            for index, candidate in enumerate(candidates, 1)
        ]
        return sorted(proposals, key=lambda item: item.priority, reverse=True)
