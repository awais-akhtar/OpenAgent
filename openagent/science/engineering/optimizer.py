"""Deterministic constrained optimisation for small engineering designs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Mapping

from .constraints import DeflectionLimit, EngineeringConstraint, StressLimit


@dataclass(frozen=True)
class DesignProblem:
    variables: Mapping[str, tuple[float, float]]
    objective: str = "minimum_mass"
    constraints: tuple[EngineeringConstraint, ...] = ()
    evaluator: Callable[[Mapping[str, float]], Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("at least one design variable is required")
        for name, bounds in self.variables.items():
            if len(bounds) != 2 or bounds[0] <= 0 or bounds[1] <= bounds[0]:
                raise ValueError(f"invalid bounds for {name}")

    def evaluate(self, design: Mapping[str, float]) -> dict[str, float]:
        if self.evaluator is not None:
            return {str(key): float(value) for key, value in self.evaluator(design).items()}
        width = float(design.get("width", 0.0))
        height = float(design.get("height", 0.0))
        length = float(design.get("length", 1.0))
        load = float(design.get("load", 1000.0))
        youngs_modulus = float(design.get("youngs_modulus", 200e9))
        inertia = width * height**3 / 12.0
        section = max(width * height, 1e-12)
        stress = 6.0 * load * length / max(width * height**2, 1e-12)
        deflection = load * length**3 / max(3.0 * youngs_modulus * inertia, 1e-12)
        return {
            "mass": width * height * length * 7850.0,
            "stress": stress,
            "deflection": deflection,
            "section_area": section,
        }


@dataclass
class EngineeringOptimizer:
    iterations: int = 800
    seed: int = 19

    def solve(self, problem: DesignProblem) -> dict[str, object]:
        if self.iterations < 1 or self.iterations > 50_000:
            raise ValueError("iterations must be between 1 and 50000")
        rng = random.Random(self.seed)
        names = list(problem.variables)
        best: dict[str, object] | None = None
        for iteration in range(self.iterations):
            design = {name: rng.uniform(*problem.variables[name]) for name in names}
            candidate = self._score(problem, design, iteration)
            if best is None or float(candidate["score"]) < float(best["score"]):
                best = candidate
        if best is None:
            raise RuntimeError("optimizer produced no candidate")
        return best

    def _score(self, problem: DesignProblem, design: Mapping[str, float], iteration: int) -> dict[str, object]:
        metrics = problem.evaluate(design)
        checks = [constraint.evaluate(design, metrics) for constraint in problem.constraints]
        violations = sum(max(0.0, -check.margin) for check in checks)
        objective = float(metrics.get(problem.objective.removeprefix("minimum_"), metrics.get("mass", float("inf"))))
        score = objective + violations * max(1.0, abs(objective)) * 1_000.0
        return {
            "status": "feasible" if all(check.passed for check in checks) else "infeasible",
            "iteration": iteration,
            "design": {str(key): round(float(value), 10) for key, value in design.items()},
            "metrics": {str(key): float(value) for key, value in metrics.items()},
            "constraints": [check.to_dict() for check in checks],
            "objective": objective,
            "score": score,
        }


def beam_problem(
    *,
    stress_limit: float = 250e6,
    deflection_limit: float = 0.005,
) -> DesignProblem:
    """Return the reference cantilever beam problem used in examples and tests."""

    return DesignProblem(
        variables={"width": (0.01, 0.1), "height": (0.02, 0.2)},
        objective="minimum_mass",
        constraints=(StressLimit(stress_limit), DeflectionLimit(deflection_limit)),
    )
