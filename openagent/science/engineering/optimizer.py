"""Deterministic constrained optimisation for small engineering designs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
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
    method: str = "random"
    fallback_to_random: bool = True

    def solve(self, problem: DesignProblem, *, method: str | None = None) -> dict[str, object]:
        if self.iterations < 1 or self.iterations > 50_000:
            raise ValueError("iterations must be between 1 and 50000")
        selected = (method or self.method).strip().lower()
        if selected not in {"random", "bayesian", "differential_evolution", "minimize"}:
            raise ValueError("method must be random, bayesian, differential_evolution, or minimize")
        started = time.perf_counter()
        if selected == "random":
            result = self._solve_random(problem)
        elif selected == "bayesian":
            result = self._solve_bayesian(problem)
        else:
            try:
                result = self._solve_scipy(problem, selected)
            except ImportError:
                if not self.fallback_to_random:
                    raise RuntimeError(f"SciPy is required for {selected}") from None
                result = self._solve_random(problem)
                result["requested_method"] = selected
                result["method_note"] = "SciPy is unavailable; deterministic random search was used instead."
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    def _solve_bayesian(self, problem: DesignProblem) -> dict[str, object]:
        """Minimise the penalised objective with a small native Gaussian process.

        The implementation deliberately uses only the standard library. An RBF
        Gaussian process models the observed score and expected improvement
        chooses each subsequent design from a deterministic candidate set.
        """

        rng = random.Random(self.seed)
        names = list(problem.variables)
        dimensions = len(names)
        budget = self.iterations
        initial_count = min(budget, max(3, 2 * dimensions + 1))
        candidate_pool_size = max(256, 128 * dimensions)
        points = self._latin_hypercube(initial_count, dimensions, rng)
        observations: list[float] = []
        results: list[dict[str, object]] = []
        numerical_fallbacks = 0

        def evaluate(point: list[float]) -> None:
            design = {
                name: problem.variables[name][0] + point[index] * (problem.variables[name][1] - problem.variables[name][0])
                for index, name in enumerate(names)
            }
            candidate = self._score(problem, design, len(results) + 1)
            observations.append(float(candidate["score"]))
            results.append(candidate)

        for point in points:
            evaluate(point)
        initial_best = min(results, key=lambda candidate: float(candidate["score"]))
        initial_best_score = float(initial_best["score"])
        initial_best_objective = float(initial_best["objective"])

        while len(results) < budget:
            candidates = self._bayesian_candidates(
                points,
                observations,
                dimensions=dimensions,
                count=candidate_pool_size,
                rng=rng,
            )
            try:
                model = self._fit_gaussian_process(points, observations)
                best_observed = min(model["targets"])
                selected_point = max(
                    candidates,
                    key=lambda point: self._expected_improvement(
                        *self._predict_gaussian_process(model, point),
                        best=best_observed,
                    ),
                )
            except (ArithmeticError, ValueError):
                # A singular covariance matrix should not abort an engineering
                # run. The deterministic candidate stream remains reproducible.
                numerical_fallbacks += 1
                selected_point = candidates[0]
            points.append(selected_point)
            evaluate(selected_point)

        best = min(results, key=lambda candidate: float(candidate["score"]))
        best.update({
            "method": "bayesian",
            "requested_method": "bayesian",
            "backend": "native-rbf-gaussian-process",
            "backend_provenance": "OpenAgent standard-library Gaussian process; no optional numerical package required.",
            "acquisition": "expected_improvement",
            "acquisition_goal": "Maximise expected improvement over the best observed penalised objective.",
            "evaluations": len(results),
            "initial_evaluations": initial_count,
            "adaptive_evaluations": len(results) - initial_count,
            "initial_best_objective": initial_best_objective,
            "initial_best_score": initial_best_score,
            "acquisition_improvement": initial_best_score - float(best["score"]),
            "candidate_pool_size": candidate_pool_size,
            "deterministic": True,
            "fallback_used": numerical_fallbacks > 0,
            "fallback_evaluations": numerical_fallbacks,
        })
        return best

    @staticmethod
    def _latin_hypercube(count: int, dimensions: int, rng: random.Random) -> list[list[float]]:
        columns: list[list[float]] = []
        for _ in range(dimensions):
            values = [(index + rng.random()) / count for index in range(count)]
            rng.shuffle(values)
            columns.append(values)
        return [[columns[dimension][row] for dimension in range(dimensions)] for row in range(count)]

    @staticmethod
    def _bayesian_candidates(
        points: list[list[float]],
        observations: list[float],
        *,
        dimensions: int,
        count: int,
        rng: random.Random,
    ) -> list[list[float]]:
        best_point = points[min(range(len(points)), key=observations.__getitem__)]
        candidates: list[list[float]] = []
        local_count = count // 2
        refinement_scales = (0.2, 0.1, 0.05, 0.02, 0.01)
        for index in range(local_count):
            radius = refinement_scales[index % len(refinement_scales)]
            candidates.append([
                min(1.0, max(0.0, value + rng.gauss(0.0, radius)))
                for value in best_point
            ])
        candidates.extend([[rng.random() for _ in range(dimensions)] for _ in range(count - local_count)])
        return candidates

    @classmethod
    def _fit_gaussian_process(cls, points: list[list[float]], values: list[float]) -> dict[str, object]:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        scale = math.sqrt(max(variance, 1e-18))
        targets = [(value - mean) / scale for value in values]
        length_scale = max(0.12, min(0.4, 0.65 / math.sqrt(max(1, len(points[0])))))
        covariance = [
            [cls._rbf(left, right, length_scale) + (1e-8 if row == column else 0.0)
             for column, right in enumerate(points)]
            for row, left in enumerate(points)
        ]
        cholesky = cls._cholesky(covariance)
        alpha = cls._cholesky_solve(cholesky, targets)
        return {
            "points": points,
            "targets": targets,
            "mean": mean,
            "scale": scale,
            "length_scale": length_scale,
            "cholesky": cholesky,
            "alpha": alpha,
        }

    @staticmethod
    def _rbf(left: list[float], right: list[float], length_scale: float) -> float:
        distance = sum((a - b) ** 2 for a, b in zip(left, right))
        return math.exp(-0.5 * distance / (length_scale * length_scale))

    @staticmethod
    def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
        size = len(matrix)
        lower = [[0.0] * size for _ in range(size)]
        for row in range(size):
            for column in range(row + 1):
                subtotal = sum(lower[row][index] * lower[column][index] for index in range(column))
                if row == column:
                    diagonal = matrix[row][row] - subtotal
                    if diagonal <= 0.0:
                        raise ArithmeticError("Gaussian-process covariance is not positive definite")
                    lower[row][column] = math.sqrt(diagonal)
                else:
                    lower[row][column] = (matrix[row][column] - subtotal) / lower[column][column]
        return lower

    @staticmethod
    def _cholesky_solve(lower: list[list[float]], values: list[float]) -> list[float]:
        size = len(values)
        forward = [0.0] * size
        for row in range(size):
            forward[row] = (values[row] - sum(lower[row][column] * forward[column] for column in range(row))) / lower[row][row]
        solution = [0.0] * size
        for row in range(size - 1, -1, -1):
            solution[row] = (
                forward[row] - sum(lower[column][row] * solution[column] for column in range(row + 1, size))
            ) / lower[row][row]
        return solution

    @classmethod
    def _predict_gaussian_process(cls, model: Mapping[str, object], point: list[float]) -> tuple[float, float]:
        training = model["points"]
        length_scale = float(model["length_scale"])
        kernel = [cls._rbf(existing, point, length_scale) for existing in training]  # type: ignore[arg-type]
        alpha = model["alpha"]
        mean = sum(value * weight for value, weight in zip(kernel, alpha))  # type: ignore[arg-type]
        lower = model["cholesky"]
        projected: list[float] = []
        for row in range(len(kernel)):
            projected.append(
                (kernel[row] - sum(lower[row][column] * projected[column] for column in range(row))) / lower[row][row]  # type: ignore[index]
            )
        variance = max(1e-12, 1.0 - sum(value * value for value in projected))
        return mean, math.sqrt(variance)

    @staticmethod
    def _expected_improvement(mean: float, standard_deviation: float, *, best: float, exploration: float = 0.01) -> float:
        improvement = best - mean - exploration
        if standard_deviation <= 1e-12:
            return max(0.0, improvement)
        z_score = improvement / standard_deviation
        density = math.exp(-0.5 * z_score * z_score) / math.sqrt(2.0 * math.pi)
        distribution = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        return improvement * distribution + standard_deviation * density

    def _solve_random(self, problem: DesignProblem) -> dict[str, object]:
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
        best["method"] = "random"
        best["evaluations"] = self.iterations
        return best

    def _solve_scipy(self, problem: DesignProblem, method: str) -> dict[str, object]:
        try:
            from scipy.optimize import differential_evolution, minimize
        except ImportError:
            raise
        names = list(problem.variables)
        bounds = [problem.variables[name] for name in names]
        evaluations = 0

        def objective(values: object) -> float:
            nonlocal evaluations
            evaluations += 1
            design = {name: float(value) for name, value in zip(names, values)}  # type: ignore[arg-type]
            return float(self._score(problem, design, evaluations)["score"])

        if method == "differential_evolution":
            outcome = differential_evolution(
                objective,
                bounds,
                seed=self.seed,
                maxiter=max(1, self.iterations // max(1, len(names) * 12)),
                popsize=8,
                polish=True,
                updating="immediate",
            )
        else:
            midpoint = [(low + high) / 2.0 for low, high in bounds]
            outcome = minimize(objective, midpoint, method="Powell", bounds=bounds, options={"maxiter": self.iterations})
        design = {name: float(value) for name, value in zip(names, outcome.x)}
        result = self._score(problem, design, evaluations)
        result.update({
            "method": method,
            "evaluations": evaluations,
            "optimizer_success": bool(outcome.success),
            "optimizer_message": str(outcome.message),
        })
        return result

    def benchmark(self, problem: DesignProblem, *, methods: tuple[str, ...] = ("random", "bayesian", "differential_evolution", "minimize")) -> dict[str, object]:
        """Run comparable deterministic solver trials and retain all outcomes."""

        trials = [self.solve(problem, method=method) for method in methods]
        feasible = [trial for trial in trials if trial["status"] == "feasible"]
        pool = feasible or trials
        best = min(pool, key=lambda trial: float(trial["objective"]))
        return {
            "methods": trials,
            "best_method": best["method"],
            "best_objective": best["objective"],
            "comparison_basis": "Lowest feasible objective; lowest penalized result when no method finds a feasible design.",
        }

    def _score(self, problem: DesignProblem, design: Mapping[str, float], iteration: int) -> dict[str, object]:
        metrics = problem.evaluate(design)
        checks = [constraint.evaluate(design, metrics) for constraint in problem.constraints]
        violations = sum(max(0.0, -check.margin) / max(abs(check.limit), 1e-12) for check in checks)
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
