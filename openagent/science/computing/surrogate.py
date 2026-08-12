"""Lightweight operator surrogate with an optional NumPy acceleration path.

This is intentionally a transparent baseline: Fourier features plus ridge
regression approximate a solver map while retaining a direct numerical solver
for verification. It is not presented as a new scientific discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Sequence


@dataclass
class ScientificSurrogate:
    backend: str = "fourier-ridge"
    regularization: float = 1e-4
    modes: int = 6

    def __post_init__(self) -> None:
        self._weights: list[list[float]] | None = None
        self._points = 0

    def fit(self, dataset: Sequence[dict[str, Sequence[float]]]) -> dict[str, object]:
        if not dataset:
            raise ValueError("dataset must contain at least one sample")
        inputs = [[float(item) for item in row["input"]] for row in dataset]
        targets = [[float(item) for item in row["target"]] for row in dataset]
        self._points = len(inputs[0])
        if any(len(row) != self._points for row in inputs + targets):
            raise ValueError("all surrogate rows must have the same length")
        features = [self._features(row) for row in inputs]
        self._weights = _ridge_fit(features, targets, self.regularization)
        return {
            "status": "fitted",
            "samples": len(dataset),
            "points": self._points,
            "feature_count": len(features[0]),
            "backend": self.backend,
        }

    def predict(self, initial: Iterable[float]) -> list[float]:
        if self._weights is None:
            raise RuntimeError("fit the surrogate before prediction")
        values = [float(item) for item in initial]
        if len(values) != self._points:
            raise ValueError(f"initial condition must contain {self._points} values")
        features = self._features(values)
        return [sum(features[index] * row[index] for index in range(len(features))) for row in self._weights]

    def compare_with_solver(self, solver: object, initial: Iterable[float]) -> dict[str, float]:
        values = [float(item) for item in initial]
        predicted_started = time.perf_counter()
        predicted = self.predict(values)
        surrogate_seconds = time.perf_counter() - predicted_started
        solver_started = time.perf_counter()
        solved = list(getattr(solver, "solve")(values))
        solver_seconds = time.perf_counter() - solver_started
        errors = [left - right for left, right in zip(predicted, solved)]
        error = math.sqrt(sum(item * item for item in errors) / max(1, len(solved)))
        mae = sum(abs(item) for item in errors) / max(1, len(solved))
        scale = max(1e-12, math.sqrt(sum(item * item for item in solved) / max(1, len(solved))))
        mean = sum(solved) / max(1, len(solved))
        total = sum((item - mean) ** 2 for item in solved)
        residual = sum(item * item for item in errors)
        return {
            "rmse": error,
            "mae": mae,
            "relative_rmse": error / scale,
            "r2": 1.0 - residual / max(total, 1e-12),
            "max_absolute_error": max(abs(item) for item in errors),
            "surrogate_seconds": surrogate_seconds,
            "solver_seconds": solver_seconds,
        }

    def _features(self, values: Sequence[float]) -> list[float]:
        mean = sum(values) / max(1, len(values))
        features = [1.0, mean, math.sqrt(sum(item * item for item in values) / max(1, len(values)))]
        limit = min(self.modes, max(1, len(values) // 2))
        for mode in range(1, limit + 1):
            features.append(sum(item * math.sin(math.pi * mode * (index + 1) / (len(values) + 1)) for index, item in enumerate(values)))
            features.append(sum(item * math.cos(math.pi * mode * (index + 1) / (len(values) + 1)) for index, item in enumerate(values)))
        return features


def _ridge_fit(features: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], regularization: float) -> list[list[float]]:
    columns = len(features[0])
    outputs = len(targets[0])
    gram = [[0.0] * columns for _ in range(columns)]
    rhs = [[0.0] * outputs for _ in range(columns)]
    for row, target in zip(features, targets):
        for left in range(columns):
            for right in range(columns):
                gram[left][right] += row[left] * row[right]
            for output in range(outputs):
                rhs[left][output] += row[left] * target[output]
    for diagonal in range(columns):
        gram[diagonal][diagonal] += max(1e-12, regularization)
    solution = [_solve_linear_system(gram, [rhs[row][output] for row in range(columns)]) for output in range(outputs)]
    return solution


def _solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    size = len(vector)
    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(size)]
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[best][pivot]) < 1e-12:
            continue
        augmented[pivot], augmented[best] = augmented[best], augmented[pivot]
        scale = augmented[pivot][pivot]
        augmented[pivot] = [value / scale for value in augmented[pivot]]
        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            if factor:
                augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[pivot])]
    return [augmented[row][-1] for row in range(size)]
