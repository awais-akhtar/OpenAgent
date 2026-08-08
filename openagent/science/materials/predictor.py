"""Small transparent property predictor for local materials exploration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from .dataset import validate_records


@dataclass
class MaterialPropertyPredictor:
    regularization: float = 1e-6

    def __post_init__(self) -> None:
        self.features: tuple[str, ...] = ()
        self.weights: list[float] = []
        self.residual_scale: float = 0.0
        self.target: str = ""

    def fit(self, records: Iterable[dict[str, object]], *, target: str) -> dict[str, object]:
        rows = validate_records(records, target=target)
        self.target = target
        feature_names = sorted({str(key) for row in rows for key, value in row.items() if key != target and _number(value)})
        if not feature_names:
            raise ValueError("materials records need at least one numeric feature")
        self.features = tuple(feature_names)
        design = [[1.0] + [float(row.get(name, 0.0)) for name in self.features] for row in rows]
        targets = [float(row[target]) for row in rows]
        self.weights = _least_squares(design, targets, self.regularization)
        errors = [self._predict_row(row) - target_value for row, target_value in zip(rows, targets)]
        self.residual_scale = math.sqrt(sum(error * error for error in errors) / max(1, len(errors)))
        return {"status": "fitted", "samples": len(rows), "features": list(self.features), "target": target}

    def predict(self, features: Mapping[str, float]) -> float:
        if not self.weights:
            raise RuntimeError("fit the predictor before prediction")
        row = [1.0] + [float(features.get(name, 0.0)) for name in self.features]
        return sum(weight * value for weight, value in zip(self.weights, row))

    def rank_candidates(self, candidates: Sequence[Mapping[str, float]], *, maximize: bool = True) -> list[dict[str, object]]:
        ranked = []
        for candidate in candidates:
            predicted = self.predict(candidate)
            ranked.append({"features": dict(candidate), "prediction": predicted, "uncertainty": self.uncertainty(candidate)})
        return sorted(ranked, key=lambda item: float(item["prediction"]), reverse=maximize)

    def uncertainty(self, features: Mapping[str, float]) -> float:
        if not self.weights:
            raise RuntimeError("fit the predictor before uncertainty estimation")
        distance = math.sqrt(sum(float(features.get(name, 0.0)) ** 2 for name in self.features))
        return self.residual_scale * (1.0 + distance / max(1.0, len(self.features)))

    def _predict_row(self, row: Mapping[str, object]) -> float:
        return self.predict({name: float(row.get(name, 0.0)) for name in self.features})


def _least_squares(matrix: Sequence[Sequence[float]], values: Sequence[float], regularization: float) -> list[float]:
    columns = len(matrix[0])
    gram = [[0.0] * columns for _ in range(columns)]
    rhs = [0.0] * columns
    for row, value in zip(matrix, values):
        for left in range(columns):
            rhs[left] += row[left] * value
            for right in range(columns):
                gram[left][right] += row[left] * row[right]
    for index in range(1, columns):
        gram[index][index] += max(1e-12, regularization)
    return _solve(gram, rhs)


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
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
            augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[pivot])]
    return [augmented[row][-1] for row in range(size)]


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
