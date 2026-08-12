"""Transparent local property prediction with distribution-aware uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from .dataset import validate_records


@dataclass
class MaterialPropertyPredictor:
    """Ridge baseline with standardized features and a bootstrap ensemble.

    The model is intentionally small and inspectable.  Uncertainty combines
    disagreement between bootstrap regressors with distance from the fitted
    feature distribution; it is not presented as a calibrated probability.
    """

    regularization: float = 1e-6
    ensemble_size: int = 16
    seed: int = 23

    def __post_init__(self) -> None:
        self.features: tuple[str, ...] = ()
        self.weights: list[float] = []
        self.ensemble_weights: list[list[float]] = []
        self.feature_means: list[float] = []
        self.feature_scales: list[float] = []
        self.training_points: list[list[float]] = []
        self.residual_scale: float = 0.0
        self.target: str = ""

    def fit(self, records: Iterable[dict[str, object]], *, target: str) -> dict[str, object]:
        rows = validate_records(records, target=target)
        self.target = target
        feature_names = sorted({str(key) for row in rows for key, value in row.items() if key != target and _number(value)})
        if not feature_names:
            raise ValueError("materials records need at least one numeric feature")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        self.features = tuple(feature_names)
        raw = [[float(row.get(name, 0.0)) for name in self.features] for row in rows]
        self.feature_means = [fmean(column) for column in zip(*raw)]
        self.feature_scales = [_sample_scale(column) for column in zip(*raw)]
        self.training_points = [self._standardize_values(values) for values in raw]
        design = [[1.0, *point] for point in self.training_points]
        targets = [float(row[target]) for row in rows]
        self.weights = _least_squares(design, targets, self.regularization)

        rng = random.Random(self.seed)
        self.ensemble_weights = []
        for _ in range(self.ensemble_size):
            indexes = [rng.randrange(len(rows)) for _ in rows]
            sample_design = [design[index] for index in indexes]
            sample_targets = [targets[index] for index in indexes]
            self.ensemble_weights.append(_least_squares(sample_design, sample_targets, self.regularization))

        errors = [self._predict_vector(point, self.weights) - actual for point, actual in zip(self.training_points, targets)]
        self.residual_scale = math.sqrt(sum(error * error for error in errors) / max(1, len(errors)))
        metrics = _regression_metrics(targets, [self._predict_vector(point, self.weights) for point in self.training_points])
        return {
            "status": "fitted",
            "samples": len(rows),
            "features": list(self.features),
            "target": target,
            "model": "standardized-ridge-bootstrap",
            "ensemble_size": self.ensemble_size,
            "training_metrics": metrics,
            "uncertainty_note": "Bootstrap disagreement plus standardized nearest-training-point distance; not a calibrated probability.",
        }

    def predict(self, features: Mapping[str, float]) -> float:
        self._require_fit()
        return self._predict_vector(self._standardize_mapping(features), self.weights)

    def predict_distribution(self, features: Mapping[str, float]) -> dict[str, float]:
        self._require_fit()
        point = self._standardize_mapping(features)
        predictions = [self._predict_vector(point, weights) for weights in self.ensemble_weights]
        mean = fmean(predictions)
        spread = math.sqrt(sum((value - mean) ** 2 for value in predictions) / max(1, len(predictions) - 1))
        distance = min(_euclidean(point, training) for training in self.training_points)
        distance_component = self.residual_scale * distance / max(1.0, math.sqrt(len(point)))
        uncertainty = math.sqrt(spread * spread + distance_component * distance_component)
        return {
            "prediction": self._predict_vector(point, self.weights),
            "ensemble_mean": mean,
            "ensemble_std": spread,
            "nearest_training_distance": distance,
            "uncertainty": uncertainty,
        }

    def evaluate(self, records: Iterable[dict[str, object]]) -> dict[str, float | int]:
        rows = validate_records(records, target=self.target)
        actual = [float(row[self.target]) for row in rows]
        predicted = [self.predict(row) for row in rows]
        return {"samples": len(rows), **_regression_metrics(actual, predicted)}

    def rank_candidates(self, candidates: Sequence[Mapping[str, float]], *, maximize: bool = True) -> list[dict[str, object]]:
        ranked = []
        for candidate in candidates:
            distribution = self.predict_distribution(candidate)
            ranked.append({"features": dict(candidate), **distribution})
        return sorted(ranked, key=lambda item: float(item["prediction"]), reverse=maximize)

    def uncertainty(self, features: Mapping[str, float]) -> float:
        return self.predict_distribution(features)["uncertainty"]

    def _standardize_mapping(self, features: Mapping[str, float]) -> list[float]:
        return self._standardize_values([float(features.get(name, 0.0)) for name in self.features])

    def _standardize_values(self, values: Sequence[float]) -> list[float]:
        return [(value - mean) / scale for value, mean, scale in zip(values, self.feature_means, self.feature_scales)]

    @staticmethod
    def _predict_vector(point: Sequence[float], weights: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(weights, [1.0, *point]))

    def _require_fit(self) -> None:
        if not self.weights:
            raise RuntimeError("fit the predictor before prediction")


def _sample_scale(values: Sequence[float]) -> float:
    mean = fmean(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return max(math.sqrt(variance), 1e-12)


def _regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    errors = [estimate - truth for truth, estimate in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / max(1, len(errors))
    rmse = math.sqrt(sum(error * error for error in errors) / max(1, len(errors)))
    mean = fmean(actual)
    denominator = sum((value - mean) ** 2 for value in actual)
    r_squared = 1.0 - (sum(error * error for error in errors) / denominator) if denominator > 1e-15 else (1.0 if rmse <= 1e-12 else 0.0)
    return {"mae": mae, "rmse": rmse, "r2": r_squared}


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


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
