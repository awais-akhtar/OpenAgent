"""Reusable engineering constraint checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    passed: bool
    value: float
    limit: float
    margin: float
    units: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "limit": self.limit,
            "margin": self.margin,
            "units": self.units,
        }


class EngineeringConstraint:
    name = "constraint"

    def evaluate(self, design: Mapping[str, float], metrics: Mapping[str, float]) -> ConstraintResult:
        raise NotImplementedError


@dataclass(frozen=True)
class StressLimit(EngineeringConstraint):
    limit: float
    name: str = "stress"
    units: str = "Pa"

    def evaluate(self, design: Mapping[str, float], metrics: Mapping[str, float]) -> ConstraintResult:
        value = float(metrics.get("stress", float("inf")))
        return ConstraintResult(self.name, value <= self.limit, value, self.limit, self.limit - value, self.units)


@dataclass(frozen=True)
class DeflectionLimit(EngineeringConstraint):
    limit: float
    name: str = "deflection"
    units: str = "m"

    def evaluate(self, design: Mapping[str, float], metrics: Mapping[str, float]) -> ConstraintResult:
        value = float(metrics.get("deflection", float("inf")))
        return ConstraintResult(self.name, value <= self.limit, value, self.limit, self.limit - value, self.units)
