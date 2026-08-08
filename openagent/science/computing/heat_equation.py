"""A small, deterministic 1-D heat equation solver for scientific workflows."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class HeatEquation1D:
    """Solve u_t = alpha * u_xx on a unit interval with fixed endpoints."""

    points: int = 32
    steps: int = 80
    alpha: float = 0.15
    dt: float = 0.0008

    def __post_init__(self) -> None:
        if self.points < 5 or self.points > 2048:
            raise ValueError("points must be between 5 and 2048")
        if self.steps < 1 or self.steps > 10_000:
            raise ValueError("steps must be between 1 and 10000")
        if self.alpha <= 0 or self.dt <= 0:
            raise ValueError("alpha and dt must be positive")
        dx = 1.0 / (self.points - 1)
        if self.alpha * self.dt / (dx * dx) > 0.5:
            raise ValueError("unstable timestep: alpha * dt / dx^2 must be <= 0.5")

    @property
    def dx(self) -> float:
        return 1.0 / (self.points - 1)

    def solve(self, initial: Iterable[float]) -> list[float]:
        values = [float(item) for item in initial]
        if len(values) != self.points:
            raise ValueError(f"initial condition must contain {self.points} values")
        values[0] = 0.0
        values[-1] = 0.0
        coefficient = self.alpha * self.dt / (self.dx * self.dx)
        for _ in range(self.steps):
            next_values = list(values)
            for index in range(1, self.points - 1):
                next_values[index] = values[index] + coefficient * (
                    values[index - 1] - 2.0 * values[index] + values[index + 1]
                )
            values = next_values
        return values

    def initial_condition(self, kind: str = "sine", *, amplitude: float = 1.0) -> list[float]:
        name = kind.strip().lower()
        result: list[float] = []
        for index in range(self.points):
            x = index / (self.points - 1)
            if name == "sine":
                value = math.sin(math.pi * x)
            elif name == "two_peak":
                value = math.sin(2.0 * math.pi * x) ** 2
            elif name == "tent":
                value = 2.0 * x if x <= 0.5 else 2.0 * (1.0 - x)
            else:
                raise ValueError("unknown initial condition; use sine, two_peak, or tent")
            result.append(float(amplitude) * value)
        return result

    def generate_dataset(self, samples: int = 32, *, seed: int = 7) -> list[dict[str, list[float]]]:
        if samples < 1 or samples > 10_000:
            raise ValueError("samples must be between 1 and 10000")
        rng = random.Random(seed)
        dataset: list[dict[str, list[float]]] = []
        for _ in range(samples):
            initial = [0.0] * self.points
            for index in range(1, self.points - 1):
                x = index / (self.points - 1)
                initial[index] = max(0.0, math.sin(math.pi * x) + rng.uniform(-0.12, 0.12))
            dataset.append({"input": initial, "target": self.solve(initial)})
        return dataset

    def energy(self, values: Iterable[float]) -> float:
        numbers = [float(item) for item in values]
        if len(numbers) != self.points:
            raise ValueError(f"values must contain {self.points} samples")
        return sum(item * item for item in numbers) * self.dx
