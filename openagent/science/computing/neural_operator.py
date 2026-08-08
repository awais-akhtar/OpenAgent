"""Small, reproducible operator-learning interfaces for scientific tools."""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class NeuralOperator(Protocol):
    family: str

    def fit(self, inputs: Sequence[Sequence[float]], outputs: Sequence[Sequence[float]]) -> dict[str, Any]: ...

    def predict(self, inputs: Sequence[float]) -> list[float]: ...


class OperatorLearningBaseline:
    """A dependency-light linear baseline shared by FNO, DeepONet and PINN wrappers.

    The names identify the interchange contract; they do not claim to implement
    the full research algorithms without their optional numerical backends.
    """

    def __init__(self, family: str) -> None:
        self.family = family
        self._mean: list[float] = []

    def fit(self, inputs: Sequence[Sequence[float]], outputs: Sequence[Sequence[float]]) -> dict[str, Any]:
        if not outputs:
            raise ValueError("outputs must contain at least one sample")
        width = len(outputs[0])
        self._mean = [sum(row[index] for row in outputs) / len(outputs) for index in range(width)]
        return {"family": self.family, "samples": len(outputs), "output_width": width, "backend": "deterministic-baseline"}

    def predict(self, inputs: Sequence[float]) -> list[float]:
        if not self._mean:
            raise RuntimeError("fit must be called before predict")
        return list(self._mean)


class FNO(OperatorLearningBaseline):
    def __init__(self) -> None:
        super().__init__("FNO")


class DeepONet(OperatorLearningBaseline):
    def __init__(self) -> None:
        super().__init__("DeepONet")


class PINN(OperatorLearningBaseline):
    def __init__(self) -> None:
        super().__init__("PINN")
