"""Operator-learning models with explicit implementation boundaries.

DeepONet is implemented as a branch/trunk operator network.  PyTorch is used
when requested and installed.  The NumPy backend is a deterministic
random-feature DeepONet: its branch contains nonlinear random features and its
trunk is a coordinate basis, while the coupling weights are learned by ridge
regression.  FNO and PINN remain named interface baselines and make no claim to
implement those algorithms.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence


class NeuralOperator(Protocol):
    family: str

    def fit(self, inputs: Sequence[Sequence[float]], outputs: Sequence[Sequence[float]]) -> dict[str, Any]: ...

    def predict(self, inputs: Sequence[float]) -> list[float]: ...


class OperatorLearningBaseline:
    """Constant-output compatibility baseline for unimplemented families."""

    implementation = "constant-output baseline; not a full neural operator"

    def __init__(self, family: str) -> None:
        self.family = family
        self._mean: list[float] = []

    def fit(self, inputs: Sequence[Sequence[float]], outputs: Sequence[Sequence[float]]) -> dict[str, Any]:
        _validate_training_rows(inputs, outputs)
        width = len(outputs[0])
        self._mean = [sum(float(row[index]) for row in outputs) / len(outputs) for index in range(width)]
        return {
            "family": self.family,
            "samples": len(outputs),
            "output_width": width,
            "backend": "deterministic-mean-baseline",
            "implementation": self.implementation,
            "is_full_implementation": False,
        }

    def predict(self, inputs: Sequence[float]) -> list[float]:
        if not self._mean:
            raise RuntimeError("fit must be called before predict")
        return list(self._mean)


class FNO(OperatorLearningBaseline):
    """FNO interchange interface backed only by an explicit mean baseline."""

    def __init__(self) -> None:
        super().__init__("FNO")


class PINN(OperatorLearningBaseline):
    """PINN interchange interface backed only by an explicit mean baseline."""

    def __init__(self) -> None:
        super().__init__("PINN")


class DeepONet:
    """Train a branch/trunk operator for fixed-grid function observations.

    ``backend="auto"`` selects PyTorch when it is installed and otherwise uses
    the deterministic NumPy random-feature implementation.  Both paths produce
    input-dependent predictions through a branch/trunk dot product.
    """

    family = "DeepONet"

    def __init__(
        self,
        *,
        backend: str = "auto",
        hidden: int = 48,
        basis: int = 24,
        epochs: int = 350,
        learning_rate: float = 0.01,
        regularization: float = 1e-6,
        seed: int = 7,
    ) -> None:
        if backend not in {"auto", "torch", "numpy"}:
            raise ValueError("backend must be auto, torch, or numpy")
        if hidden < 2 or basis < 2 or epochs < 1:
            raise ValueError("hidden and basis must be at least 2; epochs must be positive")
        self.backend = backend
        self.hidden = int(hidden)
        self.basis = int(basis)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.seed = int(seed)
        self._backend_used = ""
        self._input_width = 0
        self._output_width = 0
        self._state: dict[str, Any] = {}

    def fit(self, inputs: Sequence[Sequence[float]], outputs: Sequence[Sequence[float]]) -> dict[str, Any]:
        input_width, output_width = _validate_training_rows(inputs, outputs)
        self._input_width = input_width
        self._output_width = output_width

        # A single observation cannot identify an operator. Preserve exact
        # interpolation for compatibility while reporting the limitation.
        if len(inputs) == 1:
            self._backend_used = "single-sample-interpolant"
            self._state = {"single": [float(value) for value in outputs[0]]}
            metrics = self.evaluate(inputs, outputs)
            return self._fit_report(len(inputs), metrics, identifiable=False)

        selected = self.backend
        if selected == "auto":
            selected = "torch" if _torch_available() else "numpy"
        if selected == "torch":
            try:
                self._fit_torch(inputs, outputs)
            except ImportError as error:
                if self.backend == "torch":
                    raise RuntimeError("PyTorch backend requested but torch is not installed") from error
                self._fit_numpy(inputs, outputs)
        else:
            self._fit_numpy(inputs, outputs)
        metrics = self.evaluate(inputs, outputs)
        return self._fit_report(len(inputs), metrics, identifiable=True)

    def predict(self, inputs: Sequence[float]) -> list[float]:
        if not self._backend_used:
            raise RuntimeError("fit must be called before predict")
        values = [float(value) for value in inputs]
        if len(values) != self._input_width:
            raise ValueError(f"input function must contain {self._input_width} values")
        if self._backend_used == "single-sample-interpolant":
            return list(self._state["single"])
        if self._backend_used == "torch":
            return self._predict_torch(values)
        return self._predict_numpy(values)

    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        outputs: Sequence[Sequence[float]],
    ) -> dict[str, float]:
        _validate_training_rows(inputs, outputs)
        predicted = [self.predict(row) for row in inputs]
        expected = [[float(value) for value in row] for row in outputs]
        errors = [guess - truth for guesses, truths in zip(predicted, expected) for guess, truth in zip(guesses, truths)]
        truths = [value for row in expected for value in row]
        rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
        mae = sum(abs(error) for error in errors) / len(errors)
        scale = math.sqrt(sum(value * value for value in truths) / len(truths))
        mean = sum(truths) / len(truths)
        total = sum((value - mean) ** 2 for value in truths)
        residual = sum(error * error for error in errors)
        return {
            "rmse": rmse,
            "mae": mae,
            "relative_rmse": rmse / max(scale, 1e-12),
            "r2": 1.0 - residual / max(total, 1e-12),
            "max_absolute_error": max(abs(error) for error in errors),
        }

    def _fit_report(self, samples: int, metrics: dict[str, float], *, identifiable: bool) -> dict[str, Any]:
        return {
            "family": self.family,
            "samples": samples,
            "input_width": self._input_width,
            "output_width": self._output_width,
            "backend": self._backend_used,
            "implementation": "branch/trunk operator with learned coupling",
            "is_full_implementation": True,
            "operator_identifiable": identifiable,
            "training_metrics": metrics,
        }

    def _fit_numpy(self, inputs: Sequence[Sequence[float]], outputs: Sequence[Sequence[float]]) -> None:
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - NumPy is a science extra
            raise RuntimeError("The deterministic DeepONet backend requires NumPy") from error

        x = np.asarray(inputs, dtype=np.float64)
        y = np.asarray(outputs, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-10] = 1.0
        normalized = (x - mean) / scale
        rng = np.random.default_rng(self.seed)
        random_weights = rng.normal(0.0, 1.0 / math.sqrt(self._input_width), (self._input_width, self.hidden))
        random_bias = rng.uniform(-0.5, 0.5, self.hidden)
        branch = _numpy_branch(np, normalized, random_weights, random_bias)
        trunk = _numpy_trunk(np, self._output_width, min(self.basis, self._output_width))

        # Project observed functions onto the trunk, then learn the branch-to-
        # coefficient map. The final prediction is branch(x) @ W @ trunk(s)^T.
        coefficients = y @ np.linalg.pinv(trunk.T)
        gram = branch.T @ branch
        penalty = self.regularization * np.eye(gram.shape[0])
        penalty[0, 0] = 0.0
        coupling = np.linalg.pinv(gram + penalty) @ branch.T @ coefficients
        self._state = {
            "mean": mean,
            "scale": scale,
            "random_weights": random_weights,
            "random_bias": random_bias,
            "coupling": coupling,
            "trunk": trunk,
        }
        self._backend_used = "numpy-random-feature-deeponet"

    def _predict_numpy(self, values: Sequence[float]) -> list[float]:
        import numpy as np

        state = self._state
        normalized = (np.asarray(values, dtype=np.float64)[None, :] - state["mean"]) / state["scale"]
        branch = _numpy_branch(np, normalized, state["random_weights"], state["random_bias"])
        prediction = branch @ state["coupling"] @ state["trunk"].T
        return [float(value) for value in prediction[0]]

    def _fit_torch(self, inputs: Sequence[Sequence[float]], outputs: Sequence[Sequence[float]]) -> None:
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(outputs, dtype=torch.float32)
        mean = x.mean(dim=0)
        scale = x.std(dim=0, unbiased=False)
        scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
        coordinates = torch.linspace(-1.0, 1.0, self._output_width).reshape(-1, 1)

        class _TorchDeepONet(nn.Module):
            def __init__(self, input_width: int, hidden: int, basis: int) -> None:
                super().__init__()
                self.branch = nn.Sequential(nn.Linear(input_width, hidden), nn.Tanh(), nn.Linear(hidden, basis))
                self.trunk = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, basis))
                self.bias = nn.Parameter(torch.zeros(1))

            def forward(self, functions: Any, points: Any) -> Any:
                return self.branch(functions) @ self.trunk(points).T + self.bias

        model = _TorchDeepONet(self._input_width, self.hidden, min(self.basis, self._output_width))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)
        normalized = (x - mean) / scale
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(normalized, coordinates) - y) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        self._state = {"model": model, "mean": mean, "scale": scale, "coordinates": coordinates}
        self._backend_used = "torch-deeponet"

    def _predict_torch(self, values: Sequence[float]) -> list[float]:
        import torch

        state = self._state
        x = torch.tensor([values], dtype=torch.float32)
        with torch.no_grad():
            prediction = state["model"]((x - state["mean"]) / state["scale"], state["coordinates"])[0]
        return [float(value) for value in prediction.tolist()]


def _validate_training_rows(
    inputs: Sequence[Sequence[float]], outputs: Sequence[Sequence[float]]
) -> tuple[int, int]:
    if not inputs or not outputs:
        raise ValueError("inputs and outputs must contain at least one sample")
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must contain the same number of samples")
    input_width = len(inputs[0])
    output_width = len(outputs[0])
    if input_width < 1 or output_width < 1:
        raise ValueError("training rows must not be empty")
    if any(len(row) != input_width for row in inputs) or any(len(row) != output_width for row in outputs):
        raise ValueError("all input and output rows must have consistent widths")
    return input_width, output_width


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _numpy_branch(np: Any, normalized: Any, weights: Any, bias: Any) -> Any:
    nonlinear = np.tanh(normalized @ weights + bias)
    return np.concatenate((np.ones((normalized.shape[0], 1)), normalized, nonlinear), axis=1)


def _numpy_trunk(np: Any, output_width: int, basis: int) -> Any:
    coordinates = np.linspace(0.0, 1.0, output_width)
    columns = [np.ones(output_width)]
    mode = 1
    while len(columns) < basis:
        columns.append(np.sin(math.pi * mode * coordinates))
        if len(columns) < basis:
            columns.append(np.cos(math.pi * mode * coordinates))
        mode += 1
    return np.column_stack(columns[:basis])
