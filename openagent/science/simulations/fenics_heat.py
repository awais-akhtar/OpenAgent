"""One-dimensional heat diffusion through FEniCS or a labeled local fallback."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from typing import Any, Mapping

from ..computing.heat_equation import HeatEquation1D


@dataclass(frozen=True)
class FenicsHeatAdapter:
    points: int = 32
    steps: int = 80
    alpha: float = 0.15
    dt: float = 0.0008
    allow_fallback: bool = True

    def capability(self) -> dict[str, Any]:
        available = importlib.util.find_spec("fenics") is not None
        return {
            "name": "FEniCS heat equation",
            "available": available,
            "backend": "fenics" if available else "deterministic-finite-difference",
            "fallback": not available,
            "reason": "FEniCS Python package detected." if available else "FEniCS is not installed; the built-in finite-difference solver can be used as an explicit fallback.",
            "privacy_boundary": "The solver runs in the local Python process and does not add network access.",
        }

    def run(self, *, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        config = dict(inputs or {})
        points = int(config.get("points", self.points))
        steps = int(config.get("steps", self.steps))
        alpha = float(config.get("alpha", self.alpha))
        dt = float(config.get("dt", self.dt))
        amplitude = float(config.get("amplitude", 1.0))
        if importlib.util.find_spec("fenics") is not None:
            return self._run_fenics(points=points, steps=steps, alpha=alpha, dt=dt, amplitude=amplitude)
        if not self.allow_fallback:
            return {"status": "unavailable", **self.capability()}
        return self._run_fallback(points=points, steps=steps, alpha=alpha, dt=dt, amplitude=amplitude)

    def _run_fallback(self, *, points: int, steps: int, alpha: float, dt: float, amplitude: float) -> dict[str, Any]:
        solver = HeatEquation1D(points=points, steps=steps, alpha=alpha, dt=dt)
        initial = solver.initial_condition(amplitude=amplitude)
        final = solver.solve(initial)
        return {
            "status": "completed",
            "backend": "deterministic-finite-difference",
            "is_fenics_result": False,
            "fidelity": "reference-fallback",
            "warning": "FEniCS was unavailable. This result comes from OpenAgent's explicit finite-difference reference solver.",
            "coordinates": [index * solver.dx for index in range(points)],
            "initial": initial,
            "final": final,
            "energy_initial": solver.energy(initial),
            "energy_final": solver.energy(final),
            "parameters": {"points": points, "steps": steps, "alpha": alpha, "dt": dt},
        }

    @staticmethod
    def _run_fenics(*, points: int, steps: int, alpha: float, dt: float, amplitude: float) -> dict[str, Any]:
        import fenics as fe  # type: ignore[import-not-found]

        if points < 5 or steps < 1 or alpha <= 0 or dt <= 0:
            raise ValueError("points >= 5, steps >= 1, alpha > 0, and dt > 0 are required")
        mesh = fe.UnitIntervalMesh(points - 1)
        space = fe.FunctionSpace(mesh, "P", 1)
        boundary = fe.DirichletBC(space, fe.Constant(0.0), "on_boundary")
        previous = fe.interpolate(fe.Expression("a*sin(pi*x[0])", degree=3, a=amplitude, pi=math.pi), space)
        trial = fe.TrialFunction(space)
        test = fe.TestFunction(space)
        solution = fe.Function(space)
        form_left = trial * test * fe.dx + dt * alpha * fe.dot(fe.grad(trial), fe.grad(test)) * fe.dx
        form_right = previous * test * fe.dx
        initial_pairs = sorted(zip(mesh.coordinates().reshape(-1), previous.compute_vertex_values(mesh)))
        for _ in range(steps):
            fe.solve(form_left == form_right, solution, boundary)
            previous.assign(solution)
        final_pairs = sorted(zip(mesh.coordinates().reshape(-1), solution.compute_vertex_values(mesh)))
        coordinates = [float(x) for x, _ in final_pairs]
        initial = [float(value) for _, value in initial_pairs]
        final = [float(value) for _, value in final_pairs]
        energy_initial = _trapezoid_energy(coordinates, initial)
        energy_final = _trapezoid_energy(coordinates, final)
        return {
            "status": "completed",
            "backend": "fenics",
            "is_fenics_result": True,
            "fidelity": "finite-element",
            "coordinates": coordinates,
            "initial": initial,
            "final": final,
            "energy_initial": energy_initial,
            "energy_final": energy_final,
            "parameters": {"points": points, "steps": steps, "alpha": alpha, "dt": dt},
        }


def _trapezoid_energy(coordinates: list[float], values: list[float]) -> float:
    return sum(
        (coordinates[index + 1] - coordinates[index]) * (values[index] ** 2 + values[index + 1] ** 2) / 2.0
        for index in range(len(values) - 1)
    )
