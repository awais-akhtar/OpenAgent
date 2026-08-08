"""Routes scientific plans to local, testable tools."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
import json

from .computing.heat_equation import HeatEquation1D
from .materials.active_learning import ActiveLearner
from .materials.predictor import MaterialPropertyPredictor
from .planner import SciencePlan
from .verification.scientific_verifier import ScientificVerifier
from .execution import ScientificCodeRunner


class ScientificToolRouter:
    def __init__(self, *, repair: Callable[[str, str], str] | None = None) -> None:
        self.verifier = ScientificVerifier()
        self.repair = repair

    def execute(self, plan: SciencePlan, *, memory: Sequence[Mapping[str, Any]] = ()) -> tuple[dict[str, Any], Any]:
        if plan.workflow == "heat_equation":
            solver = HeatEquation1D()
            initial = solver.initial_condition("sine")
            output = {"solver": {"points": solver.points, "steps": solver.steps, "alpha": solver.alpha, "dt": solver.dt}, "initial": initial, "final": solver.solve(initial), "energy_initial": solver.energy(initial), "energy_final": solver.energy(solver.solve(initial))}
            return output, self.verifier.verify(output)
        if plan.workflow == "beam_optimization":
            parameters = _engineering_parameters(memory, bridge=False)
            return self._run_engineering_script(_beam_script(parameters), parameters, workflow="beam_optimization")
        if plan.workflow == "heat_exchanger_optimization":
            code = _heat_exchanger_script()
            execution = ScientificCodeRunner().run(code, retries=2, repair=self.repair)
            if execution.status != "completed":
                output = {"status": "failed", "execution": execution.to_dict(), "generated_code": code}
                return output, self.verifier.verify(output)
            result = json.loads(execution.stdout.strip().splitlines()[-1])
            constraints = [
                {"name": "effectiveness", "passed": float(result["effectiveness"]) >= 0.75},
                {"name": "pressure_drop", "passed": float(result["pressure_drop_pa"]) <= 45000.0},
            ]
            output = {"status": "completed", "design": result, "constraints": constraints, "execution": execution.to_dict(), "generated_code": code, "units": {"area": "m2", "pressure_drop": "Pa", "temperature": "K"}}
            return output, self.verifier.verify(output, units=output["units"], constraints=constraints)
        if plan.workflow == "bridge_optimization":
            parameters = _engineering_parameters(memory, bridge=True)
            output, verification = self._run_engineering_script(_bridge_script(parameters), parameters, workflow="bridge_optimization")
            output["limitations"] = ["This equivalent-girder screening model is not a code-certified bridge design. Validate load cases, connections, fatigue, buckling, dynamics, foundations, and the applicable design code before use."]
            return output, verification
        if plan.workflow == "materials_discovery":
            records = [
                {"density": 7.8, "hardness": 220.0, "strength": 620.0},
                {"density": 7.4, "hardness": 180.0, "strength": 510.0},
                {"density": 8.1, "hardness": 260.0, "strength": 710.0},
            ]
            predictor = MaterialPropertyPredictor()
            fit = predictor.fit(records, target="strength")
            candidates = [{"density": 7.2, "hardness": 175.0}, {"density": 7.7, "hardness": 235.0}, {"density": 8.0, "hardness": 280.0}]
            ranked = predictor.rank_candidates(candidates)
            next_experiment = ActiveLearner().select(predictor, candidates)
            output = {"fit": fit, "ranked_candidates": ranked, "next_experiment": next_experiment}
            return output, self.verifier.verify(output)
        return {"plan": plan.to_dict()}, self.verifier.verify(plan.to_dict())

    def _run_engineering_script(self, code: str, parameters: dict[str, Any], *, workflow: str) -> tuple[dict[str, Any], Any]:
        execution = ScientificCodeRunner().run(code, retries=2, repair=self.repair)
        if execution.status != "completed":
            output = {"status": "failed", "workflow": workflow, "execution": execution.to_dict(), "generated_code": code, "parameters": parameters}
            return output, self.verifier.verify(output)
        result = json.loads(execution.stdout.strip().splitlines()[-1])
        constraints = list(result.get("constraints", []))
        equations = list(result.get("equations", []))
        output = {
            "status": "completed",
            "workflow": workflow,
            "design": result.get("design", {}),
            "metrics": result.get("metrics", {}),
            "constraints": constraints,
            "equations": equations,
            "parameters": parameters,
            "memory_used": parameters.get("memory_used", []),
            "execution": execution.to_dict(),
            "generated_code": code,
            "units": result.get("units", {}),
        }
        verification = self.verifier.verify(output, units=output["units"], constraints=constraints, equations=equations)
        return output, verification


def _heat_exchanger_script() -> str:
    return """import json
import math

best = None
for tubes in range(20, 121, 5):
    area = tubes * math.pi * 0.019 * 2.5
    effectiveness = 1.0 - math.exp(-(8000.0 * area) / 42000.0)
    pressure_drop = 18000.0 + 1.8 * tubes * tubes
    mass = area * 2.4
    if effectiveness < 0.75 or pressure_drop > 45000.0:
        continue
    score = mass + pressure_drop / 10000.0
    candidate = {"tubes": tubes, "area_m2": area, "effectiveness": effectiveness, "pressure_drop_pa": pressure_drop, "estimated_mass_kg": mass, "score": score}
    if best is None or candidate["score"] < best["score"]:
        best = candidate

if best is None:
    raise RuntimeError("No feasible heat exchanger candidate was found")
print(json.dumps(best, sort_keys=True))
"""


def _engineering_parameters(memory: Sequence[Mapping[str, Any]], *, bridge: bool) -> dict[str, Any]:
    defaults = {
        "density": 7850.0,
        "youngs_modulus": 200e9,
        "stress_limit": 250e6,
        "deflection_limit": 0.005 if not bridge else 0.025,
        "length": 1.0 if not bridge else 20.0,
        "load": 1000.0 if not bridge else 500000.0,
    }
    aliases = {
        "density": ("density",),
        "youngs_modulus": ("young modulus", "youngs modulus", "elastic modulus"),
        "stress_limit": ("stress limit", "yield stress"),
        "deflection_limit": ("deflection limit",),
        "length": ("length", "span"),
        "load": ("load", "design load"),
    }
    used: list[dict[str, Any]] = []
    for record in memory:
        name = str(record.get("name", "")).strip().lower()
        for key, names in aliases.items():
            if name not in names:
                continue
            try:
                value = float(record.get("value"))
            except (TypeError, ValueError):
                continue
            unit = str(record.get("unit", ""))
            if key == "youngs_modulus" and unit == "GPa":
                value *= 1e9
            elif key in {"stress_limit"} and unit == "MPa":
                value *= 1e6
            elif key == "density" and unit == "g/cm3":
                value *= 1000.0
            defaults[key] = value
            used.append({"name": record.get("name"), "value": record.get("value"), "unit": unit, "source": record.get("source", "scientific-memory")})
    defaults["memory_used"] = used
    return defaults


def _beam_script(parameters: Mapping[str, Any]) -> str:
    config = json.dumps({key: value for key, value in parameters.items() if key != "memory_used"}, sort_keys=True)
    return f'''import json
import math

p = json.loads({config!r})
best = None
for width_mm in range(10, 101, 2):
    for height_mm in range(20, 201, 2):
        width = width_mm / 1000.0
        height = height_mm / 1000.0
        inertia = width * height ** 3 / 12.0
        stress = 6.0 * p["load"] * p["length"] / (width * height ** 2)
        deflection = p["load"] * p["length"] ** 3 / (3.0 * p["youngs_modulus"] * inertia)
        mass = width * height * p["length"] * p["density"]
        if stress > p["stress_limit"] or deflection > p["deflection_limit"]:
            continue
        candidate = {{"width_m": width, "height_m": height, "mass_kg": mass, "stress_pa": stress, "deflection_m": deflection}}
        if best is None or mass < best["mass_kg"]:
            best = candidate
if best is None:
    raise RuntimeError("No feasible beam candidate was found")
result = {{
    "design": {{"width_m": best["width_m"], "height_m": best["height_m"]}},
    "metrics": {{"mass_kg": best["mass_kg"], "stress_pa": best["stress_pa"], "deflection_m": best["deflection_m"]}},
    "constraints": [
        {{"name": "stress", "passed": best["stress_pa"] <= p["stress_limit"], "value": best["stress_pa"], "limit": p["stress_limit"]}},
        {{"name": "deflection", "passed": best["deflection_m"] <= p["deflection_limit"], "value": best["deflection_m"], "limit": p["deflection_limit"]}},
    ],
    "equations": [{{"expression": "stress = 6*load*length/(width*height^2)", "symbols": {{"stress": best["stress_pa"], "load": p["load"], "length": p["length"], "width": best["width_m"], "height": best["height_m"]}}}}],
    "units": {{"width": "m", "height": "m", "mass": "kg", "stress": "Pa", "deflection": "m"}},
}}
print(json.dumps(result, sort_keys=True))
'''


def _bridge_script(parameters: Mapping[str, Any]) -> str:
    config = json.dumps({key: value for key, value in parameters.items() if key != "memory_used"}, sort_keys=True)
    return f'''import json
import math

p = json.loads({config!r})
girder_count = 4
load_per_girder = p["load"] / girder_count
distributed_load = load_per_girder / p["length"]
best = None
for width_cm in range(20, 101, 5):
    for height_cm in range(50, 301, 5):
        width = width_cm / 100.0
        height = height_cm / 100.0
        inertia = width * height ** 3 / 12.0
        moment = load_per_girder * p["length"] / 8.0
        stress = moment * (height / 2.0) / inertia
        deflection = 5.0 * distributed_load * p["length"] ** 4 / (384.0 * p["youngs_modulus"] * inertia)
        mass = girder_count * width * height * p["length"] * p["density"]
        if stress > p["stress_limit"] or deflection > p["deflection_limit"]:
            continue
        candidate = {{"width_m": width, "height_m": height, "girder_count": girder_count, "mass_kg": mass, "stress_pa": stress, "deflection_m": deflection}}
        if best is None or mass < best["mass_kg"]:
            best = candidate
if best is None:
    raise RuntimeError("No feasible bridge screening candidate was found")
result = {{
    "design": {{"width_m": best["width_m"], "height_m": best["height_m"], "girder_count": girder_count}},
    "metrics": {{"mass_kg": best["mass_kg"], "stress_pa": best["stress_pa"], "deflection_m": best["deflection_m"]}},
    "constraints": [
        {{"name": "stress", "passed": best["stress_pa"] <= p["stress_limit"], "value": best["stress_pa"], "limit": p["stress_limit"]}},
        {{"name": "deflection", "passed": best["deflection_m"] <= p["deflection_limit"], "value": best["deflection_m"], "limit": p["deflection_limit"]}},
    ],
    "equations": [{{"expression": "stress = moment*(height/2)/inertia", "symbols": {{"stress": best["stress_pa"], "moment": load_per_girder*p["length"]/8.0, "height": best["height_m"], "inertia": best["width_m"]*best["height_m"]**3/12.0}}}}],
    "units": {{"span": "m", "width": "m", "height": "m", "mass": "kg", "stress": "Pa", "deflection": "m"}},
}}
print(json.dumps(result, sort_keys=True))
'''
