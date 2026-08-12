"""Validated model-assisted planning with a deterministic local fallback."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping


SUPPORTED_WORKFLOWS = frozenset({
    "heat_equation",
    "heat_exchanger_optimization",
    "bridge_optimization",
    "beam_optimization",
    "materials_discovery",
    "multimodal_fusion",
    "scientific_review",
})


@dataclass(frozen=True)
class SciencePlan:
    workflow: str
    steps: tuple[str, ...]
    assumptions: tuple[str, ...] = ()
    source: str = "deterministic"

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow": self.workflow,
            "steps": list(self.steps),
            "assumptions": list(self.assumptions),
            "source": self.source,
        }


class ScientificPlanner:
    def __init__(self, gateway: Any = None) -> None:
        self.gateway = gateway

    def create_plan(self, goal: str) -> SciencePlan:
        if self.gateway is not None:
            model_plan = self._model_plan(goal)
            if model_plan is not None:
                return model_plan
        return self._deterministic_plan(goal)

    def _model_plan(self, goal: str) -> SciencePlan | None:
        schema = {
            "workflow": sorted(SUPPORTED_WORKFLOWS),
            "steps": "array of 2 to 8 concise executable steps",
            "assumptions": "array of up to 6 explicit assumptions",
        }
        try:
            response = self.gateway.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Plan a scientific workflow. Return one JSON object only. "
                            "Use exactly one allowed workflow and do not add keys. "
                            f"Schema: {json.dumps(schema, sort_keys=True)}"
                        ),
                    },
                    {"role": "user", "content": goal.strip()},
                ],
                provider_id="auto",
                allow_external=False,
                max_output_tokens=500,
            )
            payload = _json_object(str(response.content))
            return validate_plan(payload, source="model")
        except Exception:
            return None

    def _deterministic_plan(self, goal: str) -> SciencePlan:
        text = goal.strip().lower()
        if any(word in text for word in ("heat equation", "diffusion", "temperature field", "thermal simulation")):
            return SciencePlan("heat_equation", ("define a stable finite-difference problem", "run the numerical solver", "verify finite values and energy behavior", "prepare a source-backed result"))
        if "heat exchanger" in text:
            return SciencePlan("heat_exchanger_optimization", ("gather parameters and thermal constraints", "generate a reproducible model", "run the optimisation", "evaluate equations, units, and constraints", "produce the profiled report"))
        if "bridge" in text:
            return SciencePlan("bridge_optimization", ("gather geometry, loads, material properties, and safety factors", "generate the structural model", "run the section optimisation", "evaluate stress and deflection constraints", "produce the profiled report"), ("Defaults are labelled when project memory does not provide a required structural parameter.",))
        if any(word in text for word in ("beam", "stress", "deflection", "design optimization", "design optimisation")):
            return SciencePlan("beam_optimization", ("gather parameters", "generate the beam model", "run the optimisation", "evaluate stress and deflection constraints", "produce the profiled report"))
        if any(word in text for word in ("material", "alloy", "band gap", "strength", "density")):
            return SciencePlan("materials_discovery", ("validate the local property table", "fit a transparent property predictor", "rank candidates", "select the next experiment by uncertainty"))
        if any(word in text for word in ("sensor", "thermal image", "simulation output", "multimodal", "overheating")):
            return SciencePlan("multimodal_fusion", ("collect typed local evidence", "summarise each evidence source", "compare signals", "state supported findings and open questions"))
        return SciencePlan(
            "scientific_review",
            ("define the measurable question", "separate observations from assumptions", "choose a reproducible local calculation", "verify the result", "report limitations"),
            ("No specialised scientific input was supplied; the result is a workflow plan rather than a claim about an experiment.",),
        )


def validate_plan(value: Mapping[str, Any], *, source: str = "model") -> SciencePlan:
    if set(value) - {"workflow", "steps", "assumptions"}:
        raise ValueError("The scientific plan contains unsupported fields.")
    workflow = value.get("workflow")
    if not isinstance(workflow, str) or workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError("The scientific plan selected an unsupported workflow.")
    steps = _bounded_strings(value.get("steps"), field="steps", minimum=2, maximum=8, length=180)
    assumptions = _bounded_strings(value.get("assumptions", []), field="assumptions", minimum=0, maximum=6, length=240)
    return SciencePlan(workflow, steps, assumptions, source=source)


def _bounded_strings(value: Any, *, field: str, minimum: int, maximum: int, length: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{field} must contain between {minimum} and {maximum} items.")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item.strip()) > length:
            raise ValueError(f"{field} contains an invalid item.")
        cleaned.append(item.strip())
    return tuple(cleaned)


def _json_object(value: str) -> Mapping[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else value.strip()
    if not candidate.startswith("{") or not candidate.endswith("}"):
        raise ValueError("The model did not return a JSON object.")
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("The model plan must be an object.")
    return parsed


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80]
