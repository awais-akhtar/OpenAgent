"""Deterministic planning for scientific tasks before tool execution."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SciencePlan:
    workflow: str
    steps: tuple[str, ...]
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"workflow": self.workflow, "steps": list(self.steps), "assumptions": list(self.assumptions)}


class ScientificPlanner:
    def create_plan(self, goal: str) -> SciencePlan:
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


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80]
