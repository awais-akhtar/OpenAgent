"""Convenience entry point for the reference beam optimisation workflow."""

from __future__ import annotations

from .optimizer import EngineeringOptimizer, beam_problem


def optimise_beam(*, iterations: int = 800) -> dict[str, object]:
    return EngineeringOptimizer(iterations=iterations).solve(beam_problem())


__all__ = ["optimise_beam"]
