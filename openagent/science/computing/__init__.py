"""Numerical solvers and lightweight operator surrogates."""

from .heat_equation import HeatEquation1D
from .surrogate import ScientificSurrogate

__all__ = ["HeatEquation1D", "ScientificSurrogate"]
