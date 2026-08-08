"""Equation residual checks supplied by a caller."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable, Iterable, Mapping

from ..models import ScientificFinding


def check_residuals(residuals: Iterable[float], *, tolerance: float = 1e-6) -> ScientificFinding:
    values = [abs(float(value)) for value in residuals]
    maximum = max(values, default=0.0)
    passed = maximum <= max(0.0, float(tolerance))
    return ScientificFinding(
        check="equation_residuals",
        passed=passed,
        message="Equation residuals are within tolerance." if passed else "Equation residuals exceed tolerance.",
        severity="error" if not passed else "info",
        evidence={"maximum_residual": maximum, "tolerance": float(tolerance), "count": len(values)},
    )


def evaluate_equations(
    equations: Iterable[Callable[[], float]],
    *,
    tolerance: float = 1e-6,
) -> ScientificFinding:
    return check_residuals((equation() for equation in equations), tolerance=tolerance)


def check_symbolic_equations(
    equations: Iterable[Mapping[str, Any] | str],
    *,
    tolerance: float = 1e-6,
) -> ScientificFinding:
    """Evaluate restricted algebraic equations with SymPy and report residuals."""

    rows = list(equations)
    if not rows:
        return ScientificFinding("symbolic_equations", True, "No symbolic equations were supplied.", evidence={"checked": 0})
    try:
        import sympy
    except ImportError:
        sympy = None
    residuals: list[float] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        expression = str(row.get("expression", "")) if isinstance(row, Mapping) else str(row)
        symbols = row.get("symbols", {}) if isinstance(row, Mapping) and isinstance(row.get("symbols"), Mapping) else {}
        if not _safe_expression(expression):
            errors.append(f"Equation {index + 1} contains unsupported syntax.")
            continue
        try:
            left, right = (expression.split("=", 1) + ["0"])[:2] if "=" in expression else (expression, "0")
            left, right = left.strip(), right.strip()
            numeric_symbols = {str(name): float(value) for name, value in symbols.items()}
            if sympy is not None:
                names = {name: sympy.Symbol(name) for name in set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", expression))}
                substitutions = {names[name]: value for name, value in numeric_symbols.items() if name in names}
                residual = sympy.sympify(left.replace("^", "**"), locals=names) - sympy.sympify(right.replace("^", "**"), locals=names)
                residuals.append(abs(float(residual.subs(substitutions).evalf())))
            else:
                residuals.append(abs(_evaluate_restricted(left, numeric_symbols) - _evaluate_restricted(right, numeric_symbols)))
        except (TypeError, ValueError, KeyError, SyntaxError) as exc:
            errors.append(f"Equation {index + 1} could not be evaluated: {exc}")
    maximum = max(residuals, default=float("inf") if errors else 0.0)
    passed = not errors and maximum <= max(0.0, float(tolerance))
    return ScientificFinding(
        "symbolic_equations",
        passed,
        "Symbolic equations satisfy the supplied values." if passed else "One or more symbolic equations need review.",
        "error" if not passed else "info",
        {"checked": len(rows), "evaluated": len(residuals), "maximum_residual": maximum, "tolerance": tolerance, "errors": errors, "engine": "sympy" if sympy is not None else "restricted-algebra-fallback"},
    )


def _safe_expression(value: str) -> bool:
    return bool(value.strip()) and "__" not in value and re.fullmatch(r"[A-Za-z0-9_+\-*/().=^ ]+", value) is not None


_BINARY_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_restricted(expression: str, symbols: Mapping[str, float]) -> float:
    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in symbols:
            return float(symbols[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return float(_BINARY_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return float(_UNARY_OPERATORS[type(node.op)](evaluate(node.operand)))
        raise ValueError("unsupported symbolic expression")

    return evaluate(ast.parse(expression.replace("^", "**"), mode="eval"))
