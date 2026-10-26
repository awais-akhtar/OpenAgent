"""Public P02 route-binding API backed by the canonical source matrix.

The previous implementation maintained a second hand-authored 162-row table
whose owners drifted from ``source_scope``. This module deliberately delegates
to ``route_bindings`` so admission, verification, and callers share one exact
six-field key and one ordered callable chain.
"""
from __future__ import annotations

from .route_bindings import (
    SourceReviewedRouteBinding as RouteBinding,
    get_source_reviewed_route_binding,
    source_reviewed_route_bindings,
    _resolve_callable_owner_source,
    validate_callable_owner_sources,
    validate_source_reviewed_route_bindings,
)


def all_bindings() -> tuple[RouteBinding, ...]:
    """Return the immutable exact bindings for all unique admitted tuples."""
    return source_reviewed_route_bindings()


def get_route_binding(
    family: str,
    model_type: str,
    backend: str,
    arithmetic: str,
    source_format: str,
    persisted_format: str,
) -> RouteBinding | None:
    """Return an exact source binding, or ``None`` when no tuple matches."""
    try:
        return get_source_reviewed_route_binding(
            family=family,
            model_type=model_type,
            backend=backend,
            arithmetic=arithmetic,
            source_format=source_format,
            persisted_format=persisted_format,
        )
    except KeyError:
        return None


def require_route_binding(
    family: str,
    model_type: str,
    backend: str,
    arithmetic: str,
    source_format: str,
    persisted_format: str,
) -> RouteBinding:
    """Return and validate the exact six-field binding; never guess a neighbor."""
    try:
        return get_source_reviewed_route_binding(
            family=family,
            model_type=model_type,
            backend=backend,
            arithmetic=arithmetic,
            source_format=source_format,
            persisted_format=persisted_format,
        )
    except KeyError as error:
        raise ValueError('route tuple has no P02 source-reviewed binding') from error


def validate_all_bindings() -> dict[tuple[str, ...], str]:
    """Return validation results for every canonical binding."""
    try:
        validate_source_reviewed_route_bindings()
        return {binding.route_key: "ok" for binding in all_bindings()}
    except Exception as error:
        return {binding.route_key: f"{type(error).__name__}: {error}"
                for binding in all_bindings()}


def report_coverage() -> str:
    """Describe tuple coverage without implying runtime execution evidence."""
    bindings = all_bindings()
    counts: dict[str, int] = {}
    for binding in bindings:
        counts[binding.family] = counts.get(binding.family, 0) + 1
    lines = ["P02 canonical route bindings", f"  Unique tuples: {len(bindings)}"]
    lines.extend(f"  {family}: {count}" for family, count in sorted(counts.items()))
    lines.append("  Runtime certified: 0")
    return "\n".join(lines)


__all__ = [
    "RouteBinding",
    "all_bindings",
    "get_route_binding",
    "require_route_binding",
    "validate_all_bindings",
    "validate_callable_owner_sources",
    "_resolve_callable_owner_source",
    "report_coverage",
]
