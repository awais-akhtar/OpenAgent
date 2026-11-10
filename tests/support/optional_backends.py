"""Why an optional dependency group is missing, for tests that need one.

Several engine paths and document parsers come from optional dependency groups
(``engine-native``, ``documents``, ``security``). On an install without the
group, a test that imports one at module level does not fail cleanly: pytest
reports it as a collection error, which stops the whole run, and a test that
imports it inside the body fails with a provider error that reads as a
regression in the code under test rather than an absent optional package.

Each helper keys on the import, so a package that is installed and genuinely
broken still fails loudly instead of being skipped.
"""

from __future__ import annotations

from importlib import import_module

NUMBA_REASON = "needs the engine-native dependency group (numba)"


def missing(*modules: str) -> str:
    """The skip reason naming the first absent module, or an empty string."""
    for module in modules:
        try:
            import_module(module)
        except ImportError:
            return f"needs the optional dependency {module}"
    return ""


def numba_missing() -> str:
    """The skip reason, or an empty string when numba is importable."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return NUMBA_REASON
    return ""
