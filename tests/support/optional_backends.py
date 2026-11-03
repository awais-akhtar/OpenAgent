"""Why an optional compute backend cannot be exercised on this host.

Several engine paths go through kernels from the optional ``engine-native``
dependency group. On an install without that group those tests failed with
``ProviderConfigurationError: ... No module named 'numba'``, which reads as a
regression in the gateway rather than an absent optional backend.

The import is what decides it, so a backend that is installed and genuinely
broken still fails loudly instead of being skipped.
"""

from __future__ import annotations

NUMBA_REASON = "needs the engine-native dependency group (numba)"


def numba_missing() -> str:
    """The skip reason, or an empty string when numba is importable."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return NUMBA_REASON
    return ""
