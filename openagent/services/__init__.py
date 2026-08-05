"""Service-layer facades for OpenAgent domain managers.

These modules keep route/runtime code from depending directly on every manager
method while preserving the local-first behavior of the existing managers.
"""

from __future__ import annotations

__all__ = [
    "auth",
    "calendar_sync",
    "communication",
    "documents",
    "model_ops",
]
