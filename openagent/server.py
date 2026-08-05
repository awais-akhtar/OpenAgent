"""Compatibility facade for the OpenAgent local HTTP runtime.

The stdlib HTTP implementation now lives under ``openagent.routes.runtime`` so
route modules can continue moving toward domain-owned dispatch without breaking
imports that use ``openagent.server``.
"""

from __future__ import annotations

from .routes.runtime import build_http_server, first, parse_bool, parse_int, run_server

__all__ = [
    "build_http_server",
    "first",
    "parse_bool",
    "parse_int",
    "run_server",
]
