"""Compatibility import for the canonical CPU evidence implementation.

The implementation is maintained in :mod:`cpu_evidence`; this module
preserves the original Phase 1 import path without duplicating its schemas or
capture logic.
"""
from __future__ import annotations

import sys

from . import cpu_evidence as _implementation

sys.modules[__name__] = _implementation
