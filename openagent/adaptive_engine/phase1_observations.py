"""Stable compatibility name for the canonical observation verifier."""
import sys

from . import device_observations as _implementation

# Alias the module itself (rather than copying symbols) so constants patched
# by bounded tests/tools affect the one implementation's verifier.
sys.modules[__name__] = _implementation
