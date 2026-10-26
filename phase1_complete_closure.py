#!/usr/bin/env python3
"""Compatibility entry point for the canonical Phase 1 closure verifier.

Older revisions of this script fabricated route conversions, artifact hashes,
cache measurements, and serving evidence. The canonical verifier is now the
only supported closure authority; this wrapper performs no source mutation and
creates no evidence.
"""

from phase1_verify import main


if __name__ == "__main__":
    raise SystemExit(main())
