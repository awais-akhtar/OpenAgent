"""Legacy Ollama name diagnostic; cannot certify Phase 1 routes.

Usage:
    python -m openagent.adaptive_engine.phase1_ollama_certifier_runner
    python -m openagent.adaptive_engine.phase1_ollama_certifier_runner --dry-run
    python -m openagent.adaptive_engine.phase1_ollama_certifier_runner --ollama-url http://localhost:11434
"""
from __future__ import annotations

import argparse
import sys

from .phase1_ollama_certifier import (
    DEFAULT_OLLAMA_URL,
    check_ollama_connectivity,
    map_ollama_models_to_families,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Ollama model names; this is not Phase 1 route evidence."
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama HTTP API base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check Ollama connectivity and show informational name hints only.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if not args.dry_run:
        print(
            "Ollama probes cannot certify Phase 1 routes. Use phase1_capture with "
            "the exact admitted OpenAgent checkpoint route.",
            file=sys.stderr,
        )
        return 2

    print(f"Checking Ollama connectivity at {args.ollama_url} ...")
    try:
        models = check_ollama_connectivity(args.ollama_url)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Available models ({len(models)}): {', '.join(models)}")
    model_to_family = map_ollama_models_to_families(models)

    if not model_to_family:
        print("No Ollama models matched any known Phase 1 family.")
        return 0

    print("\nInformational model-name hints only (not route or family verification):")
    for model_name, family in model_to_family.items():
        print(f"  {model_name!r} → name hint={family!r}")
    print("\nNo prompts were run and no receipt file was read or written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
