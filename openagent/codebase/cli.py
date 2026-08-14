"""Command-line access to the CodeQ generator SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .sdk import CodeQSDK, GenerationRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect software targets and CodeQ generation contracts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="List registered framework adapters.")
    inspect = subparsers.add_parser("inspect", help="Build a repository map.")
    inspect.add_argument("repository", type=Path)
    preflight = subparsers.add_parser("preflight", help="Prepare a generation and validation contract.")
    preflight.add_argument("goal")
    preflight.add_argument("--repository", type=Path)
    preflight.add_argument("--accept", action="append", default=[], help="Add a required acceptance criterion.")
    return parser


def run(argv: list[str] | None = None, *, sdk: CodeQSDK | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    args = build_parser().parse_args(argv)
    runtime = sdk or CodeQSDK()
    if args.command == "catalog":
        return runtime.registry.catalog()
    if args.command == "inspect":
        return runtime.inspect(args.repository)
    request = GenerationRequest(
        goal=args.goal,
        repository=str(args.repository or ""),
        acceptance_criteria=tuple(args.accept),
    )
    return runtime.prepare(request)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
