"""Command line entry point for OpenAgent."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import default_home
from .server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenAgent local core.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Defaults to localhost.")
    parser.add_argument("--port", default=8765, type=int, help="Port to bind. Defaults to 8765.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for local OpenAgent data. Defaults to OPENAGENT_HOME or ~/.openagent.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else default_home()
    run_server(host=args.host, port=args.port, data_dir=data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

