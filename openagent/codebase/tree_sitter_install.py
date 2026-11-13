"""Explicitly install local Tree-sitter grammars for CodeQ.

Grammar downloads are never initiated by ordinary application scans. This
module is the deliberate opt-in command for an owner who wants local parsing.
"""

from __future__ import annotations

import argparse
import sys

from .tree_sitter_index import _cache_dir, supported_language_names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install selected local CodeQ Tree-sitter grammars")
    parser.add_argument("--language", action="append", default=[], help="grammar name; repeat for multiple languages")
    parser.add_argument("--list-supported", action="store_true", help="list languages routed by CodeQ")
    args = parser.parse_args(argv)
    if args.list_supported:
        print("\n".join(supported_language_names()))
        return 0
    if not args.language:
        parser.error("select at least one --language, or use --list-supported")
    names = sorted(set(str(value).strip() for value in args.language if str(value).strip()))
    unknown = sorted(set(names) - set(supported_language_names()))
    if unknown:
        parser.error("unsupported CodeQ grammar name(s): " + ", ".join(unknown))
    try:
        import tree_sitter_language_pack as pack
    except ImportError:
        print("Install the optional dependency first: pip install 'openagent[code-index]'", file=sys.stderr)
        return 2
    pack.init(pack.PackConfig(cache_dir=str(_cache_dir())))
    try:
        count = pack.download(names)
    except Exception as exc:
        print(f"Grammar installation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    installed = set(pack.downloaded_languages())
    missing = sorted(set(names) - installed)
    if missing:
        print("Grammars not present after installation: " + ", ".join(missing), file=sys.stderr)
        return 1
    print(f"Installed {len(names)} Tree-sitter grammar(s) in {_cache_dir()} ({count} package item(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
