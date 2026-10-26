"""Refresh the master audit's complete route/source trace table.

Run from the repository root with its backend Python environment active:
    python frontend-second/scripts/refresh_master_audit.py

This is a static source trace. A matching route string is not evidence that a
request succeeds, is authorized, or was exercised by an end-to-end test.
"""

from __future__ import annotations

from pathlib import Path
import sys
from datetime import date
import ast
import re
from collections import Counter
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openagent.routes import DOMAIN_ROUTE_MODULES, route_contracts  # noqa: E402
from openagent.auth import required_scope_for_path  # noqa: E402
from scripts.audit_frontend_api_coverage import (  # noqa: E402
    GROUP_AUDIENCE,
    ROUTE_AUDIENCE_OVERRIDES,
    route_pattern,
)

MASTER = ROOT / "frontend-second" / "FRONTEND_BACKEND_MASTER_GAPS.md"
SURFACES = {
    "Original frontend": ROOT / "frontend" / "src",
    "frontend-second": ROOT / "frontend-second" / "src",
}
EXCLUDED = {"demo", "tests", "__tests__"}
EXCLUDED_SUFFIXES = (".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx", ".test.js", ".spec.js")


@lru_cache(maxsize=None)
def dispatcher(module: str, method: str) -> str:
    """Locate a dispatch symbol, without claiming its branches were executed."""
    relative = f"openagent/routes/{module}.py"
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
    symbol = f"handle_{method.lower()}"
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            return f"{relative}:{node.lineno} {symbol}"
    runtime = ROOT / "openagent/routes/runtime.py"
    for node in ast.walk(ast.parse(runtime.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.FunctionDef) and node.name == f"do_{method}":
            return f"openagent/routes/runtime.py:{node.lineno} {node.name} (fallback; branch unverified)"
    return "Dispatch symbol not found"


def bounded_pattern(path: str) -> re.Pattern[str]:
    # A base path inside a longer URL must not count as a reference to that
    # endpoint. Template segments are still static candidates, not runtime calls.
    return re.compile(route_pattern(path).pattern + r"(?=[?\"'`]|$)")


def read_sources(root: Path) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx"}:
            continue
        if path.name.endswith(EXCLUDED_SUFFIXES) or any(part.casefold() in EXCLUDED for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        sources.append((path.relative_to(ROOT).as_posix(), source))
    return sources


def main() -> None:
    contracts = route_contracts()
    if len(contracts) != len(DOMAIN_ROUTE_MODULES):
        raise SystemExit("Route-module mapping does not match the current contract registry.")
    modules = {contract["group"]: module for contract, module in zip(contracts, DOMAIN_ROUTE_MODULES, strict=True)}
    sources = {name: read_sources(root) for name, root in SURFACES.items()}

    lines = [
        f"### Complete route-level inventory — {sum(int(c['route_count']) for c in contracts)} declared entries",
        "",
        "Regenerated from the current Python registry. Dispatch symbols are source locations, not proof that every branch is reachable or correct. The scope column evaluates the current method-specific permission mapper with the declared path. A mapped scope does not prove object ownership. Production TS/JS URL matches exclude demo/tests and require an endpoint boundary; dynamic construction can still be missed. Matches are by path, not method, and do not prove requests, persistence or integration. Known dispatcher failures are recorded in the current gap ledger above.",
        "",
        "| Group | Audience | Method | Backend route pattern | Declared module / dispatch source | Required scope when auth enforced | Original frontend source candidates | frontend-second source candidates |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    total = 0
    second_group_rows: list[str] = [
        "### Route-pattern source references by backend group",
        "",
        "These counts are static production-source candidates. The screen-level API code-path list and complete file-by-route evidence follow; none of these counts is an end-to-end wiring count.",
        "",
        "| Group | Audience | Declared | Original frontend refs | frontend-second refs | frontend-second no literal |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for contract in contracts:
        group = str(contract["group"])
        counts = {name: 0 for name in sources}
        for route in contract["routes"]:
            total += 1
            method, pattern = str(route["method"]), str(route["path"])
            regex = bounded_pattern(pattern)
            matched: dict[str, list[str]] = {}
            for name, files in sources.items():
                matched[name] = [relative for relative, text in files if regex.search(text)]
                if matched[name]:
                    counts[name] += 1
            audience = ROUTE_AUDIENCE_OVERRIDES.get((group, pattern), GROUP_AUDIENCE.get(group, "User-facing / microphone-sensitive" if group == "voice" else "Review required"))
            handler = f"openagent/routes/{modules[group]}.py"
            scope = required_scope_for_path(pattern, method) or "No mapped scope; see public/bootstrap review"
            dispatch = dispatcher(modules[group], method)
            primary = "; ".join(f"`{p}`" for p in matched["Original frontend"]) or "No literal found"
            second = "; ".join(f"`{p}`" for p in matched["frontend-second"]) or "No literal found"
            lines.append(f"| {group} | {audience} | {method} | `{pattern}` | `{handler}`; `{dispatch}` | `{scope}` | {primary} | {second} |")
        declared = int(contract["route_count"])
        second_group_rows.append(
            f"| {group} | {GROUP_AUDIENCE.get(group, 'User-facing / microphone-sensitive' if group == 'voice' else 'Review required')} | {declared} | "
            f"{counts['Original frontend']} | {counts['frontend-second']} | {declared - counts['frontend-second']} |"
        )

    route_keys = [(str(route['method']), str(route['path'])) for contract in contracts for route in contract['routes']]
    print(f"Registry: {total} declarations, {len(set(route_keys))} unique method/path entries; duplicates: {sum(value - 1 for value in Counter(route_keys).values())}.")
    text = MASTER.read_text(encoding="utf-8")
    text = text.replace("**Updated:** 2026-10-15", f"**Updated:** {date.today().isoformat()}", 1)
    start = text.index("### Complete route-level inventory")
    end = text.index("## Direction C", start)
    text = text[:start] + "\n".join(lines) + "\n\n" + text[end:]
    direction_b = text.index("## Direction B")
    if "| Route group | Audience | Routes | Second refs | No literal |" in text[direction_b:]:
        summary_start = text.index("| Route group | Audience | Routes | Second refs | No literal |", direction_b)
    else:
        summary_start = text.index("### Route-pattern source references by backend group", direction_b)
    summary_end = text.index("### Complete route-level inventory", summary_start)
    text = text[:summary_start] + "\n".join(second_group_rows) + "\n\n" + text[summary_end:]
    # Preserve manually reviewed Direction B capability/permission findings.
    # Only its generated group summary and route matrix are replaced above.
    MASTER.write_text(text, encoding="utf-8")
    print(f"Refreshed {total} route rows from {len(contracts)} live contract groups.")
    for name, files in sources.items():
        count = sum(1 for contract in contracts for route in contract["routes"] if any(bounded_pattern(str(route["path"])).search(source) for _, source in files))
        print(f"{name}: {count} route-pattern candidates from {len(files)} production source files")


if __name__ == "__main__":
    main()
