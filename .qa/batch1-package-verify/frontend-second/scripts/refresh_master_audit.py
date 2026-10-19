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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openagent.routes import DOMAIN_ROUTE_MODULES, route_contracts  # noqa: E402
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
        "The route-contract module column names the module that declares each group. Runtime dispatch may execute a handler branch in that module or in `openagent/routes/runtime.py`; this table does not claim every implementation branch was inspected or tested. Source-file columns are literal route-pattern matches in production TS/JS; demo fixtures and test files are excluded. Matches are by path, not method, and are not proof of active calls, response compatibility, permissions or runtime behavior. The tests below do not exercise this entire matrix against a running Python backend.",
        "",
        "| Group | Audience | Method | Backend route pattern | Route-contract module | Original frontend source candidates | frontend-second source candidates |",
        "| --- | --- | --- | --- | --- | --- | --- |",
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
            regex = route_pattern(pattern)
            matched: dict[str, list[str]] = {}
            for name, files in sources.items():
                matched[name] = [relative for relative, text in files if regex.search(text)]
                if matched[name]:
                    counts[name] += 1
            audience = ROUTE_AUDIENCE_OVERRIDES.get((group, pattern), GROUP_AUDIENCE.get(group, "Review required"))
            handler = f"openagent/routes/{modules[group]}.py"
            primary = "; ".join(f"`{p}`" for p in matched["Original frontend"]) or "No literal found"
            second = "; ".join(f"`{p}`" for p in matched["frontend-second"]) or "No literal found"
            lines.append(f"| {group} | {audience} | {method} | `{pattern}` | `{handler}` | {primary} | {second} |")
        declared = int(contract["route_count"])
        second_group_rows.append(
            f"| {group} | {GROUP_AUDIENCE.get(group, 'Review required')} | {declared} | "
            f"{counts['Original frontend']} | {counts['frontend-second']} | {declared - counts['frontend-second']} |"
        )

    if total != 654:
        print(f"Note: registry currently contains {total} route entries (not the previously observed 654).")
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
    summary_intro = (
        "The route matrix below contains all registered patterns, handler modules, audiences and source-file references. "
        "A source reference is only a static trace candidate. A route with no literal reference is a screen-coverage review candidate, not a reason to add UI. "
        "Do not surface privileged or internal routes merely to raise counts.\n\n"
    )
    text = text[:summary_start] + "\n".join(second_group_rows) + "\n\n" + text[summary_end:]
    direction_b = text.index("## Direction B")
    summary_start = text.index("### Route-pattern source references by backend group", direction_b)
    text = text[:direction_b] + "## Direction B — backend functions not directly exposed in frontend-second\n\n" + summary_intro + text[summary_start:]
    MASTER.write_text(text, encoding="utf-8")
    print(f"Refreshed {total} route rows from {len(contracts)} live contract groups.")
    for name, files in sources.items():
        count = sum(1 for contract in contracts for route in contract["routes"] if any(route_pattern(str(route["path"])).search(source) for _, source in files))
        print(f"{name}: {count} route-pattern candidates from {len(files)} production source files")


if __name__ == "__main__":
    main()
