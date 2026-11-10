"""Generate a source-level comparison of registered backend routes and both UIs.

This reports direct route-string references only. It does not infer that a route
works, that the caller has permission, or that every unreferenced route needs UI.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from openagent.routes import route_contracts
from openagent.auth import required_scope_for_path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "frontend" / "src"
SECOND = ROOT / "frontend-second" / "src"
BACKEND_DOC = ROOT / "frontend-second" / "BACKEND_CAPABILITY_GAPS.md"
FRONTEND_DOC = ROOT / "frontend-second" / "FRONTEND_CAPABILITY_GAPS.md"

# Product audience is assigned before interpreting source-reference counts.
# A group can contain mixed operations; the per-route notes below take
# precedence where a particular route is privileged or operational.
GROUP_AUDIENCE = {
    "health": "Internal / operator",
    "core": "Internal / operator",
    "integrations": "Advanced / admin",
    "chat": "User-facing",
    "agents": "User-facing + advanced permissions",
    "workflows": "User-facing",
    "knowledge": "User-facing",
    "semantic": "Advanced user",
    "media": "User-facing",
    "local_ai": "Advanced user / admin actions",
    "enterprise": "Admin-only",
    "developer": "Advanced / admin",
    "security": "Admin-only / security-sensitive",
    "calendar": "User-facing",
    "communication": "User-facing + external-send approval",
    "assistant": "User-facing",
    "personal_os": "Advanced user",
    "production": "Internal / operator",
    "twin": "Advanced user / privacy-sensitive",
    "fact_provenance": "Advanced user / privacy-sensitive",
    "vault": "Advanced / security-sensitive",
    "scraper": "Advanced user / network-sensitive",
    "research": "User-facing",
    "finetune": "Advanced user / admin actions",
    "workspace": "User-facing",
    "auth": "Admin-only / security-sensitive",
    "document": "User-facing",
    "draft": "Advanced user",
    "worker": "Internal / operator",
    "vector_store": "Internal / operator",
    "advanced_rag": "Advanced user",
    "inference": "Internal / operator",
    "ultra_model": "Advanced user / admin actions",
    "adaptive_engine": "Advanced user / operator",
    "contacts": "User-facing",
    "backup_restore": "Admin-only / destructive confirmation",
    "model_cookbook": "Advanced user",
    "document_processing": "Advanced user",
    "image_studio": "User-facing",
    "scheduler": "User-facing + admin actions",
    "safe_terminal": "Advanced user / explicit consent",
    "mcp_runtime": "Advanced / admin",
    "tool_runtime": "Internal / operator",
    "compute_router": "Internal / operator",
    "agent_bridge": "Internal / operator",
    "skills": "Advanced user / admin actions",
    "conversation-organization": "User-facing",
    "enterprise-identity": "Admin-only / security-sensitive",
    "science": "Advanced user",
    "codebase": "User-facing + destructive/approval controls",
    "voice": "User-facing / microphone-sensitive",
    "voice_conversation": "User-facing / microphone-sensitive",
    "model_serve": "Advanced user / model execution",
    "g25_advanced": "Advanced user / privacy-sensitive",
}

ROUTE_AUDIENCE_OVERRIDES = {
    ("chat", "/api/chat/diagnose"): "Internal / operator",
    ("core", "/api/capabilities/hard-audit"): "Admin-only / operator",
    ("core", "/api/mission/run-check"): "Internal / operator",
    ("health", "/api/system/metrics"): "Admin-only / operator",
    ("health", "/api/routes/handlers"): "Internal / operator",
    ("safe_terminal", "/api/terminal/stream"): "Advanced user / explicit consent",
}

def route_pattern(path: str) -> re.Pattern[str]:
    pieces = re.split(r"(\{[^{}]+\})", path)
    expression = "".join(r"\$\{[^{}]+\}" if piece.startswith("{") and piece.endswith("}") else re.escape(piece) for piece in pieces)
    return re.compile(expression)


def production_source(root: Path) -> list[str]:
    result: list[str] = []
    for path in root.rglob("*"):
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx"}:
            continue
        if path.name.endswith((".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx", ".test.js", ".spec.js")) or any(part.casefold() in {"__tests__", "tests", "demo"} for part in path.parts):
            continue
        try:
            result.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            continue
    return result


def scan() -> tuple[list[tuple[str, str, str]], dict[str, set[tuple[str, str, str]]]]:
    contracts = route_contracts()
    endpoints = [(contract["group"], route["method"], route["path"]) for contract in contracts for route in contract["routes"]]
    matches: dict[str, set[tuple[str, str, str]]] = {}
    for name, root in (("primary", PRIMARY), ("second", SECOND)):
        texts = production_source(root)
        matches[name] = {
            endpoint for endpoint in endpoints
            if any(re.compile(route_pattern(endpoint[2]).pattern + r"(?=[?\"'`]|$)").search(text) for text in texts)
        }
    return endpoints, matches


def fmt_table(rows: list[tuple[str, ...]]) -> str:
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "| " + " | ".join("---" for _ in rows[0]) + " |", *["| " + " | ".join(row) + " |" for row in rows[1:]]])


def generate_backend(endpoints: list[tuple[str, str, str]], matches: dict[str, set[tuple[str, str, str]]]) -> str:
    contracts = route_contracts()
    group_rows = [("Route group", "Audience", "Declared", "Primary refs", "Second refs")]
    for contract in contracts:
        group = str(contract["group"])
        total = int(contract["route_count"])
        primary = sum(1 for endpoint in endpoints if endpoint[0] == group and endpoint in matches["primary"])
        second = sum(1 for endpoint in endpoints if endpoint[0] == group and endpoint in matches["second"])
        group_rows.append((group, GROUP_AUDIENCE.get(group, "Review required"), str(total), str(primary), str(second)))
    total = len(endpoints)
    unmapped = [(method, path) for _, method, path in endpoints if required_scope_for_path(path, method) is None]
    public = ", ".join(f"`{method} {path}`" for method, path in unmapped)
    return f"""# Backend API surface — generated source inventory

> Supporting route evidence only. [FRONTEND_BACKEND_MASTER_GAPS.md](FRONTEND_BACKEND_MASTER_GAPS.md) is the sole current gap, wiring and test-status map.

Updated: {date.today().isoformat()}
Source: `openagent.routes.route_contracts()` and production frontend source at this working-tree revision.

## Registered route inventory

The registry has **{total} distinct method/path patterns across {len(contracts)} groups**. The current scope mapper covers **{total - len(unmapped)}/{total}**. The unmapped routes are {public}. These are public health and authentication/bootstrap paths, not a count of working private routes. Adaptive Engine declarations are included for registry accounting only and excluded from the owner's frontend/backend gap closure.

The static scan found bounded route-pattern strings for **{len(matches['primary'])}/{total}** entries under `frontend/src` and **{len(matches['second'])}/{total}** under `frontend-second/src`. A match is not an executed request. A missing literal may be a dynamically constructed URL or a route intentionally reserved for admin/runtime use. The complete route-by-route string matrix is in [FRONTEND_CAPABILITY_GAPS.md](FRONTEND_CAPABILITY_GAPS.md), and handler/scope/file candidates are in the master file.

{fmt_table(group_rows)}

## Capability and security disposition

The current missing backend contracts, incomplete integrations, backend operations needing UI, permissions and test evidence are classified once in the [master gap map](FRONTEND_BACKEND_MASTER_GAPS.md). This generated file does not repeat those findings because a route-string scan cannot establish a capability gap or a safe UI placement.

## Evidence limits

- The counts come from declared contracts and source text, not all possible dynamic runtime branches.
- Scope mapping does not prove object ownership or approval behavior.
- Source references do not prove a frontend control works or that a Python handler persists data.
- Read the dated commands and current blockers in the master file before treating a capability as verified.
"""


def generate_frontend(endpoints: list[tuple[str, str, str]], matches: dict[str, set[tuple[str, str, str]]]) -> str:
    contracts = route_contracts()
    summaries = [("Group", "Audience", "API entries", "Primary direct refs", "Primary w/o ref", "Second direct refs", "Second w/o ref")]
    for contract in contracts:
        group = str(contract["group"])
        total = int(contract["route_count"])
        p = sum(1 for endpoint in endpoints if endpoint[0] == group and endpoint in matches["primary"])
        q = sum(1 for endpoint in endpoints if endpoint[0] == group and endpoint in matches["second"])
        summaries.append((group, GROUP_AUDIENCE.get(group, "Review required"), str(total), str(p), str(total - p), str(q), str(total - q)))

    matrix = [("Group", "Audience", "Method", "Backend route pattern", "Primary source ref", "Second source ref")]
    for group, method, path in endpoints:
        key = (group, method, path)
        audience = ROUTE_AUDIENCE_OVERRIDES.get((group, path), GROUP_AUDIENCE.get(group, "Review required"))
        matrix.append((group, audience, method, f"`{path}`", "Found" if key in matches["primary"] else "No literal", "Found" if key in matches["second"] else "No literal"))

    return f"""# Frontend route references — generated source inventory

> Supporting source inventory only. [FRONTEND_BACKEND_MASTER_GAPS.md](FRONTEND_BACKEND_MASTER_GAPS.md) holds the current gap, integration and test classifications. “Found” means only that a production source string matched.

Updated: {date.today().isoformat()}
Surfaces: `frontend/src` and `frontend-second/src`, excluding test and Demo fixture files.

## Summary

The backend registry declares **{len(endpoints)} method/path entries across {len(contracts)} groups**. Bounded route strings appear for **{len(matches['primary'])}** entries in the original frontend and **{len(matches['second'])}** in `frontend-second`. These are source-reference counts, not working integrations or missing UI counts. Adaptive Engine routes appear only for registry accounting and are excluded from the owner's gap closure.

{fmt_table(summaries)}

## Route-by-route source matrix

{fmt_table(matrix)}

## How to read the matrix

- “Found” means a path pattern appears in production source. It does not establish the HTTP method, mounted control, response schema, authorization, or a successful Python request.
- “No literal” may indicate dynamic URL construction or an intentionally internal/admin route. It does not itself establish a frontend feature gap.
- The [master map](FRONTEND_BACKEND_MASTER_GAPS.md) records verified workflows, remaining backend contracts, frontend omissions, test evidence and product/permission boundaries.
"""


def main() -> None:
    endpoints, matches = scan()
    BACKEND_DOC.write_text(generate_backend(endpoints, matches), encoding="utf-8")
    FRONTEND_DOC.write_text(generate_frontend(endpoints, matches), encoding="utf-8")
    print(f"Wrote {BACKEND_DOC.relative_to(ROOT)} and {FRONTEND_DOC.relative_to(ROOT)}")
    print(f"Declared {len(endpoints)} routes across {len(route_contracts())} groups; primary references {len(matches['primary'])}; second references {len(matches['second'])}.")


if __name__ == "__main__":
    main()
