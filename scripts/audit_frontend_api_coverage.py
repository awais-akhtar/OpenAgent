"""Generate a source-level comparison of registered backend routes and both UIs.

This reports direct route-string references only. It does not infer that a route
works, that the caller has permission, or that every unreferenced route needs UI.
"""

from __future__ import annotations

import re
from datetime import date
from collections import Counter
from pathlib import Path

from openagent.routes import route_contracts


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "frontend" / "src"
SECOND = ROOT / "frontend-second" / "src"
BACKEND_DOC = ROOT / "frontend-second" / "BACKEND_CAPABILITY_GAPS.md"
FRONTEND_DOC = ROOT / "frontend-second" / "FRONTEND_CAPABILITY_GAPS.md"

BACKEND_GAPS = [
    ("Shared workspace-to-project selection", "GET /api/codebase/project-identities now exposes CodeQ identity records, but no common user-facing contract consistently maps a selected project to chat, files, research, tasks and specialist services. Enterprise workspace IDs, CodeQ project IDs and the default service project remain distinct."),
    ("Incremental agent/workflow execution events", "CodeQ and Research expose run-event SSE, and workflows accept POST /api/workflows/event. No resumable run-event read/stream contract for agent/workflow execution output is declared."),
    ("Agent run retry and cancellation", "Agent runs can be listed and checked, but no run-specific retry or cancellation contract is declared."),
    ("Science project, executed-run and report history", "Science proposal records are persisted, but there is no complete project/report history contract and a proposal remains distinct from an executed experiment."),
    ("Personal-context source attribution", "Twin source metadata exists, but there is no source-to-fact mapping explaining which interaction produced a profile field or preference."),
    ("Fine-tuning checkpoint manifest", "Datasets, jobs and adapters exist; no authoritative checkpoint list/manifest route is declared."),
    ("Browser voice streaming", "Voice sessions, turns, STT and TTS operations exist; no browser microphone WebSocket or equivalent live capture transport is declared."),
    ("Voice media playback", "STT/TTS return operation metadata or artifacts, but no permissioned binary asset fetch/stream contract is declared for browser playback."),
    ("Chat agent selection", "The chat request accepts provider/model, attachments and external-egress consent, but has no agent_id or agent selection contract."),
    ("Incremental terminal output", "The safe terminal route returns a completed run with buffered events; no incremental response/event transport is declared."),
    ("CodeQ preview session lifecycle", "Codebase project/tree/file/run/event APIs exist; no preview URL or preview-session create/read/stop contract is declared."),
    ("Notification preferences", "Notification records can be listed and marked read, but per-user delivery preferences and channel/type settings have no storage/API contract."),
    ("Workspace item archive/delete/restore", "Workspace items can be listed, created and updated, but no dedicated delete/archive/restore operation is declared. The current UI stores link type as item metadata, so this is not counted as a missing typed-link contract."),
    ("Research source review notes", "Research source discovery and session checkpoints exist, but no API persists user review notes or annotations against a discovered source."),
    ("Assistant task editing and deletion", "Assistant tasks can be created and completed, but no task update/delete contract is declared for due dates, priority or goal relationships."),
    ("Calendar event mutation lifecycle", "Calendar event creation and free-slot lookup exist, but event update/delete and reminder acknowledgement contracts are not declared."),
    ("Calendar-scoped availability search", "Free-slot lookup returns project-level availability and does not accept a calendar identifier for a calendar-specific query."),
    ("Calendar provider lifecycle", "Provider registration and sync are represented, but no provider removal or re-authentication lifecycle contract is declared."),
    ("Stored chat media deletion", "Chat supports uploaded media references and detach-from-prompt behavior, but no user-scoped endpoint deletes a stored attachment asset."),
    ("Knowledge record detail/actions mapping", "Document detail/update/archive/delete routes exist. The Knowledge screen does not call those routes, and a mapping between indexed knowledge IDs and document IDs has not been verified."),
    ("Model benchmark history retrieval", "Benchmark results are persisted in the model_benchmarks table, but the registered API has no GET list/history contract for retrieving those records."),
]

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
}

ROUTE_AUDIENCE_OVERRIDES = {
    ("chat", "/api/chat/diagnose"): "Internal / operator",
    ("core", "/api/capabilities/hard-audit"): "Admin-only / operator",
    ("core", "/api/mission/run-check"): "Internal / operator",
    ("health", "/api/system/metrics"): "Admin-only / operator",
    ("health", "/api/routes/handlers"): "Internal / operator",
    ("safe_terminal", "/api/terminal/stream"): "Advanced user / explicit consent",
}

SECURITY_REVIEW = [
    ("Safe terminal run detail", "Fixed in this pass: the detail lookup now filters by authenticated owner when auth is enforced, and /api/terminal/ requires terminal:read or terminal:write scopes in team/enterprise mode. Local single-user behavior remains unchanged."),
    ("Research session ownership", "Research routes are behind the knowledge scope gate in enforced-auth mode, but durable session rows currently scope by project and session ID rather than owner. Confirm that project membership is the intended sharing boundary; add owner isolation if sessions are user-private."),
    ("Conversation object ownership", "Chat/conversation route auth gates API scopes, but stored conversations and some branch/export/replay operations are resolved by opaque conversation ID or project without an owner field. Confirm local single-user-only deployment or add owner-scoped storage/lookup before multi-user rollout."),
    ("Agent approval ownership", "Approval resolution accepts an approval ID without a user-facing project/owner binding in the request/service contract. Scope the approval to its requester/project before multi-user rollout."),
    ("Share grant expiry enforcement", "The share API accepts and persists an expiry timestamp, and the UI requires one; verify authorization checks enforce expiry on every access path rather than treating it as display metadata."),
    ("Project identity consistency", "The second frontend hardcodes the default service project on many screens, and CodeQ projects have their own IDs. Do not treat the frontend project string or enterprise workspace ID as an authorization boundary."),
]


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
            if any(route_pattern(endpoint[2]).search(text) for text in texts)
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
    primary_count = len(matches["primary"])
    second_count = len(matches["second"])
    wired_rows = [("Previously identified area", "Current backed capability")]
    wired_rows.extend([
        ("Notifications", "Notification records list and mark-read now use the scheduler routes; preference storage and push transport remain absent."),
        ("Knowledge collections", "Semantic collection list/create routes exist; they are separate from indexed documents and have no document-membership or rename/delete contract."),
        ("Communications", "Inbox, Sent and Drafts records are surfaced; inbox read-state uses the existing message-state route and the UI polls every 15 seconds."),
        ("Scraper history", "Stored extraction metadata has a list route. This does not persist or return full extraction bodies."),
        ("Vault access log", "Project-scoped operation metadata is returned without secret values or ciphertext."),
        ("Science proposals", "Proposal records persist with proposed/not_started semantics; proposals are not executed experiment runs."),
        ("Twin source metadata", "Interaction type/topic/date metadata is listed without original prompts; source-to-profile-fact provenance remains absent."),
        ("Calendar", "Calendar list/filter/create and user-triggered free-slot lookup are surfaced; free-slot search is project-wide because its API has no calendar-id argument."),
    ])
    gap_rows = [("Capability gap", "Backend finding")]
    gap_rows.extend(BACKEND_GAPS)
    return f"""# Backend capability gaps and API surface audit

> Supporting generated route evidence only. Use [FRONTEND_BACKEND_MASTER_GAPS.md](FRONTEND_BACKEND_MASTER_GAPS.md) as the sole current gap/status source.

Updated: {date.today().isoformat()}  
Source of truth: `openagent.routes.route_contracts()` at this repository revision.

## Route inventory

The registered contract catalog currently declares **{total} distinct method/path entries across {len(contracts)} route groups**. The user estimate of 646 is close, but the current registry reports {total}. These are route patterns (including parameterized paths), not a count of concrete URLs or proof that each handler succeeds.

The source-level scan found a direct route-pattern string in **{primary_count}/{total}** primary frontend endpoints and **{second_count}/{total}** second-frontend endpoints. Those counts are a traceability aid, not functional coverage. See [FRONTEND_CAPABILITY_GAPS.md](FRONTEND_CAPABILITY_GAPS.md) for the full {total}-row matrix and the static-scan limits.

{fmt_table(group_rows)}

## Genuine backend capability gaps

These gaps are capabilities not represented by the current backend contracts. Routes present in the registry but absent from a UI are frontend-surface gaps, not backend gaps. CodeQ and Research SSE are existing contracts and are not listed as missing capabilities.

{fmt_table(gap_rows)}

## Backend security review findings

This source-level review identifies authorization boundaries that need intentional product policy. The terminal detail exposure was closed in this pass; research's project-shared ownership model still requires confirmation before multi-user rollout.

{fmt_table([("Area", "Finding / status"), *SECURITY_REVIEW])}

## Backend capabilities already added or confirmed during gap closure

These are functional contracts, not remaining backend gaps. Their frontend coverage is listed in the route matrix.

{fmt_table(wired_rows)}

## Backend route groups with broader APIs than the current screens use

Audience classification is applied by route group before UI review. A group may mix audiences, so route-level overrides are shown in the matrix. Advanced/admin routes must stay behind intentional panels and backend authorization; internal/operator contracts should not be added to the user-facing navigation. This classification is a product-surface recommendation, not a substitute for checking the route's actual authorization.

## Verification boundary

- Inventory was read from the live Python route-contract registry, not a remembered count or documentation snapshot.
- Each contract entry is compared against production TypeScript/JavaScript under `frontend/src` and `frontend-second/src`; test files are excluded.
- The scanner reports only whether a route pattern appears as source text. Dynamic URL builders, aliases and helper-generated paths can be false negatives; a route string in source can still be unused or broken.
- Route declarations do not prove a handler branch, successful response, response-schema compatibility, authentication, authorization, streaming behavior or end-to-end usability. Those require route-specific tests and runtime/browser verification.
- The current gap audit also includes a terminal detail authorization fix and new terminal scopes. Focused tests cover owner filtering, enforcement-mode authentication, read-scope denial and full-scope access.
"""


def generate_frontend(endpoints: list[tuple[str, str, str]], matches: dict[str, set[tuple[str, str, str]]]) -> str:
    contracts = route_contracts()
    summaries = [("Group", "Audience", "API entries", "Primary direct refs", "Primary w/o ref", "Second direct refs", "Second w/o ref")]
    for contract in contracts:
        group = str(contract["group"])
        total = int(contract["route_count"])
        p = sum(1 for endpoint in endpoints if endpoint[0] == group and endpoint in matches["primary"])
        s = sum(1 for endpoint in endpoints if endpoint[0] == group and endpoint in matches["second"])
        summaries.append((group, GROUP_AUDIENCE.get(group, "Review required"), str(total), str(p), str(total - p), str(s), str(total - s)))

    matrix = [("Group", "Audience", "Method", "Backend route pattern", "Primary source ref", "Second source ref")]
    for group, method, path in endpoints:
        key = (group, method, path)
        audience = ROUTE_AUDIENCE_OVERRIDES.get((group, path), GROUP_AUDIENCE.get(group, "Review required"))
        matrix.append((group, audience, method, f"`{path}`", "Found" if key in matches["primary"] else "No literal", "Found" if key in matches["second"] else "No literal"))

    return f"""# Frontend API coverage gaps — full route audit

> Supporting route source inventory only. The authoritative current gap/status, wiring and test classifications are in [FRONTEND_BACKEND_MASTER_GAPS.md](FRONTEND_BACKEND_MASTER_GAPS.md). “Found” means only that a production source path string matched.

Updated: {date.today().isoformat()}  
Audited surfaces: primary React app (`frontend/src`) and independent second frontend (`frontend-second/src`).

## Summary

The backend contract registry declares **{len(endpoints)} method/path entries across {len(contracts)} groups**. The scanner found direct route-pattern references in **{len(matches['primary'])}** entries in the primary frontend and **{len(matches['second'])}** in the second frontend. A direct reference is not a verification of request method, response handling, permissions or working interaction.

The primary app is the existing 22-destination reference frontend. `frontend-second/` is the target production frontend. Keep their coverage distinct: a capability in one does not imply that the other frontend exposes it. Route audiences below distinguish user-facing, advanced/admin-only and internal operations; route-level overrides take precedence.

{fmt_table(summaries)}

## How to use the route matrix

- **Found**: the registered route pattern appears as a string in that frontend's production source (fixture/demo and test files are excluded). This is still not proof of an active callsite or successful request.
- **No literal**: no direct route-pattern string was found. This is a candidate frontend gap, but may be a false negative if a shared helper or dynamic URL builder constructs the path.
- Routes are not assigned “not needed” automatically. Some are privileged admin/runtime operations and should remain inaccessible unless a deliberate screen and authorization workflow are designed.
- The audience column is a UI classification aid, not proof of the route's authorization. The full list is the backend's declared contract surface; it does not prove every route has a live handler or passes its tests.

## Route-by-route matrix

{fmt_table(matrix)}

## Highest-priority frontend review areas

1. **Project scope**: both frontends commonly use `default`; no shared project selector is safe until backend project/workspace identities are formally mapped.
2. **Execution feedback**: CodeQ and Research have real SSE. Generic agent/workflow records are polled and do not expose a common incremental event stream. The second frontend must show only statuses supported by returned run records.
3. **Communications**: the second frontend now displays Inbox/Sent/Drafts records and persists read state; Inbox polls. Compare provider accounts, approvals, scheduled messages, analytics and dead-letter actions against route rows before exposing them.
4. **Calendar**: the second frontend now filters events by calendar and requests backend free-slot suggestions. Providers, conflicts, reminders, sync and iCalendar actions remain separate review items.
5. **Specialist/admin APIs**: enterprise identity, integrations, worker queues, model lifecycle, backup/restore, adaptive engine and developer/security actions need explicit role/scope review; route existence alone is not a product UI requirement.

## Recommended closure workflow

For each candidate route, inspect the route handler and service, verify its permission/scope behavior, map it to a user workflow, wire it to the intended frontend with honest loading/error/empty/success states, add focused component and backend contract tests, then update the status here. Keep missing backend capabilities in `BACKEND_CAPABILITY_GAPS.md` instead of building simulated UI responses.
"""


def main() -> None:
    endpoints, matches = scan()
    BACKEND_DOC.write_text(generate_backend(endpoints, matches), encoding="utf-8")
    FRONTEND_DOC.write_text(generate_frontend(endpoints, matches), encoding="utf-8")
    print(f"Wrote {BACKEND_DOC.relative_to(ROOT)} and {FRONTEND_DOC.relative_to(ROOT)}")
    print(f"Declared {len(endpoints)} routes across {len(route_contracts())} groups; primary references {len(matches['primary'])}; second references {len(matches['second'])}.")


if __name__ == "__main__":
    main()
