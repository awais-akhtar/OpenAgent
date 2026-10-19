# Prioritized unresolved issues — 2026-10-18

This is a decision-focused issue list from the current code and evidence. It does not treat DOM discovery, route declarations, mock tests, or screenshot counts as functional completion.

## P0 — Security blocker

### Live agent run history is unsafe and intentionally withheld

- **Control:** Agents/Home run history (`GET /api/agent-runs`).
- **Evidence:** Current runtime handler is at `openagent/routes/runtime.py:1654`; route is missing from all 654 declared contracts; `required_scope_for_path('/api/agent-runs','GET')` returns `None`; enabled auth enforcement permits unmapped scopes. `openagent/storage.py:360` returns every project when `project` is omitted.
- **Reproduction:** Isolated temporary SQLite + real local Python HTTP server with `OPENAGENT_AUTH_ENFORCED=1`, two synthetic projects. No token → HTTP 200 with both project IDs; `agents:run` token → HTTP 200 with both; `?project=alpha-test` → alpha only. Full exact output is in `TEST_RUN_LOGS/python-agent-runs-security-repro.json`.
- **Current UI:** Live history is withheld; Demo history remains fixture-only. Existing tests assert the Live UI does not request the unsafe endpoint.
- **Required decision/work:** Owner approval before changing backend behavior. Register a read contract/scope, require a project identity, enforce owner/project membership (including no-filter behavior), and add anonymous, under-scoped, authorized, cross-project and omission tests before reconnecting UI.

## P1 — Material desktop visual mismatch

- Latest default-state matrix: **84.94% strict RGBA** across 92 matched route/viewport/theme captures. The lowest full-frame routes are Agents 61.83%, Models 63.37%, Fine-tuning 70.69%, Chat 71.41%, Research 71.89%, Scraper 72.98%, Scientific AI 74.49%, Automations 74.81%, and CodeQ 78.94%.
- The lower score for nine execution-oriented idle routes includes the deliberate owner-requested suppression of canonical sample traces at rest. Do not restore fake idle activity to raise the score. Compare explicit-start/active states separately; remaining state-level mismatches are still real work.
- No region-by-region full 23-route visual review was completed in this audit. The current raw pixel comparison is a measurement, not a pass. Reproduce each cited pair from the screenshot evidence archive and inspect the full-size images.
- Milestone A remains open; no 100% fidelity claim is supported.

## P1 — Interaction inventory is not fully activated

- The catalog has 2,251 recorded control/form entries over 23 routes and 82 visible tab states. The inventory generator does not activate controls.
- 474 route/tab records have only navigation-smoke evidence. Other controls are conservatively `Not tested` unless a workflow has explicit coverage in `INTERACTION_STATE_COVERAGE_REPORT.md` and a named test.
- Prioritize unverified form validation, nested menus, notification actions, empty/loading/error branches, approval review, retry/cancel, keyboard focus recovery, file import/export, microphone permission, and destructive Live confirmations.
- Exact per-control DOM metadata, route/tab source map, status and reproduction note: `INTERACTION_CONTROL_MATRIX.csv`.

## P1 — No comprehensive React-to-Python integration run

- `tests/backend-wiring.test.tsx` and related adapter tests use mocked/intercepted API responses. Production preview checks used offline HTTP 503 stubs. They do not establish Python handler compatibility, auth configuration, persistence, or streaming in a running app.
- Existing handler unit/HTTP tests are separate from React and do not cover the entire route registry.
- Route-level evidence and scope/source candidates: `BACKEND_ROUTE_INTEGRATION_MATRIX.csv`; details and test boundaries: `BACKEND_INTEGRATION_SECURITY_AUDIT.md` and `FRONTEND_BACKEND_MASTER_GAPS.md`.
- Do not proceed into Milestone B until visual/interaction acceptance and the unsafe agent-history boundary are reviewed.

## P2 — Route inventory is source-trace, not handler-complete proof

- Current Python registry count: 654 patterns / 49 groups.
- In a conservative source scan, 39 declared paths have a request-callsite candidate, 218 have only a static source reference, and 397 have no frontend-second literal reference. Those are discovery categories, not “working integrations” or product decisions. Many APIs are intentionally internal/admin/runtime and should not be surfaced.
- The 654-row CSV records declaring module and auth-scope mapping; each row explicitly says runtime handler behavior was not individually verified. Use Python route tests and auth/source inspection before promoting any row to verified.

## P2 — Specific workflow/contract limits

See the 25 current Direction A entries and Direction C verification list in `FRONTEND_BACKEND_MASTER_GAPS.md`. Notable items include project/workspace identity consistency, missing incremental events for several job types, science executed-run/report history, fact-level provenance, Fine-tuning checkpoint manifest, browser microphone stream, terminal incremental stream, CodeQ preview lifecycle, notification preferences, and the absent scoped agent-run history contract. These are not implemented by this audit.

## Startup error report (separate from Milestone A)

The user reported recurring Windows dialog `python.exe — The application was unable to start correctly (0x0000142)`. During this handoff, `C:\Python314\python.exe` 3.14.3, `.venv` 3.14.3, and `.accelerator-venv` 3.12.10 all launched directly; backend module import passed; Workbench status reported ready; isolated direct-server and Workbench HTTP health checks each returned 200 when configured to use writable temporary runtime/data paths. The Windows Application log showed no Python application-crash event during the inspected 24-hour period. The error was therefore **not reproduced** and no source fix is claimed. One separate Python unittest process was observed in the environment; it was not terminated because it may be an active job. Exact failing executable/trigger remains unknown and requires a screenshot or exact action/command that raises the dialog.
