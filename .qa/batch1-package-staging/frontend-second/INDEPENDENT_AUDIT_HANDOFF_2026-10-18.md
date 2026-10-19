# Independent OpenAgent + CodeQ audit handoff

**Audit date:** 2026-10-18 (Europe/London)  
**Frontend target:** `frontend-second/`  
**Canonical design:** `reference/frontend-final-approved.html`  
**Pinned SHA-256:** `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`

This handoff packages the current implementation and evidence for independent review. It does not approve Milestone A: the measured default-state desktop matrix is 84.94% strict RGBA, interaction activation is incomplete, and the visual report includes owner-requested hidden idle examples. No canonical HTML, Python backend behavior, original `frontend/`, or `../web` was changed for this handoff.

## Deliverables

- `OpenAgent_Frontend_Second_Current.zip` — complete current React/TypeScript source, assets, configuration, tests, scripts, approved reference, audit reports and this handoff. Excludes dependencies, caches, generated build output, database files and credentials.
- `OpenAgent_Desktop_Screenshot_Evidence_2026-10-18.zip` — 92 full-size approved PNGs and 92 full-size React PNGs, the matching manifests and strict pixel data, plus selected matched Living UI, Vault and Agents interaction screenshots. The included README records each evidence set’s source revision and limitations.
- `BACKEND_ROUTE_INTEGRATION_MATRIX.csv` — 654 declared method/path rows. It distinguishes literal/request-site candidates from absent source references and labels verification status; a source match is not a proven or successful integration.
- `INTERACTION_CONTROL_MATRIX.csv` — 2,251 canonical control/form records across 23 default routes and 82 visible tab states. Route/tab activation is marked separately from user-action verification; unactivated controls remain `Not tested`.
- `BACKEND_INTEGRATION_SECURITY_AUDIT.md` — route/source evidence, mock versus real HTTP proof boundary, and a reproducible `/api/agent-runs` authorization/project-scope reproduction.
- `UNRESOLVED_FRONTEND_ISSUES_2026-10-18.md` — prioritized remaining visual, interaction, security and integration issues with reproduction references.
- `TEST_RUN_LOGS/` — raw TypeScript, unit, Playwright, Python security, production-boundary and build outputs from this handoff.
- `backend-source/` — read-only copies of the current Python route runtime/contract, authorization, storage and selected service modules used by the audit. It is evidence only, not a patch.

All files above are included in the source ZIP except the separate screenshot ZIP, whose full-size PNGs are deliberately kept out of the source archive.

## Desktop visual evidence

The latest available matched default-route captures were produced 2026-10-18 from the pinned in-repository reference and current React source, Chrome 155.0.8059.80, DPR 1, fixed time and seeded fixtures. They cover 23 routes × 2 desktop sizes × 2 themes = 92 pairs. Full-frame strict RGBA equality is **84.94%**. The best route is Calendar at **94.92%**; lower scores include Agents **61.83%**, Models **63.37%**, Fine-tuning **70.69%**, Chat **71.41%**, Research **71.89%**, Scraper **72.98%**, Scientific AI **74.49%**, Automations **74.81%**, and CodeQ **78.94%**.

These default captures intentionally omit the canonical’s idle sample Living UI in React. Those named routes’ idle scores therefore contain expected behavior differences; they are not active-run parity measurements and were not manipulated to conceal the pixels. The approved HTML is unchanged. The paired state-specific evidence separates intentional idle suppression from explicit-start and active states. See `VISUAL_EVIDENCE_INDEX.md` and the screenshot ZIP manifests for exact files and pair metadata. The 100% fidelity target is not met; Milestone A remains open.

## Interaction evidence

The machine-readable catalog is a DOM discovery inventory, not proof that every button works. The supplied 2,251-row matrix conservatively marks only route and visible-tab navigation as exercised by the all-workspaces smoke test; it does not upgrade a control to working merely because the control appears in the DOM or its route string occurs in source. `INTERACTION_STATE_COVERAGE_REPORT.md` records focused workflows that were actually activated, including Chat streaming/cancel/error, Fine-tuning local record lifecycle and preview state, Vault Access/Audit, Communications Demo Inbox, Calendar event creation, Knowledge collection activity, Security Demo record actions, and contextual execution previews. Other control-level coverage remains unverified and is listed in the matrix.

The current full Playwright run executed the repository’s configured suite, including a historical communications mobile smoke test; no mobile/tablet design acceptance is claimed. Desktop acceptance remains 1440×900 and 1280×800, both themes.

## Backend integration and security boundary

The live Python route registry currently declares 654 method/path patterns across 49 groups. The route CSV records source references and callsite candidates, not an assertion that 654 routes are implemented, safe, or wired. Adapter tests use intercepted responses or mocked fetch. There was no full React-to-Python workflow test in this audit. Direct Python HTTP verification was limited to a health startup smoke and the isolated `/api/agent-runs` security reproduction described in the security audit.

**Confirmed blocker — `/api/agent-runs`:** the runtime GET dispatch exists at `openagent/routes/runtime.py:1654`, but it is absent from the declared registry and `required_scope_for_path()` returns `None`. When `OPENAGENT_AUTH_ENFORCED=1`, `_enforce_auth_gate()` returns true for an unmapped scope. In a fresh temporary SQLite database with synthetic `alpha-test` and `beta-test` runs, unauthenticated `GET /api/agent-runs` returned 200 with both projects. A bearer token with only `agents:run` also returned 200 with both projects. `?project=alpha-test` returned only that project. The frontend suppresses this Live history request. No backend change was made; keep it suppressed until owner-approved scope and project/owner isolation are implemented and independently tested.

Other existing/live adapter capability claims are separated into source-wired, mock-tested, Python-handler-tested, and React-to-Python-tested. None is promoted to real integration merely because a route is declared or a 503 stub was intercepted.

## Test result summary

The exact output logs are part of the source archive. The handoff includes freshly run `npm run check`, `npm test -- --reporter=dot`, the complete desktop-priority Playwright suite, `npm run build` through the safe build script, and production Live boundary verification. Backend contract/security tests use isolated temporary data where possible. A direct HTTP security test was added as an audit command, not committed as a backend test or behavior change.

No Python integration, role membership matrix, external email/network operation, real microphone capture, production database, model download, or production cutover was performed. Historical failures remain labeled historical or owner-reported; the new run’s outcomes are reported only after it completes.

## Approval decision

Milestone A should remain **OPEN**. Review the specific visual mismatches and unverified controls, then decide whether to approve backend changes for the confirmed `/api/agent-runs` authorization/project-scope issue and other documented missing capabilities. This handoff does not begin Milestone B or implement new backend functionality.
