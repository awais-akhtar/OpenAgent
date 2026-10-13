# AionUI-inspired OpenAgent frontend transformation

## Scope and guardrails

The implementation follows the supplied AionUI transformation pack extracted from `OpenAgent_AionUI_Codex_Full_Transformation_Pack.zip`. It keeps the Python service and its same-origin APIs, existing project/data scope, route security, and the original 23 feature IDs. The target is a conversation-first React + TypeScript + Vite app using Arco Design, UnoCSS, and scoped CSS, with OpenAgent branding. The supplied AionUI code/screenshots are development references only; browser-incompatible IPC/Electron behavior and branded assets are excluded.

## Phase status

| Phase | Status | Evidence / notes |
| --- | --- | --- |
| P0 Baseline and mapping | Done (source inventory) | Read the master instructions and referenced architecture, design, API/security, reuse, phase, QA, deployment and acceptance specifications; all 23 screen dossiers were reviewed. Current `npm run check:web`, `npm run test:web` (54 tests), `npm run build:web`, and focused Python UI/route/proof tests passed using `.venv`. Playwright Chromium launch failed with Windows `spawn EPERM`; live local browser was used for route and visual smoke. |
| P1 Tokens, shell, routing, home | Implemented; visual acceptance partial | Conversation-first shell, compact navigation, project scope, history/search, command palette and query routes are implemented. Arco Design, IconPark, UnoCSS and Vite integration are in place. 1440×900 light/dark and 390/320 responsive checks were inspected; keyboard/zoom and complete screenshot matrix remain incomplete. |
| P2 Chat and streaming | Implemented in source; workflow verification partial | Chat surface uses existing Python stream/API client with provider/model, attachment, privacy gate, abort and history actions. Live send/stop, large attachments, stream reconnect/error scenarios remain unverified. |
| P3 Workspace/files/preview | Implemented; interaction verification partial | Dedicated file tree/editor/Markdown preview/details surface uses existing project-scoped APIs and mobile Edit/Preview tabs. Pane resizing and live mutation/failure-path QA remain incomplete. |
| P4 Agents/settings | Implemented in source; mutation verification partial | Dedicated agent roster/approval/permission and settings/security surfaces use real APIs; security/settings values use allowlisted summaries. Approval, dangerous-action and recovery workflows require browser tests. |
| P5 Specialist feature parity | Implemented surfaces; dossier acceptance partial | All 23 query destinations render within dedicated chat, activity, specialist, personal-work or platform-work page families. Shared compact interaction primitives remain, but no route relies on a generic fallback. Several dossier-specific workflows remain unverified; Automation builder currently supports one step at a time. |
| P6 Copy, accessibility, responsive polish | Partial | Legacy shell copy is reduced; security/settings raw JSON is replaced, theme/reduced-motion support and responsive styles exist. 320/390 widths were inspected on representative routes with no horizontal overflow; full copy, keyboard, contrast and 200% zoom audits remain. |
| P7 Integration, deployment, attribution, acceptance | Partial | Current type check, 54 web tests, production build, 8 focused Python tests and live 23-view route smoke passed. Visual inspection covered 1440×900 light/dark and workspace layout. Browser automation, stream/write workflows, full screenshot matrix, service-worker old-cache transition and license/SBOM checks remain. Full Python suite has pre-existing failures noted in QA. |

## Baseline verification

- `npm run check:web` — PASS before implementation.
- `npm run test:web` — PASS before implementation: 6 files, 23 tests.
- `npm run build:web` — PASS before implementation; output served through existing `web/` static contract.
- `.venv/Scripts/python.exe -m unittest tests.test_web_ui tests.test_route_contracts` — PASS before implementation: 7 tests.
- `python -m unittest tests.test_web_ui tests.test_route_contracts` using system Python 3.14 — environment failure: `ModuleNotFoundError: argon2`; the repository venv resolves it.
- `npm run test:e2e` — browser tests could not launch Chromium in this environment (`browserType.launch: spawn EPERM`). The live Python health smoke passed. Visual review will use the connected local browser and will be recorded separately.

## Implementation decisions

- The existing project is already React 19 + TypeScript + Vite; no framework migration is needed.
- Old `?view=...`, `project`, and `conversation_id` links must continue to resolve. The initial overview/dashboard link may open the new-chat home, while the dashboard/activity feature remains accessible.
- Sidebar history and chat timeline must not duplicate conversation lists. Conversation history is project-scoped and loaded from the existing organized-conversations API.
- No new Python behavior is assumed. Any unsupported operation remains explicitly unavailable instead of receiving a fake button/result.

## Current blockers / incomplete verification

- Playwright's Chromium process is denied by the Windows environment. Browser-level checks will be exercised through the connected desktop browser where possible, and the launch restriction will remain clearly noted.
- Full AionUI visual scoring, 23-view mutation/stream parity, accessibility, source-license/SBOM, service-worker and release acceptance gates are not yet all PASS. The 23-view live route smoke does pass, but is not full feature verification.
