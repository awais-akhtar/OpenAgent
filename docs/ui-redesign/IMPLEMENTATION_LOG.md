# OpenAgent React frontend implementation ledger

This ledger maps the current React 19 frontend to the 23 existing view IDs. The shipped UI is built with TypeScript, Vite, Tailwind CSS 3, app-owned shadcn-style controls, and Radix Dialog, Tabs, Tooltip and Separator. The original neutral design follows the reference application's broad visual language (palette, density, shell proportions, forms, tables and dialogs) without copying source code or assets. Pages are split into lazy-loaded view bundles; the initial JavaScript bundle is 339 KB (107 KB gzip). The production Playwright suite exercises each view's API method, path and payload contract against a mocked API; live Python smoke covers health, capabilities and conversation listing. Handler-specific persistence and authorization are enforced by the unchanged Python backend and are not all individually proven by the frontend E2E suite.

| View ID | React implementation | API coverage in code | Verification evidence |
|---|---|---|---|
| `dashboard` | `frontend/src/pages/DashboardPage.tsx` | health, real metrics, modules, worker jobs, approvals, inference/local AI and audit reads | Mocked frontend requests exercised; live health/capability smoke; no fabricated metric history |
| `chat` | `frontend/src/pages/ChatPage.tsx` | conversation history, models, POST SSE stream, stop, branch, archive/restore, import/export and artifacts | Mocked frontend requests exercised; live smoke covers conversation listing; mobile history starts collapsed |
| `codebase` | `frontend/src/pages/CodebasePage.tsx` | CodeQ project, guarded file writes, preflight, run/resume/cancel, snapshots, approvals and evidence | Mocked frontend requests exercised; handler side effects not all covered by live tests |
| `voice` | `frontend/src/pages/VoicePage.tsx` | actual local STT status, sessions, transcript/turns, recording and translation | Mocked frontend requests exercised; runtime availability follows backend capabilities |
| `research` | `frontend/src/pages/ResearchPage.tsx` | sessions, durable event stream/checkpoint recovery, reports, source graph, cancel/retry/branch/archive/delete | Mocked frontend requests exercised; handler side effects not all covered by live tests |
| `phd` | `frontend/src/pages/ResourcePages.tsx` (`phd`) | studio, profile, applications, documents, runs and artifacts; create/start-draft actions | Mocked frontend requests exercised; Python handler side effects not covered |
| `agents` | `frontend/src/pages/ResourcePages.tsx` (`agents`) | roster, marketplace, project permissions/approvals; request approval | Mocked frontend requests exercised; Python handler side effects not covered |
| `knowledge` | `frontend/src/pages/KnowledgePage.tsx` | status, search/RAG, documents, ingest/upload, tags, graph, timeline and diagnostics | Mocked frontend requests exercised; Radix keyboard tabs verified |
| `workspace` | `frontend/src/pages/WorkspacePage.tsx` | library, documents, versioned edits/restore, archive and honest local text export | Mocked frontend requests exercised; handler side effects not all covered by live tests |
| `automation` | `frontend/src/pages/ResourcePages.tsx` (`automation`) | workflows, builder and run history; create and dry-run actions | Mocked frontend requests exercised; Python handler side effects not covered |
| `assistant` | `frontend/src/pages/ResourcePages.tsx` (`assistant`) | brief, tasks, goals, coach and CRM; task/goal creation | Mocked frontend requests exercised; Python handler side effects not covered |
| `communications` | `frontend/src/pages/ResourcePages.tsx` (`communications`) | connectors, inbox, contacts, approvals and schedule; draft creation | Mocked frontend requests exercised; Python handler side effects not covered |
| `models` | `frontend/src/pages/ModelsPage.tsx` | providers, models, adapters, hardware/policy, discovery, register/download/delete/probe and setup jobs | Mocked frontend requests exercised; network/download operations are explicitly confirmed |
| `security` | `frontend/src/pages/SecurityPage.tsx` | policy, egress decisions, retention preview/apply, grants/revocation, masked secrets, audit seal/verify, reports and sandbox profiles/evaluation | Mocked frontend requests exercised; secret values remain masked unless deliberately revealed |
| `developer` | `frontend/src/pages/ResourcePages.tsx` (`developer`) | developer inventory; webhook and SDK actions | Mocked frontend requests exercised; Python handler side effects not covered |
| `settings` | `frontend/src/pages/SettingsPage.tsx` | persisted runtime fields, model policy, backend privacy/account/backup reads, backup create/preview/restore | Mocked frontend requests exercised; no simulated privacy policy stored in browser |
| `engine` | `frontend/src/pages/ResourcePages.tsx` (`engine`) | adaptive engine status, hardware, plans, benchmarks and roles; inspect action | Mocked frontend requests exercised; Python handler side effects not covered |
| `science` | `frontend/src/pages/ResourcePages.tsx` (`science`) | capabilities, tools, project memory and materials; note/run actions | Mocked frontend requests exercised; Python handler side effects not covered |
| `calendar` | `frontend/src/pages/ResourcePages.tsx` (`calendar`) | project-scoped status, calendars, events, providers, conflicts and reminders; event creation | Mocked frontend requests exercised; Python handler side effects not covered |
| `twin` | `frontend/src/pages/ResourcePages.tsx` (`twin`) | project-scoped context/profile/patterns/suggestions/goals; goal creation | Mocked frontend requests exercised; Python handler side effects not covered |
| `vault` | `frontend/src/pages/VaultPage.tsx` | project-scoped masked entries, encrypted storage/reveal, key rotation and encrypted export | Mocked frontend requests exercised; sensitive reveal and key operations are confirmed |
| `scraper` | `frontend/src/pages/ResourcePages.tsx` (`scraper`) | project-scoped status/jobs; explicit schedule and scrape actions | Mocked frontend requests exercised; Python handler side effects not covered |
| `finetune` | `frontend/src/pages/ResourcePages.tsx` (`finetune`) | status, templates, datasets, jobs and adapters; dataset creation | Mocked frontend requests exercised; Python handler side effects not covered |

## Verification status

- `npm run check:web` — passed (TypeScript strict check).
- `npm run build:web` — passed (Vite production build).
- `npm run test:web` — passed (23 Vitest tests across API client, streaming, navigation, theme, research recovery, and resource contracts; Vitest 4.1.11).
- `npm run build:web` — passed; Tailwind CSS 3 and Radix UI production bundle emitted to `web/` for Python static serving.
- `npm run test:e2e` — passed, 9/9 Playwright tests. UI action checks intercept the API and verify requested method, path, and payload contracts across all 23 views. These mocked frontend-contract checks do not execute all Python route side effects.
- Browser checks cover navigation/history, Radix command dialog, Radix knowledge tabs and arrow-key behavior, collapsed-sidebar labels/tooltips, theme persistence, mobile navigation, and no horizontal overflow at widths from 1440 px down to 320 px. Desktop light, desktop dark, and mobile screenshots were visually reviewed.
- Live Python smoke coverage passed for `/api/health`, `/api/capabilities`, and conversation listing. It does not cover every live API handler/action.
- `python -m unittest tests.test_route_contracts tests.test_web_ui tests.test_desktop.DesktopTests.test_workbench_uses_one_stable_default_port` — passed (8 tests); the Python server also served the generated `app.js` and `styles.css` with HTTP 200 during the final verification.
- The full Python unittest run executed 1,396 tests and reported 33 failures and 198 errors. The current run attributes these to pre-existing backend changes; they are not frontend parity evidence. This ledger does not claim those failures are repaired.
- Backend behavior for each mocked page action, including persistence, validation, authorization and egress enforcement, remains unverified unless covered by the listed live smoke checks. No additional operation-gap claims are made without a handler-by-handler audit.
- `npm audit --omit=dev` — 0 production dependency vulnerabilities. Full `npm audit` still reports 7 development dependency advisories in the Tailwind CSS 3.4 dependency tree; the requested Tailwind major is retained because the available audit fix would move to Tailwind 4. Vitest's critical advisories were removed by upgrading to 4.1.11.

## Integration notes

- `frontend/src/App.tsx` retains the active project in the URL/local storage and remounts the active page when it changes.
- Project-scoped list reads in `ResourcePages.tsx` send the selected project for agents' permission/approval lists, assistant/workflow/communication lists, calendar, PhD, science memory/datasets, scraper, twin, and vault entries. Host-global model/runtime/security/developer inventories remain unscoped.
- The Python API is authoritative for authentication, authorization, privacy/egress, and side-effect approval. Browser labels do not grant access.
