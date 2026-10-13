# OpenAgent React frontend implementation ledger

This ledger maps the current React 19 frontend to its 22 active view IDs. The PhD Application Studio is retired from this build. The frontend uses TypeScript, Vite, Arco Design and UnoCSS with app-owned, responsive AionUI-inspired styling. No AionUI branding, logos, or source assets are included. Pages are split into lazy-loaded view bundles; the latest production entry is approximately 336 KB (102 KB gzip). The Playwright suite checks deep-link navigation and representative API method, path, payload, safety and responsive contracts; live Python smoke covers health, capabilities and conversation listing. Handler-specific persistence and authorization remain enforced by the Python backend and are not all individually proven by browser tests.

| View ID | React implementation | API coverage in code | Verification evidence |
|---|---|---|---|
| `dashboard` | `frontend/src/pages/DashboardPage.tsx` | health, real metrics, modules, worker jobs, approvals, inference/local AI and audit reads | Mocked frontend requests exercised; live health/capability smoke; no fabricated metric history |
| `chat` | `frontend/src/pages/ChatPage.tsx` | conversation history, models, POST SSE stream, stop, branch, archive/restore, import/export and artifacts | Mocked frontend requests exercised; live smoke covers conversation listing; mobile history starts collapsed |
| `codebase` | `frontend/src/pages/CodebasePage.tsx` | CodeQ project, guarded file writes, preflight, run/resume/cancel, snapshots, approvals and evidence | Mocked frontend requests exercised; handler side effects not all covered by live tests |
| `voice` | `frontend/src/pages/VoicePage.tsx` | actual local STT status, sessions, transcript/turns, recording and translation | Mocked frontend requests exercised; runtime availability follows backend capabilities |
| `research` | `frontend/src/pages/ResearchPage.tsx` | sessions, durable event stream/checkpoint recovery, reports, source graph, cancel/retry/branch/archive/delete | Mocked frontend requests exercised; handler side effects not all covered by live tests |
| `agents` | `frontend/src/pages/ResourcePages.tsx` (`agents`) | roster, marketplace, project permissions/approvals; request approval | Mocked frontend requests exercised; Python handler side effects not covered |
| `knowledge` | `frontend/src/pages/KnowledgePage.tsx` | status, search/RAG, documents, ingest/upload, tags, graph, timeline and diagnostics | Mocked frontend requests exercised; accessible keyboard tabs verified |
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
- `npm run build:web` — passed; React/Vite production assets and UnoCSS output emitted to `web/` for Python static serving.
- `npm run test:e2e` — current browser suite result is recorded in [FRONTEND_QA_RESULTS.md](../FRONTEND_QA_RESULTS.md). UI action checks intercept the API and verify methods, paths, payload contracts and safety confirmations; these checks do not execute every Python route side effect.
- Browser checks cover 22 active destinations, navigation/history, command palette, accessible tabs and keyboard navigation, collapsed-sidebar names, theme persistence, mobile navigation, chat/provider/checked-agent actions and responsive overflow. Screenshots were captured for desktop and mobile in both themes; complete dossier-level visual scoring remains outstanding.
- Live Python smoke coverage passed for `/api/health`, `/api/capabilities`, and conversation listing. It does not cover every live API handler/action.
- `python -m unittest tests.test_route_contracts tests.test_web_ui tests.test_desktop.DesktopTests.test_workbench_uses_one_stable_default_port` — passed (8 tests); the Python server also served the generated `app.js` and `styles.css` with HTTP 200 during the final verification.
- Full Python discovery results and remaining failures are recorded in [FRONTEND_QA_RESULTS.md](../FRONTEND_QA_RESULTS.md); focused UI, route, chat-document, authentication and security tests must pass before release.
- Backend behavior for each mocked page action, including persistence, validation, authorization and egress enforcement, remains unverified unless covered by the listed live smoke checks. No additional operation-gap claims are made without a handler-by-handler audit.
- Third-party dependency license inventory is recorded in [FRONTEND_NPM_LICENSE_INVENTORY.md](../FRONTEND_NPM_LICENSE_INVENTORY.md); full release attribution and distribution checks remain separate gates.

## Integration notes

- `frontend/src/App.tsx` retains the active project in the URL/local storage and remounts the active page when it changes.
- Project-scoped list reads in `ResourcePages.tsx` send the selected project for agents' permission/approval lists, assistant/workflow/communication lists, calendar, science memory/datasets, scraper, twin, and vault entries. Host-global model/runtime/security/developer inventories remain unscoped.
- The Python API is authoritative for authentication, authorization, privacy/egress, and side-effect approval. Browser labels do not grant access.
