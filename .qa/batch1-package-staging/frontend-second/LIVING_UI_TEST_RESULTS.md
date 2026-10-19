# Living UI test results

## Latest verified continuation — 2026-10-18 (Calendar card fidelity)

- Corrected a real CSS reset mismatch in src/app.css: React's calendar event cards had text-width buttons, noncanonical padding/border treatment and a 1 px placement offset. They now follow the approved HTML geometry and button border behavior; canonical source remains unchanged.
- tests/e2e/calendar-event-canonical-style.spec.ts: 4/4 passed at 1440×900 and 1280×800 in light/dark, asserting approved padding, card height, themed border, radius, shadow and placement.
- Full verification: npm run check passed; npm test -- --reporter=dot passed 151/151 across 22 files; latest full Playwright passed 193/193 in 5.9 minutes; npm run build passed to isolated dist; npm run verify:production-live passed with stale Demo preference cleared, 12 attempts to offline 503 stubs, zero demo fixture requests and zero page errors. Current dated evidence: audit/2026-10-18/TEST_RESULTS_2026-10-18.md. These are frontend/mock checks, not Python integration.
- Fresh canonical/React captures: 92 matched desktop pairs, 23 routes × two sizes × two themes, Chrome 155.0.8059.80, DPR 1. Full-frame strict RGBA is 84.94%; Calendar is 94.92%. Manifests: .qa/approved-after-calendar-fix-20261018/manifest.json, .qa/react-after-calendar-fix-20261018/manifest.json, .qa/visual-match-after-calendar-fix-20261018.json. React reported 0 route/layout/runtime issues.
- Route declarations refreshed: 654 entries / 49 groups, 249 original-frontend and 257 frontend-second literal route candidates. Static matches are not verified integrations.
- Live GET /api/agent-runs remains withheld: runtime handler exists but is absent from the declared registry; required_scope_for_path() returns no scope and omitting project returns records across projects. See the master gap map. No backend code changed.
- Milestone A remains open; the whole app still has material visual and interaction differences.

## Latest full frontend verification — Advanced Demo rail actions — 2026-10-18

- `npm run check`: passed.
- `npm test -- --reporter=dot`: **142/142 passed across 22 files**.
- `npm run test:e2e -- --workers=1 --retries=0`: **154/154 passed** in 5.2 minutes. The new Advanced Demo rail cases cover successful and failed exports/resets at 1440×900 and 1280×800, light/dark; they assert no unhandled page errors and zero `/api/` calls.
- `npm run build`: passed to isolated `frontend-second/dist`. Safe-build verified `../web` remained unchanged and `public/demo` fixtures were excluded. Existing Monaco chunk-size advisory remains.
- Production preview: `npm exec vite preview -- --host 127.0.0.1 --port 5187 --strictPort`, then `$env:OPENAGENT_PRODUCTION_PREVIEW_ORIGIN='http://127.0.0.1:5187'; npm run verify:production-live` — passed; stale Demo preference cleared, 8 Live API requests intercepted with offline 503 stubs, 0 demo fixture requests, 0 page errors. This is not Python integration.
- No fresh pixel comparison or real Python integration was run. Milestone A remains open; the 23-screen visual acceptance and exhaustive interaction inventory are not claimed complete.

## Communications activity rail correction — full regression rerun — 2026-10-18

- The Communications Demo right rail now uses the same locally persisted event source as the workspace activity card. The focused `npx playwright test tests/e2e/communications-demo-record-actions.spec.ts --workers=1 --retries=0` passed **4/4** at both desktop sizes and themes and asserts the saved draft appears in the rail after reload with a nonzero activity count; the flow still makes zero `/api/` calls.
- Full post-change regression: `npm run check` passed; `npm test -- --reporter=dot` passed **142/142 across 22 files**; `npm run test:e2e -- --workers=1 --retries=0` passed **154/154 in 5.1 minutes**; `npm run build` passed to isolated `dist/`, with `../web` unchanged and `public/demo` omitted. One earlier concurrent full-suite run had a Chat delayed-response timeout (153/154); that exact test passed 4/4 in isolation, and the subsequent full run passed 154/154 with no concurrent build. This transient test failure and clean rerun are both retained in the run evidence; no source change to Chat was made.
- Production preview rerun after the final source build: stale Demo preference cleared; 8 API requests were intercepted with 503 offline stubs; 0 demo-fixture requests; 0 page errors. This remains a frontend mode-boundary check, not a Python integration test.

## Knowledge Demo reset/activity fix — 2026-10-18

- `npm run check` passed; `npm test -- --reporter=dot` passed **142/142 across 22 files**; focused `npx playwright test tests/e2e/knowledge-demo-collections.spec.ts --workers=1 --retries=0` passed **1/1**; full `npm run test:e2e -- --workers=1 --retries=0` passed **154/154 in 6.1 minutes**.
- The Knowledge test validates local collection creation, search/detail, activity persistence through reload, actual reset restoring seed records and clearing activity, and zero Demo `/api/` calls.
- `npm run build` passed to isolated `dist/`; safe-build confirmed `../web` unchanged and `public/demo` excluded. Production preview rerun passed: stale Demo preference cleared, 8 Live API attempts intercepted as 503 offline stubs, 0 demo-fixture requests, 0 page errors. This is not a live Python integration test. No pixel comparison was run; Milestone A remains open.

Status date: 2026-10-18. The latest final-source verification is summarized first; older dated sections are historical and superseded where totals differ. No backend implementation was performed. The canonical HTML remains pinned to SHA-256 `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`.

## Latest complete regression rerun — 2026-10-18

- `npm run check`: passed (TypeScript project check).
- `npm test -- --reporter=dot`: passed **142/142 tests across 22 files**.
- `npm run test:e2e -- --workers=1 --retries=0`: passed **146/146** in 5.4 minutes on the current source tree. This includes all 23 route smoke coverage, Chat streaming-state cases, contextual Living UI, Communications Demo record workflows, Advanced workspaces, Fine-tuning, Vault, CodeQ, Models, Voice, and desktop viewport/theme cases. The legacy suite also runs compact-width smoke tests; no mobile/tablet visual changes were made.
- `npm run build`: passed to isolated `frontend-second/dist`. Safe-build verified that `../web` was unchanged and `public/demo` fixtures were excluded. Vite emitted the existing large Monaco chunk advisory.
- `npm exec vite preview -- --host 127.0.0.1 --port 5187 --strictPort`, followed by `$env:OPENAGENT_PRODUCTION_PREVIEW_ORIGIN='http://127.0.0.1:5187'; npm run verify:production-live`: passed against that build. A stale Demo preference was cleared; 8 API calls were intercepted with offline 503 responses; there were 0 demo-fixture requests, 0 page errors, and no `demo_mode_request_blocked` result. This is a production-mode boundary test, not Python backend integration.
- Canonical HTML SHA-256 verified as `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`. No visual pixel matrix or live Python integration was run in this regression pass. Milestone A remains open; backend gaps remain tracked in `FRONTEND_BACKEND_MASTER_GAPS.md`.

## Chat lifecycle revalidation — 2026-10-18

- Re-ran `npx playwright test tests/e2e/chat-runtime-states.spec.ts --workers=1 --retries=0`: **48/48 passed** at 1440×900 and 1280×800 in both themes. The cases cover clean idle, immediate response, delayed first content, progressive chunks, neutral empty completion, completion while history refresh is delayed, error, cancel, conversation switch/late-response rejection, final completion, and Demo zero-API behavior.
- Re-ran `npm run check` (passed), `npm test -- --reporter=dot` (**142/142 across 22 files**), full `npm run test:e2e -- --workers=1 --retries=0` (**146/146**), and `npm run build` (passed to isolated `dist/`; existing Monaco chunk-size warning). Production preview regression passed: stale Demo preference cleared; 8 API attempts were intercepted with offline 503s; 0 demo-fixture requests; 0 page errors.
- No Chat source change was needed: review confirmed a 500 ms pending indicator that disappears on the first assistant content, stream terminal, cancel, error, or conversation switch; optimistic rows use stable local IDs and stale requests are sequence/abort guarded. Tests use intercepted SSE and offline stubs; they do not establish real Python integration. No new screenshot/pixel comparison was captured.

## Communications Demo mode-boundary correction — 2026-10-18

- Fixed a duplicate `ConnectionRail` sync-history query that was route-gated but not Demo/Live-gated. It now runs only when the Communications route is in Live mode. Live continues to use existing `GET /api/communication/sync-runs` (`openagent/routes/communication_routes.py::handle_get`); Demo activity remains local.
- `npx playwright test tests/e2e/communications-demo-record-actions.spec.ts --workers=1 --retries=0`: **4/4 passed**, including a zero-API assertion after initial content readiness and after Demo actions at 1440×900 and 1280×800, light/dark. `npx playwright test tests/e2e/all-workspaces-functional-smoke.spec.ts --workers=1 --retries=0`: **1/1 passed**, all 23 workspaces/tabs with no Demo API requests or browser errors.
- After the fix: `npm run check` passed; `npm test -- --reporter=dot` passed **142/142 across 22 files**; `npm run build` passed in isolated `dist/`; full `npm run test:e2e -- --workers=1 --retries=0` passed **146/146**; production-preview stale-Demo check passed with 8 offline 503 stubs, 0 demo-fixture requests, and 0 page errors. No backend integration test or pixel comparison was run. Milestone A remains open.

## Latest full verification — Communications Demo record actions, 2026-10-18

- `npm run check`: passed. `npm test -- --reporter=dot`: **142/142 across 22 files**. `npm run test:e2e -- --workers=1 --retries=0`: **146/146**, including 4 Communications Demo cases at 1440×900 and 1280×800 in light/dark. The workflow tests details, reply prefill, edit, archive, duplicate draft, delete cancel/confirm, compose validation/save, Escape/focus recovery, local activity persistence and zero Demo `/api/` requests.
- `npm run build`: passed to isolated `dist/`; safe-build verified `../web` unchanged and `public/demo` omitted. Existing Monaco chunk-size warning remains.
- `$env:OPENAGENT_PRODUCTION_PREVIEW_ORIGIN='http://127.0.0.1:5181'; npm run verify:production-live`: passed against the just-built production preview. Stale Demo preference cleared; 8 API calls received offline 503 stubs; 0 `/demo/` requests; Integration Guide showed no Demo privacy/export copy; 0 page errors. This is not live Python integration.
- Communications source audit found existing `POST /api/communication/messages/state` archive/read fields and draft/reply POST contracts, but the Live Inbox UI currently does not wire Archive/Reply/Compose to them. `communication_routes.ROUTES` has no message edit/delete/duplicate contracts. These are recorded in `FRONTEND_BACKEND_MASTER_GAPS.md`; no backend code changed.
- No visual pixel comparison or real Python integration was run. Canonical hash remains unchanged. Milestone A remains open.

## Communications Demo query-isolation regression — 2026-10-18

- `CommunicationsInbox` now disables the Live sync-history query while Demo mode is active. The local Inbox activity list remains backed by actual Demo fixture events; Live mode retains the existing sync-runs query and its 30-second polling interval.
- `npm run check` passed; `npm test -- --reporter=dot` passed **142/142 across 22 files**; `npm run test:e2e -- tests/e2e/communications-demo-record-actions.spec.ts --workers=1 --retries=0` passed **4/4** across 1440×900 and 1280×800 in light/dark; `npm run build` passed to isolated `dist/`, confirming `../web` unchanged and `public/demo` omitted.
- This targeted check did not rerun the complete Playwright suite, production-preview test, pixel comparison, or real Python integration. The earlier full suite remains 146/146 before this query-isolation change.

## Current backend audit-route verification — 2026-10-18

- Current source maps `GET /api/audit` to `security:admin` (`openagent/auth.py::required_scope_for_path`). `tests.test_security_acceptance.ProtectedApiAcceptanceTests.test_cross_project_audit_feed_requires_security_admin_scope` ran against an isolated temporary database and HTTP server: no token returned 401, a `security:read` token returned 403, and a `security:admin` token returned 200 with synthetic events from two projects. The full `python -m unittest tests.test_security_acceptance -v` command passed **5/5**.
- The route still returns a cross-project feed when no project filter is supplied. The React Security Audit list remains withheld; no production role/project membership matrix or frontend-to-Python UI workflow was tested. This verifies the local backend's enforced-auth behavior, not deployment configuration.
- After correcting the React Live Audit explanation, source comments and gap classification, `npm run check` passed; `npm test -- --reporter=dot -- tests/backend-wiring.test.tsx` passed **142/142**; focused Advanced E2E passed **20/20**; and the full Playwright suite rerun passed **146/146**. The full run includes Live Personal Assistant and Security Audit no-list-request checks plus all Demo workflows. `npm run build` passed to isolated `dist/` and safe-build confirmed `../web` unchanged and `public/demo` omitted. Production-preview stale-Demo verification passed with 8 offline 503 API stubs, zero demo-fixture requests and zero page errors. No screenshot comparison was rerun for this copy-only UI change.

## Latest Chat lifecycle verification — 2026-10-18

- `npm run check`: passed. `npm test -- --reporter=dot`: **142/142 passed across 22 files**. `npm run test:e2e -- --workers=1 --retries=0`: **142/142 passed**. `npm run build`: passed to isolated `dist/`; existing Monaco chunk-size advisory only.
- Chat E2E: **48/48 cases** from `tests/e2e/chat-runtime-states.spec.ts` passed at 1440×900 and 1280×800, both themes. Covers idle, immediate and delayed first response, progressive chunks, empty response, detached/delayed history refresh, failure, cancellation, conversation switching and completion. Demo idle is verified to issue zero `/api/` calls. Live cases use mocked SSE responses; no live Python integration was tested.
- Source review confirmed 500 ms delayed pending indicator, removal when the first assistant content arrives, stable optimistic row IDs, terminal completion independent of history refresh, neutral empty-response notice, and request-sequence/abort guards against stale conversation updates. No code change was required in this handoff because these behaviors were already present in the current source.
- Canonical SHA verified unchanged. No screenshot or pixel comparison was run for this Chat behavior pass; it is interaction evidence, not visual-parity evidence. Milestone A remains open.

## Latest final-source verification — Calendar dialog parity, 2026-10-18

- Calendar details match the canonical title/description modal, Cancel and Close actions; the primary Close displays the preview-only toast. Escape/close restores focus to the invoking event.
- `npm run check` passed; `npm test -- --reporter=dot` passed **142/142 across 22 files**; focused Calendar E2E passed **5/5** including a mocked Live record; full `npm run test:e2e -- --workers=1 --retries=0` passed **130/130**.
- `npm run build` passed to isolated `dist`; safe-build confirmed `../web` unchanged and `/demo` fixtures absent. The existing Monaco chunk-size warning remains.
- Production preview safety check passed on this final build: stale Demo was cleared; 8 API calls were intercepted by offline 503 stubs; 0 demo fixture requests and 0 page errors. This is not Python integration.
- Registry re-read: **654 declared method/path entries across 49 groups**. Canonical source hash unchanged. Calendar Demo verification covered invalid time, valid local create, canonical modal, toast, Escape focus recovery, reload persistence and zero Demo API calls at 1440×900 and 1280×800 in both themes. The mocked Live case verifies backend-sourced event copy and no Demo toast; it is not Python integration. No screenshot pixel comparison was generated for this interaction correction.

## Subsequent regression — Fine-tuning Live write retry and suite rerun, 2026-10-18

- `npm run check` passed.
- `npm run test -- --maxWorkers=1` passed **142/142 tests across 22 files**. New component coverage verifies Fine-tuning dataset/example/adapter form values survive a mocked 403 and clear only after a mocked 201 retry.
- `npm run test:e2e -- --workers=1` passed **125/125** on the final source/test tree. An earlier run had one Personal Assistant activity dialog miss; an isolated run of that case passed, its trigger was scoped to the activity region, the focused Advanced suite passed **20/20**, and then the full suite passed.
- `npm run build` passed to isolated `dist/`; `safe-build.mjs` confirmed `../web` unchanged and `public/demo` fixtures absent. The existing Monaco large-chunk warning remains.
- `npm run verify:production-live` passed on the built app: stale Demo preference was cleared, 8 API attempts hit offline 503 stubs, 0 demo fixture requests, 0 page errors, and the Integration Guide did not show Demo privacy/export controls. This is a production-preview boundary check, not Python integration.
- No screenshot/pixel comparison or Python live workflow was run for this form retry correction. Canonical hash remains unchanged; broader visual/interaction acceptance and Milestone A remain open.

## Subsequent frontend regression — Adaptive Engine Live rendering, 2026-10-18

- `npm run check` passed after the source change.
- `npm run test -- --maxWorkers=1` passed **141/141** across 22 files. New mocked regression covers Adaptive Engine Live Overview, Hardware, Jobs and Plans using the current endpoint payload shapes.
- `npm run test:e2e -- --workers=1` passed **125/125**. This complete desktop suite ran after the source edit; no focused visual screenshot capture was performed for Adaptive Engine.
- `npm run build` passed to isolated `frontend-second/dist`; Python-served `../web` was unchanged and `public/demo` fixture JSON was absent. Existing Monaco large-chunk warning remains.
- `npm run verify:production-live` passed against `npx vite preview --host 127.0.0.1 --port 5181 --strictPort`: stale Demo was cleared, 8 API calls received offline 503 stubs, 0 `/demo/` requests, no sample trace/export UI and 0 browser errors. The preview server must be started separately because no `preview` npm script exists.
- Adaptive Engine payload/handler shape was checked in `openagent/routes/adaptive_engine_routes.py` and `openagent/adaptive_engine/runtime.py` / `hardware.py`; the test uses mocked responses only. No Python server, authenticated session, production integration, or backend persistence was exercised. This UI correction uses existing routes and adds no backend capability.
- The earlier security route test and audit exposure remain isolated source/runtime evidence as described below. Historical broad visual scores remain diagnostic; no new pixel-parity or screenshot claim is made here.

## Historical pre-fix regression — Live audit-feed isolation, 2026-10-18 (superseded by Current backend audit-route verification above)

- TypeScript: `npm run check` passed.
- Unit suite: `npm test -- --reporter=dot` passed **140/140**, 22 files.
- Full Playwright: `$env:OPENAGENT_UI_ORIGIN='http://127.0.0.1:5190'; npm run test:e2e -- --workers=1 --reporter=line` passed **125/125**. Includes Security Audit list withholding and Personal Assistant Live no-audit-feed/no-simulated-activity checks at 1440×900 and 1280×800, both themes (8 focused cases total).
- Production build: `npm run build` passed into isolated `dist`; safe-build confirmed `../web` unchanged and `public/demo` fixture JSON excluded. Existing Monaco large-chunk warning remains.
- Production preview: `npm run verify:production-live` passed with stale Demo preference seeded; 8 API requests were intercepted with offline 503 responses, zero `/demo/` requests, no Demo export/trace UI, and zero browser errors. This is not Python integration evidence.
- Historical backend source evidence found no mapped scope and an optional project filter. This was the source state before the `security:admin` change; the handler's global-list behavior without a project filter remains.
- Historical route-scope check returned `None security:read security:admin`. It was a direct code check, not an HTTP middleware test, and is superseded by the current scope check above.
- Historical isolated runtime check returned HTTP 200 to an unauthenticated `GET /api/audit` with synthetic cross-project events. That was the pre-fix behavior; it is superseded by the current 401/403/200 acceptance test above.
- No visual recapture was needed for the non-layout change. Latest available broad visual baseline remains 344 pairs at 84.48% strict RGBA; it is historical/diagnostic, so Milestone A stays open.

### Historical follow-up — 2026-10-17 (superseded by the 2026-10-18 regression above)

After the Personal Assistant activity-state correction, `npm run check` passed; `npm test -- --reporter=dot` passed **140/140 across 22 files**; the focused Advanced demo E2E passed **12/12**; the full Playwright suite passed **117/117**; and `npm run build` passed to isolated `dist/`. The Assistant activity workflow is exercised at 1440×900 and 1280×800, light and dark. `$env:OPENAGENT_PRODUCTION_PREVIEW_ORIGIN='http://127.0.0.1:5181'; npm run verify:production-live` passed against the production preview: 8 `/api/` requests received offline 503 stubs, **0** `/demo/` fixture requests, no Demo privacy/export UI, and 0 browser errors. This is not a real Python backend test. Canonical HTML SHA remains unchanged. Broad paired screenshot matrix was not rerun after the activity-card correction, so the prior 344-pair 84.48% strict RGBA result remains historical diagnostic evidence only.

### Verification re-run — 2026-10-17

Previous verification before the Personal Assistant activity follow-up: `npm run check` passed; `npm test -- --reporter=dot` passed **140/140 (22 files)**; `npm run test:e2e -- --workers=1 --reporter=line` passed **113/113**; and the isolated build passed. The production preview check passed after rerunning with the server active. The latest verification, including the activity source edit, is recorded above.

## Historical full regression — 2026-10-17

| Check | Exact command | Result |
|---|---|---|
| TypeScript | `npm run check` | Passed, exit 0. |
| Unit/component suite | `npm test -- --reporter=dot` | Passed, **140/140 across 22 files**. |
| Fine-tuning focused E2E | `$env:OPENAGENT_UI_ORIGIN='http://127.0.0.1:5180'; npm run test:e2e -- --workers=1 --reporter=line tests/e2e/finetune-demo-interactions.spec.ts` | Passed **7/7**. New cases verify search reset across collections; unique, persisted repeated edit events; and cancelled/completed preview traces do not replay after Jobs → Results → Jobs. Existing four viewport/theme CRUD workflows verify zero Demo `/api/` requests and Demo export. |
| Full Playwright | `npm run test:e2e -- --workers=1 --reporter=line` | Passed, **117/117**. Includes a Home → Chat regression that loads the reference fixture collection before opening Chat; it verifies one clean selector option, zero Demo `/api/` calls, and no React missing-key warning, plus Personal Assistant activity at both desktop sizes and themes. |
| Isolated production build | `npm run build` | Passed in `frontend-second/dist`; safe-build verified `../web` unchanged and `public/demo` fixtures absent. Existing Monaco large-chunk advisory remains. |
| Production Live mode / Integration Guide | `npx vite preview --host 127.0.0.1 --port 5183` plus `$env:OPENAGENT_PRODUCTION_PREVIEW_ORIGIN='http://127.0.0.1:5183'; npm run verify:production-live` | Passed. Stale Demo preference cleared; 8 `/api/` calls were intercepted with offline 503 responses; **0 `/demo/` requests**, no Demo privacy/export UI, truthful Live export unavailable status, 0 page errors. This is not Python backend integration. |
| Focused desktop evidence | Previously captured with `$env:OPENAGENT_UI_ORIGIN='http://127.0.0.1:5180'; node scripts/capture-targeted-corrections.mjs` | 20 Demo screenshots at 1440×900 and 1280×800, light/dark; zero `/api/` requests. These are behavioral captures, not a fresh canonical-vs-React comparison after the Chat cache change. Manifest: `.qa/targeted-corrections-20261017/manifest.json`. |

This continuation also namespaced Home's reference recent-chat cache away from the Chat conversation cache and guards the Chat selector against missing IDs. Existing demo/live guards, SSE behavior, canonical HTML and backend adapters were retained. No fresh broad canonical-vs-React pixel matrix was generated; existing matrices remain diagnostic and Milestone A remains open.

## Historical Milestone A continuation — 2026-10-17 (superseded by the 2026-10-18 regression above)

This later update supersedes test/capture counts below where they differ. It adds truthful local Fine-tuning activity, removes unsourced generic activity copy, cleans up the user-facing integration guide, fixes Knowledge demo collection cache/tab behavior, and strengthens Fine-tuning and Knowledge E2E flows. No Python or production frontend source was changed.

| Check | Exact command | Final result |
|---|---|---|
| TypeScript | `npm run check` | Passed, exit 0. |
| Unit/component suite | `npm run test -- --maxWorkers=1` | Passed, **140/140 across 22 files**. |
| Full Playwright suite | `$env:OPENAGENT_UI_ORIGIN='http://127.0.0.1:5180'; npm run test:e2e -- --workers=1` | Passed, **61/61** after final source/test edits. This includes all-route smoke, Fine-tuning four-tab CRUD/local preview at both desktop sizes/themes, Knowledge collection invalid/valid create and reload persistence, Vault activation, and contextual trace workflows. |
| Isolated production build | `npm run build` | Passed to `frontend-second/dist`; safe-build reported `../web` unchanged and `public/demo` fixtures absent. Existing Monaco chunk-size warning remains. |
| Desktop tab comparisons | `$env:OPENAGENT_UI_ORIGIN='http://127.0.0.1:5180'; $env:OPENAGENT_DESKTOP_TABS_OUTPUT='.qa/milestone-a-current-20261017/tab-states'; node scripts/capture-desktop-tab-states.mjs` | **344 matched pairs**; 84.48% strict RGBA; 0 unmatched tabs, 0 Demo API calls, 0 browser errors. |
| Living UI comparisons | `$env:OPENAGENT_LIVING_UI_ROUTES=''; $env:OPENAGENT_UI_ORIGIN='http://127.0.0.1:5180'; node scripts/capture-living-ui-evidence.mjs` | **144 matched pairs / 80 readiness checks**; 82.48% overall; 74.73% intentionally changed idle; 83.82% explicit-start behavior divergence; 85.60% active states; 0 Demo API calls and 0 browser errors. |

Both visual runs used Chrome 155.0.8059.80, DPR 1, the pinned canonical reference, fixed clock, desktop 1440×900 and 1280×800, and light/dark themes. The screen matrix and named-state captures are separate. These measured scores do **not** establish 100% visual fidelity; Milestone A remains open.

The production stale-Demo preview regression documented below was not rerun in this continuation. The guard files and implementation were unchanged, but the newer build was not independently run through that focused production-preview check. No real Python backend was run or live integration verified.

## Final verification

| Check | Exact command | Result |
|---|---|---|
| TypeScript | `npm run check` | Passed, exit 0. |
| Unit/component suite | `npm run test -- --maxWorkers=1` | Passed: **140/140 tests across 22 files**. Includes development Demo blocking of JSON GET/POST, FormData, SSE POST and SSE GET when localStorage reads throw; also authenticated SSE GET/POST parsing checks. |
| Full Playwright suite | `npm run test:e2e -- --workers=1 --reporter=line` | Passed: **57/57 tests**, 2.0 minutes. Includes the full desktop regression suite; repository also retains legacy responsive/mobile smoke cases, with no mobile/tablet design changes in this task. |
| Isolated production build | `npm run build` | Passed. Output is `frontend-second/dist`; safe-build confirmed `../web` unchanged and `public/demo` fixtures absent. Vite still reports the existing large Monaco chunk warning. |
| Production stale-Demo regression | Start preview with `npm exec vite preview -- --host 127.0.0.1 --port 5182 --strictPort`; then `$env:OPENAGENT_PRODUCTION_PREVIEW_ORIGIN='http://127.0.0.1:5182'; node scripts/verify-production-live-context.mjs` | Passed against the final build. The browser preloaded `localStorage['oa-second-data-mode']='demo'`; production removed the stale key, rendered Live CodeQ, attempted **3 API requests** answered by offline **503** stubs, showed no `demo_mode_request_blocked`, no sample trace/playback, and had 0 page errors. This is not Python integration. |
| Regression script syntax | `node --check scripts/verify-production-live-context.mjs` | Passed. |

## API boundary details

- Production `readDataMode()` now clears a stale development Demo preference where storage is available and always returns in-memory Live. The request guard ignores persisted Demo state in production while retaining a defensive block if effective Zustand state were ever Demo.
- Development continues to fail closed on an in-memory Demo selection, even when localStorage is unavailable. The unit test asserts **zero fetch calls** for JSON GET, JSON POST, FormData POST, SSE POST and SSE GET.
- SSE POST now preserves `Accept: text/event-stream`; shared header creation no longer overwrites an explicit Accept value. Unit tests verify bearer auth, SSE content negotiation and event parsing for POST, and bearer auth plus cursor URL/event parsing for GET.
- Earlier contextual behavior remains intact: simulated traces are operation-triggered in Demo, CodeQ/Research traces default compact, and sample execution is absent in Live.

## Visual evidence and limits

The preceding paired desktop evidence remains in `.qa/living-ui-desktop-evidence-20261017-current/`; this safety-only patch did not alter layout or animation. Its 80 pairs reported 79.22% weighted strict RGBA equality, split into 74.71% intentionally changed idle, 83.82% explicit-start behavior divergences (descriptive), and 83.70% aligned active states. These are prior contextual captures, not new screenshots from this safety regression, and do not establish full 23-screen acceptance.

No Python backend was started. Requests in the production-preview check were intercepted and returned 503. No live API, database persistence, permission matrix, SSE delivery/reconnect, or real execution was tested. No backend capability was added. Milestone A remains open; no 100% visual or Live integration claim is made.
# Milestone A implementation continuation — 2026-10-17

- Scope: Fine-tuning Demo interactions in `src/screens/AdvancedScreen.tsx`; no backend changes, production cutover, canonical HTML edits, or mobile/tablet design work.
- `npm run check`: passed (TypeScript project build).
- `npm test -- --reporter=dot`: passed **140/140**, 22 test files.
- `npm run test:e2e -- --workers=1 --reporter=line`: passed **61/61** after final test-selector correction. This includes 23-route smoke, Fine-tuning CRUD and training preview at 1440×900 and 1280×800 in light/dark, Vault Access/Audit activation, and offline Live/UI-isolation checks. First exploratory run was 59/61 due one ambiguous Fine-tuning text locator and one nonreproduced Vault activation timing miss; focused rerun passed 10/10 and the clean full rerun passed 61/61.
- `npm run build`: passed in isolated `frontend-second/dist`; `safe-build.mjs` verified `../web` unchanged and `public/demo` fixture JSON absent. Existing Monaco chunk size advisory remains.
- `npm run verify:production-live` against a production preview seeded with stale `oa-second-data-mode=demo`: passed. Production cleared the stale preference, attempted 3 `/api/` requests answered by 503 stubs, showed no synthetic trace/playback and had zero page errors. This verifies frontend request behavior only, not Python integration.
- No real Python backend was started or tested. Live requests in existing Playwright checks use mocked/offline responses; this is not live integration evidence.
- Desktop paired tab screenshot capture: `node scripts/capture-desktop-tab-states.mjs` with `OPENAGENT_APPROVED_HTML=reference/frontend-final-approved.html`, `OPENAGENT_UI_ORIGIN=http://127.0.0.1:5180`, output `.qa/milestone-a-final-20261017/tab-states`; 328 tab pairs captured with 0 unmatched tabs, 0 Demo API calls, 0 browser errors, strict RGBA 84.46%. Added initial captures for the four routes without tabs, yielding 344 total pairs and 84.55% raw weighted RGBA. Intentional idle Living UI divergence is included in the raw result and must be separated from active-state comparison. Full comparison report and paired screenshots are packaged separately. Pinned canonical hash: `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`.
- Fine-tuning interaction specifics are in `INTERACTION_STATE_COVERAGE_REPORT.md`. Remaining per-control visual/behavioral coverage is partial; Milestone A remains OPEN.

## Historical frontend safety regression — 2026-10-18 (scope finding superseded by Current backend audit-route verification above)

- `npm run check`: passed.
- `npm test -- --reporter=dot`: **140/140 passed**, 22 files.
- PowerShell command `$env:OPENAGENT_UI_ORIGIN='http://127.0.0.1:5190'; npm run test:e2e -- --workers=1 --reporter=line`: **121/121 passed**. The isolated Vite dev server was started with `npm run dev -- --host 127.0.0.1 --port 5190 --strictPort`. An initial attempt against a reused 5180 server failed to mount the app because the broad Playwright route matcher `**/api/**` also intercepted the source module `/src/api/client.ts`; the matcher was corrected to intercept only URL paths beginning `/api/`, and the clean complete run passed.
- Focused Live Security Audit: **4/4 passed**, confirming zero GET `/api/audit` calls and successful mocked existing chain verification at 1440×900 and 1280×800, light/dark.
- `npm run build`: passed to `dist/`. `safe-build.mjs` verified `../web` unchanged and excluded `public/demo` fixtures. Monaco chunk size warning remains.
- `npm run verify:production-live`: passed against local Vite production preview, with stale Demo preference seeded; 8 API requests were answered by offline 503 stubs, zero demo fixture requests, no sample trace/playback, no Demo export UI, and zero page errors. This is a production frontend safety regression, not a Python integration test.
- At the time of this earlier regression, the backend source audit found no mapped read scope and the Live list was withheld. The scope finding is now superseded by the `security:admin` correction and acceptance test above; the list remains withheld due to unresolved project-scope policy. No Python service was run in that earlier frontend continuation.

### Final rerun after adding the Personal Assistant Live safeguard

- `npm run check`: passed.
- `npm test -- --reporter=dot`: **140/140 passed**, 22 files.
- `$env:OPENAGENT_UI_ORIGIN='http://127.0.0.1:5190'; npm run test:e2e -- --workers=1 --reporter=line`: **125/125 passed**. This includes 4 Personal Assistant Live cases and 4 Security Audit Live cases, each across 1440×900 and 1280×800, light/dark.
- `npm run build`: passed; isolated `dist/`, `../web` unchanged and `public/demo` omitted. Existing Monaco chunk size advisory remains.
- `npm run verify:production-live`: passed after starting `npx vite preview --host 127.0.0.1 --port 5181 --strictPort`; stale Demo preference removed, 8 API requests received offline 503 responses, 0 demo fixture requests, no sample traces/export UI in Live, 0 page errors.
- Visual evidence was not recaptured for this non-layout guard/test change. Latest broad matrix remains the historical 344-pair, 84.48% strict RGBA result; Milestone A remains open. No live Python integration was tested.

## Calendar event dialog parity and accessibility regression — 2026-10-18

- Restored the canonical event dialog content: selected event title, the approved visual-fixture description, Cancel and Close actions. The primary Close action emits the canonical “Visual preview only. No backend changes were saved.” toast. Removed extra event-field content and the redundant second close action. The dialog restores keyboard focus to the event button that opened it after Escape or close. The Radix dialog root stays mounted while its portal is conditionally shown, and the opener is retained explicitly so focus recovery is reliable for grid, Upcoming, and Events-list triggers.
- Focused Playwright: `npm run test:e2e -- --workers=1 --retries=0 tests/e2e/calendar-demo-create.spec.ts` passed **4/4**, covering create validation, local save, canonical detail modal/toast, Escape close/focus restoration, reload persistence and zero Demo API calls at 1440×900 and 1280×800 in light/dark.
- Final checks after this source edit: `npm run check` passed; `npm test -- --reporter=dot` passed **142/142 tests across 22 files**; `npm run test:e2e -- --workers=1 --retries=0` passed **129/129**; `npm run build` passed with isolated output to `dist/` (existing Monaco chunk-size warning); production-preview stale-Demo regression passed against this build with **8 offline 503 API stubs, 0 demo fixture requests and 0 page errors**.
- Backend registry re-read: **654 declared method/path entries across 49 groups**. This is a declaration inventory, not evidence of live operation. No Python server or live calendar integration was tested. The canonical approved HTML hash remains `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`.

