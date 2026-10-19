# Batch 2A implementation and verification results

Date: 2026-10-19  
Scope: existing frontend/backend contracts only. No new Python endpoint or authorization policy was introduced in this work. The approved HTML remains unchanged at SHA-256 `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`.

## Implemented

- Added an isolated local React-to-Python smoke harness using the real React app, Vite proxy, authenticated local Python HTTP server, temporary SQLite/runtime directories, two synthetic users, a deterministic in-process provider, and a disposable CodeQ repository. No external providers are called.
- Verified Chat streaming success and persistence, provider failure, cancellation; CodeQ create/run, authenticated SSE cursor reconnection, cancellation, controlled failure and resume; Research create/report/checkpoints/events/cancellation; Agent run/history/approval controls; safe Models, Files and Calendar reads; document import/read/download; Knowledge ingestion; Communications reply draft/archive without sending; Calendar create/read.
- Research cancellation now reads and renders the persisted cancellation checkpoint after the existing cancellation request, rather than waiting for a later refresh.
- Communications Inbox Live reply and archive use existing draft and message-state contracts and give visible feedback. No send operation is invoked.
- Kept Demo fixture operations local and Live API adapters on existing routes. Existing CodeQ/Research SSE auth and cursor handling remain in place.
- Added/updated the smoke fixture, end-to-end smoke harness, UI tests and master gap findings. The production-preview regression script now starts and stops its own isolated preview when no origin is supplied.

## Exact verification on the final source revision

Commands were run from `D:\openagent\frontend-second` unless a repository-root command is shown.

| Command | Result |
|---|---|
| `npm run check` | PASS — TypeScript project build/check exited 0. |
| `npm test -- --reporter=dot` | PASS — 22 test files, **152 tests passed**. |
| `npm run test:e2e -- --workers=1 --retries=0` | PASS — **193 passed**, 0 failed, 6.2 minutes; one worker, isolated Vite server. No build ran concurrently. |
| `npm run build` | PASS — production output at `frontend-second/dist`; safe-build verified `../web` unchanged and `public/demo` excluded. Vite emitted the existing Monaco bundle-size advisory (3.71 MB minified); it did not fail the build. |
| `npm run verify:production-live` | PASS — fresh production Live browser cleared stale Demo preference, attempted 12 API calls against offline 503 stubs, requested 0 Demo fixtures, showed no Demo-only Integration Guide privacy/export copy, and had 0 page errors. This validates the frontend mode boundary, **not** Python integration. |
| `.venv\Scripts\python.exe -m unittest tests.test_agent_run_access tests.test_auth_policy tests.test_auth_enforcement tests.test_project_identity_contract tests.test_codebase_frameworks tests.test_codebase_agent -v` (repo root) | PASS — **61 tests**, 40.716 seconds. Includes the Batch 1 Agent history authorization and CodeQ project namespace regression suites. |
| `npm run test:real-python` | PASS as a smoke-harness process (exit 0); **129 API responses**, 0 Demo fixture requests, external providers disabled. Individual workflow results and three security blockers are below. |
| `node --check scripts/real-python-smoke.mjs` | PASS — syntax check exited 0. |
| `git diff --check -- frontend-second/src/screens/WorkspaceScreens.tsx frontend-second/src/screens/AdvancedScreen.tsx frontend-second/FRONTEND_BACKEND_MASTER_GAPS.md frontend-second/scripts/real-python-smoke.mjs` | PASS — no whitespace errors in the reviewed changed files. |

The full Playwright run produced expected Vite proxy `ECONNREFUSED` diagnostics for Live-mode tests because that suite intentionally has no Python service attached. The relevant tests assert honest unavailable/error states; this is not evidence of a successful Python integration. The separate real-Python smoke used an actual local server.

## Real React-to-Python smoke results

Passed against the local Python server and actual React UI:

- Chat: response stream and persisted messages; provider failure; cancellation.
- CodeQ: project and run lifecycle; authenticated event stream/cursor reconnect; cancel; controlled failure; resume.
- Agents: authorized history and reload, run-only token denied history, approval request/rejection, cross-owner history denied.
- Files: document import/read/download; media upload; basic read.
- Knowledge: text ingestion and multipart upload in the submitting owner session.
- Communications: reply draft persisted and message archived; no send endpoint called.
- Calendar: read and create persisted.
- Models: safe read-only data.
- Research: create, report, checkpoints, stream events and cancellation checkpoint/terminal cancellation in the owner session.

The run also proved these cross-user exposure failures using distinct authenticated synthetic users granted the relevant read scopes:

1. **Research — BLOCKED:** another user with `knowledge:read` listed the first user's Research session and read its report by ID (HTTP 200). Handler path: `openagent/routes/runtime.py`; current Research storage/lifecycle is not bound to authenticated owner. The UI/API must not be treated as safe for multi-user history until an owner/project policy is approved and implemented.
2. **Knowledge — BLOCKED:** another user with `knowledge:read` listed the first user's indexed item under the same project name (HTTP 200). Relevant source: `openagent/routes/knowledge_routes.py` and Knowledge storage. Project-name filtering does not prove principal ownership.
3. **Uploaded media — BLOCKED:** another user with `media:read` listed the first user's uploaded synthetic media under the same project name (HTTP 200). Relevant source: `openagent/routes/media_routes.py` and `openagent/media.py`; asset lookup lacks an authenticated owner predicate.

No Python authorization was weakened or changed for these three newly observed boundaries. The handoff's Research stop-and-flag instruction was followed; the same evidence was recorded for Knowledge and media because the real smoke exposed equivalent isolation problems. These are unresolved security blockers and require an approved owner-binding design before multi-user exposure.

## Source files touched for Batch 2A

Frontend and reports:

- `frontend-second/src/screens/WorkspaceScreens.tsx`
- `frontend-second/src/screens/AdvancedScreen.tsx`
- `frontend-second/src/screens/HomeChat.tsx`
- `frontend-second/src/screens/OperationsScreens.tsx`
- `frontend-second/scripts/real-python-smoke.mjs`
- `frontend-second/scripts/verify-production-live-context.mjs`
- `frontend-second/tests/backend-wiring.test.tsx`
- `frontend-second/tests/e2e/agents-history-live-security.spec.ts`
- `frontend-second/tests/workspaceBackend.test.tsx`
- `frontend-second/FRONTEND_BACKEND_MASTER_GAPS.md`
- `frontend-second/BATCH2A_IMPLEMENTATION_RESULTS.md`

Local smoke support and Batch 1 Agent history security source included for reproducibility:

- `tests/support/real_integration_server.py`
- `openagent/auth.py`, `openagent/multi_agent.py`, `openagent/routes/agents_routes.py`, `openagent/routes/__init__.py`, `openagent/routes/runtime.py`, `openagent/storage.py`, `openagent/migrations.py`
- `tests/test_agent_run_access.py`, `tests/test_auth_policy.py`

The working tree contains unrelated pre-existing edits outside these files. They were not reverted or included in the targeted source package.

## Status

Existing-contract wiring is implemented and its smoke evidence is recorded. The smoke is **not** a claim that every route or every Live workflow is complete. Batch 2A is not fully security-accepted: Research, Knowledge and uploaded-media cross-user isolation remain blocked. No new backend capability was added, and the approved canonical HTML was not modified. The Batch 1 `GET /api/agent-runs` authorization work remains intact.
