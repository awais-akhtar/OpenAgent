# Batch 1 implementation and verification results — 2026-10-19

## Scope and disposition

Implemented the explicitly approved security change for `GET /api/agent-runs` and ran isolated React-to-local-Python integration checks. No new Python capability beyond the approved route protection was added. The canonical HTML was not edited. Live Agent history is enabled only for the project-scoped authenticated contract and renders a non-disclosing 404 as empty history.

The approved Agent-history security fix, full desktop Playwright suite, relevant Python regressions, and nine-workflow React-to-Python smoke suite passed. The smoke covers CodeQ cancellation/resume and Research stream consumption/cancellation. This does not establish production-provider behavior.

## Implemented security changes

- Declared `GET /api/agent-runs` in the agent route contract registry.
- Added `agents:read` without granting it implicitly to existing token/session defaults. `agents:run` does not authorize history reads.
- Under enforced authentication, the handler requires a principal and read scope, requires a non-empty explicit project, ignores caller-supplied owner and returns non-disclosing 404 for unknown, unowned, or another user's project.
- Added project namespace ownership binding and persisted the authenticated owner on new agent-run rows. A submitted project string cannot reassign an existing owner binding. Existing CodeQ project ownership is checked before an unbound namespace can be claimed.
- Added owner/project pagination and migrations 24–25. Legacy runs without proven owners remain inaccessible in enforced mode; no automatic attribution is performed.
- Preserved single-user local behavior when auth enforcement is disabled.
- Enabled the existing React Live history query only after security tests passed. Demo history remains isolated.

## Verification results

Commands ran from `D:\openagent` or `D:\openagent\frontend-second` on 2026-10-19. Python used `.venv\Scripts\python.exe`; Playwright used installed Chrome with its isolated temporary profile.

| Command | Result | Notes |
|---|---|---|
| `.\.venv\Scripts\python.exe -m py_compile openagent\auth.py openagent\migrations.py openagent\storage.py openagent\agents.py openagent\routes\agents_routes.py openagent\routes\runtime.py tests\test_agent_run_access.py tests\support\real_integration_server.py` | PASS | Modified Python modules and integration helper compile. |
| `.\.venv\Scripts\python.exe -m unittest tests.test_agent_run_access tests.test_auth_enforcement tests.test_auth_policy tests.test_migrations tests.test_route_contracts tests.test_chat_and_workflows tests.test_chat_research tests.test_research_lifecycle tests.test_server` | PASS, 74/74 | Rerun against final source, including the CodeQ namespace ownership guard and Agent-history security cases. One non-fatal Chroma deprecation warning appeared. |
| `npm run check` | PASS | TypeScript check. |
| `npm run test -- --reporter=dot` | PASS, 151/151 across 22 files | Final rerun after Research active-action stacking correction. |
| `npm run test:e2e -- --grep "Chat delayed stream uses only a compact pending indicator" --workers=1 --retries=0 --reporter=list` | PASS, 4/4 | 1440×900 and 1280×800, light/dark. Response remains held until `Working…` is visible. |
| `npm run test:e2e -- --grep "approved chooser region" --workers=1 --retries=0 --reporter=list --trace retain-on-failure` | PASS, 1/1 | Fixture-ready comparison across both desktop sizes and themes. |
| `npm run test:e2e -- --workers=1 --retries=0 --reporter=list` | PASS, 193/193 in 6.2 minutes | Fresh runner-owned Vite server; final Playwright process exit code 0. See `PLAYWRIGHT_BATCH1_FINAL.log`. |
| `npm run test:real-python` | PASS, all 9 workflows; process exit code 0 | Actual React over Vite proxy to temporary authenticated Python HTTP server; 74 API responses, zero Demo fixture requests, external providers disabled. Final stdout is recorded in `BATCH1_REAL_PYTHON_SMOKE_FINAL.log`. |
| `npm run build` | PASS | Final isolated output `frontend-second/dist`; `../web` unchanged and `public/demo` fixtures excluded. Existing Monaco bundle-size warning remains. |
| Start `npx vite preview --host 127.0.0.1 --port 5181 --strictPort`, then `npm run verify:production-live` | PASS | Stale localStorage Demo preference was cleared; Live attempted 12 API calls intercepted as offline 503s, zero `/demo/` requests and zero browser errors. Not a Python integration test. |

## Playwright blocker investigation

1. **Chat timing race:** the former test used a fixed 1.5-second route delay, which still depended on browser scheduling. It now holds the intercepted SSE response behind a promise, waits for the actual delayed `Working…` status, verifies the compact styling and absence of fabricated activity, then releases the response. All four focused desktop/theme cases and all Chat cases in the full suite passed. Product spinner timing was not changed.
2. **Chooser fixture readiness:** the React dialog became visible before its fixture query finished. The early capture was 245 px high, while the fully populated canonical chooser was 439.56 px. The test now waits for named fixture rows before capture. The same-state comparison passed for both sizes and themes. Paired captures are saved under `evidence/project-picker-20261018/`.
3. **Windows server lifecycle:** debug output showed Chromium exiting successfully while a Playwright-managed npm/Vite server wrapper remained alive and withheld the final runner summary. `scripts/run-e2e.mjs` now starts Vite directly, passes its origin to Playwright, records test output, and shuts down the owned process (using a Windows `taskkill /T /F` fallback when needed). The complete suite returned exit code 0.
4. **Research cancellation hit target:** hit-testing identified the absolute `New research` summary as the element receiving clicks at the Cancel button's center. The Research title/action row now has a scoped stacking layer that keeps its child controls clickable while allowing uncovered areas to pass pointer events through to the positioned creation control. The real smoke asserts the center hit target is the Cancel button and confirms the authenticated cancellation request and persisted cancelled state/checkpoint. The same layering leaves the New research control itself clickable.
5. **Calendar E2E ambiguity:** created events are intentionally represented in both the calendar grid and Upcoming list. The broad accessible-name locator selected both, causing strict-mode failures; the test now targets the Upcoming row specifically for create/detail/reload and the grid event for the Live detail case. Focused Calendar verification passed 5/5, and the final complete suite passed.

A separate Chrome `--version` probe attempted to use the locked default user profile and printed access-denied diagnostics. Playwright launched installed Chrome successfully with a unique temporary profile. Its Google Update registry warning is non-fatal.

## Real React-to-Python smoke matrix

`npm run test:real-python` starts `tests/support/real_integration_server.py` with temporary SQLite/runtime storage, enforced auth, two synthetic users, least-privilege tokens, a disposable repository and deterministic in-process provider. React runs in Live mode through Vite proxy.

| Workflow | Result and evidence |
|---|---|
| Chat | PASS — React POST to `/api/chat/stream`; Python persisted user and assistant messages. |
| Chat provider failure | PASS — Python SSE error displayed; no synthetic success. |
| Chat cancellation | PASS — React aborted authenticated SSE; fixture provider was local. |
| CodeQ | PASS — React project/run, authorized event cursor/SSE request, React cancellation and persisted cancelled status; controlled local provider failure, then React Resume returned 202 for the same run ID. Reconnection after browser/network loss was not tested. |
| Research | PASS — session created with `allow_external=false`, persisted session/report/checkpoint schema, authenticated event stream and queued/planning event rendered in Activity; React cancellation returned successfully and Python persisted cancelled status/checkpoint. |
| Agents | PASS — authorized owner-bound history rendered; unrelated token returned 403; another user's project returned concealed 404. |
| Models | PASS — basic `/api/models` read using a limited token; no download started. |
| Files | PASS — basic `/api/documents/library` read using a limited token. |
| Calendar | PASS — basic `/api/calendar/events` read using a limited token. |

This is real local HTTP integration against Python, not a mocked API response. It does not establish production provider behavior or every workspace workflow. Research cancellation now passes against the real UI and Python server after correcting the obscured button hit target.

## Route inventory snapshot

Current Python registry: **655 declared method/path entries across 49 route groups**, from `openagent.routes.route_contracts()`. Registry declaration is not proof that each route was called or fully verified.

## Remaining limitations

1. The real smoke did not exercise CodeQ stream reconnection after browser/network loss or external model providers. Downloads and external side effects were intentionally excluded.
2. Legacy agent-run records remain ownerless and inaccessible under enforced authentication. No attribution migration was attempted.
3. Existing frontend/backend gaps remain documented in `FRONTEND_BACKEND_MASTER_GAPS.md`; no speculative backend capability was implemented.

## Reproduction

From repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_run_access tests.test_auth_enforcement tests.test_auth_policy tests.test_migrations tests.test_route_contracts tests.test_chat_and_workflows tests.test_chat_research tests.test_research_lifecycle tests.test_server
```

From `frontend-second/`:

```powershell
npm run check
npm run test -- --reporter=dot
npm run test:e2e -- --workers=1 --retries=0 --reporter=list
npm run test:real-python
npm run build
```

Do not implement additional Python capabilities without the owner's next approval.
