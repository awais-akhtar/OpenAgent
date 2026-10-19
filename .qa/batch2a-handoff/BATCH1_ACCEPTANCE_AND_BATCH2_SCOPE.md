# OpenAgent + CodeQ — Batch 1 closure and Batch 2 implementation handoff

**Review date:** 2026-10-19  
**Review basis:** `OpenAgent_Batch1_Updated(2).zip` and the original unedited final logs.  
**Decision:** **Batch 1 accepted for its approved, isolated security/integration scope**. **Not** a production sign-off or Milestone A completion.

## Evidence accepted

- ZIP: 181 entries; integrity `testzip()` passes; 8 relevant Python modules independently syntax-compiled; canonical reference hash independently rechecked: `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`.
- Final full Playwright log explicitly ends `193 passed (6.2m)` and `Playwright process exited with code 0`.
- Final logs/reports document Python regressions 74/74, frontend units 151/151, TypeScript/build success and real React-to-temporary-Python smoke 9/9 with 74 API responses. We inspected the smoke script and log; these tests were executed by Codex, not re-run independently in this environment.
- `GET /api/agent-runs` now maps to `agents:read`, checks explicit project and recorded ownership, returns 404 for unowned/cross-owner projects and isolates legacy ownerless records. The relevant Python code, tests, and migrations were inspected. Existing sessions do not silently inherit `agents:read`.
- CodeQ Resume and Research cancellation were exercised by real React controls against the isolated Python server; browser/network-loss reconnect and production providers were not.

## High-priority exact-source gap discovered during review

**Agent live run → history inconsistency:** `frontend-second/src/screens/OperationsScreens.tsx` posts the main Live Agent run to `POST /api/agents/run-checked` and invalidates `['agent-runs']`. `openagent/routes/agents_routes.py` handles `/run-checked` via `deps.multi_agent.run_agent(...)`; that visible handler does **not** call `store.save_agent_run(...)`. The smoke's history fixture was instead created using `POST /api/agents/run` (`frontend-second/scripts/real-python-smoke.mjs`), which persists owner-bound rows through `openagent/agents.py`. It is **not proved** from the supplied partial backend source that `/run-checked` never persists elsewhere, but the primary React workflow is **not validated** end-to-end against its actual submission route. Add the exact UI→`/run-checked`→GET authorized history test; if persistence is absent, fix using the existing service/contract after security review. Do not alter approved `agents:read` ownership rules or claim the history UI works for normal logins without the new scope.

**Outdated Home copy:** `frontend-second/src/screens/HomeChat.tsx` still tells users 'Agent run history is unavailable until its backend route has verified project authorization.' That was true before Batch 1, but is now stale. Home's Live history query remains disabled. Replace with accurate product copy (e.g. 'Agent run history is available in Agents for an authorized project with agents:read; Home does not display it yet') or connect the authorized Home query after a real security/UX test. Do not inject fixture data into Live.

**Access UX:** `openagent/auth.py::_user_scopes()` intentionally excludes `agents:read` from default sessions; `AgentsScreen` automatically requests history in Live. Verify how an eligible user obtains a least-privilege token/read grant, and test that a 403 is displayed clearly. Do not silently broaden session scopes or existing tokens.

**Cross-service project identity:** `claim_agent_run_project` protects existing CodeQ project-owner records and first-claims other agent namespaces. It does not itself establish membership in all other workspace services. Test namespace collisions and cross-service ownership before treating the project ID as a universal authorization contract.

## BATCH 2A — implement a meaningful existing-contracts functionality batch, not another audit

Codex should use its *complete current repository* for implementation. Keep a single outcome table with `DONE`, `BLOCKED`, `FAILED`, and `NOT TESTED`, plus reproducible tests. Make real code changes, not merely gap inventories.

1. **Agents / Home (P1):** repair and verify React's actual `/api/agents/run-checked` flow, approval/denial, authenticated owner-bound run history, persistence after reload and wrong-owner 404. Confirm the legitimate `agents:read` grant UX; correct the outdated Home message. Test separately with tokens that have `agents:run` only, `agents:read` only, both and neither. Do not widen history access.
2. **CodeQ (P1):** test run, fail, cancel, resume, streaming cursor and *browser/network interruption and resubscription*, using a disposable Git repo and local provider. Fix frontend adapter/state bugs using declared contracts only. Verify no duplicate/old SSE events and truthful terminal status.
3. **Research (P1):** verify create, progress/event cursor, report/checkpoints, cancel/retry and reload persistence with a fake local provider. Run two-user/project negative tests; stop and flag approval if current backend project ownership is inadequate rather than assuming the React view secures it.
4. **Files and Knowledge (P1):** test actual existing upload/import/read/download, size/type rejection, data persistence, owner/project isolation, errors and confirmation. Correct frontend wiring where Python contracts already exist. Any missing delete/archive endpoint must be recorded separately for explicit backend approval.
5. **Communications and Calendar (P1):** wire available Live inbox read/reply/draft/archive and calendar create/read contracts, with local fake transport and no real messages; test permission/validation/persistence. Do not invent delete/edit contracts absent from Python or send externally.
6. **Models and other screens (P2):** exercise safe read/status and failure/retry with local provider stubs, verify honest absent capabilities; no downloads, GPU operations or training in acceptance tests.
7. **Visual/interaction completion:** fix demonstrated desktop visual defects in equivalent active states and genuinely broken controls; preserve contextual Living UI and clean Chat. Do not restore idle sample traces to increase pixel scores. Keep immutable canonical HTML, 1440x900/1280x800 light/dark.

### Batch 2A acceptance criteria

- Complete source files/patches, not only Markdown. A named E2E test must click the real React control and observe real localhost Python response and persisted state (where supported).
- Negative auth cases for every newly connected sensitive workflow; no tokens/secrets in logs. No changes to a new Python route/service contract without explicit owner approval.
- Full `npm run check`, unit, Playwright and isolated build rerun on final revision, plus relevant Python regression and expanded `npm run test:real-python` output with exact assertions and exit statuses.
- Any unresolved capability must have a concrete blocker, missing method/path or schema proof and separately estimated change, rather than being described as 'done'.
- Return **one** updated source ZIP, changed-file list, test summary and only the meaningful remaining blockers. Proceed to next batch once verified; do not start another inventory-only round.

## Scope controls

Batch 1 was authorized for the Agent-run history security fix and synthetic localhost testing. **Batch 2A is limited to integrating and repairing existing contracts**; additional Python backend capabilities or authorization-policy expansions need a separate approval. No production cutover, legacy frontend or original `../web` changes, original canonical HTML edits, mobile redesign or uncontrolled side effects.
