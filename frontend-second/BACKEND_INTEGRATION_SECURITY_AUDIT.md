# Backend integration and authorization audit — 1–2 November 2026

This is supporting evidence for the [authoritative frontend/backend gap map](FRONTEND_BACKEND_MASTER_GAPS.md). It describes the **current working tree**, not the superseded 18 October snapshot. Adaptive Engine implementation gaps are outside this audit.

The master map's **L01–L13 closure board** defines the final local-release checks. In particular, L06 and L11 require real owner/project denial tests for each release-critical React action; the 779/783 scope count below cannot substitute for those tests.

## Proof levels

| Level | What it proves | What it does not prove |
| --- | --- | --- |
| Declared | A method/path is in `openagent.routes.route_contracts()` | The handler works or is safe |
| Scope-mapped | `required_scope_for_path()` returns a scope | Ownership, approval and project checks |
| Source-wired | A production React caller exists | A user action reached Python successfully |
| Python-tested | A handler or local HTTP request passed with synthetic data | The React control works |
| React-to-Python tested | The actual React app called an isolated local Python server and asserted behavior | External providers, hardware, every permission and every screen |

The generated [backend inventory](BACKEND_CAPABILITY_GAPS.md) and [783-row frontend route-reference matrix](FRONTEND_CAPABILITY_GAPS.md) are review leads, not success counts. The master file records workflow-level evidence and remaining gaps.

## Current registry and gate

`route_contracts()` declares **783 unique method/path entries in 54 groups**. **779** return a required scope. The four unmapped entries are `GET /api/health` and the auth register/login/refresh bootstrap routes. Scope coverage must still be paired with object and project authorization tests.

The historical unauthenticated `GET /api/agent-runs` finding is **closed for the tested contract**. The route is declared, maps to `agents:read`, requires a nonempty `project`, and checks the authenticated owner's claim before returning owner-filtered rows. Missing project returns 400; another owner's project returns 404. Existing ownerless rows are not silently attributed under enforced authentication. Agents uses this route with an explicit project; Home does not request unrestricted history. Current focused regression includes `tests.test_agent_run_access`, `tests.test_auth_policy` and `tests.test_auth_enforcement` in the [20-test log](.qa/full-gap-audit-20261101/python-targeted.log). This is not a full authorization review of every route.

G25, Model serving and Voice conversations are registered and scope-mapped, with mounted React controls and API adapters. The G25 graph owner/edge and RAG owner cases have focused Python HTTP evidence in the master map. Broader React-to-Python selected-state, permission, persistence and restart checks remain open. CodeQ's read-only question route has focused service/route tests, but its React click-through remains unverified.

## Boundaries still open

- A literal endpoint in React is not an executed request. The bounded scan found 413 second-frontend and 243 original-frontend route-pattern candidates; it does not establish 413 working actions.
- Security Audit uses `security:admin`. Ordinary `security:read` access is not implied.
- Research owner-bound sessions are accessible with the proper scope. Legacy ownerless sessions remain inaccessible in enforced authentication until a safe attribution policy is approved.
- CodeQ repository IDs do not establish a verified shared project identity across Chat, Files, Research, Assistant and other services.
- Demo fixtures, intercepted browser requests and offline 503 responses do not prove Live Python integration. A current-tree isolated React-to-Python smoke **did** pass 49 selected workflows with authentication enforced; it is not an exhaustive route/control/permission matrix.
- External sending, model launch/download, arbitrary training quality, microphone hardware, provider credentials and full desktop visual acceptance are not certified by registry or unit tests.

## Checks on this working tree

- `.venv\Scripts\python.exe frontend-second/scripts/refresh_master_audit.py`: 783 declarations, 783 unique routes, 54 groups, 243/413 bounded source candidates.
- `.venv\Scripts\python.exe -m unittest tests.test_auth_policy tests.test_auth_enforcement tests.test_agent_run_access tests.test_g25_model_serve_voice_ownership tests.test_research_owner_access tests.test_route_contracts -q`: **20 passed**, exit 0. [Log](.qa/full-gap-audit-20261101/python-targeted.log).
- `cd frontend-second; npm run check`: exit 0; `npm test -- --reporter=dot --maxWorkers=1`: **205 passed / 26 files** after the final source edits. [TypeScript](.qa/full-gap-audit-20261101/typescript-final.log), [Vitest](.qa/full-gap-audit-20261101/units-final.log).
- Full desktop Playwright: **269 passed, exit 0** on the source revision before the later CodeQ backend edits. [Log](.qa/full-gap-audit-20261101/playwright-full-final.log). Many cases use intercepted responses; selected cases start a disposable Python server.
- Complete isolated React-to-Python smoke: **49 workflows / 505 API responses / zero Demo fixture requests, exit 0** with authentication enforced and external providers disabled, also before the later CodeQ edits. [Log](.qa/full-gap-audit-20261101/real-python-smoke-final2.log). It verifies selected G25, Voice conversation and Model serving *plan* actions, not speech processing or a launched model.
- After the CodeQ tool/teardown changes, three focused regressions passed. The subsequent full Python discovery ran 2,827 tests in 604.224 seconds: 8 failures, 14 errors, 81 skipped, exit 1. The exact failure cases, causes and next fixes are grouped in L01 of the master map. The separately rerun CodeQ route/contract suite passed 8/8.
- Isolated production build: **PASS**, `frontend-second/dist` only, unchanged `../web`, no Demo fixtures. [Log](.qa/full-gap-audit-20261101/build.log).
- Fresh full desktop pixel/state comparison was **not executed here**. The current Python discovery failed as recorded above; historical frontend results remain dated in the master map.
