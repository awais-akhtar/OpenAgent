# Batch 2B implementation results — single-owner personal remote

Date: 2026-10-19

Environment: Windows 11; Python 3.14.3 in `.venv`; Node v24.14.0; Google Chrome 155.0.8059.80.

## Delivered

- Added explicit `single_user`/`personal_remote` startup posture. Personal remote is opt-in, always authenticated, loopback-bound, and requires an HTTPS-terminated proxy declaration plus allowed Host names. Unknown deployment modes and unsafe auth/network combinations fail startup.
- Enforced the one-owner invariant at startup and for private requests. First-owner setup requires a high-entropy bootstrap proof. Additional account creation is denied. Databases with multiple/ambiguous accounts are refused without changing data.
- Personal-remote requests authenticate the installation owner before handlers run. The private route set fails closed for unmapped routes and requires explicit scopes. Public remote paths are limited to health, login, refresh, and bootstrap registration.
- Applied the same installation-owner gate to Research, Knowledge, Media, SSE, and other private paths. Foreign-installation credentials fail authentication. Personal remote rejects raw server-path media ingestion while retaining byte uploads.
- Added Owner sign-in, optional TOTP code, current-session view, refresh, and sign-out under Settings → Advanced. Tokens remain tab-scoped. The real integration smoke exercises this React path.
- Adapted the supplied smoke fail-closed behavior: any `BLOCKED` or `FAIL` record makes the command exit nonzero.
- Preserved multiple CodeQ repository support, the Batch 1 agent-history restrictions, canonical HTML, old `frontend/`, Python-served `web/`, and the existing Live API/SSE adapters.

## Verification commands and results

All commands ran in `D:\openagent` or the listed frontend directory using the repository `.venv`, Node.js and installed Chrome.

| Command | Result |
|---|---|
| `npm run test:real-python` (in `frontend-second`) | Exit 0; 18/18 rows PASS; 128 HTTP API responses; 0 blockers; 0 Demo-fixture requests. Uses actual React + actual loopback Python HTTP server, disposable SQLite/runtime directories, synthetic owner and separate foreign-installation token, deterministic in-process provider; no external services. Controlled proxy headers simulate the HTTPS proxy boundary; this does not certify a deployed TLS/VPN. |
| `npm run check` (in `frontend-second`) | Exit 0, TypeScript project references clean. |
| `npm test -- --reporter=dot` (in `frontend-second`) | Exit 0; 152 tests in 22 files passed. |
| `npm run test:e2e -- --workers=1 --retries=0` (in `frontend-second`) | Exit 0; 193 desktop Playwright tests passed in 6.1 minutes. |
| `npm run build` (in `frontend-second`) | Exit 0; isolated `frontend-second/dist` build verified; no demo fixture assets bundled; `../web` hash unchanged. Vite reports the expected large Monaco worker/chunk advisory. |
| `npm run verify:production-live` (in `frontend-second`) | Exit 0; 12 safely stubbed 503 API requests, zero Demo fixture requests, zero page errors; this is a production-preview regression, not a Python integration test. |
| `.\.venv\Scripts\python.exe -m unittest tests.test_personal_remote_security tests.test_agent_run_access tests.test_auth_policy tests.test_auth_enforcement tests.test_auth_routes tests.test_auth tests.test_knowledge_routes tests.test_media_routes tests.test_research_lifecycle tests.test_project_identity_contract tests.test_source_scope_inventory -v` | Exit 0; 109 Python security, route, auth, domain-lifecycle, identity and scope-inventory tests passed. |
| `Get-FileHash frontend-second/reference/frontend-final-approved.html -Algorithm SHA256` | SHA-256 `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1` (unchanged). |

The first smoke attempt after adding terminal-cancellation verification exited nonzero because it inspected the Research worker before the controlled provider call completed. The test now waits for terminal `cancelled` and still requires that exact terminal status. The final run passed. No product cancellation requirement was weakened.

## Acceptance status

The Batch 2B application/security implementation and local verification pass. The batch is **not fully deployment-accepted**: a real private HTTPS proxy/VPN and access from another physical device were not exercised. This remains unverified by design; no public endpoint was exposed.

## Exact remaining boundaries

- The actual remote TLS reverse proxy/VPN and second physical device were not deployed or tested. The Python server rejects non-loopback binding, checks Host and forwarded protocol headers, and was exercised with controlled localhost proxy headers. Do not enable public exposure.
- Existing legacy `team`/`enterprise` multi-user modes are not secured by this solo architecture; cross-account Research/Knowledge/Media object filtering remains unsupported in those modes. Do not present these modes as approved personal remote.
- Personal remote refuses a multi-account/ambiguous database. No account deletion, owner reassignment, or destructive data migration was performed.
- Batch 2C core controls still need per-workflow verification; the smoke validates representative flows, not every control on all 23 screens.
- The Visual Milestone remains open. This turn did not rerun the canonical desktop screenshot comparison matrix.

## Changed implementation areas

- `openagent/config.py`
- `openagent/auth.py`
- `openagent/routes/runtime.py`
- `openagent/routes/auth_routes.py`
- `openagent/routes/media_routes.py`
- `frontend-second/src/screens/OperationsScreens.tsx`
- `frontend-second/vite.config.ts`
- `frontend-second/scripts/real-python-smoke.mjs`
- `tests/test_personal_remote_security.py`
- `tests/support/real_integration_server.py`
- `docs/deployment.md`
- `frontend-second/FRONTEND_BACKEND_MASTER_GAPS.md`
- `ACCEPTANCE_TRACKER.csv`

