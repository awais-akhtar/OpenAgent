# Backend integration and authorization audit — 2026-10-18

## Scope and proof categories

This is a read-only review of the current `frontend-second` source and Python route/authorization/storage source. It does not implement backend changes. The generated `BACKEND_ROUTE_INTEGRATION_MATRIX.csv` has one row per currently declared method/path (654 total) and includes: group/audience, declaring route module, `required_scope_for_path()` result, frontend-second and original-frontend source line references, test-file source references, and a conservative integration category.

The scan categories are deliberately not success labels:

- **Request callsite candidate**: path-like source plus a nearby `apiGet`/`apiPost`/other shared client or `fetch` call. This is a static review lead, not proof of an executed request.
- **Static reference only**: route pattern appears in production source, without a nearby call signature under the scan rule. Could be metadata/configuration or an indirect call.
- **No frontend literal**: no matching production-source path pattern. This is only a review candidate; internal/runtime/admin routes need not have UI.
- **Mock-tested**: test exercises the React request path with controlled responses. It does not prove Python handler compatibility.
- **Python handler/HTTP tested**: a backend test invokes a Python handler or local server. It does not prove React-to-Python wiring.
- **React-to-Python tested**: requires the actual React app to call a running local Python server and verify the returned data/action. No full feature workflow met this bar in this audit.

The static scan found 39 request-callsite candidates, 218 static references, and 397 patterns with no frontend-second literal. Those totals are heuristic discovery counts and must not be read as route coverage, successful wiring, or UI requirements. Current API adapters are primarily tested using mocks; production preview checks use offline 503 stubs. No broad React-to-Python integration run was performed.

## Screen-level map and verification status

| Screen | React source | Backend groups / important paths | Current evidence classification |
|---|---|---|---|
| Home / Dashboard | `src/screens/HomeChat.tsx`, `src/App.tsx` | health, codebase projects, conversations, agents, events, calendar, tasks | Request adapters and UI handling have mocked/source tests. Demo shell records use local fixtures. `/api/agent-runs` is Demo fixture-only; Live query is disabled. No full Python dashboard integration. |
| Chat | `src/screens/HomeChat.tsx`, `src/api/client.ts` | `POST /api/chat/stream` (SSE), conversations/messages/search/export/replay | Streaming lifecycle tests use intercepted SSE. Auth header/request guard/client behavior is unit-tested. No live Python model/provider run. |
| Agents | `src/screens/AgentsDemoWorkspace.tsx`, `src/screens/OperationsScreens.tsx` | `/api/agents`, `/api/agents/run`, approvals/delegation; unsafe undeclared `/api/agent-runs` history | API request/UI adapters use controlled mocks. Demo trace is local and explicit. Live history is deliberately withheld. Direct Python security reproduction confirms the history route is unsafe (below). |
| Research | `src/screens/WorkspaceScreens.tsx` | research sessions, report/checkpoint/source actions; `/api/research/sessions/{id}/stream` SSE | Source and mocked adapter tests cover session lifecycle and event parsing. No real Python research run in this audit. |
| Files / Workspace | `src/screens/WorkspaceScreens.tsx` | documents, media, workspace items, knowledge; archive/version/import routes | Mock-backed API contract tests. Upload, external filesystems and destructive Live persistence were not executed against Python. |
| Automations | `src/screens/OperationsScreens.tsx` | workflow CRUD/run, scheduler tasks and schedules | Dry-run/confirmation and request mapping are mocked; Demo simulations are local. No real scheduled job execution. |
| Calendar | `src/screens/WorkspaceScreens.tsx` | calendar events, free slots, reminders, iCalendar import, providers | Event create behavior in Demo and mocked Live detail are tested. Python persistence/provider synchronization is not React-to-Python verified. |
| Models | `src/screens/OperationsScreens.tsx` | local AI catalog/status/jobs/setup/download/benchmark routes | Mocked adapter behavior and explicit Demo workflows. No actual model download or runtime installation was performed. |
| Settings | `src/screens/OperationsScreens.tsx`, `src/App.tsx` | auth scopes, runtime, backup, model/privacy/security settings | Selected mock request flows; no production identity/permission matrix. |
| CodeQ | `src/screens/WorkspaceScreens.tsx` | codebase project/tree/file/run/approval/cancel/resume/events/snapshot; terminal | Monaco retained. API/event handling source and mocks exist. No real Python CodeQ job/terminal session was run. Preview lifecycle is a documented missing/limited capability. |
| Knowledge | `src/screens/WorkspaceScreens.tsx` | knowledge search/ingest/index/graph/tags/timeline and semantic collections | Demo collection workflows are local; API paths have mocks. ID/schema mapping to Files/document actions remains unverified. |
| Voice | `src/screens/AdvancedScreen.tsx` | media and voice session/persona/turn/STT/TTS/upload | Mocked service responses; browser microphone capture and real playback asset delivery were not tested. |
| Personal Assistant | `src/screens/AdvancedScreen.tsx` | assistant tasks/goals/daily brief and scheduler | Mocked Live adapter tests and local Demo CRUD/event tests. No real user/project authorization matrix. |
| Communications | `src/screens/AdvancedScreen.tsx` | inbox, drafts, contacts, calendar, OAuth/sync/approval/scheduled send | Demo inbox actions are local and E2E-tested. Live queries/mutations are mocked; no real message was sent. |
| Security | `src/screens/AdvancedScreen.tsx` | egress, shares, sessions, audit and chain verification | Audit list is withheld in Live. Existing `/api/audit` scope is `security:admin`; Python tests verify route behavior in a temporary DB, but deployment config/project membership are not verified. |
| Developer | `src/screens/AdvancedScreen.tsx` | API keys/scopes, webhooks, MCP, tools and extensions | Selected request handling mocked; credentials were not created against a live backend. |
| Adaptive Engine | `src/screens/AdvancedScreen.tsx` | status/hardware/jobs/plans/compute | Four read views verified with mocks; no real accelerator or job execution. |
| Scientific AI | `src/screens/AdvancedScreen.tsx` | experiments/tools/memory/materials/catalog/proposals/execute | Mocked proposals/catalog; Demo runs explicitly simulated. Executed-run/report history remains a documented gap. |
| Personal Context | `src/screens/AdvancedScreen.tsx` | twin profile/patterns/suggestions/sources/goals/preferences | Mocked response display and confirmation behavior; fact-level provenance is absent. |
| Vault | `src/screens/AdvancedScreen.tsx` | vault status/entries/access log/reveal/rotate/export | Demo records are synthetic and secret-free; selected Live API paths use mocks. No live secret was stored or revealed. |
| Web Scraper | `src/screens/AdvancedScreen.tsx` | status/jobs/history/extraction/schedule/run-due | Demo local job previews; mocked Live responses. No external network scrape executed. |
| Fine-tuning | `src/screens/AdvancedScreen.tsx` | datasets/examples/jobs/adapters/activation/cancel | Demo job/checkpoint previews explicitly simulated. Mocked supported endpoints. No actual training/checkpoint created. |
| All Tools | `src/screens/OperationsScreens.tsx` | Local navigation/search | Frontend-only; no API expected. |

The complete 654-row method/path source inventory remains linked in `FRONTEND_BACKEND_MASTER_GAPS.md` and is reproduced as CSV in the source handoff. The screen grouping above is contextual guidance; it does not imply each route in a group is used by the named screen.

## Confirmed `/api/agent-runs` authorization and scope defect

### Source trace

1. `openagent/routes/runtime.py:1654` dispatches `GET /api/agent-runs`.
2. The handler reads `project = first(params, "project")`, which can be absent, then calls `store.list_agent_runs(limit=limit, project=project)` at line 1658.
3. `openagent/auth.py:909` `required_scope_for_path()` has no exact or prefix mapping for `/api/agent-runs`; direct evaluation returns `None`.
4. `openagent/routes/runtime.py:4085` `_enforce_auth_gate()` is active only when `OPENAGENT_AUTH_ENFORCED=1`. At line 4096 it returns `True` for `scope is None`, so the unmapped route remains accessible even with enforcement enabled.
5. `openagent/storage.py:360` applies `WHERE project = ?` only when a project is supplied. When omitted, it runs `SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?`.
6. `openagent/routes/*` contract registry currently has 654 declarations / 49 groups and contains no `GET /api/agent-runs` pattern.
7. React Live history is suppressed in `src/screens/OperationsScreens.tsx` and `src/screens/HomeChat.tsx`; `tests/e2e/agents-history-live-security.spec.ts` asserts the UI does not issue the request. The test uses intercepted/mock browser requests and is not backend auth proof.

### Reproduction executed for this handoff

Using the real Python `build_http_server()` on an ephemeral localhost port, `OPENAGENT_AUTH_ENFORCED=1`, an isolated `TemporaryDirectory` SQLite store, and two synthetic agent-run records (`alpha-test`, `beta-test`):

| Request | Observed result |
|---|---|
| `GET /api/agent-runs` without Authorization | HTTP 200; both `alpha-test` and `beta-test` are present |
| `GET /api/agent-runs` with an `agents:run` bearer token | HTTP 200; both project records are present |
| `GET /api/agent-runs?project=alpha-test` | HTTP 200; only `alpha-test` is present |
| Registry declaration | Absent |
| `required_scope_for_path(..., "GET")` | `None` |

Full JSON output is included in `TEST_RUN_LOGS/python-agent-runs-security-repro.json`. This proves cross-project retrieval is possible through the unfiltered endpoint in the tested local configuration. It does not prove remote exposure or production deployment settings. The route must remain suppressed in Live. No Python source was edited.

## Security boundaries and approval requirements

- Preserve the shared API client’s auth headers and Demo request blocking. Mock tests of headers do not validate a real identity provider or deployed token policy.
- Never enable `/api/agent-runs` history in React until a declared scope and project/owner policy are implemented and tested. Any behavior fix is an owner-approved backend change.
- Security Audit list uses the existing `security:admin` mapping, but cross-project list semantics still need a product/security decision; UI remains withheld.
- Do not expose routes classified internal/runtime, admin-only, security-sensitive, destructive, or approval-gated based on route count.
- No routes were added, no auth policy was weakened, no data was written outside temporary audit stores, and no real credentials or user content were used.
