# Frontend API contracts and safety semantics

## Scope and evidence

This document inventories the client behavior and endpoint families found in the current frontend source, cross-checked against the screen dossiers and supplied API inventory. It is not a complete listing of every Python route, and source references do not prove runtime handler behavior. The route inventory reports 672 method/path declarations across 50 modules; declarations alone are not a tested contract. Exact handlers remain authoritative.

## Shared HTTP and stream clients

`frontend/src/lib/react/api.ts` exposes generic `get<T>`, `post<T>`, `put<T>`, `patch<T>`, and `delete<T>` helpers. It enforces same-origin paths beginning `/api/`, rejects `//` and newline paths, sends `credentials: 'same-origin'`, and attaches the `oa-auth-token` browser session token as a Bearer token when present. Object bodies are JSON encoded; `FormData` and `Blob` are passed through. `ApiError` retains HTTP status and request ID. GET retries once for network failure or 429/502/503/504 and uses a 45-second timeout with caller abort support.

`frontend/src/lib/react/stream.ts` implements fetch-based SSE readers for GET/POST. It parses split UTF-8 frames, supports event/data/id fields and final partial frames, applies the same-origin/auth pattern, and aborts/releases the reader on cancellation. The GET stream error path is less diagnostic than the generic API error path.

The client is **generic rather than domain-typed**: many calls use `api.get<unknown>()`, then permissive `record()`/`records()` helpers. It does not provide comprehensive runtime response-schema validation. Static type annotations and tests for request configuration are not equivalent to validating Python responses.

## Screen endpoint map

| Screen | Frontend API contract families observed |
|---|---|
| Dashboard / Activity | GET `/api/health`, `/api/system/metrics`, `/api/modules/health`, `/api/inference/status`, `/api/local-ai/status`, `/api/audit`, `/api/agents/approvals`, `/api/worker/jobs` |
| Chat and shell history | GET `/api/conversations/organized`, `/api/models`, `/api/chat/search`, `/api/conversation/messages`; POST `/api/chat/stream`, `/api/conversations/bulk`, `/api/chat/branch`, `/api/chat/import`, `/api/chat/artifact`; GET `/api/chat/replay`, `/api/chat/export`; POST `/api/knowledge/upload` |
| CodeQ | GET `/api/codebase/projects`, project tree/file, runs/detail/events; POST folder-picker/project/scan/preflight/run/cancel/resume/snapshot restore/approval resolve/file write; event stream is fetched with cursor/timeout parameters |
| Voice | GET `/api/voice/status`, `/personas`, `/sessions`, `/turns`; POST sessions/STT/personas/transcript/TTS/translate/close |
| Research | GET `/api/research/sessions` and session event stream; POST session creation and `/api/chat/research-record` |
| Agents | GET `/api/agents`, `/marketplace`, `/permissions`, `/approvals`; POST `/run-checked`, approval request/resolve, `/delegate`, permission grant |
| Knowledge | GET knowledge status/documents/tags/graph/timeline and RAG status; POST hybrid search/synthesis/upload/ingest |
| Workspace | GET `/api/workspace/items`, `/api/documents/library`, `/api/document/{id}`, versions; POST create/archive/restore/update and PUT document |
| Automations | GET `/api/workflows`, `/api/workflows/builder`, `/api/workflow-runs`; POST/PUT workflow and POST `/api/workflows/run` |
| Assistant | GET `/api/assistant/daily-brief`, `/tasks`, `/goals`, `/coach`, `/crm`; POST task create/complete, goal, meeting brief, CRM record |
| Communications | GET `/api/communication/connectors`, `/inbox`, `/contacts`, `/calendar`, `/approvals`, `/scheduled`; POST draft, request/resolve approval, send |
| Models | GET `/api/models`, `/api/local-ai/models`, `/hardware`, `/model-policy`, adapters/marketplace/setup/jobs; POST discover/setup/download/register/delete/cancel and probes |
| Security | GET `/api/security/status`, `/egress`, `/retention`, `/secrets`, `/shares`, `/privacy-reports`, `/api/audit`, audit-chain verify, sandbox profiles; POST egress/secret/sandbox/report/audit-chain/retention/share mutations |
| Developer | GET developer status/schema/API keys/webhooks/tools/plugins/MCP/extensions; POST registration/configuration operations |
| Settings | GET runtime/model-policy/security/auth/backups; POST runtime save, backup create, restore dry-run, restore |
| Adaptive Engine | GET adaptive-engine status/hardware/plans/benchmarks/roles/direct status; POST inspect/benchmark/compile |
| Science | GET science status/tools/memory/material datasets/catalog; POST memory, dataset analysis, bounded run |
| Calendar | GET calendar status/calendars/events/providers/conflicts/reminders; POST event/calendar/free-slots/conflict resolve/sync |
| Personal Context / Twin | GET twin status/profile/patterns/suggestions/goals; POST goal/predict/interaction/consolidate |
| Vault | GET vault status/entries; POST store/retrieve/rotate-key/export |
| Scraper | GET scraper status/jobs; POST schedule/scrape/run-due |
| Fine-tuning | GET finetune status/templates/datasets/jobs/adapters; POST dataset/example/job/adapter registration/activation |

See [FRONTEND_FEATURE_PARITY.md](FRONTEND_FEATURE_PARITY.md) for the screen-by-screen implementation assessment and [22_ENDPOINT_TO_SURFACE_MATRIX.md](../CODEX_AIONUI_TRANSFORMATION/22_ENDPOINT_TO_SURFACE_MATRIX.md) for the supplied wiring targets.

## Privacy and safety behavior represented in source

- Project scope is sent on many reads and mutations from the active `project` query or saved `oa-project` value. The Workspace route subtree is keyed by active project in the shell; Workspace also clears document selection, content, versions, and library rows on a scope change and guards late document reads. A deterministic component regression test covers the page transition and the encoded project-scoped library requests. These client behaviors do not establish backend authorization: the Python service remains the authorization and data-isolation authority, and no frontend-only selection is a security boundary.
- Chat sends default to local-only/external access disabled unless the user explicitly opts in. The chat page passes the scope into `/api/chat/stream`; source tests cover payload construction but not the live backend policy.
- Communications sends are separated into draft, approval request, approval resolution, then send. The send request includes `allow_external:false`; server enforcement still governs delivery.
- Workflow run supports dry-run and a separate confirmed live action; configured payload sets `allow_network:false`. The JSON steps field remains a usability/parity concern.
- Scraper requests default external access off, request robots compliance, and leave egress/SSRF enforcement to the backend.
- CodeQ write includes the file's expected SHA-256 and requires an explicit confirmation; operation approvals, runs, and snapshots call dedicated API routes.
- Vault secrets are submitted to server endpoints, reveal requires a passphrase and source schedules concealment after 15 seconds; source review does not prove memory clearing, transport behavior or project isolation.
- Security/settings and privileged actions remain server-backed. Security and Settings summaries use allowlisted, human-readable fields and omit sensitive nested policy details and identifiers; some specialist surfaces may still show API-shaped data. Avoid treating a UI label, payload flag, or client confirmation as equivalent to backend security enforcement.

## Verification boundary

The source tree includes API-client, SSE, chat, specialist, resource, workspace and project-scope tests. The latest targeted verification for this audit ran the TypeScript check and the API/SSE/chat/agent-workflow/workspace suites; the new Workspace project-change case uses mocked responses and makes no live mutations. It proves request scope and UI clearing, not server-side isolation. This audit did not validate response schemas comprehensively, simulate protected-route auth/egress denials against Python, exercise every protected route, or verify real stream cancel/reconnect against the backend. The previously listed PhD Application Studio is retired, leaving 22 active view destinations. See [FRONTEND_QA_RESULTS.md](FRONTEND_QA_RESULTS.md) for the broader recorded checks and limitations.
