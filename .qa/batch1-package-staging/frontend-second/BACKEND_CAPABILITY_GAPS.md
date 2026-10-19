# Backend capability gaps and API surface audit

> Supporting generated route evidence only. Use [FRONTEND_BACKEND_MASTER_GAPS.md](FRONTEND_BACKEND_MASTER_GAPS.md) as the sole current gap/status source.

Updated: 2026-10-18  
Source of truth: `openagent.routes.route_contracts()` at this repository revision.

## Route inventory

The registered contract catalog currently declares **654 distinct method/path entries across 49 route groups**. The user estimate of 646 is close, but the current registry reports 654. These are route patterns (including parameterized paths), not a count of concrete URLs or proof that each handler succeeds.

The source-level scan found a direct route-pattern string in **249/654** primary frontend endpoints and **257/654** second-frontend endpoints. Those counts are a traceability aid, not functional coverage. See [FRONTEND_CAPABILITY_GAPS.md](FRONTEND_CAPABILITY_GAPS.md) for the full 654-row matrix and the static-scan limits.

| Route group | Audience | Declared | Primary refs | Second refs |
| --- | --- | --- | --- | --- |
| health | Internal / operator | 5 | 3 | 1 |
| core | Internal / operator | 9 | 0 | 2 |
| integrations | Advanced / admin | 13 | 0 | 0 |
| chat | User-facing | 12 | 11 | 10 |
| agents | User-facing + advanced permissions | 11 | 10 | 9 |
| workflows | User-facing | 13 | 7 | 7 |
| knowledge | User-facing | 14 | 9 | 11 |
| semantic | Advanced user | 10 | 0 | 2 |
| media | User-facing | 20 | 11 | 11 |
| local_ai | Advanced user / admin actions | 30 | 19 | 19 |
| enterprise | Admin-only | 14 | 0 | 0 |
| developer | Advanced / admin | 23 | 17 | 15 |
| security | Admin-only / security-sensitive | 25 | 25 | 24 |
| calendar | User-facing | 15 | 10 | 12 |
| communication | User-facing + external-send approval | 37 | 14 | 21 |
| assistant | User-facing | 13 | 10 | 6 |
| personal_os | Advanced user | 18 | 0 | 0 |
| production | Internal / operator | 34 | 0 | 0 |
| twin | Advanced user / privacy-sensitive | 11 | 10 | 10 |
| vault | Advanced / security-sensitive | 8 | 7 | 8 |
| scraper | Advanced user / network-sensitive | 6 | 5 | 6 |
| research | User-facing | 14 | 3 | 8 |
| finetune | Advanced user / admin actions | 10 | 10 | 9 |
| workspace | User-facing | 6 | 3 | 3 |
| auth | Admin-only / security-sensitive | 18 | 1 | 2 |
| document | User-facing | 10 | 9 | 9 |
| draft | Advanced user | 5 | 0 | 0 |
| worker | Internal / operator | 8 | 2 | 0 |
| vector_store | Internal / operator | 5 | 0 | 0 |
| advanced_rag | Advanced user | 4 | 1 | 2 |
| inference | Internal / operator | 14 | 3 | 1 |
| ultra_model | Advanced user / admin actions | 12 | 0 | 0 |
| adaptive_engine | Advanced user / operator | 29 | 10 | 5 |
| contacts | User-facing | 7 | 0 | 0 |
| backup_restore | Admin-only / destructive confirmation | 4 | 4 | 4 |
| model_cookbook | Advanced user | 11 | 0 | 0 |
| document_processing | Advanced user | 14 | 2 | 2 |
| image_studio | User-facing | 15 | 0 | 0 |
| scheduler | User-facing + admin actions | 21 | 0 | 5 |
| safe_terminal | Advanced user / explicit consent | 3 | 0 | 3 |
| mcp_runtime | Advanced / admin | 7 | 0 | 0 |
| tool_runtime | Internal / operator | 2 | 0 | 0 |
| compute_router | Internal / operator | 2 | 0 | 2 |
| agent_bridge | Internal / operator | 9 | 0 | 0 |
| skills | Advanced user / admin actions | 17 | 0 | 0 |
| conversation-organization | User-facing | 9 | 4 | 3 |
| enterprise-identity | Admin-only / security-sensitive | 15 | 0 | 0 |
| science | Advanced user | 19 | 9 | 7 |
| codebase | User-facing + destructive/approval controls | 23 | 20 | 18 |

## Genuine backend capability gaps

These gaps are capabilities not represented by the current backend contracts. Routes present in the registry but absent from a UI are frontend-surface gaps, not backend gaps. CodeQ and Research SSE are existing contracts and are not listed as missing capabilities.

| Capability gap | Backend finding |
| --- | --- |
| Shared workspace-to-project selection | GET /api/codebase/project-identities now exposes CodeQ identity records, but no common user-facing contract consistently maps a selected project to chat, files, research, tasks and specialist services. Enterprise workspace IDs, CodeQ project IDs and the default service project remain distinct. |
| Incremental agent/workflow execution events | CodeQ and Research expose run-event SSE, and workflows accept POST /api/workflows/event. No resumable run-event read/stream contract for agent/workflow execution output is declared. |
| Agent run retry and cancellation | Agent runs can be listed and checked, but no run-specific retry or cancellation contract is declared. |
| Science project, executed-run and report history | Science proposal records are persisted, but there is no complete project/report history contract and a proposal remains distinct from an executed experiment. |
| Personal-context source attribution | Twin source metadata exists, but there is no source-to-fact mapping explaining which interaction produced a profile field or preference. |
| Fine-tuning checkpoint manifest | Datasets, jobs and adapters exist; no authoritative checkpoint list/manifest route is declared. |
| Browser voice streaming | Voice sessions, turns, STT and TTS operations exist; no browser microphone WebSocket or equivalent live capture transport is declared. |
| Voice media playback | STT/TTS return operation metadata or artifacts, but no permissioned binary asset fetch/stream contract is declared for browser playback. |
| Chat agent selection | The chat request accepts provider/model, attachments and external-egress consent, but has no agent_id or agent selection contract. |
| Incremental terminal output | The safe terminal route returns a completed run with buffered events; no incremental response/event transport is declared. |
| CodeQ preview session lifecycle | Codebase project/tree/file/run/event APIs exist; no preview URL or preview-session create/read/stop contract is declared. |
| Notification preferences | Notification records can be listed and marked read, but per-user delivery preferences and channel/type settings have no storage/API contract. |
| Workspace item archive/delete/restore | Workspace items can be listed, created and updated, but no dedicated delete/archive/restore operation is declared. The current UI stores link type as item metadata, so this is not counted as a missing typed-link contract. |
| Research source review notes | Research source discovery and session checkpoints exist, but no API persists user review notes or annotations against a discovered source. |
| Assistant task editing and deletion | Assistant tasks can be created and completed, but no task update/delete contract is declared for due dates, priority or goal relationships. |
| Calendar event mutation lifecycle | Calendar event creation and free-slot lookup exist, but event update/delete and reminder acknowledgement contracts are not declared. |
| Calendar-scoped availability search | Free-slot lookup returns project-level availability and does not accept a calendar identifier for a calendar-specific query. |
| Calendar provider lifecycle | Provider registration and sync are represented, but no provider removal or re-authentication lifecycle contract is declared. |
| Stored chat media deletion | Chat supports uploaded media references and detach-from-prompt behavior, but no user-scoped endpoint deletes a stored attachment asset. |
| Knowledge record detail/actions mapping | Document detail/update/archive/delete routes exist. The Knowledge screen does not call those routes, and a mapping between indexed knowledge IDs and document IDs has not been verified. |
| Model benchmark history retrieval | Benchmark results are persisted in the model_benchmarks table, but the registered API has no GET list/history contract for retrieving those records. |

## Backend security review findings

This source-level review identifies authorization boundaries that need intentional product policy. The terminal detail exposure was closed in this pass; research's project-shared ownership model still requires confirmation before multi-user rollout.

| Area | Finding / status |
| --- | --- |
| Safe terminal run detail | Fixed in this pass: the detail lookup now filters by authenticated owner when auth is enforced, and /api/terminal/ requires terminal:read or terminal:write scopes in team/enterprise mode. Local single-user behavior remains unchanged. |
| Research session ownership | Research routes are behind the knowledge scope gate in enforced-auth mode, but durable session rows currently scope by project and session ID rather than owner. Confirm that project membership is the intended sharing boundary; add owner isolation if sessions are user-private. |
| Conversation object ownership | Chat/conversation route auth gates API scopes, but stored conversations and some branch/export/replay operations are resolved by opaque conversation ID or project without an owner field. Confirm local single-user-only deployment or add owner-scoped storage/lookup before multi-user rollout. |
| Agent approval ownership | Approval resolution accepts an approval ID without a user-facing project/owner binding in the request/service contract. Scope the approval to its requester/project before multi-user rollout. |
| Share grant expiry enforcement | The share API accepts and persists an expiry timestamp, and the UI requires one; verify authorization checks enforce expiry on every access path rather than treating it as display metadata. |
| Project identity consistency | The second frontend hardcodes the default service project on many screens, and CodeQ projects have their own IDs. Do not treat the frontend project string or enterprise workspace ID as an authorization boundary. |

## Backend capabilities already added or confirmed during gap closure

These are functional contracts, not remaining backend gaps. Their frontend coverage is listed in the route matrix.

| Previously identified area | Current backed capability |
| --- | --- |
| Notifications | Notification records list and mark-read now use the scheduler routes; preference storage and push transport remain absent. |
| Knowledge collections | Semantic collection list/create routes exist; they are separate from indexed documents and have no document-membership or rename/delete contract. |
| Communications | Inbox, Sent and Drafts records are surfaced; inbox read-state uses the existing message-state route and the UI polls every 15 seconds. |
| Scraper history | Stored extraction metadata has a list route. This does not persist or return full extraction bodies. |
| Vault access log | Project-scoped operation metadata is returned without secret values or ciphertext. |
| Science proposals | Proposal records persist with proposed/not_started semantics; proposals are not executed experiment runs. |
| Twin source metadata | Interaction type/topic/date metadata is listed without original prompts; source-to-profile-fact provenance remains absent. |
| Calendar | Calendar list/filter/create and user-triggered free-slot lookup are surfaced; free-slot search is project-wide because its API has no calendar-id argument. |

## Backend route groups with broader APIs than the current screens use

Audience classification is applied by route group before UI review. A group may mix audiences, so route-level overrides are shown in the matrix. Advanced/admin routes must stay behind intentional panels and backend authorization; internal/operator contracts should not be added to the user-facing navigation. This classification is a product-surface recommendation, not a substitute for checking the route's actual authorization.

## Verification boundary

- Inventory was read from the live Python route-contract registry, not a remembered count or documentation snapshot.
- Each contract entry is compared against production TypeScript/JavaScript under `frontend/src` and `frontend-second/src`; test files are excluded.
- The scanner reports only whether a route pattern appears as source text. Dynamic URL builders, aliases and helper-generated paths can be false negatives; a route string in source can still be unused or broken.
- Route declarations do not prove a handler branch, successful response, response-schema compatibility, authentication, authorization, streaming behavior or end-to-end usability. Those require route-specific tests and runtime/browser verification.
- The current gap audit also includes a terminal detail authorization fix and new terminal scopes. Focused tests cover owner filtering, enforcement-mode authentication, read-scope denial and full-scope access.
