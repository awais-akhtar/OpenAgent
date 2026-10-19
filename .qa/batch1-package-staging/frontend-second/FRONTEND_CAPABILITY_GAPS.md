# Frontend API coverage gaps — full route audit

> Supporting route source inventory only. The authoritative current gap/status, wiring and test classifications are in [FRONTEND_BACKEND_MASTER_GAPS.md](FRONTEND_BACKEND_MASTER_GAPS.md). “Found” means only that a production source path string matched.

Updated: 2026-10-18  
Audited surfaces: primary React app (`frontend/src`) and independent second frontend (`frontend-second/src`).

## Summary

The backend contract registry declares **654 method/path entries across 49 groups**. The scanner found direct route-pattern references in **249** entries in the primary frontend and **257** in the second frontend. A direct reference is not a verification of request method, response handling, permissions or working interaction.

The primary app is the existing 22-destination reference frontend. `frontend-second/` is the target production frontend. Keep their coverage distinct: a capability in one does not imply that the other frontend exposes it. Route audiences below distinguish user-facing, advanced/admin-only and internal operations; route-level overrides take precedence.

| Group | Audience | API entries | Primary direct refs | Primary w/o ref | Second direct refs | Second w/o ref |
| --- | --- | --- | --- | --- | --- | --- |
| health | Internal / operator | 5 | 3 | 2 | 1 | 4 |
| core | Internal / operator | 9 | 0 | 9 | 2 | 7 |
| integrations | Advanced / admin | 13 | 0 | 13 | 0 | 13 |
| chat | User-facing | 12 | 11 | 1 | 10 | 2 |
| agents | User-facing + advanced permissions | 11 | 10 | 1 | 9 | 2 |
| workflows | User-facing | 13 | 7 | 6 | 7 | 6 |
| knowledge | User-facing | 14 | 9 | 5 | 11 | 3 |
| semantic | Advanced user | 10 | 0 | 10 | 2 | 8 |
| media | User-facing | 20 | 11 | 9 | 11 | 9 |
| local_ai | Advanced user / admin actions | 30 | 19 | 11 | 19 | 11 |
| enterprise | Admin-only | 14 | 0 | 14 | 0 | 14 |
| developer | Advanced / admin | 23 | 17 | 6 | 15 | 8 |
| security | Admin-only / security-sensitive | 25 | 25 | 0 | 24 | 1 |
| calendar | User-facing | 15 | 10 | 5 | 12 | 3 |
| communication | User-facing + external-send approval | 37 | 14 | 23 | 21 | 16 |
| assistant | User-facing | 13 | 10 | 3 | 6 | 7 |
| personal_os | Advanced user | 18 | 0 | 18 | 0 | 18 |
| production | Internal / operator | 34 | 0 | 34 | 0 | 34 |
| twin | Advanced user / privacy-sensitive | 11 | 10 | 1 | 10 | 1 |
| vault | Advanced / security-sensitive | 8 | 7 | 1 | 8 | 0 |
| scraper | Advanced user / network-sensitive | 6 | 5 | 1 | 6 | 0 |
| research | User-facing | 14 | 3 | 11 | 8 | 6 |
| finetune | Advanced user / admin actions | 10 | 10 | 0 | 9 | 1 |
| workspace | User-facing | 6 | 3 | 3 | 3 | 3 |
| auth | Admin-only / security-sensitive | 18 | 1 | 17 | 2 | 16 |
| document | User-facing | 10 | 9 | 1 | 9 | 1 |
| draft | Advanced user | 5 | 0 | 5 | 0 | 5 |
| worker | Internal / operator | 8 | 2 | 6 | 0 | 8 |
| vector_store | Internal / operator | 5 | 0 | 5 | 0 | 5 |
| advanced_rag | Advanced user | 4 | 1 | 3 | 2 | 2 |
| inference | Internal / operator | 14 | 3 | 11 | 1 | 13 |
| ultra_model | Advanced user / admin actions | 12 | 0 | 12 | 0 | 12 |
| adaptive_engine | Advanced user / operator | 29 | 10 | 19 | 5 | 24 |
| contacts | User-facing | 7 | 0 | 7 | 0 | 7 |
| backup_restore | Admin-only / destructive confirmation | 4 | 4 | 0 | 4 | 0 |
| model_cookbook | Advanced user | 11 | 0 | 11 | 0 | 11 |
| document_processing | Advanced user | 14 | 2 | 12 | 2 | 12 |
| image_studio | User-facing | 15 | 0 | 15 | 0 | 15 |
| scheduler | User-facing + admin actions | 21 | 0 | 21 | 5 | 16 |
| safe_terminal | Advanced user / explicit consent | 3 | 0 | 3 | 3 | 0 |
| mcp_runtime | Advanced / admin | 7 | 0 | 7 | 0 | 7 |
| tool_runtime | Internal / operator | 2 | 0 | 2 | 0 | 2 |
| compute_router | Internal / operator | 2 | 0 | 2 | 2 | 0 |
| agent_bridge | Internal / operator | 9 | 0 | 9 | 0 | 9 |
| skills | Advanced user / admin actions | 17 | 0 | 17 | 0 | 17 |
| conversation-organization | User-facing | 9 | 4 | 5 | 3 | 6 |
| enterprise-identity | Admin-only / security-sensitive | 15 | 0 | 15 | 0 | 15 |
| science | Advanced user | 19 | 9 | 10 | 7 | 12 |
| codebase | User-facing + destructive/approval controls | 23 | 20 | 3 | 18 | 5 |

## How to use the route matrix

- **Found**: the registered route pattern appears as a string in that frontend's production source (fixture/demo and test files are excluded). This is still not proof of an active callsite or successful request.
- **No literal**: no direct route-pattern string was found. This is a candidate frontend gap, but may be a false negative if a shared helper or dynamic URL builder constructs the path.
- Routes are not assigned “not needed” automatically. Some are privileged admin/runtime operations and should remain inaccessible unless a deliberate screen and authorization workflow are designed.
- The audience column is a UI classification aid, not proof of the route's authorization. The full list is the backend's declared contract surface; it does not prove every route has a live handler or passes its tests.

## Route-by-route matrix

| Group | Audience | Method | Backend route pattern | Primary source ref | Second source ref |
| --- | --- | --- | --- | --- | --- |
| health | Internal / operator | GET | `/api/health` | Found | Found |
| health | Admin-only / operator | GET | `/api/system/metrics` | Found | No literal |
| health | Internal / operator | GET | `/api/ui/build` | No literal | No literal |
| health | Internal / operator | GET | `/api/modules/health` | Found | No literal |
| health | Internal / operator | GET | `/api/routes/handlers` | No literal | No literal |
| core | Internal / operator | GET | `/api/capabilities` | No literal | No literal |
| core | Internal / operator | GET | `/api/core/runtime` | No literal | No literal |
| core | Internal / operator | GET | `/api/mission/features` | No literal | No literal |
| core | Internal / operator | GET | `/api/context` | No literal | No literal |
| core | Internal / operator | GET | `/api/events` | No literal | Found |
| core | Internal / operator | GET | `/api/chat/diagnose` | No literal | No literal |
| core | Internal / operator | POST | `/api/events` | No literal | Found |
| core | Admin-only / operator | POST | `/api/capabilities/hard-audit` | No literal | No literal |
| core | Internal / operator | POST | `/api/mission/run-check` | No literal | No literal |
| integrations | Advanced / admin | GET | `/api/integrations/status` | No literal | No literal |
| integrations | Advanced / admin | GET | `/api/integrations/pairing-codes` | No literal | No literal |
| integrations | Advanced / admin | POST | `/api/integrations/pairing-codes` | No literal | No literal |
| integrations | Advanced / admin | POST | `/api/integrations/pairing-codes/redeem` | No literal | No literal |
| integrations | Advanced / admin | GET | `/api/integrations/preferences` | No literal | No literal |
| integrations | Advanced / admin | POST | `/api/integrations/preferences` | No literal | No literal |
| integrations | Advanced / admin | GET | `/api/integrations/diagnostics` | No literal | No literal |
| integrations | Advanced / admin | POST | `/api/integrations/diagnostics` | No literal | No literal |
| integrations | Advanced / admin | GET | `/api/integrations/backups` | No literal | No literal |
| integrations | Advanced / admin | POST | `/api/integrations/backups` | No literal | No literal |
| integrations | Advanced / admin | GET | `/api/integrations/safe-shell` | No literal | No literal |
| integrations | Advanced / admin | POST | `/api/integrations/safe-shell` | No literal | No literal |
| integrations | Advanced / admin | GET | `/api/integrations/parity` | No literal | No literal |
| chat | User-facing | GET | `/api/conversations` | Found | Found |
| chat | User-facing | GET | `/api/messages` | No literal | Found |
| chat | User-facing | GET | `/api/conversation/messages` | Found | No literal |
| chat | User-facing | GET | `/api/chat/search` | Found | No literal |
| chat | User-facing | GET | `/api/chat/export` | Found | Found |
| chat | User-facing | GET | `/api/chat/replay` | Found | Found |
| chat | User-facing | POST | `/api/chat` | Found | Found |
| chat | User-facing | POST | `/api/chat/stream` | Found | Found |
| chat | User-facing | POST | `/api/chat/branch` | Found | Found |
| chat | User-facing | POST | `/api/chat/import` | Found | Found |
| chat | User-facing | POST | `/api/chat/artifact` | Found | Found |
| chat | User-facing | POST | `/api/chat/research-record` | Found | Found |
| agents | User-facing + advanced permissions | GET | `/api/agents` | Found | Found |
| agents | User-facing + advanced permissions | POST | `/api/agents/run` | Found | Found |
| agents | User-facing + advanced permissions | POST | `/api/agents/run-checked` | Found | Found |
| agents | User-facing + advanced permissions | POST | `/api/agents/delegate` | Found | Found |
| agents | User-facing + advanced permissions | POST | `/api/agents/parallel` | No literal | No literal |
| agents | User-facing + advanced permissions | GET | `/api/agents/marketplace` | Found | No literal |
| agents | User-facing + advanced permissions | GET | `/api/agents/permissions` | Found | Found |
| agents | User-facing + advanced permissions | GET | `/api/agents/approvals` | Found | Found |
| agents | User-facing + advanced permissions | POST | `/api/agents/permissions/grant` | Found | Found |
| agents | User-facing + advanced permissions | POST | `/api/agents/approvals/request` | Found | Found |
| agents | User-facing + advanced permissions | POST | `/api/agents/approvals/resolve` | Found | Found |
| workflows | User-facing | GET | `/api/workflows` | Found | Found |
| workflows | User-facing | POST | `/api/workflows` | Found | Found |
| workflows | User-facing | PUT | `/api/workflows/{id}` | Found | Found |
| workflows | User-facing | DELETE | `/api/workflows/{id}` | Found | Found |
| workflows | User-facing | GET | `/api/workflows/builder` | Found | Found |
| workflows | User-facing | GET | `/api/workflows/visual` | No literal | No literal |
| workflows | User-facing | POST | `/api/workflows/run` | Found | No literal |
| workflows | User-facing | POST | `/api/workflows/{id}/run` | No literal | Found |
| workflows | User-facing | POST | `/api/workflows/run-due` | No literal | No literal |
| workflows | User-facing | POST | `/api/workflows/event` | No literal | No literal |
| workflows | User-facing | GET | `/api/workflows/layout` | No literal | No literal |
| workflows | User-facing | POST | `/api/workflows/layout` | No literal | No literal |
| workflows | User-facing | GET | `/api/workflow-runs` | Found | Found |
| knowledge | User-facing | GET | `/api/knowledge/status` | Found | Found |
| knowledge | User-facing | GET | `/api/knowledge/documents` | Found | Found |
| knowledge | User-facing | GET | `/api/knowledge/search` | No literal | Found |
| knowledge | User-facing | GET | `/api/knowledge/graph` | Found | Found |
| knowledge | User-facing | GET | `/api/knowledge/tags` | Found | Found |
| knowledge | User-facing | GET | `/api/knowledge/timeline` | Found | Found |
| knowledge | User-facing | POST | `/api/knowledge/ingest` | Found | Found |
| knowledge | User-facing | POST | `/api/knowledge/upload` | Found | Found |
| knowledge | User-facing | POST | `/api/knowledge/semantic-search` | No literal | Found |
| knowledge | User-facing | POST | `/api/knowledge/hybrid-search` | Found | Found |
| knowledge | User-facing | POST | `/api/knowledge/synthesize` | Found | Found |
| knowledge | User-facing | POST | `/api/knowledge/compress` | No literal | No literal |
| knowledge | User-facing | POST | `/api/knowledge/replay` | No literal | No literal |
| knowledge | User-facing | POST | `/api/embeddings` | No literal | No literal |
| semantic | Advanced user | GET | `/api/semantic/status` | No literal | No literal |
| semantic | Advanced user | GET | `/api/semantic/memories` | No literal | No literal |
| semantic | Advanced user | POST | `/api/semantic/memories` | No literal | No literal |
| semantic | Advanced user | GET | `/api/semantic/collections` | No literal | Found |
| semantic | Advanced user | POST | `/api/semantic/collections` | No literal | Found |
| semantic | Advanced user | GET | `/api/semantic/graph-analytics` | No literal | No literal |
| semantic | Advanced user | POST | `/api/semantic/consolidate` | No literal | No literal |
| semantic | Advanced user | POST | `/api/semantic/vectors` | No literal | No literal |
| semantic | Advanced user | POST | `/api/semantic/vector-search` | No literal | No literal |
| semantic | Advanced user | POST | `/api/semantic/rerank` | No literal | No literal |
| media | User-facing | GET | `/api/media/status` | No literal | No literal |
| media | User-facing | GET | `/api/media/assets` | No literal | Found |
| media | User-facing | GET | `/api/media/analyses` | No literal | No literal |
| media | User-facing | POST | `/api/media/ingest` | No literal | Found |
| media | User-facing | POST | `/api/media/analyze` | No literal | No literal |
| media | User-facing | GET | `/api/voice/status` | Found | Found |
| media | User-facing | GET | `/api/voice/personas` | Found | Found |
| media | User-facing | GET | `/api/voice/sessions` | Found | Found |
| media | User-facing | GET | `/api/voice/turns` | Found | Found |
| media | User-facing | POST | `/api/voice/personas` | Found | Found |
| media | User-facing | POST | `/api/voice/sessions` | Found | Found |
| media | User-facing | POST | `/api/voice/vosk/setup` | No literal | No literal |
| media | User-facing | POST | `/api/voice/clone` | No literal | No literal |
| media | User-facing | POST | `/api/voice/sessions/close` | Found | Found |
| media | User-facing | POST | `/api/voice/transcript` | Found | No literal |
| media | User-facing | POST | `/api/voice/transcribe-upload` | No literal | No literal |
| media | User-facing | POST | `/api/voice/upload` | No literal | No literal |
| media | User-facing | POST | `/api/voice/stt` | Found | Found |
| media | User-facing | POST | `/api/voice/tts` | Found | Found |
| media | User-facing | POST | `/api/voice/translate` | Found | No literal |
| local_ai | Advanced user / admin actions | GET | `/api/models` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/model-policy` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/settings/runtime` | Found | Found |
| local_ai | Advanced user / admin actions | POST | `/api/settings/runtime` | Found | Found |
| local_ai | Advanced user / admin actions | POST | `/api/models/discover` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/jobs` | Found | Found |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/jobs` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/jobs/{id}` | No literal | No literal |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/jobs/cancel` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/status` | Found | No literal |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/setup/status` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/smooth-runtime` | No literal | No literal |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/models` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/models/{id}/probe` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/hardware` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/adapters` | Found | Found |
| local_ai | Advanced user / admin actions | GET | `/api/local-ai/marketplace` | Found | Found |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/models/register` | Found | Found |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/models/download` | Found | Found |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/models/setup/job` | Found | Found |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/llamacpp/register` | No literal | No literal |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/models/auto-quantize` | No literal | No literal |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/models/delete` | Found | Found |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/models/{id}/dependencies` | No literal | No literal |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/models/{id}/tools` | No literal | No literal |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/benchmark` | No literal | Found |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/cluster-plan` | No literal | No literal |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/models/quantize` | No literal | No literal |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/models/setup-chat` | No literal | No literal |
| local_ai | Advanced user / admin actions | POST | `/api/local-ai/ollama/pull` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/status` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/users` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/roles` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/workspaces` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/policies` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/sso` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/backups` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/disaster-recovery` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/workspace/members` | No literal | No literal |
| enterprise | Admin-only | GET | `/api/enterprise/org-memory` | No literal | No literal |
| enterprise | Admin-only | POST | `/api/enterprise/access/check` | No literal | No literal |
| enterprise | Admin-only | POST | `/api/enterprise/audit/export` | No literal | No literal |
| enterprise | Admin-only | POST | `/api/enterprise/policies/evaluate` | No literal | No literal |
| enterprise | Admin-only | POST | `/api/enterprise/sso/validate` | No literal | No literal |
| developer | Advanced / admin | GET | `/api/developer/status` | Found | Found |
| developer | Advanced / admin | GET | `/api/developer/schema` | Found | No literal |
| developer | Advanced / admin | GET | `/api/developer/webhooks` | Found | Found |
| developer | Advanced / admin | GET | `/api/developer/webhook-deliveries` | No literal | Found |
| developer | Advanced / admin | GET | `/api/developer/plugins` | Found | Found |
| developer | Advanced / admin | GET | `/api/developer/tools` | Found | Found |
| developer | Advanced / admin | GET | `/api/developer/mcp` | Found | Found |
| developer | Advanced / admin | GET | `/api/developer/mcp/servers` | Found | Found |
| developer | Advanced / admin | GET | `/api/developer/extensions` | Found | Found |
| developer | Advanced / admin | GET | `/api/developer/sdk` | Found | No literal |
| developer | Advanced / admin | GET | `/api/developer/api-keys` | Found | Found |
| developer | Advanced / admin | POST | `/api/developer/api-keys` | Found | Found |
| developer | Advanced / admin | POST | `/api/developer/api-keys/verify` | No literal | No literal |
| developer | Advanced / admin | POST | `/api/developer/webhooks` | Found | Found |
| developer | Advanced / admin | POST | `/api/developer/webhooks/dispatch` | No literal | No literal |
| developer | Advanced / admin | POST | `/api/developer/plugins` | Found | Found |
| developer | Advanced / admin | POST | `/api/developer/tools` | Found | Found |
| developer | Advanced / admin | POST | `/api/developer/tools/invoke` | No literal | No literal |
| developer | Advanced / admin | POST | `/api/developer/mcp/servers` | Found | Found |
| developer | Advanced / admin | POST | `/api/developer/extensions` | Found | Found |
| developer | Advanced / admin | POST | `/api/developer/sdk` | Found | No literal |
| developer | Advanced / admin | POST | `/api/graphql` | No literal | No literal |
| developer | Advanced / admin | POST | `/api/developer/graphql` | No literal | No literal |
| security | Admin-only / security-sensitive | GET | `/api/audit` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/status` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/secrets` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/retention` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/shares` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/egress` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/audit-chain` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/audit-chain/verify` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/privacy-reports` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/privacy-report` | Found | Found |
| security | Admin-only / security-sensitive | GET | `/api/security/sandbox-profiles` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/secrets` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/audit-chain/seal` | Found | No literal |
| security | Admin-only / security-sensitive | POST | `/api/security/audit-chain/verify` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/privacy-report` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/egress/request` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/egress/resolve` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/retention` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/retention/apply` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/sandbox-profiles` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/sandbox/evaluate` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/secrets/reveal` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/secrets/rotate` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/shares` | Found | Found |
| security | Admin-only / security-sensitive | POST | `/api/security/shares/resolve` | Found | Found |
| calendar | User-facing | GET | `/api/calendar/status` | Found | No literal |
| calendar | User-facing | GET | `/api/calendar/calendars` | Found | Found |
| calendar | User-facing | GET | `/api/calendar/events` | Found | Found |
| calendar | User-facing | POST | `/api/calendar/events` | Found | Found |
| calendar | User-facing | POST | `/api/calendar/ical-import` | No literal | No literal |
| calendar | User-facing | POST | `/api/calendar/ical-export` | No literal | No literal |
| calendar | User-facing | GET | `/api/calendar/free-slots` | Found | Found |
| calendar | User-facing | GET | `/api/calendar/reminders` | Found | Found |
| calendar | User-facing | GET | `/api/calendar/optimize` | No literal | Found |
| calendar | User-facing | GET | `/api/calendar/providers` | Found | Found |
| calendar | User-facing | POST | `/api/calendar/providers/register` | No literal | Found |
| calendar | User-facing | GET | `/api/calendar/conflicts` | Found | Found |
| calendar | User-facing | POST | `/api/calendar/conflicts/{conflict_id}/resolve` | Found | Found |
| calendar | User-facing | POST | `/api/calendar/sync` | Found | Found |
| calendar | User-facing | GET | `/api/calendar/sync/{sync_job_id}` | No literal | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/connectors` | Found | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/inbox` | Found | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/drafts` | Found | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/sent` | No literal | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/contacts` | Found | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/calendar` | Found | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/analytics` | No literal | No literal |
| communication | User-facing + external-send approval | GET | `/api/communication/sync-runs` | No literal | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/dead-letters` | No literal | No literal |
| communication | User-facing + external-send approval | GET | `/api/communication/oauth/providers` | No literal | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/oauth/accounts` | No literal | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/oauth/callback` | No literal | No literal |
| communication | User-facing + external-send approval | GET | `/api/communication/oauth/callbacks` | No literal | No literal |
| communication | User-facing + external-send approval | GET | `/api/communication/oauth/mailbox-policy` | No literal | No literal |
| communication | User-facing + external-send approval | GET | `/api/communication/provider-certification` | No literal | No literal |
| communication | User-facing + external-send approval | GET | `/api/communication/attachments` | No literal | No literal |
| communication | User-facing + external-send approval | GET | `/api/communication/approvals` | Found | Found |
| communication | User-facing + external-send approval | GET | `/api/communication/scheduled` | Found | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/connectors/configure` | No literal | No literal |
| communication | User-facing + external-send approval | POST | `/api/communication/contacts` | Found | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/messages/import` | No literal | No literal |
| communication | User-facing + external-send approval | POST | `/api/communication/drafts/reply` | No literal | No literal |
| communication | User-facing + external-send approval | POST | `/api/communication/drafts` | Found | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/send/request-approval` | Found | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/send/resolve` | Found | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/send` | Found | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/calendar` | Found | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/sync` | No literal | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/sync-resilient` | No literal | No literal |
| communication | User-facing + external-send approval | POST | `/api/communication/oauth/configure` | No literal | No literal |
| communication | User-facing + external-send approval | POST | `/api/communication/oauth/accounts/configure` | No literal | No literal |
| communication | User-facing + external-send approval | POST | `/api/communication/oauth/authorize-url` | No literal | No literal |
| communication | User-facing + external-send approval | POST | `/api/communication/oauth/refresh` | No literal | No literal |
| communication | User-facing + external-send approval | POST | `/api/communication/attachments` | No literal | No literal |
| communication | User-facing + external-send approval | POST | `/api/communication/messages/state` | No literal | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/scheduled` | Found | Found |
| communication | User-facing + external-send approval | POST | `/api/communication/scheduled/process` | No literal | Found |
| assistant | User-facing | GET | `/api/assistant/tasks` | Found | Found |
| assistant | User-facing | GET | `/api/assistant/goals` | Found | Found |
| assistant | User-facing | GET | `/api/assistant/daily-brief` | Found | Found |
| assistant | User-facing | GET | `/api/assistant/coach` | Found | No literal |
| assistant | User-facing | GET | `/api/assistant/crm` | Found | No literal |
| assistant | User-facing | POST | `/api/assistant/profile` | No literal | No literal |
| assistant | User-facing | POST | `/api/assistant/goals` | Found | Found |
| assistant | User-facing | POST | `/api/assistant/tasks` | Found | Found |
| assistant | User-facing | POST | `/api/assistant/tasks/complete` | Found | Found |
| assistant | User-facing | POST | `/api/assistant/meeting/brief` | Found | No literal |
| assistant | User-facing | POST | `/api/assistant/meeting/summary` | No literal | No literal |
| assistant | User-facing | POST | `/api/assistant/travel/plan` | No literal | No literal |
| assistant | User-facing | POST | `/api/assistant/crm` | Found | No literal |
| personal_os | Advanced user | GET | `/api/personal-os/status` | No literal | No literal |
| personal_os | Advanced user | GET | `/api/personal-os/presets` | No literal | No literal |
| personal_os | Advanced user | POST | `/api/personal-os/presets` | No literal | No literal |
| personal_os | Advanced user | GET | `/api/personal-os/signatures` | No literal | No literal |
| personal_os | Advanced user | POST | `/api/personal-os/signatures` | No literal | No literal |
| personal_os | Advanced user | GET | `/api/personal-os/drafts` | No literal | No literal |
| personal_os | Advanced user | POST | `/api/personal-os/drafts` | No literal | No literal |
| personal_os | Advanced user | GET | `/api/personal-os/drafts/{id}/versions` | No literal | No literal |
| personal_os | Advanced user | GET | `/api/personal-os/notes` | No literal | No literal |
| personal_os | Advanced user | POST | `/api/personal-os/notes` | No literal | No literal |
| personal_os | Advanced user | GET | `/api/personal-os/gallery` | No literal | No literal |
| personal_os | Advanced user | POST | `/api/personal-os/gallery` | No literal | No literal |
| personal_os | Advanced user | POST | `/api/personal-os/gallery/favorite` | No literal | No literal |
| personal_os | Advanced user | GET | `/api/personal-os/compare` | No literal | No literal |
| personal_os | Advanced user | POST | `/api/personal-os/compare/start` | No literal | No literal |
| personal_os | Advanced user | POST | `/api/personal-os/compare/vote` | No literal | No literal |
| personal_os | Advanced user | GET | `/api/personal-os/export` | No literal | No literal |
| personal_os | Advanced user | POST | `/api/personal-os/import` | No literal | No literal |
| production | Internal / operator | GET | `/api/production/status` | No literal | No literal |
| production | Internal / operator | GET | `/api/production/policies` | No literal | No literal |
| production | Internal / operator | GET | `/api/production/queue` | No literal | No literal |
| production | Internal / operator | GET | `/api/production/audit-anchors` | No literal | No literal |
| production | Internal / operator | GET | `/api/runtime-tools/status` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/policies` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/policies/evaluate` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/queue` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/queue/run-next` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/audit-anchor` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/vector/indexes` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/vector/upsert` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/vector/search` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/semantic/science` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/rag/rerank` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/oidc/validate` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/crypto/encrypt` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/crypto/decrypt` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/connectors/adapters` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/connectors/poll` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/automation/desktop` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/automation/browser` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/packages/app` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/marketplace/agents` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/gpu/plan` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/finetune/jobs` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/proactive/jobs` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/proactive/run` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/federated/rounds` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/sync/devices` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/sync/bundles` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/sdk/publish` | No literal | No literal |
| production | Internal / operator | POST | `/api/production/sandbox/run` | No literal | No literal |
| production | Internal / operator | POST | `/api/runtime-tools/browser-test` | No literal | No literal |
| twin | Advanced user / privacy-sensitive | GET | `/api/twin/status` | Found | Found |
| twin | Advanced user / privacy-sensitive | GET | `/api/twin/profile` | Found | Found |
| twin | Advanced user / privacy-sensitive | GET | `/api/twin/patterns` | Found | Found |
| twin | Advanced user / privacy-sensitive | GET | `/api/twin/suggestions` | Found | Found |
| twin | Advanced user / privacy-sensitive | GET | `/api/twin/sources` | No literal | Found |
| twin | Advanced user / privacy-sensitive | GET | `/api/twin/goals` | Found | Found |
| twin | Advanced user / privacy-sensitive | POST | `/api/twin/profile` | Found | Found |
| twin | Advanced user / privacy-sensitive | POST | `/api/twin/interaction` | Found | No literal |
| twin | Advanced user / privacy-sensitive | POST | `/api/twin/predict` | Found | Found |
| twin | Advanced user / privacy-sensitive | POST | `/api/twin/goals` | Found | Found |
| twin | Advanced user / privacy-sensitive | POST | `/api/twin/consolidate` | Found | Found |
| vault | Advanced / security-sensitive | GET | `/api/vault/status` | Found | Found |
| vault | Advanced / security-sensitive | GET | `/api/vault/entries` | Found | Found |
| vault | Advanced / security-sensitive | GET | `/api/vault/access-log` | No literal | Found |
| vault | Advanced / security-sensitive | POST | `/api/vault/entries` | Found | Found |
| vault | Advanced / security-sensitive | POST | `/api/vault/store` | Found | Found |
| vault | Advanced / security-sensitive | POST | `/api/vault/retrieve` | Found | Found |
| vault | Advanced / security-sensitive | POST | `/api/vault/rotate-key` | Found | Found |
| vault | Advanced / security-sensitive | POST | `/api/vault/export` | Found | Found |
| scraper | Advanced user / network-sensitive | GET | `/api/scraper/status` | Found | Found |
| scraper | Advanced user / network-sensitive | GET | `/api/scraper/jobs` | Found | Found |
| scraper | Advanced user / network-sensitive | GET | `/api/scraper/history` | No literal | Found |
| scraper | Advanced user / network-sensitive | POST | `/api/scraper/scrape` | Found | Found |
| scraper | Advanced user / network-sensitive | POST | `/api/scraper/schedule` | Found | Found |
| scraper | Advanced user / network-sensitive | POST | `/api/scraper/run-due` | Found | Found |
| research | User-facing | GET | `/api/research/status` | No literal | No literal |
| research | User-facing | GET | `/api/research/sessions` | Found | Found |
| research | User-facing | GET | `/api/research/sessions/{id}` | Found | Found |
| research | User-facing | GET | `/api/research/sessions/{id}/report` | No literal | Found |
| research | User-facing | GET | `/api/research/sessions/{id}/checkpoints` | No literal | Found |
| research | User-facing | GET | `/api/research/sessions/{id}/source-graph` | No literal | Found |
| research | User-facing | GET | `/api/research/sessions/{id}/stream` | No literal | Found |
| research | User-facing | POST | `/api/research/run` | No literal | No literal |
| research | User-facing | POST | `/api/research/sessions` | Found | Found |
| research | User-facing | POST | `/api/research/sessions/{id}/cancel` | No literal | Found |
| research | User-facing | POST | `/api/research/sessions/{id}/retry` | No literal | No literal |
| research | User-facing | POST | `/api/research/sessions/{id}/branch` | No literal | No literal |
| research | User-facing | POST | `/api/research/sessions/{id}/archive` | No literal | No literal |
| research | User-facing | POST | `/api/research/sessions/{id}/delete` | No literal | No literal |
| finetune | Advanced user / admin actions | GET | `/api/finetune/status` | Found | Found |
| finetune | Advanced user / admin actions | GET | `/api/finetune/datasets` | Found | Found |
| finetune | Advanced user / admin actions | POST | `/api/finetune/datasets` | Found | Found |
| finetune | Advanced user / admin actions | POST | `/api/finetune/examples` | Found | Found |
| finetune | Advanced user / admin actions | GET | `/api/finetune/jobs` | Found | Found |
| finetune | Advanced user / admin actions | POST | `/api/finetune/jobs` | Found | Found |
| finetune | Advanced user / admin actions | GET | `/api/finetune/adapters` | Found | Found |
| finetune | Advanced user / admin actions | POST | `/api/finetune/adapters` | Found | Found |
| finetune | Advanced user / admin actions | POST | `/api/finetune/adapters/activate` | Found | Found |
| finetune | Advanced user / admin actions | GET | `/api/finetune/templates` | Found | No literal |
| workspace | User-facing | GET | `/api/workspace/items` | Found | Found |
| workspace | User-facing | POST | `/api/workspace/items` | Found | Found |
| workspace | User-facing | POST | `/api/workspace/items/update` | Found | Found |
| workspace | User-facing | POST | `/api/workspace/templates/instantiate` | No literal | No literal |
| workspace | User-facing | GET | `/api/workspace/dashboard` | No literal | No literal |
| workspace | User-facing | GET | `/api/workspace/canvas` | No literal | No literal |
| auth | Admin-only / security-sensitive | POST | `/api/auth/register` | No literal | No literal |
| auth | Admin-only / security-sensitive | POST | `/api/auth/login` | No literal | No literal |
| auth | Admin-only / security-sensitive | POST | `/api/auth/refresh` | No literal | No literal |
| auth | Admin-only / security-sensitive | POST | `/api/auth/logout` | No literal | No literal |
| auth | Admin-only / security-sensitive | GET | `/api/auth/me` | Found | No literal |
| auth | Admin-only / security-sensitive | GET | `/api/auth/sessions` | No literal | Found |
| auth | Admin-only / security-sensitive | POST | `/api/auth/sessions/revoke` | No literal | No literal |
| auth | Admin-only / security-sensitive | POST | `/api/auth/2fa/setup` | No literal | No literal |
| auth | Admin-only / security-sensitive | POST | `/api/auth/2fa/verify` | No literal | No literal |
| auth | Admin-only / security-sensitive | POST | `/api/auth/2fa/disable` | No literal | No literal |
| auth | Admin-only / security-sensitive | GET | `/api/auth/scopes` | No literal | Found |
| auth | Admin-only / security-sensitive | GET | `/api/auth/users` | No literal | No literal |
| auth | Admin-only / security-sensitive | GET | `/api/auth/roles` | No literal | No literal |
| auth | Admin-only / security-sensitive | GET | `/api/auth/policy` | No literal | No literal |
| auth | Admin-only / security-sensitive | GET | `/api/auth/audit` | No literal | No literal |
| auth | Admin-only / security-sensitive | GET | `/api/auth/api-tokens` | No literal | No literal |
| auth | Admin-only / security-sensitive | POST | `/api/auth/api-tokens` | No literal | No literal |
| auth | Admin-only / security-sensitive | DELETE | `/api/auth/api-tokens/{token_id}` | No literal | No literal |
| document | User-facing | POST | `/api/document` | Found | Found |
| document | User-facing | GET | `/api/documents/library` | Found | Found |
| document | User-facing | GET | `/api/document/{doc_id}` | Found | Found |
| document | User-facing | PUT | `/api/document/{doc_id}` | Found | Found |
| document | User-facing | PATCH | `/api/document/{doc_id}` | Found | Found |
| document | User-facing | POST | `/api/document/{doc_id}/archive` | Found | Found |
| document | User-facing | DELETE | `/api/document/{doc_id}` | Found | Found |
| document | User-facing | GET | `/api/document/{doc_id}/versions` | Found | Found |
| document | User-facing | GET | `/api/document/{doc_id}/version/{num}` | No literal | No literal |
| document | User-facing | POST | `/api/document/{doc_id}/restore/{num}` | Found | Found |
| draft | Advanced user | GET | `/api/editor-drafts` | No literal | No literal |
| draft | Advanced user | POST | `/api/editor-drafts` | No literal | No literal |
| draft | Advanced user | GET | `/api/editor-drafts/{draft_id}` | No literal | No literal |
| draft | Advanced user | PUT | `/api/editor-drafts/{draft_id}` | No literal | No literal |
| draft | Advanced user | DELETE | `/api/editor-drafts/{draft_id}` | No literal | No literal |
| worker | Internal / operator | GET | `/api/worker/status` | No literal | No literal |
| worker | Internal / operator | GET | `/api/worker/jobs` | Found | No literal |
| worker | Internal / operator | GET | `/api/worker/jobs/{id}` | No literal | No literal |
| worker | Internal / operator | GET | `/api/worker/dead-letters` | No literal | No literal |
| worker | Internal / operator | POST | `/api/worker/jobs` | Found | No literal |
| worker | Internal / operator | POST | `/api/worker/run-once` | No literal | No literal |
| worker | Internal / operator | POST | `/api/worker/jobs/{id}/cancel` | No literal | No literal |
| worker | Internal / operator | POST | `/api/worker/jobs/{id}/retry` | No literal | No literal |
| vector_store | Internal / operator | GET | `/api/vector-store/status` | No literal | No literal |
| vector_store | Internal / operator | GET | `/api/vector-store/collections` | No literal | No literal |
| vector_store | Internal / operator | POST | `/api/vector-store/upsert` | No literal | No literal |
| vector_store | Internal / operator | POST | `/api/vector-store/search` | No literal | No literal |
| vector_store | Internal / operator | POST | `/api/vector-store/benchmark` | No literal | No literal |
| advanced_rag | Advanced user | GET | `/api/rag/status` | Found | Found |
| advanced_rag | Advanced user | POST | `/api/rag/query` | No literal | No literal |
| advanced_rag | Advanced user | POST | `/api/rag/sync` | No literal | Found |
| advanced_rag | Advanced user | POST | `/api/rag/benchmark` | No literal | No literal |
| inference | Internal / operator | GET | `/api/inference/status` | Found | No literal |
| inference | Internal / operator | GET | `/health` | Found | Found |
| inference | Internal / operator | GET | `/metrics` | Found | No literal |
| inference | Internal / operator | GET | `/v1/models` | No literal | No literal |
| inference | Internal / operator | GET | `/v1/models/{model}` | No literal | No literal |
| inference | Internal / operator | GET | `/v1/colibri/capabilities` | No literal | No literal |
| inference | Internal / operator | POST | `/v1/brio` | No literal | No literal |
| inference | Internal / operator | POST | `/v1/systemone` | No literal | No literal |
| inference | Internal / operator | POST | `/v1/chat/completions` | No literal | No literal |
| inference | Internal / operator | POST | `/v1/completions` | No literal | No literal |
| inference | Internal / operator | POST | `/v1/messages` | No literal | No literal |
| inference | Internal / operator | POST | `/api/inference/benchmark` | No literal | No literal |
| inference | Internal / operator | POST | `/api/inference/cache/clear` | No literal | No literal |
| inference | Internal / operator | POST | `/api/inference/warmup` | No literal | No literal |
| ultra_model | Advanced user / admin actions | GET | `/api/ultra-model/status` | No literal | No literal |
| ultra_model | Advanced user / admin actions | GET | `/api/ultra-model/manifests` | No literal | No literal |
| ultra_model | Advanced user / admin actions | GET | `/api/ultra-model/plans` | No literal | No literal |
| ultra_model | Advanced user / admin actions | GET | `/api/ultra-model/sessions` | No literal | No literal |
| ultra_model | Advanced user / admin actions | POST | `/api/ultra-model/manifests` | No literal | No literal |
| ultra_model | Advanced user / admin actions | POST | `/api/ultra-model/plan` | No literal | No literal |
| ultra_model | Advanced user / admin actions | POST | `/api/ultra-model/route` | No literal | No literal |
| ultra_model | Advanced user / admin actions | POST | `/api/ultra-model/calibrate` | No literal | No literal |
| ultra_model | Advanced user / admin actions | POST | `/api/ultra-model/launch` | No literal | No literal |
| ultra_model | Advanced user / admin actions | POST | `/api/ultra-model/warmup` | No literal | No literal |
| ultra_model | Advanced user / admin actions | POST | `/api/ultra-model/verify` | No literal | No literal |
| ultra_model | Advanced user / admin actions | POST | `/api/ultra-model/sessions/{id}/stop` | No literal | No literal |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/status` | Found | Found |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/hardware` | Found | Found |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/plans` | Found | Found |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/benchmarks` | Found | No literal |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/tuning` | No literal | No literal |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/direct/status` | Found | Found |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/direct/calibrations` | No literal | No literal |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/role-options` | No literal | No literal |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/roles` | Found | No literal |
| adaptive_engine | Advanced user / operator | GET | `/api/adaptive-engine/orchestrations` | No literal | Found |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/inspect` | Found | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/compile` | Found | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/benchmark` | Found | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/autotune` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/kv/session` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/kv/append` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/kv/release` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/scheduler/queue` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/scheduler/next` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/direct/load` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/direct/load-llm` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/direct/infer` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/direct/generate` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/direct/benchmark` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/direct/calibrate` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/direct/unload` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/direct/stop` | No literal | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/roles` | Found | No literal |
| adaptive_engine | Advanced user / operator | POST | `/api/adaptive-engine/orchestrate` | No literal | No literal |
| contacts | User-facing | GET | `/api/contacts` | No literal | No literal |
| contacts | User-facing | POST | `/api/contacts` | No literal | No literal |
| contacts | User-facing | GET | `/api/contacts/search` | No literal | No literal |
| contacts | User-facing | POST | `/api/contacts/import` | No literal | No literal |
| contacts | User-facing | GET | `/api/contacts/export` | No literal | No literal |
| contacts | User-facing | PATCH | `/api/contacts/{id}` | No literal | No literal |
| contacts | User-facing | DELETE | `/api/contacts/{id}` | No literal | No literal |
| backup_restore | Admin-only / destructive confirmation | GET | `/api/backups` | Found | Found |
| backup_restore | Admin-only / destructive confirmation | POST | `/api/backups/create` | Found | Found |
| backup_restore | Admin-only / destructive confirmation | POST | `/api/backups/restore-dry-run` | Found | Found |
| backup_restore | Admin-only / destructive confirmation | POST | `/api/backups/restore` | Found | Found |
| model_cookbook | Advanced user | GET | `/api/model-cookbook/hosts` | No literal | No literal |
| model_cookbook | Advanced user | POST | `/api/model-cookbook/hosts` | No literal | No literal |
| model_cookbook | Advanced user | GET | `/api/model-cookbook/cached` | No literal | No literal |
| model_cookbook | Advanced user | POST | `/api/model-cookbook/serve` | No literal | No literal |
| model_cookbook | Advanced user | POST | `/api/model-cookbook/adopt` | No literal | No literal |
| model_cookbook | Advanced user | POST | `/api/model-cookbook/stop/{id}` | No literal | No literal |
| model_cookbook | Advanced user | GET | `/api/model-cookbook/logs/{id}` | No literal | No literal |
| model_cookbook | Advanced user | GET | `/api/model-cookbook/hardware-fit` | No literal | No literal |
| model_cookbook | Advanced user | GET | `/api/model-cookbook/certifications` | No literal | No literal |
| model_cookbook | Advanced user | POST | `/api/model-cookbook/certify` | No literal | No literal |
| model_cookbook | Advanced user | POST | `/api/model-cookbook/certifications/{id}/revoke` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/documents/import-pdf` | Found | Found |
| document_processing | Advanced user | POST | `/api/documents/import` | Found | Found |
| document_processing | Advanced user | POST | `/api/document/{id}/extract-pdf-text` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/document/{id}/render-preview` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/document/{id}/render-page` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/document/{id}/annotations` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/document/{id}/ocr-overlay` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/document/{id}/redact` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/document/{id}/export-pdf` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/document/{id}/export-docx` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/document/{id}/export-pptx` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/document/{id}/export-xlsx` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/documents/compare` | No literal | No literal |
| document_processing | Advanced user | POST | `/api/documents/export-zip` | No literal | No literal |
| image_studio | User-facing | GET | `/api/gallery/library` | No literal | No literal |
| image_studio | User-facing | GET | `/api/gallery/{id}/operations` | No literal | No literal |
| image_studio | User-facing | GET | `/api/gallery/{id}/versions` | No literal | No literal |
| image_studio | User-facing | GET | `/api/gallery/albums` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/upload` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/{id}/rotate` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/{id}/replace` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/upscale` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/albums` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/albums/{id}/assets` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/{id}/metadata` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/batch-tags` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/{id}/transform` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/{id}/undo` | No literal | No literal |
| image_studio | User-facing | POST | `/api/gallery/deduplicate` | No literal | No literal |
| scheduler | User-facing + admin actions | GET | `/api/tasks` | No literal | Found |
| scheduler | User-facing + admin actions | GET | `/api/tasks/notifications` | No literal | Found |
| scheduler | User-facing + admin actions | GET | `/api/tasks/{id}` | No literal | No literal |
| scheduler | User-facing + admin actions | GET | `/api/tasks/{id}/runs` | No literal | No literal |
| scheduler | User-facing + admin actions | GET | `/api/tasks/{id}/destinations` | No literal | No literal |
| scheduler | User-facing + admin actions | GET | `/api/tasks/{id}/deliveries` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks` | No literal | Found |
| scheduler | User-facing + admin actions | POST | `/api/tasks/parse` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/run-due` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/notifications` | No literal | Found |
| scheduler | User-facing + admin actions | POST | `/api/tasks/notifications/{id}/read` | No literal | Found |
| scheduler | User-facing + admin actions | POST | `/api/tasks/{id}/pause` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/{id}/resume` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/{id}/stop` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/{id}/revert` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/{id}/clear-cache` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/{id}/dependencies` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/{id}/webhook` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/{id}/destinations` | No literal | No literal |
| scheduler | User-facing + admin actions | POST | `/api/tasks/{id}/delete` | No literal | No literal |
| scheduler | User-facing + admin actions | PATCH | `/api/tasks/{id}` | No literal | No literal |
| safe_terminal | Advanced user / explicit consent | GET | `/api/terminal/runs` | No literal | Found |
| safe_terminal | Advanced user / explicit consent | GET | `/api/terminal/runs/{id}` | No literal | Found |
| safe_terminal | Advanced user / explicit consent | POST | `/api/terminal/stream` | No literal | Found |
| mcp_runtime | Advanced / admin | GET | `/api/mcp/servers` | No literal | No literal |
| mcp_runtime | Advanced / admin | GET | `/api/mcp/tools` | No literal | No literal |
| mcp_runtime | Advanced / admin | GET | `/api/mcp/permissions` | No literal | No literal |
| mcp_runtime | Advanced / admin | POST | `/api/mcp/servers` | No literal | No literal |
| mcp_runtime | Advanced / admin | POST | `/api/mcp/servers/{id}/reconnect` | No literal | No literal |
| mcp_runtime | Advanced / admin | POST | `/api/mcp/tools/{id}/invoke` | No literal | No literal |
| mcp_runtime | Advanced / admin | PATCH | `/api/mcp/tools/{id}` | No literal | No literal |
| tool_runtime | Internal / operator | GET | `/api/tools/runtime` | No literal | No literal |
| tool_runtime | Internal / operator | POST | `/api/tools/runtime/execute` | No literal | No literal |
| compute_router | Internal / operator | GET | `/api/compute/status` | No literal | Found |
| compute_router | Internal / operator | POST | `/api/compute/plan` | No literal | Found |
| agent_bridge | Internal / operator | GET | `/api/agent-bridge/capabilities` | No literal | No literal |
| agent_bridge | Internal / operator | GET | `/api/agent-bridge/tasks` | No literal | No literal |
| agent_bridge | Internal / operator | POST | `/api/agent-bridge/tasks` | No literal | No literal |
| agent_bridge | Internal / operator | GET | `/api/agent-bridge/memory` | No literal | No literal |
| agent_bridge | Internal / operator | POST | `/api/agent-bridge/memory` | No literal | No literal |
| agent_bridge | Internal / operator | GET | `/api/agent-bridge/calendar/events` | No literal | No literal |
| agent_bridge | Internal / operator | POST | `/api/agent-bridge/calendar/events` | No literal | No literal |
| agent_bridge | Internal / operator | GET | `/api/agent-bridge/documents` | No literal | No literal |
| agent_bridge | Internal / operator | POST | `/api/agent-bridge/documents` | No literal | No literal |
| skills | Advanced user / admin actions | GET | `/api/skills` | No literal | No literal |
| skills | Advanced user / admin actions | GET | `/api/skills/{id}` | No literal | No literal |
| skills | Advanced user / admin actions | GET | `/api/skills/{id}/permissions` | No literal | No literal |
| skills | Advanced user / admin actions | GET | `/api/skills/{id}/dependency-scan` | No literal | No literal |
| skills | Advanced user / admin actions | GET | `/api/skills/marketplace` | No literal | No literal |
| skills | Advanced user / admin actions | GET | `/api/skills/installations` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/import-url` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/{id}/test` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/{id}/package` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/{id}/export` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/{id}/sandbox` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/{id}/publish` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/marketplace/{id}/install` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/marketplace/{id}/remove` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/installations/{id}/update` | No literal | No literal |
| skills | Advanced user / admin actions | POST | `/api/skills/installations/{id}/uninstall` | No literal | No literal |
| conversation-organization | User-facing | GET | `/api/conversations/organized` | Found | Found |
| conversation-organization | User-facing | GET | `/api/conversations/folders` | No literal | No literal |
| conversation-organization | User-facing | GET | `/api/conversations/{id}/branches` | No literal | No literal |
| conversation-organization | User-facing | GET | `/api/conversations/{id}/compactions` | No literal | No literal |
| conversation-organization | User-facing | POST | `/api/conversations/bulk` | Found | No literal |
| conversation-organization | User-facing | POST | `/api/conversations/folders` | No literal | No literal |
| conversation-organization | User-facing | POST | `/api/conversations/{id}/compact` | No literal | No literal |
| conversation-organization | User-facing | PATCH | `/api/conversations/{id}` | Found | Found |
| conversation-organization | User-facing | DELETE | `/api/conversations/{id}` | Found | Found |
| enterprise-identity | Admin-only / security-sensitive | GET | `/api/auth/identity/status` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | GET | `/api/auth/passkeys` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | GET | `/api/auth/devices` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | GET | `/api/auth/federation/policies` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/passkeys/registration/options` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/passkeys/registration/verify` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/passkeys/authentication/options` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/passkeys/authentication/verify` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/recovery/rotate` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/devices/trust` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/devices/{id}/revoke` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/devices/verify` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/federation/policies` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | POST | `/api/auth/federation/evaluate` | No literal | No literal |
| enterprise-identity | Admin-only / security-sensitive | DELETE | `/api/auth/passkeys/{id}` | No literal | No literal |
| science | Advanced user | GET | `/api/science/status` | Found | No literal |
| science | Advanced user | GET | `/api/science/tools` | Found | No literal |
| science | Advanced user | GET | `/api/science/memory` | Found | Found |
| science | Advanced user | GET | `/api/science/experiments` | No literal | Found |
| science | Advanced user | GET | `/api/science/materials/datasets` | Found | Found |
| science | Advanced user | GET | `/api/science/materials/catalog` | Found | Found |
| science | Advanced user | POST | `/api/science/run` | Found | No literal |
| science | Advanced user | POST | `/api/science/multimodal` | No literal | No literal |
| science | Advanced user | POST | `/api/science/experiments` | No literal | Found |
| science | Advanced user | POST | `/api/science/execute` | No literal | No literal |
| science | Advanced user | POST | `/api/science/memory` | Found | Found |
| science | Advanced user | POST | `/api/science/materials/datasets` | Found | Found |
| science | Advanced user | POST | `/api/science/materials/analyse` | No literal | No literal |
| science | Advanced user | POST | `/api/science/materials/catalog/import` | No literal | No literal |
| science | Advanced user | POST | `/api/science/materials/catalog/analyse` | Found | No literal |
| science | Advanced user | POST | `/api/science/operators/deeponet` | No literal | No literal |
| science | Advanced user | POST | `/api/science/engineering/optimize` | No literal | No literal |
| science | Advanced user | POST | `/api/science/verify` | No literal | No literal |
| science | Advanced user | POST | `/api/science/simulate` | No literal | No literal |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/frameworks` | No literal | No literal |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/projects` | Found | Found |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/project-identities` | No literal | No literal |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/projects/{id}` | Found | Found |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/projects/{id}/tree` | Found | Found |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/projects/{id}/file` | Found | Found |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/runs` | Found | Found |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/runs/{id}` | Found | Found |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/runs/{id}/events` | Found | Found |
| codebase | User-facing + destructive/approval controls | GET | `/api/codebase/runs/{id}/events/stream` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/projects` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/folder-picker` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/preflight` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/projects/{id}/scan` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/projects/{id}/file` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/projects/{id}/paths` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/projects/{id}/tools` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/runs` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/runs/{id}/cancel` | Found | No literal |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/runs/{id}/resume` | Found | No literal |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/runs/{id}/tools` | No literal | No literal |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/approvals/{id}/resolve` | Found | Found |
| codebase | User-facing + destructive/approval controls | POST | `/api/codebase/runs/{id}/snapshots/{snapshot_id}/restore` | Found | Found |

## Highest-priority frontend review areas

1. **Project scope**: both frontends commonly use `default`; no shared project selector is safe until backend project/workspace identities are formally mapped.
2. **Execution feedback**: CodeQ and Research have real SSE. Generic agent/workflow records are polled and do not expose a common incremental event stream. The second frontend must show only statuses supported by returned run records.
3. **Communications**: the second frontend now displays Inbox/Sent/Drafts records and persists read state; Inbox polls. Compare provider accounts, approvals, scheduled messages, analytics and dead-letter actions against route rows before exposing them.
4. **Calendar**: the second frontend now filters events by calendar and requests backend free-slot suggestions. Providers, conflicts, reminders, sync and iCalendar actions remain separate review items.
5. **Specialist/admin APIs**: enterprise identity, integrations, worker queues, model lifecycle, backup/restore, adaptive engine and developer/security actions need explicit role/scope review; route existence alone is not a product UI requirement.

## Recommended closure workflow

For each candidate route, inspect the route handler and service, verify its permission/scope behavior, map it to a user workflow, wire it to the intended frontend with honest loading/error/empty/success states, add focused component and backend contract tests, then update the status here. Keep missing backend capabilities in `BACKEND_CAPABILITY_GAPS.md` instead of building simulated UI responses.
