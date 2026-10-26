# Frontend ↔ Backend Gaps

> Generated after completing the pixel-by-pixel frontend alignment pass.
> **Do not remove existing frontend or backend code** — this file documents
> what needs to be built / wired up so the two sides connect end-to-end.

---

## 1. Chat & Conversations (`HomeChat.tsx`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Message `serverId` for copy/actions | `line.serverId` is set only when backend confirms the message | `POST /api/chat/messages` response must include a stable `id` field |
| "More actions" dropdown on AI messages | Rendered only when `line.serverId` exists and not streaming | Backend message ID is prerequisite |
| Chat inspector — **Files tab** (live mode) | Shows "No workspace connected" empty state in live mode | `GET /api/workspace/items?project=default` — workspace file listing endpoint; connect a CodeQ project to surface files |
| Chat inspector — **Tools tab** | Placeholder tool-call list | Backend must return `tool_calls` or `steps` array on each message |
| Chat inspector — **Context tab** | Shows context-window token count when backend reports it | `GET /api/chat/sessions/:id/context` or include `context_tokens` in message stream |
| Streaming cancel | `AbortController` wired to `POST /api/chat/messages` | Backend must respect `Connection: close` / abort and not persist partial messages |
| Voice message input | UI has mic button in composer | `POST /api/voice/sessions` + MediaRecorder → transcript → insert text into composer |

---

## 2. Agent Directory (`OperationsScreens.tsx`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Agent avatars / icons | Always renders generic `Bot` icon | `GET /api/agents` response should include `icon` or `avatar_url` field |
| Agent status badges | No live status shown on directory cards | Include `status` (idle / running / error) in agent list response |
| Run agent directly from directory | No "Run" button on directory cards | Implement after backend can accept `POST /api/agents/:id/run` |
| Agent capability tags | Description only; no tag chips | Add `capabilities: string[]` to agent schema |

---

## 3. Automations (`OperationsScreens.tsx`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Live automation schedule editing | Configuration tab is read-only | `PATCH /api/automations/:id` with `{ trigger, trigger_type, schedule }` |
| Automation enable/disable toggle | No toggle wired on detail view | `PATCH /api/automations/:id` with `{ enabled: bool }` |
| Real-time run status | Polling every 30 s via React Query | Prefer WebSocket or SSE from `/api/automations/:id/runs/stream` |
| Network-consent live run | UI has checkbox + confirm button | `POST /api/automations/:id/run` accepts `{ dry_run: false, allow_network: bool }` — **already implemented, just needs testing** |

---

## 4. Models & Providers (`OperationsScreens.tsx`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Model download progress | Shows spinner only | `GET /api/local-ai/models/:id/download-status` — return `{ progress_pct, eta_seconds }` |
| Benchmark history | Only latest benchmark shown | `GET /api/local-ai/benchmarks?model_id=:id&limit=20` |
| Provider credential input | No API-key input per provider | `POST /api/models/providers/:id/credentials` accepting `{ api_key }` |
| Live model selection persistence | Stored in `localStorage` only | `POST /api/model-policy/select` with `{ model_id }` to persist across devices |

---

## 5. Calendar (`WorkspaceScreens.tsx`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Event persistence | Demo uses local fixtures; live uses `POST /api/communication/calendar` | Backend must accept `{ title, start_at, end_at, description, attendees }` |
| Recurring events | UI renders single events only | Backend should return `recurrence_rule` (iCal RRULE); frontend can expand |
| External calendar sync | No OAuth flow | `GET /api/communication/oauth/providers` → OAuth → `POST /api/communication/oauth/accounts` — **endpoint exists, flow not wired in Calendar UI** |
| Event reminders | No reminder UI | `notifications` array on event object; send via `POST /api/tasks/notifications` |

---

## 6. Files & Knowledge (`WorkspaceScreens.tsx`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Document versioning restore | UI has "Restore" button per version | `POST /api/library/documents/:id/restore` with `{ version }` — **endpoint wired, needs backend implementation** |
| Full-text preview (live) | `docContent` from `GET /api/library/documents/:id/content` | Ensure backend returns plain-text or markdown body; PDF/DOCX extraction is needed server-side |
| Import (PDF/DOCX/etc.) | `POST /api/library/documents/import` with multipart form | Backend must parse and extract text; return a `doc_id` |
| Shared workspace links | Links tab makes `GET /api/workspace/items?kind=note` | Backend must differentiate `workspace_type: link` vs `note` in the same endpoint |
| AI document summary | Placeholder card shown in demo | `POST /api/library/documents/:id/summarize` → returns `{ summary }` |

---

## 7. Research (`WorkspaceScreens.tsx`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Web search execution | `POST /api/research/search` | Backend needs a search engine integration (SerpAPI / Brave / Tavily) |
| Source citation cards | Research articles reference sources array | `GET /api/research/:id` response should include `sources: [{ url, title, snippet }]` |
| Research-to-chat handoff | "Use in conversation" button present | `POST /api/chat/messages` with `context_doc_ids: [id]` to inject research into chat context |
| Graph view | Placeholder tab in inspector | Backend should return `GET /api/research/:id/graph` with nodes/edges; frontend renders D3 |
| Checkpoint / snapshot | Checkpoint tab is demo-only | `POST /api/research/:id/checkpoint` + `GET /api/research/:id/checkpoints` |

---

## 8. CodeQ (`WorkspaceScreens.tsx`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Project open / folder picker | Uses `window.showDirectoryPicker()` (browser API) | Backend should also expose `POST /api/codeq/projects/open { path }` for non-browser clients |
| Run streaming output | Bottom panel polls `GET /api/codeq/runs/:id/output` | Prefer SSE endpoint `/api/codeq/runs/:id/stream` for real-time lines |
| Preflight check | `POST /api/codeq/preflight` | Must detect framework, package manager, and code-generation contract |
| File save (dirty) | Editor is read-only; save requires `PATCH /api/codeq/projects/:id/files/:path` | Backend needs file-write endpoint; frontend `fileDirty` flag already tracks unsaved changes |
| Terminal / shell panel | No terminal in current UI | Add a `Shell` tab to `codeq-bottom` wired to WebSocket `/api/codeq/projects/:id/shell` |
| Search within project | `POST /api/codeq/projects/:id/search` | Backend must do grep/ripgrep on project tree and return `[{ path, line, preview }]` |

---

## 9. Voice (`AdvancedScreen.tsx` — route: `voice`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Recording upload | `MediaRecorder` captures audio → needs upload | `POST /api/voice/sessions/:id/recordings` (multipart, `audio/webm`) |
| Transcript retrieval | `GET /api/voice/sessions` — fetches list | `GET /api/voice/sessions/:id/transcript` — backend must run ASR (Whisper / cloud) |
| Persona list | Dropdown in Voice settings tab | `GET /api/voice/personas` — returns `[{ id, name, voice_model }]` |
| TTS playback | No playback button on transcripts | `POST /api/voice/tts` with `{ text, persona_id }` → returns audio blob |

---

## 10. Assistant (`AdvancedScreen.tsx` — route: `assistant`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Daily brief generation | `GET /api/assistant/daily-brief` | Backend must aggregate: calendar events + tasks + emails + weather |
| Task creation from brief | "Add to tasks" button in demo | `POST /api/assistant/tasks` with `{ title, goal_id?, due_at? }` |
| Goal progress | Progress bar rendered from `progress` field | Backend must calculate and return `progress_pct` per goal |

---

## 11. Communications (`AdvancedScreen.tsx` — route: `communications`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| OAuth account connect | "Connected accounts" tab has UI shell | Full OAuth flow: `GET /api/communication/oauth/providers` → redirect → `POST /api/communication/oauth/callback` |
| Send email | No compose UI in current build | Add compose dialog wired to `POST /api/communication/send` |
| Draft save | Drafts tab reads `/api/communication/drafts` | `POST /api/communication/drafts` + `PATCH /api/communication/drafts/:id` |
| Approval actions | Approvals listed in Drafts tab | `POST /api/communication/approvals/:id/approve` + `/reject` |
| Real-time inbox | Polls every 15 s | Prefer WebSocket or SSE `/api/communication/inbox/stream` |

---

## 12. Security (`AdvancedScreen.tsx` — route: `security`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Revoke access grant | Access tab lists grants but no revoke button | `DELETE /api/security/shares/:id` |
| Terminate session | Sessions tab lists sessions | `DELETE /api/auth/sessions/:id` |
| Audit log export | No export button | `GET /api/audit?project=default&format=csv` |

---

## 13. Developer (`AdvancedScreen.tsx` — route: `developer`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Create API key | "Issued API keys" card, no create flow | `POST /api/developer/api-keys` with `{ name, scopes[] }` → returns `{ key }` once |
| Revoke API key | No revoke button on key rows | `DELETE /api/developer/api-keys/:id` |
| Register webhook | No create webhook UI | `POST /api/developer/webhooks` with `{ url, events[] }` |
| Redeliver webhook | No redeliver button on delivery rows | `POST /api/developer/webhook-deliveries/:id/redeliver` |
| MCP server add | MCP tab lists servers, no add UI | `POST /api/developer/mcp/servers` with `{ name, command, env }` |

---

## 14. Adaptive Engine (`AdvancedScreen.tsx` — route: `engine`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Launch orchestration | No "Run" button on Jobs tab | `POST /api/adaptive-engine/orchestrations` with `{ plan_id, compute_target }` |
| Hardware utilisation live | Hardware tab is static | SSE or polling on `GET /api/adaptive-engine/hardware/metrics` |
| Plan create/edit | Plans are read-only | `POST /api/adaptive-engine/plans` + `PATCH /api/adaptive-engine/plans/:id` |

---

## 15. Science (`AdvancedScreen.tsx` — route: `science`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Science Projects tab | `null` endpoint — tab shows demo only | `GET /api/science/projects?project=default` |
| Run experiment | Form wired to `POST /api/science/experiments` | Backend must execute the experiment and return `execution_id` |
| Dataset upload | No upload UI on Data tab | `POST /api/science/materials/datasets` (multipart CSV/JSON) |
| Report download | No download button on Reports | `GET /api/science/executions/:id/report?format=pdf` |

---

## 16. Digital Twin (`AdvancedScreen.tsx` — route: `twin`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Profile sync | Profile data from `GET /api/twin/profile` | Backend must persist profile writes: `PATCH /api/twin/profile` |
| Pattern learning | Patterns listed; no approve/reject | `POST /api/twin/patterns/:id/approve` + `/reject` |
| Data deletion | "Data controls" tab has delete button | `DELETE /api/twin/profile` (GDPR-style wipe) |

---

## 17. Vault (`AdvancedScreen.tsx` — route: `vault`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Add secret | No create-secret UI | `POST /api/vault/entries` with `{ name, value, tags[] }` |
| Read / reveal secret | No reveal button | `GET /api/vault/entries/:id/value` — requires auth; return once |
| Delete secret | No delete button on entry rows | `DELETE /api/vault/entries/:id` |

---

## 18. Scraper (`AdvancedScreen.tsx` — route: `scraper`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Extraction form | "New extraction" tab has URL + config form | `POST /api/scraper/jobs` with `{ url, selectors[], output_format }` |
| Job progress polling | Jobs tab shows status | `GET /api/scraper/jobs/:id` — must return `{ status, progress_pct }` |
| Result download | No download button on Results | `GET /api/scraper/history/:id/export?format=json` |

---

## 19. Fine-tune (`AdvancedScreen.tsx` — route: `finetune`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Dataset upload | Datasets tab lists; no upload UI | `POST /api/finetune/datasets` (multipart JSONL) |
| Launch job | Jobs tab, no launch UI | `POST /api/finetune/jobs` with `{ dataset_id, base_model, config }` |
| Checkpoints tab | Endpoint is `null` — shows demo only | `GET /api/finetune/checkpoints?job_id=:id` |
| Adapter deploy | Results tab shows adapters | `POST /api/local-ai/adapters/load` with `{ adapter_id }` |

---

## 20. Settings (`OperationsScreens.tsx`)

| Gap | Frontend state | Backend needed |
|-----|---------------|----------------|
| Notification preference write | Live mode calls `POST /api/integrations/preferences` | Backend must persist per-project preference keys |
| Runtime settings | `POST /api/settings/runtime` | Backend must accept and apply process-level env overrides without restart |
| Backup restore | Calls `POST /api/backups/restore` | Requires elevated auth + safety-copy creation before replacing DB |
| Privacy egress list | `GET /api/security/egress` returns pending requests | Add `POST /api/security/egress/:id/approve` + `/deny` |

---

## Priority order (suggested)

1. **Chat backend** — message IDs, streaming cancel, tool-call metadata  
2. **File import / preview** — unlocks most of Files screen in live mode  
3. **Communication OAuth** — unlocks inbox, calendar sync, send  
4. **Voice ASR pipeline** — recording upload + transcript return  
5. **CodeQ file-write + run SSE** — makes CodeQ actually usable  
6. **API key CRUD** — unblocks Developer screen entirely  
7. **Vault secret CRUD** — straightforward CRUD, high security value  
8. **Calendar event persistence** — needed before comms sync matters  
9. **Twin profile PATCH + pattern approve/reject**  
10. All remaining advanced screens (Science, Engine, Scraper, Finetune)
# Superseded

This older file is retained as historical audit material. Use [`frontend-second/FRONTEND_BACKEND_MASTER_GAPS.md`](frontend-second/FRONTEND_BACKEND_MASTER_GAPS.md) as the current authoritative frontend/backend gap map; its dated source-backed status supersedes claims in this document.

