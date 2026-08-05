# Feature Status

OpenAgent is being built in layers. This file tracks what is implemented versus what is currently Completed or planned.

The 2026-07-12 operational closure adds durable research SSE, batch image
tagging, task output delivery records, a signed local skill marketplace with
declarative sandbox runs, signed model endpoint certification, and an enforced
Playwright CI gate. See [LIFECYCLE_COMPLETION.md](LIFECYCLE_COMPLETION.md) for
the runnable routes, tests, and privacy boundaries.

The 2026-07-15 PhD Application Studio adds an owner-scoped applicant profile,
private FTS5/FAISS/Chroma evidence retrieval, a position library, separately
approved public web research and external model boundaries, local multi-model
draft/review/finalization, revisioned deliverables, local authenticity review,
and styled DOCX/PDF/Markdown/ZIP exports. See
[PHD_APPLICATION_STUDIO.md](PHD_APPLICATION_STUDIO.md).

## Implemented

- Local HTTP API
- SQLite storage
- Context events
- Context search and snapshots
- Persistent conversations
- Local chat memory
- Private PhD applicant profile and reusable academic data
- Local PhD source-document vault with extraction, hashing, and deduplication
- PhD position library and professor/lab public research with per-run approval
- Evidence-grounded PhD CV, SOP, motivation letter, cover letter, email, proposal, statement, interview, checklist, and custom generation
- Owner-isolated PhD RAG with selected-document filtering and source lineage
- Local multi-model academic drafting, independent review, and finalization trace
- PhD deliverable revision history, local grounding review, and DOCX/PDF/Markdown/ZIP exports
- Conversation branching
- Conversation search
- Chat export and import
- Conversation replay
- Offline local model provider
- Ollama model adapter
- llama.cpp model adapter
- OpenAI-compatible model adapter
- Model routing policy
- Model fallback policy
- Streaming model responses
- Provider privacy warnings
- Local embedding interface
- Document ingestion
- Semantic chunking
- SQLite FTS knowledge index with fallback search
- SQLite vector adapter for local embeddings
- Source attribution
- Entity extraction
- Relationship graph
- Memory compression
- Context replay
- Automatic document tagging
- Knowledge timeline generation
- Hybrid retrieval
- Source-attributed knowledge synthesis
- Local semantic memory records
- Profile-scoped memory facts
- Semantic memory version records
- Semantic memory consolidation
- Local vector collections and vector records
- Local hybrid reranking
- Local graph analytics snapshots
- PDF ingestion with local text scan
- Image ingestion with local header understanding
- Audio and video metadata ingestion
- Built-in starter agent registry
- Agent permissions
- Agent-to-agent delegation
- Parallel agent execution
- Reflection and verification hooks
- Human approval checkpoints
- Agent run persistence
- Audit event persistence
- Agent marketplace manifests
- Manual workflow definitions
- Local workflow runs
- Workflow definition validation
- Scheduled workflow triggers
- Event-driven workflow triggers
- Conditional workflow logic
- Background workflow worker
- Browser fetch workflow action with network approval
- File read, write, and list workflow actions
- SQLite database read workflow action
- CSV spreadsheet read and write workflow actions
- Report generation workflow action
- Monitoring check and alert workflow actions
- API request workflow action with network approval
- Rollback records and file rollback hooks
- Visual workflow builder schema and graph model
- Drag-and-drop workflow layout editor with audit events
- Email connector manifest
- CalDAV calendar support
- Communication connector manifests for Slack, Discord, Telegram, Signal, Teams, SMS, and WhatsApp
- Contact management
- Unified inbox model
- Message search and summarization
- Follow-up detection
- Drafting and approval-gated sending
- Calendar event storage
- Communication analytics
- Provider-payload communication sync worker
- Provider-backed email IMAP polling worker
- Provider-backed email SMTP sender
- Communication OAuth provider onboarding and authorization URL generation
- Communication OAuth token exchange and refresh behind explicit external approval
- Sanitized local OAuth account records with no token leakage in API responses
- Gmail, Microsoft, and generic IMAP/SMTP mailbox policy normalization
- XOAUTH2 IMAP and SMTP authentication when an OAuth account is configured
- Provider-payload calendar sync worker
- CalDAV HTTP PUT push with auth headers, VCALENDAR payloads, and failed-job reporting
- CalDAV REPORT parsing with sync-token persistence
- CalDAV remote ETag and href mapping
- CalDAV If-Match update checks for pushed local edits
- CalDAV remote delete handling from REPORT responses
- CalDAV recurrence exception preservation through RECURRENCE-ID metadata
- Local model registry
- Model download manager
- Quantization metadata
- Benchmark runner
- Hardware detection
- GPU acceleration adapter registry
- Multi-node inference planning
- Local model marketplace
- Role-based access control
- Opt-in route auth enforcement across the major API route groups
- Authenticated owner scoping for editor documents, drafts, Personal OS records, calendar sync records, and communication records
- Default v2 response envelope with legacy top-level compatibility keys
- Multi-user workspace records
- Organization memory
- Audit exports
- Enterprise admin audit events
- SSO configuration records
- SSO HS256 token validation implementation
- Backup manifests and database snapshots
- Disaster recovery plans
- Policy management and evaluation
- Kubernetes deployment manifests
- GraphQL-style read API
- Local API key records
- Webhook registry and delivery logs
- Plugin manifest registry
- Tool registry and invocation logs
- MCP manifest and server configuration records
- Extension package catalog
- Generated Python SDK artifact
- Agent, workflow, and plugin SDK starter helpers
- Local integration hub with pairing codes, preferences, diagnostics, backups, safe local command checks, and parity matrix
- Personal OS surface for local presets, signatures, versioned drafts, notes/reminders, gallery metadata, response comparison, and export/import
- Encrypted local secret vault implementation
- Secret reveal and rotation controls
- Data retention policies
- Permission-based sharing grants
- Selective egress approval records
- Tamper-evident audit-chain sealing
- Privacy transparency reports
- Sandbox policy profiles
- Static web workspace
- Thin `openagent.server` facade with HTTP runtime under `openagent.routes`
- Modular health/system/ui/module-health route contracts
- Modular communication route contracts
- Modular calendar route contracts
- Service-layer package for auth identity/session/scope/token operations, document lifecycle, calendar sync, communication providers, and model lifecycle operations
- Communication web UI for OAuth account setup, authorization URL generation, token refresh boundary checks, resilient sync runs, and dead-letter review
- Calendar web UI for sync-provider registration, sync-token status, pull/push/bidirectional sync, conflict policy, and conflict review
- Module-script compatibility bridge for advanced dashboard controls that still use inline handlers
- Native Windows desktop shell
- Native desktop workbench with mission modules for workspace, knowledge, assistant, models, voice/media, automation, security/developer, chat, agents, memory, and audit
- No-GUI desktop feature manifest and feature smoke checks
- Windows installer, launcher, and uninstall scripts
- Workspace object model for folders, documents, notes, wiki pages, dashboards, research canvases, whiteboards, mind maps, and templates
- Workspace dashboard and canvas graph APIs
- Workspace template instantiation
- Assistant user profiles
- Assistant tasks and goals
- Meeting preparation and meeting summaries
- Travel planning
- CRM relationship, lead, opportunity, and contact intelligence records
- Daily brief and productivity coaching
- Voice personas
- Voice sessions
- Voice transcript records
- Wake-word and always-listening metadata controls
- Local voiceprint clone records
- Local speech-to-text audio ingestion
- Local TTS WAV generation
- Local translation records
- Command palette
- Global keyboard shortcuts
- Local capability coverage report at `/api/capabilities`
- Executable hard audit at `/api/capabilities/hard-audit`
- Production semantic model governance implementation
- External model privacy policy boundaries
- Production queue worker implementation
- Durable SQLite job queue with leases, retries, cancel/retry controls, and dead letters
- `openagent-worker` CLI for local background execution
- Worker API routes for status, jobs, run-once, cancel, retry, and dead-letter review
- Production policy engine
- Append-only audit anchoring
- Production RAG reranking implementation
- Standalone local vector store with SQLite collections and chunk lineage
- Deterministic CPU-safe local embedding router
- Vector search, upsert, collection, status, and benchmark API routes
- FAISS CPU cosine accelerator rebuilt lazily from authoritative SQLite vectors
- Persistent local ChromaDB mirror with caller-supplied embeddings and disabled anonymized telemetry
- LatticeRAG query expansion, FTS5/FAISS/Chroma fusion, consensus scoring, MMR diversity, context packing, conflict flags, citations, and generation-aware cache invalidation
- Ultra-scale sharded GGUF manifests, bandwidth-aware resident/out-of-core/private-RPC planning, loopback-only llama.cpp lifecycle, warm prompt/KV caches, speculative decoding, complexity routing, and grounded output rejection
- Adaptive heterogeneous model compiler with safe GGUF, SafeTensors, ONNX, and OpenVINO frontends
- Graph, tensor, and kernel-level model IR with phase-specific lowering candidates
- Transfer-aware CPU/GPU/NPU placement with migration hysteresis and separate prefill/decode plans
- Sensitivity-weighted mixed-precision planning with explicit conversion-required state
- Isolated OpenVINO/ONNX accelerator runtime and reproducible Windows setup script
- Verified Intel CPU, Arc GPU, and Intel NPU execution paths with numerical reference checks
- Vendor-backed ONNX/OpenVINO model compilation with local cache and audited evidence
- Persistent pipe-only OpenVINO CPU/GPU/NPU worker with resident compiled models and infer requests
- Native OpenVINO GenAI token streaming with first-token and total-latency telemetry
- Project-scoped model-role profiles for chat, code, reasoning, research, supervisor, draft, vision, and voice
- Audited multi-model sequences and bounded reasoning-code-supervisor revision workflows with explicit external-provider authorization
- Exact-model multi-device differential calibration with measured winner persistence
- Direct local OpenVINO GenAI chat provider and guarded offline/authorized-remote conversion workflow
- Paged KV metadata with shared-prefix copy-on-write and priority eviction
- Deadline-aware continuous microbatch scheduler with power modes
- Local measured device autotuning with numerical correctness gates
- Adaptive CPU inference planner with bounded prompts, task-specific output budgets, core-aware threading, warm sessions, response caching, and first-token telemetry
- RAG and inference status, query, sync, warmup, cache, and benchmark API routes plus structured Knowledge/Developer UI controls
- Standalone contacts CRM API with CRUD, search, import, export, duplicate identity collapse, and message linkage counts
- Local backup archives with manifest, database snapshot, restore dry-run, confirmed restore, and safety-copy behavior
- Physical route handler registry with one domain handler module per major runtime domain
- Auth deployment mode policy with team/enterprise default-on enforcement
- Auth users, roles, policy, and auth audit API surfaces
- Provider certification matrix, local OAuth callback capture, callback history, and attachment ingestion
- Model cookbook hosts, cached model discovery, safe serve planning, adopted process records, logs, and hardware-fit profiles
- Deep document processing for local PDF/text/Office import, previews, text extraction, and ZIP export bundles
- Image Studio local gallery upload, library, rotate, replace, upscale-derivative records, operation history, and audit events
- Unified scheduler with scheduled tasks, pause/resume, due-run processing, notifications, task runs, and worker `tasks.run_due` handler
- Safe terminal API with cwd boundary checks, no-shell execution, output redaction, local durable logs, and buffered stream events
- MCP runtime server registry, reconnect state, tool listing, per-tool toggles, and permission history records
- Agent bridge API for scoped local tasks, memory, calendar events, and documents
- Connector adapter architecture
- Connector polling workers
- Live external provider polling gate
- Deployment/package manifest architecture
- OIDC/JWKS token validation
- Local SDK package publishing artifacts
- Webhook retry queue jobs
- Audited local cryptography envelopes
- Production vector database implementation
- Semantic memory science metrics
- Production OCR, vision, audio, and video metadata analysis path
- Desktop automation action runtime
- Desktop browser automation action runtime
- Mobile app/PWA package manifest
- Desktop app package manifest with native `openagent-desktop` entrypoint
- External/community agent marketplace records
- End-to-end encrypted local envelopes
- Enterprise federation policy hooks
- Sandboxed plugin execution records
- Production GPU model management plans
- Fine-tuning and LoRA job records
- Hardware security key and secure enclave metadata records
- Proactive assistant jobs
- Federated private aggregation
- Multi-device encrypted sync bundles
- Docker skeleton
- Source package manifest including docs, web assets, deployment manifests, and Windows installer assets
- Standard-library tests

## Completed

No items from the current local build rule are incomplete after Stage 15. External hosted services, store distribution, signed Windows installers, and hardware-backed integrations are explicit deployment integrations, not hidden default behavior.

## No Absent Local Items

No current roadmap item is completely absent from the local codebase after Stage 15. Production deployments can still replace local implementations with provider-specific adapters, hardware security modules, signed installers, mobile store builds, and certified cryptography without changing the local-first trust model.

## Build Rule

No feature should be marked implemented until it is:

- Runnable locally
- Covered by tests for critical behavior
- Documented with its privacy boundary
- Auditable when it performs an action
