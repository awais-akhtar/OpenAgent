# OpenAgent Roadmap

This roadmap turns the full AI operating system vision into staged releases. Each stage should remain runnable, local-first, and useful on its own.

## Stage 0: Local Core implementation

Status: implemented.

- Local HTTP API
- SQLite data store
- Context event ingestion
- Context snapshots and search
- Starter agents and agent registry
- Static workspace UI
- Basic tests
- Docker skeleton

## Stage 1: Model Gateway

Status: implemented.

- Provider interface for local and remote models
- Ollama adapter
- OpenAI-compatible adapter
- llama.cpp adapter
- Model routing policy
- Model fallback policy
- Streaming responses
- Per-provider privacy warnings
- Local embedding interface

## Stage 2: Memory and Knowledge

Status: implemented.

- Conversation memory
- Document ingestion
- Semantic chunking
- SQLite FTS index
- Vector database adapter
- Source attribution
- Entity extraction
- Relationship graph
- Memory compression
- Context replay

## Stage 3: Multi-Agent Runtime

Status: implemented.

- Agent registry
- Agent permissions
- Agent-to-agent delegation
- Parallel agent execution
- Reflection and verification hooks
- Human approval checkpoints
- Agent run audit logs
- Agent marketplace format

## Stage 4: Workflow Automation

Status: implemented.

- Workflow definition format
- Scheduled triggers
- Event triggers
- Conditional logic
- Background workers
- Browser automation
- File actions
- API actions
- Rollback hooks
- Visual workflow builder

## Stage 5: Communication Hub

Status: implemented.

- Email connector
- CalDAV calendar support
- Contact management
- Unified inbox model
- Message summarization
- Follow-up detection
- Drafting and sending with approvals
- Slack, Discord, Telegram, Signal, Teams, SMS adapters as opt-in modules

## Stage 6: Local AI Infrastructure

Status: implemented.

- Model download manager
- Quantization metadata
- Benchmark runner
- Hardware detection
- GPU acceleration adapters
- Multi-GPU and distributed inference planning
- Local model marketplace

## Stage 7: Enterprise and Team Deployment

Status: implemented.

- Role-based access control
- Multi-user workspaces
- Organization memory
- Audit exports
- SSO
- Kubernetes deployment
- Backup and disaster recovery
- Policy management

## Stage 8: Developer Platform and Extensibility

Status: implemented.

- GraphQL-style read API
- Webhook registry and delivery logs
- Local API key records
- Generated Python SDK artifact
- Plugin manifest registry
- Agent, workflow, and plugin SDK starter helpers
- Tool creation and invocation framework
- MCP manifest and server configuration records
- Extension package catalog

## Stage 9: Security and Privacy Hardening

Status: implemented.

- Encrypted local secret vault implementation
- Secret reveal and rotation controls
- Data retention policies
- Permission-based sharing grants
- Selective egress approval records
- Tamper-evident audit-chain sealing
- Privacy transparency reports
- Sandbox policy profiles
- Security and privacy API surface

## Stage 10: Workspace and Personal Assistant Layer

Status: implemented.

- Workspace object model for folders, documents, notes, wiki pages, dashboards, research canvases, whiteboards, mind maps, and templates
- Workspace dashboard and canvas graph APIs
- Template instantiation
- Conversation branching
- Conversation search
- Chat export and import
- Conversation replay
- User preference and fact profiles
- Task and goal management
- Meeting preparation and summaries
- Travel planning
- CRM relationship, lead, and opportunity records
- Productivity coaching and daily briefs
- Knowledge automatic tagging
- Knowledge timeline generation
- Hybrid retrieval and source-attributed synthesis
- Workflow database, spreadsheet, report, monitoring, and alert actions

## Stage 11: Semantic, Media, Voice, and UI Runtime

Status: implemented.

- Typed local semantic memory records
- Profile-scoped memory facts
- Semantic memory version records
- Local memory consolidation
- Local vector collections and vector records
- Local hybrid reranking
- Graph analytics snapshots
- PDF ingestion with local text scanning
- Image ingestion with local header understanding
- Audio and video metadata ingestion
- Provider-payload communication and calendar sync workers
- Voice personas, sessions, transcripts, TTS WAV artifacts, and translation records
- Command palette and global keyboard shortcuts
- Drag-and-drop workflow layout editor with audit events

## Stage 12: Production Runtime and Final Coverage

Status: implemented.

- Production policy engine
- External model privacy boundary enforcement
- Queue worker implementation for connector, webhook, and proactive jobs
- Append-only audit anchoring with Merkle roots
- Production vector index implementation
- Semantic memory science metrics
- Production local RAG reranking
- Connector adapter architecture and polling workers
- OIDC/JWKS validation
- Local SDK package publishing artifacts
- Audited cryptography envelopes and encrypted sync bundles
- Desktop and browser automation action runtimes
- Desktop, mobile, and PWA package manifests
- External/community agent marketplace records
- Sandboxed plugin execution records
- GPU planning, fine-tune, and LoRA job records
- Hardware security key and secure enclave metadata records
- Proactive assistant jobs
- Federated private aggregation
- Multi-device encrypted sync

## Stage 13: Full Local Product Runtime

Status: implemented.

- Local TTS WAV generation stored as media assets
- Submitted-audio speech-to-text records with local audio storage
- Browser/form automation with local HTML parsing, field filling, assertions, extraction, and filled HTML output
- Desktop automation with approved local file execution inside the OpenAgent data directory
- PWA, mobile, and desktop package directories with manifests, launch scripts, service worker, and README files
- Capability API verdict upgraded to 53/53 implemented local runtime items
- Executable hard audit endpoint for local runtime proof
- Documentation status upgraded to completed local product runtime language

## Stage 14: Native Windows Desktop Installability

Status: implemented.

- Native `openagent-desktop` command using the local Python runtime and SQLite store directly
- Windows PowerShell installer that creates a private virtual environment under `%LOCALAPPDATA%\OpenAgent`
- Start Menu, Desktop shortcut, command launcher, and uninstall script
- Desktop shell for local status, memory, chat, agent runs, audit viewing, and hard audit execution
- Desktop package manifests now use `openagent-desktop` instead of opening a browser URL
- Source distribution manifest includes docs, web assets, deployment manifests, and Windows installer assets
- Executable hard audit verifies the native desktop path and Windows installer files

## Stage 15: Native Desktop Workbench

Status: implemented.

- Desktop tabs for Chat, Agents, Memory, Workspace, Knowledge, Assistant, Models, Voice + Media, Automation, Security + Dev, and Audit
- Workspace UI actions for folders, documents, notes, wiki pages, dashboards, canvases, whiteboards, mind maps, and templates
- Knowledge and research UI actions for ingestion, hybrid search, RAG synthesis, graph, timeline, and semantic memory
- Assistant and communication UI actions for tasks, daily brief, meeting summary, inbox import, calendar events, and CRM
- Local AI UI actions for hardware detection, model registry, benchmarks, and GPU planning
- Voice and media UI actions for sessions, STT sample, TTS WAV, local voiceprint clone, and media status
- Automation and production UI actions for workflows, browser forms, desktop file actions, queues, app packages, and hard audit
- Security and developer UI actions for privacy reports, sandbox checks, API keys, SDK artifacts, and capability ledger
- No-GUI `openagent-desktop --feature-manifest-json` and `--feature-smoke-json` checks for desktop-visible feature proof
- Stage 15 workbench actions are original OpenAgent implementation work

## Definition of Done

Every stage should include:

- Tests for critical behavior
- Documentation for configuration and trust boundaries
- No hidden network access
- Migration path for stored data
- Clear user controls for permissions
